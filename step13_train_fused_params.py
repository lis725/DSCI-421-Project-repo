# step13_train_fused_params.py
import os
import re
import random
import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import models

from step10_dataset_raster import CapRasterDataset
from project_paths import PATTERN_ROOT
from parallel_utils import clean_state_dict, maybe_wrap_model, state_dict_for_save

D_MAX = 7

ROOT = str(PATTERN_ROOT)

TYPE_INFO = {
    "type1": {
        "data_dir": os.path.join(ROOT, "type1_data"),
        "text_path": os.path.join(ROOT, "type1.text"),
    },
    "type2": {
        "data_dir": os.path.join(ROOT, "type2_data"),
        "text_path": os.path.join(ROOT, "type2.text"),
    },
    "type3": {
        "data_dir": os.path.join(ROOT, "type3_data"),
        "text_path": os.path.join(ROOT, "type3.text"),
    },
}


def _extract_idx_from_name(name: str) -> int:
    m = re.search(r"BEM_INPUT_(\d+)_", name)
    if not m:
        raise ValueError(f"Cannot parse idx from filename: {name}")
    return int(m.group(1))


def _read_text_rows(text_path: str) -> List[Tuple[List[float], List[float]]]:
    rows = []
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            left, right = line.split("|", 1)
            left_nums = [x for x in left.strip().split() if x]
            right_nums = [x for x in right.strip().split() if x]
            if len(left_nums) < 4 or len(right_nums) < 1:
                continue
            try:
                p = list(map(float, left_nums[:4]))
                y = list(map(float, right_nums))
            except Exception:
                continue
            rows.append((p, y))
    return rows


def build_param_map(type_name: str) -> Dict[str, torch.Tensor]:
    info = TYPE_INFO[type_name]
    data_dir = info["data_dir"]
    text_path = info["text_path"]

    files = [
        fn for fn in os.listdir(data_dir)
        if fn.lower().endswith(".txt") and fn.startswith("BEM_INPUT_")
    ]
    files = sorted(files, key=_extract_idx_from_name)

    rows = _read_text_rows(text_path)
    if len(rows) != len(files):
        raise RuntimeError(f"[{type_name}] .text rows ({len(rows)}) != files ({len(files)}).")

    mp: Dict[str, torch.Tensor] = {}
    for fn, (p, _) in zip(files, rows):
        mp[fn] = torch.tensor(p, dtype=torch.float32)
    return mp


class CapRasterParamDataset(torch.utils.data.Dataset):
    def __init__(self, type_name: str, param_map: Dict[str, torch.Tensor]):
        super().__init__()
        self.type_name = type_name
        self.base = CapRasterDataset(type_name)
        self.param_map = param_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, m, tid, idx, path = self.base[i]
        bn = os.path.basename(path)
        if bn not in self.param_map:
            raise KeyError(f"[{self.type_name}] param not found for {bn}")
        p = self.param_map[bn]
        return x, y, m, tid, p, idx, path


class MultiHeadResNetWithParams(nn.Module):
    def __init__(self, in_ch=7, p_dim=4, p_embed=64):
        super().__init__()
        self.backbone = models.resnet18(weights=None)

        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_ch,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.p_mlp = nn.Sequential(
            nn.Linear(p_dim, p_embed),
            nn.ReLU(inplace=True),
            nn.Linear(p_embed, p_embed),
            nn.ReLU(inplace=True),
        )

        fused_dim = feat_dim + p_embed
        self.head1 = nn.Linear(fused_dim, 7)
        self.head2 = nn.Linear(fused_dim, 5)
        self.head3 = nn.Linear(fused_dim, 5)

    def forward(self, x, tid, p):
        feat = self.backbone(x)
        pfeat = self.p_mlp(p)
        fused = torch.cat([feat, pfeat], dim=1)

        out = x.new_zeros((x.size(0), D_MAX))
        for t in (0, 1, 2):
            mt = (tid == t)
            if mt.any():
                f = fused[mt]
                if t == 0:
                    out[mt, :7] = self.head1(f)
                elif t == 1:
                    out[mt, :5] = self.head2(f)
                else:
                    out[mt, :5] = self.head3(f)
        return out


def masked_mse(pred, y, mask):
    diff2 = (pred - y) ** 2
    diff2 = diff2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    mae_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    mae_cnt = {0: 0.0, 1: 0.0, 2: 0.0}

    for x, y, m, tid, p, idx, path in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        tid = tid.to(device)
        p = p.to(device)

        pred = model(x, tid, p)
        loss = masked_mse(pred, y, m)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        abs_err = (pred - y).abs() * m
        for t in (0, 1, 2):
            mt = (tid == t)
            if mt.any():
                mae_sum[t] += abs_err[mt].sum().item()
                mae_cnt[t] += m[mt].sum().item()

    avg_loss = total_loss / max(1, total_count)
    mae = {t: (mae_sum[t] / max(1.0, mae_cnt[t])) for t in (0, 1, 2)}
    return avg_loss, mae


def stratified_split_indices(tids: List[int], val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    by_type = {0: [], 1: [], 2: []}
    for i, t in enumerate(tids):
        by_type[int(t)].append(i)

    train_idx, val_idx = [], []
    for t in (0, 1, 2):
        idxs = by_type[t]
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_ratio))
        n_val = max(1, min(n_val, len(idxs) - 1))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_datasets():
    print("[Step13] building param maps...")
    pm1 = build_param_map("type1")
    pm2 = build_param_map("type2")
    pm3 = build_param_map("type3")

    ds1 = CapRasterParamDataset("type1", pm1)
    ds2 = CapRasterParamDataset("type2", pm2)
    ds3 = CapRasterParamDataset("type3", pm3)
    ds_all = ConcatDataset([ds1, ds2, ds3])

    tids = []
    for i in range(len(ds_all)):
        _, _, _, tid, _, _, _ = ds_all[i]
        tids.append(int(tid))

    train_idx, val_idx = stratified_split_indices(tids, val_ratio=0.2, seed=42)

    def _count(idxs):
        c = {0: 0, 1: 0, 2: 0}
        for i in idxs:
            c[tids[i]] += 1
        return c

    print(f"Total: {len(ds_all)} Train: {len(train_idx)} Val: {len(val_idx)}")
    print("Train type counts:", _count(train_idx))
    print("Val   type counts:", _count(val_idx))

    return Subset(ds_all, train_idx), Subset(ds_all, val_idx)


def run_training(device, train_ds, val_ds, epochs, batch_size, num_workers, ckpt_dir, multi_gpu=False):
    torch.manual_seed(42)
    random.seed(42)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("=" * 70)
    print(f"[Run] device={device}")
    if device == "cuda":
        print(f"[Run] GPU: {torch.cuda.get_device_name(0)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    model = maybe_wrap_model(MultiHeadResNetWithParams(in_ch=7, p_dim=4, p_embed=64), device, multi_gpu=multi_gpu)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    patience = 12
    bad = 0

    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running, seen = 0.0, 0

        for x, y, m, tid, p, idx, path in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            tid = tid.to(device)
            p = p.to(device)

            pred = model(x, tid, p)
            loss = masked_mse(pred, y, m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(1, seen)
        val_loss, val_mae = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        if device == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start
        lr = opt.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | time={epoch_time:.2f}s | lr={lr:.2e} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| val_mae t1={val_mae[0]:.6f} t2={val_mae[1]:.6f} t3={val_mae[2]:.6f}"
        )

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            bad = 0

            device_ckpt = os.path.join(ckpt_dir, f"best_fused_params_{device}.pt")
            torch.save({"model": state_dict_for_save(model)}, device_ckpt)
            print(f"  saved: {device_ckpt}")

            if device == "cuda":
                main_ckpt = os.path.join(ckpt_dir, "best_fused_params.pt")
                torch.save({"model": state_dict_for_save(model)}, main_ckpt)
                print(f"  saved: {main_ckpt}")
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs. best_val={best_val:.6f}")
                break

    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - total_start
    print(
        f"[Done] device={device} epochs_run={epoch} total_train_time={total_time:.2f}s "
        f"best_epoch={best_epoch} best_val={best_val:.6f}"
    )

    return {
        "device": device,
        "epochs_run": epoch,
        "total_train_time": total_time,
        "best_epoch": best_epoch,
        "best_val": best_val,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="both", choices=["cuda", "cpu", "both"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--multi_gpu", action="store_true", help="Use nn.DataParallel when multiple CUDA GPUs are visible.")
    args = parser.parse_args()

    if args.device in ("cuda", "both") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    print(f"[Config] ROOT={ROOT}")
    print(f"[Config] requested_device={args.device}")
    print(f"[Config] epochs={args.epochs} batch_size={args.batch_size}")

    train_ds, val_ds = build_datasets()

    devices = ["cuda", "cpu"] if args.device == "both" else [args.device]
    results = []

    for device in devices:
        result = run_training(
            device=device,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ckpt_dir=args.ckpt_dir,
            multi_gpu=args.multi_gpu,
        )
        results.append(result)

    print("=" * 70)
    print("[Summary]")
    for r in results:
        print(
            f"{r['device']:>4s}: time={r['total_train_time']:.2f}s "
            f"epochs={r['epochs_run']} best_val={r['best_val']:.6f}"
        )

    by_device = {r["device"]: r for r in results}
    if "cpu" in by_device and "cuda" in by_device:
        speedup = by_device["cpu"]["total_train_time"] / by_device["cuda"]["total_train_time"]
        print(f"[Speedup] CPU/GPU = {speedup:.2f}x")


if __name__ == "__main__":
    main()



