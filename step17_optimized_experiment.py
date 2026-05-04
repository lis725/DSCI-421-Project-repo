# step17_optimized_experiment.py
import os
import re
import time
import random
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset
from torchvision import models

import step10_dataset_raster as raster_mod
from step10_dataset_raster import CapRasterDataset
from project_paths import PATTERN_ROOT
from parallel_utils import clean_state_dict, maybe_wrap_model, state_dict_for_save

D_MAX = 7
DEFAULT_ROOT = str(PATTERN_ROOT)


def set_root(root: str):
    raster_mod.ROOT = os.path.abspath(root)


def extract_idx(name: str) -> int:
    m = re.search(r"BEM_INPUT_(\d+)_", name)
    if not m:
        raise ValueError(f"Cannot parse idx from filename: {name}")
    return int(m.group(1))


def read_text_rows(text_path: str):
    rows = []
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            left, right = line.split("|", 1)
            left_nums = left.strip().split()
            right_nums = right.strip().split()
            if len(left_nums) < 4 or len(right_nums) < 1:
                continue
            try:
                p = list(map(float, left_nums[:4]))
            except Exception:
                continue
            rows.append(p)
    return rows


def build_param_map(root: str, type_name: str) -> Dict[str, torch.Tensor]:
    data_dir = os.path.join(root, f"{type_name}_data")
    text_path = os.path.join(root, f"{type_name}.text")
    files = [
        fn for fn in os.listdir(data_dir)
        if fn.lower().endswith(".txt") and fn.startswith("BEM_INPUT_")
    ]
    files = sorted(files, key=extract_idx)
    rows = read_text_rows(text_path)
    if len(files) != len(rows):
        raise RuntimeError(f"{type_name}: files={len(files)} rows={len(rows)} mismatch")
    return {fn: torch.tensor(p, dtype=torch.float32) for fn, p in zip(files, rows)}


class ParamDataset(torch.utils.data.Dataset):
    def __init__(self, type_name: str, param_map: Dict[str, torch.Tensor]):
        self.base = CapRasterDataset(type_name)
        self.type_name = type_name
        self.param_map = param_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, m, tid, idx, path = self.base[i]
        bn = os.path.basename(path)
        p = self.param_map[bn]
        return x, y, m, tid, p, idx


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

        out = torch.zeros((x.size(0), D_MAX), device=x.device, dtype=torch.float32)
        for t in (0, 1, 2):
            mt = tid == t
            if mt.any():
                f = fused[mt]
                if t == 0:
                    out[mt, :7] = self.head1(f).float()
                elif t == 1:
                    out[mt, :5] = self.head2(f).float()
                else:
                    out[mt, :5] = self.head3(f).float()
        return out


def masked_mse(pred, y, mask):
    diff2 = (pred - y) ** 2
    diff2 = diff2 * mask
    return diff2.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    for x, y, m, tid, p, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        tid = tid.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)

        pred = model(x, tid, p)
        loss = masked_mse(pred, y, m)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    return total_loss / max(1, total_count)


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


def build_baseline_dataset(root: str):
    set_root(root)
    pm1 = build_param_map(root, "type1")
    pm2 = build_param_map(root, "type2")
    pm3 = build_param_map(root, "type3")

    ds1 = ParamDataset("type1", pm1)
    ds2 = ParamDataset("type2", pm2)
    ds3 = ParamDataset("type3", pm3)
    ds_all = ConcatDataset([ds1, ds2, ds3])

    tids = []
    for i in range(len(ds_all)):
        _, _, _, tid, _, _ = ds_all[i]
        tids.append(int(tid))

    train_idx, val_idx = stratified_split_indices(tids, val_ratio=0.2, seed=42)
    return ds_all, train_idx, val_idx


def materialize_to_memory(ds_all):
    start = time.perf_counter()
    xs, ys, ms, tids, ps, idxs = [], [], [], [], [], []

    for i in range(len(ds_all)):
        x, y, m, tid, p, idx = ds_all[i]
        xs.append(x)
        ys.append(y)
        ms.append(m)
        tids.append(tid)
        ps.append(p)
        idxs.append(torch.tensor(idx, dtype=torch.long))

    cache_time = time.perf_counter() - start

    cached = TensorDataset(
        torch.stack(xs, dim=0).contiguous(),
        torch.stack(ys, dim=0).contiguous(),
        torch.stack(ms, dim=0).contiguous(),
        torch.stack(tids, dim=0).contiguous(),
        torch.stack(ps, dim=0).contiguous(),
        torch.stack(idxs, dim=0).contiguous(),
    )
    return cached, cache_time


def make_loaders(dataset, train_idx, val_idx, batch_size, num_workers, pin_memory):
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_scenario(name, dataset, train_idx, val_idx, device, epochs, batch_size, num_workers, use_amp, multi_gpu=False):
    torch.manual_seed(42)
    random.seed(42)

    pin_memory = device == "cuda"
    train_loader, val_loader = make_loaders(
        dataset, train_idx, val_idx, batch_size, num_workers, pin_memory
    )

    model = maybe_wrap_model(MultiHeadResNetWithParams(), device, multi_gpu=multi_gpu)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    best_val = float("inf")
    total_start = time.perf_counter()

    print("=" * 80)
    print(f"[Scenario] {name} | device={device} | amp={use_amp}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running, seen = 0.0, 0

        for x, y, m, tid, p, idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                pred = model(x, tid, p)
                loss = masked_mse(pred, y, m)

            if use_amp and device == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(1, seen)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        best_val = min(best_val, val_loss)

        if device == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch:03d} | time={epoch_time:.2f}s | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - total_start
    print(f"[Done] {name}: total_time={total_time:.2f}s best_val={best_val:.6f}")

    return {
        "name": name,
        "time": total_time,
        "best_val": best_val,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true", help="Use nn.DataParallel when multiple CUDA GPUs are visible.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is not available.")

    print(f"[Config] root={args.root}")
    print(f"[Config] device={args.device}")
    print(f"[Config] epochs={args.epochs}")
    if args.device == "cuda":
        print(f"[Config] GPU={torch.cuda.get_device_name(0)}")

    print("[Build] baseline on-the-fly dataset")
    baseline_ds, train_idx, val_idx = build_baseline_dataset(args.root)

    print("[Build] optimized in-memory cached dataset")
    cached_ds, cache_time = materialize_to_memory(baseline_ds)
    print(f"[Cache] materialization_time={cache_time:.2f}s samples={len(cached_ds)}")

    results = []

    results.append(
        train_scenario(
            name="baseline_on_the_fly",
            dataset=baseline_ds,
            train_idx=train_idx,
            val_idx=val_idx,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=False,
            multi_gpu=args.multi_gpu,
        )
    )

    results.append(
        train_scenario(
            name="optimized_cached_memory",
            dataset=cached_ds,
            train_idx=train_idx,
            val_idx=val_idx,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=False,
            multi_gpu=args.multi_gpu,
        )
    )

    if args.device == "cuda":
        results.append(
            train_scenario(
                name="optimized_cached_memory_amp",
                dataset=cached_ds,
                train_idx=train_idx,
                val_idx=val_idx,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_amp=True,
                multi_gpu=args.multi_gpu,
            )
        )

    print("=" * 80)
    print("[Optimization Summary]")
    base_time = results[0]["time"]

    for r in results:
        speedup = base_time / r["time"]
        print(
            f"{r['name']}: time={r['time']:.2f}s "
            f"speedup_vs_baseline={speedup:.2f}x "
            f"best_val={r['best_val']:.6f}"
        )

    print(f"[Cache preprocessing time] {cache_time:.2f}s")
    print("[Note] cached speedup excludes one-time preprocessing unless stated otherwise.")


if __name__ == "__main__":
    main()



