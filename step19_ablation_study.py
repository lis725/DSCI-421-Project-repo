# step19_ablation_study.py
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


def set_root(root):
    raster_mod.ROOT = os.path.abspath(root)


def extract_idx(name):
    m = re.search(r"BEM_INPUT_(\d+)_", name)
    if not m:
        raise ValueError(f"Cannot parse idx from filename: {name}")
    return int(m.group(1))


def read_params(text_path):
    rows = []
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            left, _ = line.split("|", 1)
            nums = left.strip().split()
            if len(nums) < 4:
                continue
            try:
                rows.append(list(map(float, nums[:4])))
            except Exception:
                continue
    return rows


def build_param_map(root, type_name) -> Dict[str, torch.Tensor]:
    data_dir = os.path.join(root, f"{type_name}_data")
    text_path = os.path.join(root, f"{type_name}.text")

    files = [
        fn for fn in os.listdir(data_dir)
        if fn.lower().endswith(".txt") and fn.startswith("BEM_INPUT_")
    ]
    files = sorted(files, key=extract_idx)

    rows = read_params(text_path)
    if len(files) != len(rows):
        raise RuntimeError(f"{type_name}: files={len(files)} rows={len(rows)} mismatch")

    return {fn: torch.tensor(p, dtype=torch.float32) for fn, p in zip(files, rows)}


class ParamDataset(torch.utils.data.Dataset):
    def __init__(self, type_name, param_map):
        self.base = CapRasterDataset(type_name)
        self.param_map = param_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, m, tid, idx, path = self.base[i]
        p = self.param_map[os.path.basename(path)]
        return x, y, m, tid, p, idx


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


def build_cached_dataset(root):
    set_root(root)

    datasets = []
    for type_name in ["type1", "type2", "type3"]:
        pm = build_param_map(root, type_name)
        datasets.append(ParamDataset(type_name, pm))

    ds_all = ConcatDataset(datasets)

    xs, ys, ms, tids, ps, idxs = [], [], [], [], [], []

    for i in range(len(ds_all)):
        x, y, m, tid, p, idx = ds_all[i]
        xs.append(x)
        ys.append(y)
        ms.append(m)
        tids.append(tid)
        ps.append(p)
        idxs.append(torch.tensor(idx, dtype=torch.long))

    tids_int = [int(t) for t in tids]
    train_idx, val_idx = stratified_split_indices(tids_int, val_ratio=0.2, seed=42)

    cached = TensorDataset(
        torch.stack(xs).contiguous(),
        torch.stack(ys).contiguous(),
        torch.stack(ms).contiguous(),
        torch.stack(tids).contiguous(),
        torch.stack(ps).contiguous(),
        torch.stack(idxs).contiguous(),
    )

    return cached, train_idx, val_idx


class ParamsOnlyModel(nn.Module):
    def __init__(self, p_dim=4, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(p_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.head1 = nn.Linear(hidden, 7)
        self.head2 = nn.Linear(hidden, 5)
        self.head3 = nn.Linear(hidden, 5)

    def forward(self, x, tid, p):
        feat = self.shared(p)
        out = torch.zeros((p.size(0), D_MAX), device=p.device, dtype=torch.float32)

        for t in (0, 1, 2):
            mt = tid == t
            if mt.any():
                f = feat[mt]
                if t == 0:
                    out[mt, :7] = self.head1(f)
                elif t == 1:
                    out[mt, :5] = self.head2(f)
                else:
                    out[mt, :5] = self.head3(f)

        return out


class RasterOnlyModel(nn.Module):
    def __init__(self, in_ch=7):
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

        self.head1 = nn.Linear(feat_dim, 7)
        self.head2 = nn.Linear(feat_dim, 5)
        self.head3 = nn.Linear(feat_dim, 5)

    def forward(self, x, tid, p):
        feat = self.backbone(x)
        out = torch.zeros((x.size(0), D_MAX), device=x.device, dtype=torch.float32)

        for t in (0, 1, 2):
            mt = tid == t
            if mt.any():
                f = feat[mt]
                if t == 0:
                    out[mt, :7] = self.head1(f)
                elif t == 1:
                    out[mt, :5] = self.head2(f)
                else:
                    out[mt, :5] = self.head3(f)

        return out


class RasterParamsModel(nn.Module):
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
                    out[mt, :7] = self.head1(f)
                elif t == 1:
                    out[mt, :5] = self.head2(f)
                else:
                    out[mt, :5] = self.head3(f)

        return out


def masked_mse(pred, y, mask):
    y = y.float()
    mask = mask.float()

    diff2 = (pred.float() - y) ** 2
    diff2 = diff2 * mask

    return diff2.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()

    total_loss, total_count = 0.0, 0
    mae_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    mae_cnt = {0: 0.0, 1: 0.0, 2: 0.0}

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

        abs_err = (pred - y).abs() * m

        for t in (0, 1, 2):
            mt = tid == t
            if mt.any():
                mae_sum[t] += abs_err[mt].sum().item()
                mae_cnt[t] += m[mt].sum().item()

    avg_loss = total_loss / max(1, total_count)
    mae = {t: mae_sum[t] / max(1.0, mae_cnt[t]) for t in (0, 1, 2)}

    return avg_loss, mae


def train_one(name, model, train_loader, val_loader, device, epochs, multi_gpu=False):
    torch.manual_seed(42)
    random.seed(42)

    model = maybe_wrap_model(model, device, multi_gpu=multi_gpu)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val = float("inf")
    best_mae = None
    start = time.perf_counter()

    print("=" * 80)
    print(f"[Ablation] {name}")

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

        if val_loss < best_val:
            best_val = val_loss
            best_mae = val_mae

        if device == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:03d} | time={epoch_time:.2f}s | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| val_mae t1={val_mae[0]:.6f} t2={val_mae[1]:.6f} t3={val_mae[2]:.6f}"
        )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        "name": name,
        "time": elapsed,
        "best_val": best_val,
        "mae_t1": best_mae[0],
        "mae_t2": best_mae[1],
        "mae_t3": best_mae[2],
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
    print(f"[Config] batch_size={args.batch_size}")
    print(f"[Config] num_workers={args.num_workers}")

    if args.device == "cuda":
        print(f"[Config] GPU={torch.cuda.get_device_name(0)}")

    print("[Build] cached dataset")
    dataset, train_idx, val_idx = build_cached_dataset(args.root)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    experiments = [
        ("params_only", ParamsOnlyModel()),
        ("raster_only", RasterOnlyModel()),
        ("raster_plus_params", RasterParamsModel()),
    ]

    results = []

    for name, model in experiments:
        result = train_one(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=args.epochs,
            multi_gpu=args.multi_gpu,
        )
        results.append(result)

    print("=" * 80)
    print("[Ablation Summary]")

    for r in results:
        print(
            f"{r['name']}: "
            f"time={r['time']:.2f}s "
            f"best_val={r['best_val']:.6f} "
            f"mae_t1={r['mae_t1']:.6f} "
            f"mae_t2={r['mae_t2']:.6f} "
            f"mae_t3={r['mae_t3']:.6f}"
        )


if __name__ == "__main__":
    main()



