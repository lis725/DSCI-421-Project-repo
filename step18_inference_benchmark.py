# step18_inference_benchmark.py
import os
import re
import time
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

import step10_dataset_raster as raster_mod
from step10_dataset_raster import CapRasterDataset
from project_paths import BEST_FUSED_CKPT, PATTERN_ROOT
from parallel_utils import clean_state_dict, maybe_wrap_model, state_dict_for_save

D_MAX = 7
DEFAULT_ROOT = str(PATTERN_ROOT)
DEFAULT_CKPT = str(BEST_FUSED_CKPT)


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


def materialize_dataset(root):
    set_root(root)

    xs, tids, ps, idxs = [], [], [], []

    for type_name in ["type1", "type2", "type3"]:
        pm = build_param_map(root, type_name)
        ds = CapRasterDataset(type_name)

        for i in range(len(ds)):
            x, y, m, tid, idx, path = ds[i]
            bn = os.path.basename(path)

            xs.append(x)
            tids.append(tid)
            ps.append(pm[bn])
            idxs.append(torch.tensor(idx, dtype=torch.long))

    return TensorDataset(
        torch.stack(xs).contiguous(),
        torch.stack(tids).contiguous(),
        torch.stack(ps).contiguous(),
        torch.stack(idxs).contiguous(),
    )


@torch.no_grad()
def benchmark(model, loader, device, warmup, repeats, use_amp, multi_gpu=False):
    model = maybe_wrap_model(model, device, multi_gpu=multi_gpu)
    model.eval()

    for _ in range(warmup):
        for x, tid, p, idx in loader:
            x = x.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                _ = model(x, tid, p)
        if device == "cuda":
            torch.cuda.synchronize()

    total_samples = 0
    start = time.perf_counter()

    for _ in range(repeats):
        for x, tid, p, idx in loader:
            x = x.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                _ = model(x, tid, p)
            total_samples += x.size(0)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = total_samples / elapsed

    return elapsed, throughput, total_samples


def load_model(ckpt_path):
    model = MultiHeadResNetWithParams()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--multi_gpu", action="store_true", help="Benchmark inference with nn.DataParallel when multiple CUDA GPUs are visible.")
    args = parser.parse_args()

    print(f"[Config] root={args.root}")
    print(f"[Config] ckpt={args.ckpt}")
    print(f"[Config] batch_size={args.batch_size}")
    print(f"[Config] repeats={args.repeats}")

    print("[Build] materializing inference tensors")
    dataset = materialize_dataset(args.root)
    loader_cpu = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loader_gpu = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    results = []

    model_cpu = load_model(args.ckpt)
    cpu_time, cpu_tput, cpu_samples = benchmark(
        model_cpu, loader_cpu, "cpu", args.warmup, args.repeats, use_amp=False, multi_gpu=False
    )
    results.append(("cpu", "fp32", cpu_time, cpu_tput, cpu_samples))

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

        model_gpu = load_model(args.ckpt)
        gpu_time, gpu_tput, gpu_samples = benchmark(
            model_gpu, loader_gpu, "cuda", args.warmup, args.repeats, use_amp=False, multi_gpu=False
        )
        results.append(("cuda", "fp32", gpu_time, gpu_tput, gpu_samples))

        model_amp = load_model(args.ckpt)
        amp_time, amp_tput, amp_samples = benchmark(
            model_amp, loader_gpu, "cuda", args.warmup, args.repeats, use_amp=True, multi_gpu=False
        )
        results.append(("cuda", "amp", amp_time, amp_tput, amp_samples))

        if args.multi_gpu and torch.cuda.device_count() > 1:
            model_multi = load_model(args.ckpt)
            multi_time, multi_tput, multi_samples = benchmark(
                model_multi, loader_gpu, "cuda", args.warmup, args.repeats, use_amp=False, multi_gpu=True
            )
            results.append(("cuda:all", "fp32", multi_time, multi_tput, multi_samples))

    print("=" * 80)
    print("[Inference Benchmark Summary]")
    for device, mode, elapsed, throughput, samples in results:
        print(
            f"{device:>4s} {mode:>4s}: "
            f"time={elapsed:.4f}s samples={samples} throughput={throughput:.2f} samples/s"
        )

    by_key = {(d, m): (t, th) for d, m, t, th, s in results}
    if ("cpu", "fp32") in by_key and ("cuda", "fp32") in by_key:
        speedup = by_key[("cpu", "fp32")][0] / by_key[("cuda", "fp32")][0]
        print(f"[Speedup] GPU FP32 vs CPU FP32 = {speedup:.2f}x")

    if ("cpu", "fp32") in by_key and ("cuda", "amp") in by_key:
        speedup = by_key[("cpu", "fp32")][0] / by_key[("cuda", "amp")][0]
        print(f"[Speedup] GPU AMP vs CPU FP32 = {speedup:.2f}x")

    if ("cpu", "fp32") in by_key and ("cuda:all", "fp32") in by_key:
        speedup = by_key[("cpu", "fp32")][0] / by_key[("cuda:all", "fp32")][0]
        print(f"[Speedup] Multi-GPU FP32 vs CPU FP32 = {speedup:.2f}x")

    if ("cuda", "fp32") in by_key and ("cuda:all", "fp32") in by_key:
        speedup = by_key[("cuda", "fp32")][0] / by_key[("cuda:all", "fp32")][0]
        print(f"[Speedup] Multi-GPU FP32 vs single-GPU FP32 = {speedup:.2f}x")


if __name__ == "__main__":
    main()




