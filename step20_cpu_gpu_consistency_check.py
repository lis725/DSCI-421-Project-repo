# step20_cpu_gpu_consistency_check.py
import argparse
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from parallel_utils import clean_state_dict
from project_paths import BEST_FUSED_CKPT
from step13_train_fused_params import (
    CapRasterParamDataset,
    MultiHeadResNetWithParams,
    build_param_map,
)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_dataset():
    datasets = []
    for type_name in ["type1", "type2", "type3"]:
        datasets.append(CapRasterParamDataset(type_name, build_param_map(type_name)))
    return ConcatDataset(datasets)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(BEST_FUSED_CKPT))
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this consistency check requires a GPU.")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = load_checkpoint(args.ckpt)
    state = clean_state_dict(ckpt["model"])

    cpu_model = MultiHeadResNetWithParams(in_ch=7, p_dim=4, p_embed=64)
    cpu_model.load_state_dict(state, strict=True)
    cpu_model.eval()

    gpu_model = MultiHeadResNetWithParams(in_ch=7, p_dim=4, p_embed=64).cuda()
    gpu_model.load_state_dict(state, strict=True)
    gpu_model.eval()

    dataset = build_dataset()
    n = min(args.num_samples, len(dataset))
    loader = DataLoader(Subset(dataset, list(range(n))), batch_size=n, shuffle=False, num_workers=0)

    x, y, mask, tid, p, idx, path = next(iter(loader))

    pred_cpu = cpu_model(x, tid, p).cpu()
    pred_gpu = gpu_model(x.cuda(), tid.cuda(), p.cuda()).cpu()

    diff = (pred_cpu - pred_gpu).abs() * mask
    valid = mask.bool()
    max_abs = diff[valid].max().item()
    mean_abs = diff[valid].mean().item()

    print("CPU/GPU consistency check")
    print(f"checkpoint: {args.ckpt}")
    print(f"samples: {n}")
    print(f"sample indices: {[int(i) for i in idx.tolist()]}")
    print(f"max_abs_diff: {max_abs:.10f}")
    print(f"mean_abs_diff: {mean_abs:.10f}")
    print("status: PASS" if max_abs < 1e-4 else "status: REVIEW")


if __name__ == "__main__":
    main()
