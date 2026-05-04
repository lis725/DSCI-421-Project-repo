# step14_infer_export.py
import os
import re
import csv
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from step10_dataset_raster import CapRasterDataset
from project_paths import BEST_FUSED_CKPT, PATTERN_ROOT, SUBMIT_OUT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MAX = 7

CKPT_PATH = str(BEST_FUSED_CKPT)
ROOT = str(PATTERN_ROOT)

TYPE_INFO = {
    "type1": {
        "data_dir": os.path.join(ROOT, "type1_data"),
        "text_path": os.path.join(ROOT, "type1.text"),
        "out_dim": 7,
        "y_names": ["c12", "c13", "c1e", "c1tr", "c1tl", "c1br", "c1bl"],
    },
    "type2": {
        "data_dir": os.path.join(ROOT, "type2_data"),
        "text_path": os.path.join(ROOT, "type2.text"),
        "out_dim": 5,
        "y_names": ["c12", "c1e", "c1t", "c1b", "c1d2"],
    },
    "type3": {
        "data_dir": os.path.join(ROOT, "type3_data"),
        "text_path": os.path.join(ROOT, "type3.text"),
        "out_dim": 5,
        "y_names": ["c12", "c1e", "c1t", "c1b", "c1d2"],
    },
}

# -------------------------
# helpers
# -------------------------
def _extract_idx_from_name(name: str) -> int:
    m = re.search(r"BEM_INPUT_(\d+)_", name)
    if not m:
        raise ValueError(f"Cannot parse idx from filename: {name}")
    return int(m.group(1))

def _read_text_params_only(text_path: str) -> List[List[float]]:
    """
    rows: params4 only
    each row: a b c d | y...
    """
    rows = []
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            left, _ = line.split("|", 1)
            left_nums = [x for x in left.strip().split() if x]
            if len(left_nums) < 4:
                continue
            try:
                p = list(map(float, left_nums[:4]))
            except Exception:
                continue
            rows.append(p)
    return rows

def build_param_map(type_name: str) -> Dict[str, torch.Tensor]:
    info = TYPE_INFO[type_name]
    data_dir = info["data_dir"]
    text_path = info["text_path"]

    files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith(".txt") and fn.startswith("BEM_INPUT_")]
    files = sorted(files, key=_extract_idx_from_name)

    rows = _read_text_params_only(text_path)
    if len(rows) != len(files):
        raise RuntimeError(f"[{type_name}] .text rows ({len(rows)}) != files ({len(files)})")

    mp = {}
    for fn, p in zip(files, rows):
        mp[fn] = torch.tensor(p, dtype=torch.float32)
    return mp

# -------------------------
# dataset wrapper (x, tid) + params
# -------------------------
class CapRasterParamInferDataset(torch.utils.data.Dataset):
    def __init__(self, type_name: str, param_map: Dict[str, torch.Tensor]):
        super().__init__()
        self.base = CapRasterDataset(type_name)
        self.type_name = type_name
        self.param_map = param_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, m, tid, idx, path = self.base[i]
        bn = os.path.basename(path)
        p = self.param_map[bn]
        return x, tid, p, idx, bn, path

# -------------------------
# model (same as step13)
# -------------------------
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

# -------------------------
# inference + export
# -------------------------
@torch.no_grad()
def infer_one_type(type_name: str, model: nn.Module, out_dir: str):
    info = TYPE_INFO[type_name]
    out_dim = info["out_dim"]
    y_names = info["y_names"]

    pm = build_param_map(type_name)
    ds = CapRasterParamInferDataset(type_name, pm)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    records = []  # (idx, basename, pred_list)
    for x, tid, p, idx, bn, path in loader:
        x = x.to(DEVICE)
        tid = tid.to(DEVICE)
        p = p.to(DEVICE)

        pred_pad = model(x, tid, p)              # (B,7)
        pred = pred_pad[:, :out_dim].cpu()       # (B,out_dim)

        for i in range(pred.size(0)):
            records.append((int(idx[i]), bn[i], pred[i].tolist()))

    records.sort(key=lambda z: z[0])

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"pred_{type_name}.csv")
    txt_path = os.path.join(out_dir, f"pred_{type_name}.txt")

    # csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "file"] + y_names)
        for idd, bn, y in records:
            w.writerow([idd, bn] + [f"{v:.8f}" for v in y])

    # txt (one line per sample, only values)
    with open(txt_path, "w", encoding="utf-8") as f:
        for idd, bn, y in records:
            f.write(" ".join(f"{v:.8f}" for v in y) + "\n")

    print(f"[OK] {type_name}: wrote {len(records)} rows ->")
    print(f"     {csv_path}")
    print(f"     {txt_path}")

    return records, y_names

def main():
    model = MultiHeadResNetWithParams(in_ch=7, p_dim=4, p_embed=64).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_dir = str(SUBMIT_OUT)
    all_rows = []

    for tname in ["type1", "type2", "type3"]:
        recs, y_names = infer_one_type(tname, model, out_dir)
        # store merged with type tag
        for idx, bn, y in recs:
            all_rows.append((tname, idx, bn, y))

    # merged csv (方便检查/提交)
    merged_path = os.path.join(out_dir, "pred_all.csv")
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "idx", "file", "y_values"])
        for tname, idx, bn, y in sorted(all_rows, key=lambda z: (z[0], z[1])):
            w.writerow([tname, idx, bn, " ".join(f"{v:.8f}" for v in y)])

    print(f"[OK] merged -> {merged_path}")

if __name__ == "__main__":
    main()
