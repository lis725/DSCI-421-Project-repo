import os
import re
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader

from project_paths import PATTERN_ROOT

ROOT = str(PATTERN_ROOT)

PAT = re.compile(r"BEM_INPUT_(\d+)_\d+\.txt$", re.IGNORECASE)

LABELS = {
    "type1": ["c12", "c13", "c1e", "c1tr", "c1tl", "c1br", "c1bl"],
    "type2": ["c12", "c1e", "c1t", "c1b", "c1d2"],
    "type3": ["c12", "c1e", "c1t", "c1b", "c1d2"],
}
TYPE_ID = {"type1": 0, "type2": 1, "type3": 2}

def load_gt_table(type_name: str):
    """
    returns list of dict rows (1-based index aligned with BEM_INPUT idx)
    """
    path = os.path.join(ROOT, f"{type_name}.text")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    header = lines[0]
    data = lines[1:]

    # parse header: split by "|" into params and caps
    left, right = header.split("|")
    cap_names = right.strip().split()
    cap_names = [c.lower() for c in cap_names]

    rows = []
    for ln in data:
        l, r = ln.split("|")
        cap_vals = r.strip().split()
        d = {}
        for name, val in zip(cap_names, cap_vals):
            d[name] = float(val)
        rows.append(d)
    return rows  # rows[idx-1]

class LabelsOnlyDataset(Dataset):
    def __init__(self, type_name: str):
        self.type_name = type_name
        self.dir = os.path.join(ROOT, f"{type_name}_data")
        self.files = sorted(glob.glob(os.path.join(self.dir, "BEM_INPUT_*.txt")))
        self.gt_rows = load_gt_table(type_name)
        assert len(self.files) == len(self.gt_rows), f"{type_name}: files={len(self.files)} gt_rows={len(self.gt_rows)}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        m = PAT.search(os.path.basename(path))
        idx = int(m.group(1))
        row = self.gt_rows[idx-1]
        y = [row[k] for k in LABELS[self.type_name]]
        return (
            path,
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(TYPE_ID[self.type_name], dtype=torch.long),
            idx,
        )

def main():
    # quick sanity checks
    for t in ["type1", "type2", "type3"]:
        ds = LabelsOnlyDataset(t)
        print(f"{t}: N={len(ds)}  y_dim={len(LABELS[t])}")
        # print a few random samples
        for j in [0, 1, len(ds)-1]:
            path, y, tid, idx = ds[j]
            print(f"  idx={idx:>2} tid={tid.item()} file={os.path.basename(path)} y={y.tolist()}")

    # combined loader (just to confirm batching)
    all_ds = torch.utils.data.ConcatDataset([LabelsOnlyDataset("type1"), LabelsOnlyDataset("type2"), LabelsOnlyDataset("type3")])
    loader = DataLoader(all_ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    paths, ys, tids, idxs = batch
    print("\nBatch check:")
    print("  paths:", [os.path.basename(p) for p in paths])
    print("  ys.shape:", ys.shape)
    print("  tids:", tids.tolist())
    print("  idxs:", idxs.tolist())

if __name__ == "__main__":
    main()

