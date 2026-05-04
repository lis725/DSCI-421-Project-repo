import os
import re
import glob
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from project_paths import PATTERN_ROOT

ROOT = str(PATTERN_ROOT)
PAT = re.compile(r"BEM_INPUT_(\d+)_\d+\.txt$", re.IGNORECASE)

LABELS = {
    "type1": ["c12", "c13", "c1e", "c1tr", "c1tl", "c1br", "c1bl"],  # 7
    "type2": ["c12", "c1e", "c1t", "c1b", "c1d2"],                   # 5
    "type3": ["c12", "c1e", "c1t", "c1b", "c1d2"],                   # 5
}
TYPE_ID = {"type1": 0, "type2": 1, "type3": 2}
D_MAX = 7

def load_gt_table(type_name: str):
    path = os.path.join(ROOT, f"{type_name}.text")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    header = lines[0]
    data = lines[1:]
    _, right = header.split("|")
    cap_names = [c.lower() for c in right.strip().split()]

    rows = []
    for ln in data:
        _, r = ln.split("|")
        vals = r.strip().split()
        d = {name: float(val) for name, val in zip(cap_names, vals)}
        rows.append(d)
    return rows

class LabelsOnlyDataset(Dataset):
    def __init__(self, type_name: str):
        self.type_name = type_name
        self.type_id = TYPE_ID[type_name]
        self.dir = os.path.join(ROOT, f"{type_name}_data")
        self.files = sorted(glob.glob(os.path.join(self.dir, "BEM_INPUT_*.txt")))
        self.gt_rows = load_gt_table(type_name)
        assert len(self.files) == len(self.gt_rows)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        m = PAT.search(os.path.basename(path))
        idx = int(m.group(1))
        row = self.gt_rows[idx-1]
        y = torch.tensor([row[k] for k in LABELS[self.type_name]], dtype=torch.float32)
        return path, y, torch.tensor(self.type_id, dtype=torch.long), idx

def mixed_collate(batch):
    """
    batch: list of (path, y(5or7), type_id, idx)
    returns:
      paths: list[str]
      y_pad: (B, 7)
      mask : (B, 7)  1 for valid dims
      type_id: (B,)
      idx: (B,)
    """
    paths, ys, tids, idxs = zip(*batch)
    B = len(batch)
    y_pad = torch.zeros(B, D_MAX, dtype=torch.float32)
    mask  = torch.zeros(B, D_MAX, dtype=torch.float32)

    for i, y in enumerate(ys):
        d = y.numel()
        y_pad[i, :d] = y
        mask[i, :d] = 1.0

    type_id = torch.stack(tids, dim=0)
    idx = torch.tensor(idxs, dtype=torch.long)
    return list(paths), y_pad, mask, type_id, idx

def main():
    ds = ConcatDataset([LabelsOnlyDataset("type1"), LabelsOnlyDataset("type2"), LabelsOnlyDataset("type3")])
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=mixed_collate)
    paths, y_pad, mask, type_id, idx = next(iter(loader))

    print("Batch OK")
    print("  y_pad.shape:", y_pad.shape)
    print("  mask.shape :", mask.shape)
    print("  type_id   :", type_id.tolist())
    print("  idx       :", idx.tolist())
    print("  first y_pad:", y_pad[0].tolist())
    print("  first mask :", mask[0].tolist())

if __name__ == "__main__":
    main()

