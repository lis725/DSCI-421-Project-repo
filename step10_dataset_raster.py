import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image, ImageDraw

from project_paths import PATTERN_ROOT

ROOT = str(PATTERN_ROOT)

PAT = re.compile(r"BEM_INPUT_(\d+)_\d+\.txt$", re.IGNORECASE)
name_line = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

LABELS = {
    "type1": ["c12", "c13", "c1e", "c1tr", "c1tl", "c1br", "c1bl"],  # 7
    "type2": ["c12", "c1e", "c1t", "c1b", "c1d2"],                   # 5
    "type3": ["c12", "c1e", "c1t", "c1b", "c1d2"],                   # 5
}
TYPE_ID = {"type1": 0, "type2": 1, "type3": 2}
D_MAX = 7

# channels we will rasterize
CH_KEYS = ["c1", "c2", "c3", "c2e", "botleft", "botright", "die_edge"]
C = len(CH_KEYS)

def is_int(s: str) -> bool:
    try:
        int(s.strip()); return True
    except:
        return False

def parse_bem_input(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f.readlines()]

    i = 0
    shapes = {}
    while i < len(raw):
        ln = raw[i].strip()
        if ln == "":
            i += 1
            continue
        if is_int(ln):
            i += 1
            continue
        if not name_line.match(ln):
            i += 1
            continue

        name = ln.lower()
        i += 1
        if i >= len(raw) or not is_int(raw[i]):
            continue
        k = int(raw[i]); i += 1

        pts = []
        for _ in range(k):
            if i >= len(raw): break
            parts = raw[i].split(); i += 1
            if len(parts) < 2: continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except:
                pass

        if len(pts) >= 3:
            shapes.setdefault(name, []).append(pts)
    return shapes

def compute_bbox(shapes):
    xs, ys = [], []
    for polys in shapes.values():
        for poly in polys:
            for x, y in poly:
                xs.append(x); ys.append(y)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax - xmin < 1e-9: xmax = xmin + 1e-9
    if ymax - ymin < 1e-9: ymax = ymin + 1e-9
    return xmin, xmax, ymin, ymax

def world_to_pixel(poly, xmin, xmax, ymin, ymax, H, W):
    out = []
    for x, y in poly:
        px = (x - xmin) / (xmax - xmin) * (W - 1)
        py = (y - ymin) / (ymax - ymin) * (H - 1)
        out.append((px, py))
    return out

def rasterize_fill(polys, bbox, H, W):
    xmin, xmax, ymin, ymax = bbox
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        pix = world_to_pixel(poly, xmin, xmax, ymin, ymax, H, W)
        draw.polygon(pix, outline=1, fill=1)
    return np.array(img, dtype=np.float32)

def rasterize_edges(polys, bbox, H, W):
    """draw polygon outlines only"""
    xmin, xmax, ymin, ymax = bbox
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        pix = world_to_pixel(poly, xmin, xmax, ymin, ymax, H, W)
        # close the loop
        if len(pix) >= 2:
            draw.line(pix + [pix[0]], fill=1, width=1)
    return np.array(img, dtype=np.float32)

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

class CapRasterDataset(Dataset):
    def __init__(self, type_name: str, H=256, W=256):
        self.type_name = type_name
        self.type_id = TYPE_ID[type_name]
        self.H, self.W = H, W
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

        shapes = parse_bem_input(path)
        bbox = compute_bbox(shapes)

        # die polys for edge channel
        die_polys = []
        for k, polys in shapes.items():
            if k.startswith("die"):
                die_polys.extend(polys)

        ch_arrays = []
        # conductor fills
        for k in ["c1", "c2", "c3", "c2e", "botleft", "botright"]:
            ch_arrays.append(rasterize_fill(shapes.get(k, []), bbox, self.H, self.W))
        # die edges
        ch_arrays.append(rasterize_edges(die_polys, bbox, self.H, self.W))

        x = np.stack(ch_arrays, axis=0)  # (C,H,W)
        x = torch.from_numpy(x).float()

        row = self.gt_rows[idx-1]
        y_raw = torch.tensor([row[k] for k in LABELS[self.type_name]], dtype=torch.float32)

        # pad to 7 + mask
        y_pad = torch.zeros(D_MAX, dtype=torch.float32)
        mask = torch.zeros(D_MAX, dtype=torch.float32)
        d = y_raw.numel()
        y_pad[:d] = y_raw
        mask[:d] = 1.0

        return x, y_pad, mask, torch.tensor(self.type_id, dtype=torch.long), idx, path

def main():
    ds = ConcatDataset([CapRasterDataset("type1"), CapRasterDataset("type2"), CapRasterDataset("type3")])
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    x, y_pad, mask, tid, idx, path = next(iter(loader))
    print("Batch shapes:")
    print("  x:", x.shape)         # (B,C,H,W)
    print("  y:", y_pad.shape)     # (B,7)
    print("  mask:", mask.shape)   # (B,7)
    print("  tid:", tid.tolist())
    print("  idx:", idx.tolist())
    print("  paths:", [os.path.basename(p) for p in path])

    # quick sparsity check
    print("  x mean per channel:", x.mean(dim=(0,2,3)).tolist())

if __name__ == "__main__":
    main()
