import os
import re
import numpy as np
import torch
from PIL import Image, ImageDraw

from project_paths import PATTERN_ROOT

FILE = str(PATTERN_ROOT / 'type1_data' / 'BEM_INPUT_10_43652.txt')

name_line = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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
    # avoid zero range
    if xmax - xmin < 1e-9: xmax = xmin + 1e-9
    if ymax - ymin < 1e-9: ymax = ymin + 1e-9
    return xmin, xmax, ymin, ymax

def world_to_pixel(poly, xmin, xmax, ymin, ymax, H, W):
    out = []
    for x, y in poly:
        px = (x - xmin) / (xmax - xmin) * (W - 1)
        py = (y - ymin) / (ymax - ymin) * (H - 1)
        # PIL uses (x,y) with y downward; that's fine as long as consistent
        out.append((px, py))
    return out

def rasterize_channel(polys, bbox, H, W):
    xmin, xmax, ymin, ymax = bbox
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        pix = world_to_pixel(poly, xmin, xmax, ymin, ymax, H, W)
        draw.polygon(pix, outline=1, fill=1)
    arr = np.array(img, dtype=np.float32)  # (H,W) 0/1
    return arr

def main():
    shapes = parse_bem_input(FILE)
    bbox = compute_bbox(shapes)

    # channel groups
    def get_polys(key):
        return shapes.get(key, [])

    die_polys = []
    for k, polys in shapes.items():
        if k.startswith("die"):
            die_polys.extend(polys)

    channels = [
        ("c1", get_polys("c1")),
        ("c2", get_polys("c2")),
        ("c3", get_polys("c3")),
        ("c2e", get_polys("c2e")),
        ("botleft", get_polys("botleft")),
        ("botright", get_polys("botright")),
        ("die", die_polys),
        ("air_layer", get_polys("air_layer")),
    ]

    H, W = 256, 256
    xs = []
    for name, polys in channels:
        arr = rasterize_channel(polys, bbox, H, W)
        xs.append(arr)
        print(f"{name:10s}  polys={len(polys):3d}  fill_ratio={arr.mean():.6f}")

    x = np.stack(xs, axis=0)  # (C,H,W)
    x_t = torch.from_numpy(x)
    print("x shape:", x_t.shape, "dtype:", x_t.dtype)

    # save a quick visualization (sum of channels) to check it isn't empty
    sum_img = (x.sum(axis=0) > 0).astype(np.uint8) * 255
    Image.fromarray(sum_img).save("debug_raster_sum.png")
    print("Saved debug_raster_sum.png in current working dir.")

if __name__ == "__main__":
    main()

