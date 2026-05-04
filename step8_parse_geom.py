import os
import re

from project_paths import PATTERN_ROOT

FILE = str(PATTERN_ROOT / 'type1_data' / 'BEM_INPUT_10_43652.txt')

num_line = re.compile(r"^\s*[-+]?\d+(\.\d+)?\s*$")
name_line = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def is_int(s: str) -> bool:
    try:
        int(s.strip())
        return True
    except:
        return False

def parse_bem_input(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f.readlines()]

    i = 0
    shapes = {}  # name -> list of polys, each poly list[(x,y)]
    while i < len(raw):
        ln = raw[i].strip()
        if ln == "":
            i += 1
            continue

        # skip lone numbers (block counts like "5", "79", etc.)
        if is_int(ln):
            i += 1
            continue

        # must be a name token
        if not name_line.match(ln):
            # unknown junk line, skip
            i += 1
            continue

        name = ln
        i += 1
        # after name: may be a number (vertex count)
        if i >= len(raw) or not is_int(raw[i]):
            # no vertex count -> skip this name
            continue
        k = int(raw[i])
        i += 1

        pts = []
        for _ in range(k):
            if i >= len(raw):
                break
            parts = raw[i].split()
            i += 1
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
                pts.append((x, y))
            except:
                pass

        if len(pts) >= 3:
            shapes.setdefault(name.lower(), []).append(pts)

    return shapes

def main():
    shapes = parse_bem_input(FILE)
    print("Parsed:", os.path.basename(FILE))
    print("Keys:", sorted(shapes.keys()))
    for k in sorted(shapes.keys()):
        polys = shapes[k]
        n_poly = len(polys)
        n_pts = sum(len(p) for p in polys)
        print(f"  {k:10s}  polys={n_poly:2d}  total_pts={n_pts:4d}  first_poly_pts={len(polys[0]) if polys else 0}")
        if polys:
            xs = [x for x, y in polys[0]]
            ys = [y for x, y in polys[0]]
            print(f"    first_poly_bbox: x[{min(xs):.4f},{max(xs):.4f}] y[{min(ys):.4f},{max(ys):.4f}]")

if __name__ == "__main__":
    main()

