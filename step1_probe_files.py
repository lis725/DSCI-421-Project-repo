import os
import glob

from project_paths import PATTERN_ROOT

ROOT = str(PATTERN_ROOT)

def preview(path, n=25):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except Exception as e:
        return f"[READ ERROR] {e}"

def classify(head: str) -> str:
    h = head.lower()
    # geometry signatures
    if "#begin" in h or "number of master points" in h:
        return "GEOM(official_begin_end)"
    if any(k in h for k in ["botleft", "botright", "die", "boundary", "c1", "c2", "c3"]):
        # your block style likely
        return "GEOM(block_style?)"
    # GT signatures
    if any(k in h for k in ["c12", "c2e", "c1e", "c1b", "c1d2", "c1tl", "c1tr", "c1bl", "c1br"]):
        return "GT(capacitance)"
    # unknown
    return "UNKNOWN"

def main():
    for t in ["type1", "type2", "type3"]:
        tdir = os.path.join(ROOT, f"{t}_data")
        print("\n" + "="*80)
        print(f"[{t}] dir: {tdir}")
        if not os.path.isdir(tdir):
            print("  (missing)")
            continue

        files = sorted(glob.glob(os.path.join(tdir, "*.txt")))
        print(f"  txt count: {len(files)}")
        if not files:
            continue

        # show first 3 files preview
        for p in files[:3]:
            head = preview(p, n=25)
            kind = classify(head)
            print("\n---")
            print(f"FILE: {os.path.basename(p)}  |  {kind}")
            print(head)

        # quick stats
        geom, gt, unk = 0, 0, 0
        for p in files:
            head = preview(p, n=25)
            k = classify(head)
            if k.startswith("GEOM"):
                geom += 1
            elif k.startswith("GT"):
                gt += 1
            else:
                unk += 1
        print("\n  Summary:")
        print(f"    GEOM: {geom}")
        print(f"    GT  : {gt}")
        print(f"    UNK : {unk}")

if __name__ == "__main__":
    main()

