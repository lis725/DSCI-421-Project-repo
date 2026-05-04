import os
import re
import glob

from project_paths import PATTERN_ROOT

ROOT = str(PATTERN_ROOT)
PAT = re.compile(r"BEM_INPUT_(\d+)_\d+\.txt$", re.IGNORECASE)

def read_text_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # 绗竴琛屾槸琛ㄥご锛堝寘鍚?|锛夛紝璺宠繃
    if "|" in lines[0]:
        lines = lines[1:]
    return lines

def sample_indices(tdir, k=8):
    files = sorted(glob.glob(os.path.join(tdir, "BEM_INPUT_*.txt")))
    out = []
    for p in files:
        m = PAT.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    out = sorted(out, key=lambda x: x[0])
    picks = out[:k] + out[-k:] if len(out) > 2*k else out
    return picks

def main():
    for t in ["type1", "type2", "type3"]:
        tdir = os.path.join(ROOT, f"{t}_data")
        ttext = os.path.join(ROOT, f"{t}.text")
        print("\n" + "="*90)
        print(f"[{t}]")
        print(" data:", tdir)
        print(" text:", ttext)

        if not os.path.exists(ttext):
            print("  !! missing .text")
            continue
        if not os.path.isdir(tdir):
            print("  !! missing data dir")
            continue

        lines = read_text_lines(ttext)
        print(f"  .text rows (without header): {len(lines)}")

        picks = sample_indices(tdir, k=6)
        for idx, path in picks:
            # idx assumed 1-based
            if 1 <= idx <= len(lines):
                row = lines[idx-1]
                print(f"  idx={idx:>3}  file={os.path.basename(path)}")
                print(f"       row={row}")
            else:
                print(f"  idx={idx:>3}  file={os.path.basename(path)}  -> OUT OF RANGE")

if __name__ == "__main__":
    main()

