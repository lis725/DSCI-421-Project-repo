import os
import glob

from project_paths import PATTERN_ROOT

ROOT = str(PATTERN_ROOT)

def show_tree(path):
    print(f"\n[LIST] {path}")
    for p in sorted(os.listdir(path)):
        full = os.path.join(path, p)
        if os.path.isdir(full):
            print(f"  <DIR>  {p}")
        else:
            print(f"  <FILE> {p}")

def preview(path, n=80):
    print("\n" + "="*80)
    print(f"[PREVIEW] {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(n):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip("\n"))
    except Exception as e:
        print("READ ERROR:", e)

def main():
    show_tree(ROOT)

    # find *.text / *.txt at root level
    cand = sorted(glob.glob(os.path.join(ROOT, "*.text"))) + sorted(glob.glob(os.path.join(ROOT, "*.txt")))
    print(f"\nFound {len(cand)} root text files.")
    for p in cand:
        print(" ", os.path.basename(p))

    # preview up to 3 important ones
    for name in ["type1.text", "type2.text", "type3.text"]:
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            preview(p, n=120)

if __name__ == "__main__":
    main()

