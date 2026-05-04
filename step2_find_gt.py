import os

from project_paths import PROJECT_ROOT

BASE = str(PROJECT_ROOT)

KEYS = ["c12", "c2e", "c1e", "c1b", "c1d2", "c1tl", "c1tr", "c1bl", "c1br"]

def looks_like_gt(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in KEYS)

def main():
    hits = []
    for root, _, files in os.walk(BASE):
        for fn in files:
            if not fn.lower().endswith((".txt", ".csv", ".dat", ".tsv")):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    head = "".join([f.readline() for _ in range(60)])
                if looks_like_gt(head):
                    hits.append((path, head[:400].replace("\n", "\\n")))
            except Exception:
                pass

    print(f"Found {len(hits)} potential GT files.")
    for p, preview in hits[:30]:
        print("\n---")
        print(p)
        print(preview)

if __name__ == "__main__":
    main()

