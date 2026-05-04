# step15_make_submission.py
import os
import csv
import math

from project_paths import SUBMIT_OUT

ROOT = str(SUBMIT_OUT)

PRED_FILES = {
    "type1": os.path.join(ROOT, "pred_type1.csv"),
    "type2": os.path.join(ROOT, "pred_type2.csv"),
    "type3": os.path.join(ROOT, "pred_type3.csv"),
}

# ====== choose output mode ======
# values_only:   each line only numbers
# with_filename: each line: <file> <numbers...>
# csv_wide:      one merged csv with fixed columns across all types (missing filled 0)
MODE = "with_filename"

OUT_PATH = os.path.join(ROOT, f"submission_{MODE}.txt" if MODE != "csv_wide" else "submission_wide.csv")

# unified columns for csv_wide
WIDE_COLS = ["type", "idx", "file",
             "c12", "c13", "c1e", "c1tr", "c1tl", "c1br", "c1bl",
             "c1t", "c1b", "c1d2"]

def read_pred_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        # header: idx,file,<y_names...>
        y_names = header[2:]
        for line in r:
            idx = int(line[0])
            fn = line[1]
            y = [float(v) for v in line[2:]]
            rows.append((idx, fn, y_names, y))
    rows.sort(key=lambda z: z[0])
    return rows

def is_finite_list(xs):
    return all((isinstance(v, (int, float)) and math.isfinite(v)) for v in xs)

def main():
    all_rows = []
    for tname, csv_path in PRED_FILES.items():
        rows = read_pred_csv(csv_path)
        if tname == "type1":
            expect_dim = 7
        else:
            expect_dim = 5

        # sanity checks
        assert len(rows) > 0, f"{tname} empty!"
        for idx, fn, y_names, y in rows:
            assert len(y) == expect_dim, f"{tname} {fn} dim {len(y)} != {expect_dim}"
            assert is_finite_list(y), f"{tname} {fn} has non-finite values: {y}"

        for idx, fn, y_names, y in rows:
            all_rows.append((tname, idx, fn, y_names, y))

    # global sanity
    assert len(all_rows) == 64 + 48 + 32, f"Total rows {len(all_rows)} != 144"

    # write
    if MODE in ("values_only", "with_filename"):
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for tname, idx, fn, y_names, y in sorted(all_rows, key=lambda z: (z[0], z[1])):
                if MODE == "with_filename":
                    f.write(fn + " ")
                f.write(" ".join(f"{v:.8f}" for v in y) + "\n")
        print(f"[OK] wrote -> {OUT_PATH}")

    elif MODE == "csv_wide":
        with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(WIDE_COLS)

            for tname, idx, fn, y_names, y in sorted(all_rows, key=lambda z: (z[0], z[1])):
                d = {k: 0.0 for k in WIDE_COLS}
                d["type"] = tname
                d["idx"] = idx
                d["file"] = fn
                for k, v in zip(y_names, y):
                    d[k] = v
                w.writerow([d[c] for c in WIDE_COLS])
        print(f"[OK] wrote -> {OUT_PATH}")

    else:
        raise ValueError(f"Unknown MODE={MODE}")

if __name__ == "__main__":
    main()

