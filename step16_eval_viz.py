# step16_eval_viz.py
# -*- coding: utf-8 -*-
"""
Evaluate + visualize predictions vs ground truth for 3 types.

Robust against:
- GT text having header row containing 'c12' etc
- GT text using '|' separator or spaces
- Pred txt having either:
    (A) lines with "BEM_INPUT_...txt <floats...>"
    (B) ONLY floats per line (no fname)  <-- your current pred_type1.txt
"""

import os
import re
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from project_paths import EVAL_OUT, PATTERN_ROOT, PROJECT_ROOT, SUBMIT_OUT


# -----------------------------
# Config
# -----------------------------
TYPE_DIM = {0: 7, 1: 5, 2: 5}
TYPE_NAME = {0: "type1", 1: "type2", 2: "type3"}

TARGET_NAMES = {
    0: ["c12", "c13", "c1e", "c1tr", "c1tl", "c1br", "c1bl"],
    1: ["c12", "c1e", "c1t", "c1b", "c1d2"],
    2: ["c12", "c1e", "c1t", "c1b", "c1d2"],
}

FNAME_RE = re.compile(r"(BEM_INPUT_\d+_\d+\.txt)")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _split_after_bar(s: str) -> str:
    if "|" in s:
        return s.split("|", 1)[1]
    return s


def _parse_idx_from_fname(fname: str) -> int:
    # BEM_INPUT_{idx}_{id}.txt
    parts = fname.split("_")
    return int(parts[2])


# -----------------------------
# Readers
# -----------------------------
def read_gt_text(gt_path: str, t: int) -> pd.DataFrame:
    """
    Read GT from type?.text
    Return df: idx, gt0..gt{d-1}
    Assumption: GT numeric rows are in idx order 1..N
    """
    need = TYPE_DIM[t]
    rows: List[List[float]] = []

    if not os.path.exists(gt_path):
        raise RuntimeError(f"[GT] file not found: {gt_path}")

    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            low = line.lower()
            # skip header / non-data
            if ("c12" in low) or ("mstwidth" in low) or ("type1" in low) or ("type2" in low) or ("type3" in low):
                continue

            tail = _split_after_bar(line).strip()
            nums = FLOAT_RE.findall(tail)
            if len(nums) < need:
                continue

            rows.append([float(x) for x in nums[:need]])

    if len(rows) == 0:
        with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
            head = [next(f, "").rstrip("\n") for _ in range(10)]
        raise RuntimeError(
            f"[GT] parse failed: {gt_path} no valid numeric rows.\n"
            f"head preview:\n" + "\n".join(head)
        )

    df = pd.DataFrame(rows, columns=[f"gt{k}" for k in range(need)])
    df.insert(0, "idx", np.arange(1, len(df) + 1, dtype=int))
    return df


def read_pred_txt(pred_path: str, t: int) -> pd.DataFrame:
    """
    Read pred from pred_type?.txt
    Supports two formats:
      A) each valid line contains fname + floats
      B) each line contains ONLY floats (no fname) -> idx by line order (1..N)

    Return df columns:
      - format A: fname, idx, pred0..pred{d-1}
      - format B: idx, pred0..pred{d-1} (fname filled as empty)
    """
    need = TYPE_DIM[t]

    if not os.path.exists(pred_path):
        raise RuntimeError(f"[PRED] file not found: {pred_path}")

    # First pass: detect if any filename exists
    has_fname = False
    with open(pred_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if FNAME_RE.search(raw):
                has_fname = True
                break

    rows = []
    if has_fname:
        # Format A
        with open(pred_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                m = FNAME_RE.search(s)
                if not m:
                    continue
                fname = m.group(1)
                tail = s[m.end():]
                nums = FLOAT_RE.findall(tail)
                if len(nums) < need:
                    continue
                pred = [float(x) for x in nums[:need]]
                idx = _parse_idx_from_fname(fname)

                row = {"fname": fname, "idx": idx}
                for k in range(need):
                    row[f"pred{k}"] = pred[k]
                rows.append(row)

    else:
        # Format B: only floats per line, idx by line order
        line_idx = 0
        with open(pred_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                nums = FLOAT_RE.findall(s)
                if len(nums) < need:
                    continue
                line_idx += 1
                pred = [float(x) for x in nums[:need]]
                row = {"fname": "", "idx": line_idx}
                for k in range(need):
                    row[f"pred{k}"] = pred[k]
                rows.append(row)

    if len(rows) == 0:
        with open(pred_path, "r", encoding="utf-8", errors="ignore") as f:
            head = [next(f, "").rstrip("\n") for _ in range(10)]
        raise RuntimeError(
            f"[PRED] parse failed: {pred_path} no valid prediction rows.\n"
            f"head preview:\n" + "\n".join(head)
        )

    df = pd.DataFrame(rows)
    df["idx"] = df["idx"].astype(int)
    return df


# -----------------------------
# Metrics
# -----------------------------
def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = _r2_score(gt, pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# -----------------------------
# Viz
# -----------------------------
def plot_scatter(gt: np.ndarray, pred: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(gt, pred, s=20, alpha=0.8)
    mn = float(min(gt.min(), pred.min()))
    mx = float(max(gt.max(), pred.max()))
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel("GT")
    plt.ylabel("Pred")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_err_hist(err: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=30, alpha=0.9)
    plt.xlabel("Pred - GT")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Evaluate
# -----------------------------
def evaluate_one_type(t: int, gt_path: str, pred_path: str, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    need = TYPE_DIM[t]
    type_str = TYPE_NAME[t]
    names = TARGET_NAMES[t]

    gt_df = read_gt_text(gt_path, t)
    pred_df = read_pred_txt(pred_path, t)

    # Merge by idx
    df = pd.merge(pred_df, gt_df, on="idx", how="inner")
    if len(df) == 0:
        raise RuntimeError(
            f"[{type_str}] merge empty.\n"
            f"  pred rows={len(pred_df)} idx range=({pred_df['idx'].min()}..{pred_df['idx'].max()})\n"
            f"  gt   rows={len(gt_df)} idx range=({gt_df['idx'].min()}..{gt_df['idx'].max()})\n"
            f"  -> Check whether pred/gt order and counts match."
        )

    df = df.sort_values("idx").reset_index(drop=True)

    # Per-dim metrics + plots
    per_dim_rows = []
    for k in range(need):
        gt = df[f"gt{k}"].to_numpy(dtype=np.float64)
        pr = df[f"pred{k}"].to_numpy(dtype=np.float64)
        met = compute_metrics(gt, pr)

        nm = names[k] if k < len(names) else f"y{k}"
        per_dim_rows.append({
            "type": type_str,
            "dim": k,
            "name": nm,
            "mae": met["mae"],
            "rmse": met["rmse"],
            "r2": met["r2"],
        })

        plot_scatter(gt, pr, f"{type_str} | {nm} | GT vs Pred",
                     os.path.join(out_dir, f"scatter_{type_str}_{nm}.png"))
        plot_err_hist(pr - gt, f"{type_str} | {nm} | Error (Pred-GT)",
                      os.path.join(out_dir, f"error_hist_{type_str}_{nm}.png"))

    per_dim_df = pd.DataFrame(per_dim_rows)

    # Overall
    all_gt = np.concatenate([df[f"gt{k}"].to_numpy(np.float64) for k in range(need)], axis=0)
    all_pr = np.concatenate([df[f"pred{k}"].to_numpy(np.float64) for k in range(need)], axis=0)
    overall = compute_metrics(all_gt, all_pr)

    # Save metrics txt
    txt_path = os.path.join(out_dir, f"metrics_{type_str}.txt")
    with open(txt_path, "w", encoding="utf-8") as w:
        w.write(f"[{type_str}] samples={len(df)} dim={need}\n")
        w.write(f"OVERALL: MAE={overall['mae']:.6g} RMSE={overall['rmse']:.6g} R2={overall['r2']}\n\n")
        w.write("PER-DIM:\n")
        for r in per_dim_rows:
            w.write(f"  {r['name']:<6} | MAE={r['mae']:.6g} RMSE={r['rmse']:.6g} R2={r['r2']}\n")

    # Per-sample
    abs_err_mat = np.stack(
        [(df[f"pred{k}"] - df[f"gt{k}"]).abs().to_numpy(np.float64) for k in range(need)],
        axis=1
    )
    df_out = df[["idx", "fname"]].copy()
    df_out["mae_sample"] = abs_err_mat.mean(axis=1)
    df_out["maxae_sample"] = abs_err_mat.max(axis=1)
    for k in range(need):
        nm = names[k] if k < len(names) else f"y{k}"
        df_out[f"abs_err_{nm}"] = abs_err_mat[:, k]

    df_out.to_csv(os.path.join(out_dir, f"per_sample_{type_str}.csv"),
                  index=False, encoding="utf-8-sig")

    return per_dim_df, df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--pred_dir", type=str, default=str(SUBMIT_OUT))
    parser.add_argument("--gt_dir", type=str, default=str(PATTERN_ROOT))
    parser.add_argument("--out_dir", type=str, default=str(EVAL_OUT))
    args = parser.parse_args()

    root = args.root
    pred_dir = args.pred_dir if os.path.isabs(args.pred_dir) else os.path.join(root, args.pred_dir)
    gt_dir = args.gt_dir if os.path.isabs(args.gt_dir) else os.path.join(root, args.gt_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    GT_PATH = {
        0: os.path.join(gt_dir, "type1.text"),
        1: os.path.join(gt_dir, "type2.text"),
        2: os.path.join(gt_dir, "type3.text"),
    }
    PRED_PATH = {
        0: os.path.join(pred_dir, "pred_type1.txt"),
        1: os.path.join(pred_dir, "pred_type2.txt"),
        2: os.path.join(pred_dir, "pred_type3.txt"),
    }

    all_dim_metrics = []
    all_sample_rows = []

    for t in (0, 1, 2):
        type_str = TYPE_NAME[t]
        this_out = os.path.join(out_dir, type_str)
        print(f"[Eval] {type_str}")
        print(f"  GT  : {GT_PATH[t]}")
        print(f"  Pred: {PRED_PATH[t]}")

        per_dim_df, per_sample_df = evaluate_one_type(t, GT_PATH[t], PRED_PATH[t], this_out)

        all_dim_metrics.append(per_dim_df)
        ps = per_sample_df.copy()
        ps.insert(0, "type", type_str)
        all_sample_rows.append(ps)

    dim_df = pd.concat(all_dim_metrics, ignore_index=True)
    dim_csv = os.path.join(out_dir, "summary_all_types.csv")
    dim_df.to_csv(dim_csv, index=False, encoding="utf-8-sig")

    sample_df = pd.concat(all_sample_rows, ignore_index=True)
    sample_csv = os.path.join(out_dir, "per_sample_all_types.csv")
    sample_df.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote:\n  {dim_csv}\n  {sample_csv}\n  plots under: {out_dir}")


if __name__ == "__main__":
    main()
