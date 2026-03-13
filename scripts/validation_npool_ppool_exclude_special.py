#!/usr/bin/env python3
"""
Compute npool and ppool prediction quality excluding "special value" grid cells.

Special-value cells: grid cells where ALL 16 PFTs have npool==10 and ppool==1
(ground truth). Excluding them answers: what is R²/RMSE when we validate only
on the non-constant (non-background) grid cells?

Usage:
  python scripts/validation_npool_ppool_exclude_special.py cnp_results/run_YYYYMMDD_HHMMSS
  python scripts/validation_npool_ppool_exclude_special.py cnp_results/run_20260225_013116_global_natveg --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NPOOL_SPECIAL = 10.0
PPOOL_SPECIAL = 1.0
TOL = 1e-5


def _is_special_row_npool(df_gt: pd.DataFrame) -> np.ndarray:
    """True for rows where all 16 PFT columns are == 10."""
    cols = [c for c in df_gt.columns if c.startswith("Y_npool_pft")]
    if len(cols) != 16:
        return np.zeros(len(df_gt), dtype=bool)
    arr = df_gt[cols].values.astype(float)
    return np.all(np.abs(arr - NPOOL_SPECIAL) < TOL, axis=1)


def _is_special_row_ppool(df_gt: pd.DataFrame) -> np.ndarray:
    """True for rows where all 16 PFT columns are == 1."""
    cols = [c for c in df_gt.columns if c.startswith("Y_ppool_pft")]
    if len(cols) != 16:
        return np.zeros(len(df_gt), dtype=bool)
    arr = df_gt[cols].values.astype(float)
    return np.all(np.abs(arr - PPOOL_SPECIAL) < TOL, axis=1)


def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None):
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    n = len(y_true)
    if n < 2:
        return float("nan"), float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = np.sqrt(ss_res / n)
    return r2, rmse


def run(run_dir: Path, output_path: Path = None):
    run_dir = Path(run_dir)
    pred_dir = run_dir / "cnp_predictions"
    gt_npool_path = pred_dir / "pft_1d_ground_truth" / "ground_truth_Y_npool.csv"
    gt_ppool_path = pred_dir / "pft_1d_ground_truth" / "ground_truth_Y_ppool.csv"
    pred_npool_path = pred_dir / "pft_1d_predictions" / "predictions_Y_npool.csv"
    pred_ppool_path = pred_dir / "pft_1d_predictions" / "predictions_Y_ppool.csv"

    for p in [gt_npool_path, gt_ppool_path, pred_npool_path, pred_ppool_path]:
        if not p.exists():
            print(f"Missing: {p}", file=sys.stderr)
            return None

    gt_npool = pd.read_csv(gt_npool_path)
    gt_ppool = pd.read_csv(gt_ppool_path)
    pred_npool = pd.read_csv(pred_npool_path)
    pred_ppool = pd.read_csv(pred_ppool_path)

    n = len(gt_npool)
    if n != len(gt_ppool) or n != len(pred_npool) or n != len(pred_ppool):
        print("Row count mismatch between GT and predictions.", file=sys.stderr)
        return None

    # Special-value mask: exclude rows where ALL PFTs are (npool==10 and ppool==1)
    special_npool = _is_special_row_npool(gt_npool)
    special_ppool = _is_special_row_ppool(gt_ppool)
    special_both = special_npool & special_ppool
    n_special = int(special_both.sum())
    n_keep = n - n_special
    keep_mask = ~special_both

    # PFT columns
    cols_npool = [c for c in gt_npool.columns if c.startswith("Y_npool_pft")]
    cols_ppool = [c for c in gt_ppool.columns if c.startswith("Y_ppool_pft")]
    if len(cols_npool) != 16 or len(cols_ppool) != 16:
        print("Expected 16 PFT columns.", file=sys.stderr)
        return None

    # Flatten to (n*16,) for overall metrics
    gt_npool_flat = gt_npool[cols_npool].values.ravel()
    pred_npool_flat = pred_npool[cols_npool].values.ravel()
    gt_ppool_flat = gt_ppool[cols_ppool].values.ravel()
    pred_ppool_flat = pred_ppool[cols_ppool].values.ravel()

    # Row-level mask expanded to (n*16,): row i contributes 16 elements
    keep_flat_npool = np.repeat(keep_mask, 16)
    keep_flat_ppool = np.repeat(keep_mask, 16)

    # Metrics: all cells vs excluding special
    r2_npool_all, rmse_npool_all = _r2_rmse(gt_npool_flat, pred_npool_flat, None)
    r2_npool_excl, rmse_npool_excl = _r2_rmse(gt_npool_flat, pred_npool_flat, keep_flat_npool)
    r2_ppool_all, rmse_ppool_all = _r2_rmse(gt_ppool_flat, pred_ppool_flat, None)
    r2_ppool_excl, rmse_ppool_excl = _r2_rmse(gt_ppool_flat, pred_ppool_flat, keep_flat_ppool)

    out = {
        "run_dir": str(run_dir),
        "n_test_cells": n,
        "n_special_cells_excluded": n_special,
        "n_cells_kept": n_keep,
        "npool": {
            "r2_all_cells": float(r2_npool_all),
            "rmse_all_cells": float(rmse_npool_all),
            "r2_excluding_special": float(r2_npool_excl),
            "rmse_excluding_special": float(rmse_npool_excl),
        },
        "ppool": {
            "r2_all_cells": float(r2_ppool_all),
            "rmse_all_cells": float(rmse_ppool_all),
            "r2_excluding_special": float(r2_ppool_excl),
            "rmse_excluding_special": float(rmse_ppool_excl),
        },
    }

    # Print report
    print()
    print("=== NPOOL / PPOOL validation: excluding special-value grid cells ===")
    print(f"Run: {run_dir}")
    print(f"Test grid cells: {n}")
    print(f"Special-value cells excluded (all PFTs npool==10 and ppool==1): {n_special} ({100*n_special/n:.1f}%)")
    print(f"Cells kept for validation: {n_keep}")
    print()
    print("NPOOL:")
    print(f"  R²   (all cells):     {r2_npool_all:.4f}")
    print(f"  R²   (excl. special): {r2_npool_excl:.4f}")
    print(f"  RMSE (all cells):     {rmse_npool_all:.4f}")
    print(f"  RMSE (excl. special): {rmse_npool_excl:.4f}")
    print()
    print("PPOOL:")
    print(f"  R²   (all cells):     {r2_ppool_all:.4f}")
    print(f"  R²   (excl. special): {r2_ppool_excl:.4f}")
    print(f"  RMSE (all cells):     {rmse_ppool_all:.4f}")
    print(f"  RMSE (excl. special): {rmse_ppool_excl:.4f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nReport written to {output_path}")

    return out


def main():
    ap = argparse.ArgumentParser(description="NPOOL/PPOOL validation excluding special-value grid cells.")
    ap.add_argument("run_dir", type=str, help="Path to run directory (e.g. cnp_results/run_YYYYMMDD_HHMMSS)")
    ap.add_argument("--output", "-o", type=str, help="Write JSON report to this file")
    args = ap.parse_args()
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = (REPO_ROOT / args.run_dir).resolve()
    out_path = Path(args.output).resolve() if args.output else None
    run(run_path, out_path)


if __name__ == "__main__":
    main()
