#!/usr/bin/env python3
"""
Compute npool and ppool prediction quality per PFT, considering only grid cells
where that PFT is present (PCT_NAT_PFT_k > 0).

Where a PFT has zero coverage, npool/ppool are typically the special values
(10 and 1); including those cells would distort the metric. So for npool PFT k
(and ppool PFT k), we validate only on grid cells with PCT_NAT_PFT_k > 0.

Usage:
  python scripts/validation_npool_ppool_per_pft.py cnp_results/run_YYYYMMDD_HHMMSS
  python scripts/validation_npool_ppool_per_pft.py cnp_results/run_20260226_114546_nofilter --output report.json --pct-min 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    n = len(y_true)
    if n < 2:
        return float("nan"), float("nan"), int(n)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = np.sqrt(ss_res / n)
    return r2, rmse, int(n)


def run(run_dir: Path, pct_min: float = 0.0, output_path: Path = None):
    run_dir = Path(run_dir)
    pred_dir = run_dir / "cnp_predictions"
    static_path = pred_dir / "test_static_inverse.csv"
    gt_npool_path = pred_dir / "pft_1d_ground_truth" / "ground_truth_Y_npool.csv"
    gt_ppool_path = pred_dir / "pft_1d_ground_truth" / "ground_truth_Y_ppool.csv"
    pred_npool_path = pred_dir / "pft_1d_predictions" / "predictions_Y_npool.csv"
    pred_ppool_path = pred_dir / "pft_1d_predictions" / "predictions_Y_ppool.csv"

    for p in [static_path, gt_npool_path, gt_ppool_path, pred_npool_path, pred_ppool_path]:
        if not p.exists():
            print(f"Missing: {p}", file=sys.stderr)
            return None

    static = pd.read_csv(static_path)
    gt_npool = pd.read_csv(gt_npool_path)
    gt_ppool = pd.read_csv(gt_ppool_path)
    pred_npool = pd.read_csv(pred_npool_path)
    pred_ppool = pd.read_csv(pred_ppool_path)

    n = len(static)
    if n != len(gt_npool) or n != len(gt_ppool) or n != len(pred_npool) or n != len(pred_ppool):
        print("Row count mismatch.", file=sys.stderr)
        return None

    # PCT columns: PCT_NAT_PFT_1 .. PCT_NAT_PFT_16 (1-based PFT index)
    pct_cols = [f"PCT_NAT_PFT_{k}" for k in range(1, 17)]
    if not all(c in static.columns for c in pct_cols):
        print("PCT_NAT_PFT_1..16 not found in test_static_inverse.csv", file=sys.stderr)
        return None

    gt_npool_cols = [f"Y_npool_pft{k}" for k in range(1, 17)]
    gt_ppool_cols = [f"Y_ppool_pft{k}" for k in range(1, 17)]
    if not all(c in gt_npool.columns for c in gt_npool_cols) or not all(c in gt_ppool.columns for c in gt_ppool_cols):
        print("Expected Y_npool_pft1..16 and Y_ppool_pft1..16 in GT/pred CSVs.", file=sys.stderr)
        return None

    results = {"run_dir": str(run_dir), "pct_min": pct_min, "n_test_cells": n}
    npool_per_pft = []
    ppool_per_pft = []

    for pft_idx in range(1, 17):
        # Mask: only grid cells where this PFT is present
        pct_col = f"PCT_NAT_PFT_{pft_idx}"
        pct_vals = pd.to_numeric(static[pct_col], errors="coerce").fillna(0).values
        mask = pct_vals > pct_min
        n_valid = int(mask.sum())

        # NPOOL for this PFT
        gt_col = f"Y_npool_pft{pft_idx}"
        r2_n, rmse_n, n_used = _r2_rmse(
            gt_npool.loc[mask, gt_col].values,
            pred_npool.loc[mask, gt_col].values,
        )
        npool_per_pft.append({
            "pft": pft_idx,
            "n_cells_with_pft": n_valid,
            "r2": float(r2_n),
            "rmse": float(rmse_n),
        })

        # PPOOL for this PFT
        gt_col_p = f"Y_ppool_pft{pft_idx}"
        r2_p, rmse_p, _ = _r2_rmse(
            gt_ppool.loc[mask, gt_col_p].values,
            pred_ppool.loc[mask, gt_col_p].values,
        )
        ppool_per_pft.append({
            "pft": pft_idx,
            "n_cells_with_pft": n_valid,
            "r2": float(r2_p),
            "rmse": float(rmse_p),
        })

    results["npool_per_pft"] = npool_per_pft
    results["ppool_per_pft"] = ppool_per_pft

    # Aggregate: mean R² over PFTs (only PFTs with enough valid cells, e.g. n >= 10)
    min_cells = 10
    r2_npool_list = [x["r2"] for x in npool_per_pft if x["n_cells_with_pft"] >= min_cells and np.isfinite(x["r2"])]
    r2_ppool_list = [x["r2"] for x in ppool_per_pft if x["n_cells_with_pft"] >= min_cells and np.isfinite(x["r2"])]
    results["npool_mean_r2_over_pfts"] = float(np.mean(r2_npool_list)) if r2_npool_list else None
    results["ppool_mean_r2_over_pfts"] = float(np.mean(r2_ppool_list)) if r2_ppool_list else None

    # Print report
    print()
    print("=== NPOOL / PPOOL validation per PFT (only cells with PCT_NAT_PFT_k > pct_min) ===")
    print(f"Run: {run_dir}")
    print(f"pct_min: {pct_min} (include grid cells where PCT_NAT_PFT_k > {pct_min})")
    print(f"Test grid cells: {n}")
    print()
    print("NPOOL per PFT (only cells where this PFT is present):")
    print("  PFT   n_cells   R²       RMSE")
    for x in npool_per_pft:
        print(f"  {x['pft']:2d}    {x['n_cells_with_pft']:5d}   {x['r2']:7.4f}   {x['rmse']:.4f}")
    print(f"  Mean R² (PFTs with ≥{min_cells} cells): {results['npool_mean_r2_over_pfts']}")
    print()
    print("PPOOL per PFT (only cells where this PFT is present):")
    print("  PFT   n_cells   R²       RMSE")
    for x in ppool_per_pft:
        print(f"  {x['pft']:2d}    {x['n_cells_with_pft']:5d}   {x['r2']:7.4f}   {x['rmse']:.4f}")
    print(f"  Mean R² (PFTs with ≥{min_cells} cells): {results['ppool_mean_r2_over_pfts']}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport written to {output_path}")

    return results


def main():
    ap = argparse.ArgumentParser(
        description="NPOOL/PPOOL validation per PFT: only grid cells with PCT_NAT_PFT_k > pct_min."
    )
    ap.add_argument("run_dir", type=str, help="Path to run directory")
    ap.add_argument("--output", "-o", type=str, help="Write JSON report to this file")
    ap.add_argument("--pct-min", type=float, default=0.0, help="Minimum PCT_NAT_PFT_k to include (default 0, i.e. > 0)")
    args = ap.parse_args()
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        # Resolve relative to cwd so "." means the current (run) directory
        run_path = (Path.cwd() / run_path).resolve()
    out_path = Path(args.output).resolve() if args.output else None
    run(run_path, pct_min=args.pct_min, output_path=out_path)


if __name__ == "__main__":
    main()
