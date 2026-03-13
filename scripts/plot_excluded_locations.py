#!/usr/bin/env python3
"""
Plot cpool, litr1c_vr, and other variables at gridcell locations excluded by the natveg filter
(PCT_NATVEG=0 or PCT_NAT_PFT_0=100). Reads test_static_inverse.csv to get excluded row indices,
then plots GT vs Pred at those rows only.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_excluded_indices(results_dir):
    """Load test_static_inverse.csv and return boolean mask and indices of excluded gridcells."""
    path = Path(results_dir) / "cnp_predictions" / "test_static_inverse.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    pct_natveg = df["PCT_NATVEG"].values.astype(float)
    pct_pft0 = df["PCT_NAT_PFT_0"].values.astype(float)
    include = (pct_natveg > 0) & (pct_pft0 < 100)
    excluded = ~include
    return excluded, df


def drop_coords(df):
    """Drop Longitude, Latitude (or variants) for numeric extraction."""
    for c in ["Longitude", "Latitude", "longitude", "latitude", "lon", "lat", "Long", "Lat"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


def plot_pft1d_at_excluded(results_dir, excluded_idx, out_dir, var_name="cpool"):
    """Plot PFT 1D variable (e.g. cpool) at excluded locations: GT vs Pred scatter (all PFTs)."""
    gt_path = Path(results_dir) / "cnp_predictions" / "pft_1d_ground_truth" / f"ground_truth_Y_{var_name}.csv"
    pred_path = Path(results_dir) / "cnp_predictions" / "pft_1d_predictions" / f"predictions_Y_{var_name}.csv"
    if not gt_path.exists() or not pred_path.exists():
        print(f"  Skip {var_name}: files not found")
        return
    gt = drop_coords(pd.read_csv(gt_path))
    pred = drop_coords(pd.read_csv(pred_path))
    gt_vals = gt.iloc[excluded_idx].values.flatten()
    pred_vals = pred.iloc[excluded_idx].values.flatten()
    valid = ~(np.isnan(gt_vals) | np.isnan(pred_vals))
    gt_vals = gt_vals[valid]
    pred_vals = pred_vals[valid]
    if len(gt_vals) < 2:
        print(f"  Skip {var_name}: too few valid points at excluded locations")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_vals, pred_vals, alpha=0.7, s=40)
    mn = min(gt_vals.min(), pred_vals.min())
    mx = max(gt_vals.max(), pred_vals.max())
    ax.plot([mn, mx], [mn, mx], "r--", label="1:1")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{var_name} at excluded locations (n={len(gt_vals)} points, 36 gridcells × PFTs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"excluded_locations_{var_name}_gt_vs_pred.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_soil2d_at_excluded(results_dir, excluded_idx, out_dir, var_name="litr1c_vr"):
    """Plot 2D soil variable at excluded locations: GT vs Pred scatter (all layers flattened)."""
    gt_path = Path(results_dir) / "cnp_predictions" / "soil_2d_ground_truth" / f"ground_truth_Y_{var_name}.csv"
    pred_path = Path(results_dir) / "cnp_predictions" / "soil_2d_predictions" / f"predictions_Y_{var_name}.csv"
    if not gt_path.exists() or not pred_path.exists():
        print(f"  Skip {var_name}: files not found")
        return
    gt = drop_coords(pd.read_csv(gt_path))
    pred = drop_coords(pd.read_csv(pred_path))
    gt_vals = gt.iloc[excluded_idx].values.flatten()
    pred_vals = pred.iloc[excluded_idx].values.flatten()
    valid = ~(np.isnan(gt_vals) | np.isnan(pred_vals))
    gt_vals = gt_vals[valid]
    pred_vals = pred_vals[valid]
    if len(gt_vals) < 2:
        print(f"  Skip {var_name}: too few valid points at excluded locations")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_vals, pred_vals, alpha=0.7, s=40)
    mn = min(gt_vals.min(), pred_vals.min())
    mx = max(gt_vals.max(), pred_vals.max())
    ax.plot([mn, mx], [mn, mx], "r--", label="1:1")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{var_name} at excluded locations (n={len(gt_vals)} points, 36 gridcells × layers)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"excluded_locations_{var_name}_gt_vs_pred.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot variables at natveg-filter excluded locations")
    parser.add_argument("results_dir", nargs="?", default=".", help="Results directory (e.g. cnp_results/run_xxx)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (default: results_dir/analysis/excluded_locations)")
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "analysis" / "excluded_locations"
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_mask, static_df = get_excluded_indices(results_dir)
    excluded_idx = np.where(excluded_mask)[0]
    n_excl = len(excluded_idx)
    print(f"Excluded gridcells: {n_excl}")
    print(f"Output directory: {out_dir}")

    # PFT 1D
    for var in ["cpool", "npool", "ppool"]:
        plot_pft1d_at_excluded(results_dir, excluded_idx, out_dir, var_name=var)

    # 2D soil (litr1*, soil1*, etc. that exist in this run)
    for var in ["litr1c_vr", "litr1n_vr", "litr1p_vr", "soil1c_vr", "soil1n_vr", "soil1p_vr", "primp_vr"]:
        plot_soil2d_at_excluded(results_dir, excluded_idx, out_dir, var_name=var)

    # Optional: save a small table of excluded (lat, lon) for reference
    loc_path = out_dir / "excluded_locations_lat_lon.csv"
    if "Latitude" in static_df.columns and "Longitude" in static_df.columns:
        static_df.loc[excluded_mask, ["Latitude", "Longitude", "PCT_NATVEG", "PCT_NAT_PFT_0"]].to_csv(loc_path, index=False)
        print(f"  Saved {loc_path}")
    print("Done.")


if __name__ == "__main__":
    main()
