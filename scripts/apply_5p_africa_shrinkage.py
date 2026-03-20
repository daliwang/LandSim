#!/usr/bin/env python3
"""
Africa-only 5P shrinkage correction for bias-corrected Phase2 predictions.

This script reads ground truth and bias-corrected 5P prediction CSVs from a
CNP run directory, computes per-layer shrinkage factors for the Africa region,
and writes corrected CSVs where 5P values in Africa are scaled down toward the
observed range without touching Amazon or other regions.

Expected directory structure under --run-dir:

  <run-dir>/cnp_inference_entire_dataset/cnp_predictions/
    soil_2d_ground_truth/ground_truth_Y_<var>.csv
    soil_2d_predictions_5P_bias_corrected_phase2/predictions_Y_<var>_bias_corrected.csv

Output:
  <run-dir>/cnp_inference_entire_dataset/cnp_predictions/
    soil_2d_predictions_5P_bias_corrected_africa_shrinkage/
      predictions_Y_<var>_bias_corrected_africa_shrinkage.csv

Usage (example):

  python scripts/apply_5p_africa_shrinkage.py \
    --run-dir cnp_results/run_20260311_204845_phase3_tworegions
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Africa region box (lat_min, lat_max, lon_min, lon_max)
AFRICA_BOX = (-15.0, 15.0, 0.0, 30.0)

# Per-variable maximum shrinkage factors (Africa only). Values in (0, 1].
# These enforce stronger shrinking where we know overprediction is severe.
MAX_FACTOR_BY_VAR = {
    "solutionp_vr": 0.2,  # very aggressive shrink
    "secondp_vr": 0.5,
    "primp_vr": 0.5,
    # occlp_vr and labilep_vr use default cap of 1.0
}

# Quantile used to cap corrected values relative to Africa GT (per layer).
AFRICA_GT_CAP_QUANTILE = 90.0

FIVE_P: List[str] = [
    "labilep_vr",
    "occlp_vr",
    "solutionp_vr",
    "secondp_vr",
    "primp_vr",
]


def _africa_mask(df: pd.DataFrame) -> np.ndarray:
    """Return boolean mask selecting rows within the Africa lat/lon box."""
    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("DataFrame must contain 'Longitude' and 'Latitude' columns.")
    lon = df["Longitude"].to_numpy()
    lat = df["Latitude"].to_numpy()
    lat_min, lat_max, lon_min, lon_max = AFRICA_BOX
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def _per_layer_columns(df: pd.DataFrame, var: str) -> List[str]:
    """Return ordered list of layer columns for a given 5P var (Y_<var>_col1_layer*)."""
    prefix = f"Y_{var}_col1_layer"
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols = sorted(cols, key=lambda c: int(c.split("layer")[-1]))
    return cols


def compute_africa_shrinkage_factors(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, var: str
) -> Dict[str, float]:
    """Compute per-layer shrinkage factors for Africa for one variable.

    For each layer, we look at Africa rows where pred > 0 and gt > 0 and compute
    ratio = gt / pred. The shrinkage factor is median(ratio), clamped to [0, 1]
    and also to a per-variable MAX_FACTOR if provided (to avoid overly weak
    shrinkage in regimes with extreme overprediction).
    If there are too few valid points, we fall back to factor=1.0 (no change).
    """
    if gt_df.shape != pred_df.shape:
        raise ValueError(f"Shape mismatch for {var}: GT {gt_df.shape}, PRED {pred_df.shape}")

    mask_af = _africa_mask(gt_df)
    layer_cols = _per_layer_columns(gt_df, var)
    factors: Dict[str, float] = {}

    for col in layer_cols:
        gt_vals = gt_df.loc[mask_af, col].to_numpy(dtype=float)
        pred_vals = pred_df.loc[mask_af, col].to_numpy(dtype=float)

        valid = (gt_vals > 0.0) & (pred_vals > 0.0) & np.isfinite(gt_vals) & np.isfinite(pred_vals)
        if valid.sum() < 10:
            factors[col] = 1.0
            continue

        ratios = gt_vals[valid] / pred_vals[valid]
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            factors[col] = 1.0
            continue

        median_ratio = float(np.median(ratios))
        # Base clamp to [0, 1]
        factor = max(0.0, min(1.0, median_ratio))
        # Apply per-variable maximum factor where defined
        max_factor = MAX_FACTOR_BY_VAR.get(var)
        if max_factor is not None:
            factor = min(factor, max_factor)
        factors[col] = factor

    return factors


def apply_africa_shrinkage(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    var: str,
    factors: Dict[str, float],
) -> pd.DataFrame:
    """Apply per-layer shrinkage to Africa rows for one variable.

    corrected = pred * factor in Africa; outside Africa, predictions unchanged.
    We also clip corrected values to [0, q_GT_Africa_layer] as a safety cap,
    where q is AFRICA_GT_CAP_QUANTILE (e.g. 90th percentile).
    """
    out_df = pred_df.copy()
    mask_af = _africa_mask(pred_df)
    layer_cols = _per_layer_columns(pred_df, var)

    for col in layer_cols:
        factor = factors.get(col, 1.0)
        if factor >= 0.9999:
            continue

        gt_vals = gt_df.loc[mask_af, col].to_numpy(dtype=float)
        if gt_vals.size == 0 or not np.isfinite(gt_vals).any():
            cap = None
        else:
            cap = float(
                np.nanpercentile(
                    gt_vals[np.isfinite(gt_vals)], AFRICA_GT_CAP_QUANTILE
                )
            )

        vals = out_df.loc[mask_af, col].to_numpy(dtype=float)
        corrected = vals * factor
        if cap is not None and cap > 0.0:
            corrected = np.minimum(corrected, cap)
        corrected = np.maximum(corrected, 0.0)
        out_df.loc[mask_af, col] = corrected

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply Africa-only shrinkage to 5P bias-corrected Phase2 predictions "
            "(reduces overly large Africa 5P values while leaving other regions unchanged)."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "CNP run directory containing cnp_inference_entire_dataset/cnp_predictions/ "
            "with soil_2d_ground_truth and soil_2d_predictions_5P_bias_corrected_phase2."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default="soil_2d_predictions_5P_bias_corrected_africa_shrinkage",
        help=(
            "Subdirectory name under cnp_inference_entire_dataset/cnp_predictions where "
            "Africa-shrunk 5P CSVs will be written."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    predictions_root = run_dir / "cnp_inference_entire_dataset" / "cnp_predictions"
    gt_dir = predictions_root / "soil_2d_ground_truth"
    bias_dir = predictions_root / "soil_2d_predictions_5P_bias_corrected_phase2"
    out_dir = predictions_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.is_dir():
        raise SystemExit(f"Ground-truth dir not found: {gt_dir}")
    if not bias_dir.is_dir():
        raise SystemExit(f"Bias-corrected predictions dir not found: {bias_dir}")

    print(f"Run dir: {run_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Bias-corrected 5P dir: {bias_dir}")
    print(f"Output (Africa shrinkage) dir: {out_dir}")

    all_params = {}

    for var in FIVE_P:
        gt_path = gt_dir / f"ground_truth_Y_{var}.csv"
        pred_path = bias_dir / f"predictions_Y_{var}_bias_corrected.csv"
        if not gt_path.is_file() or not pred_path.is_file():
            print(f"Skipping {var}: missing GT or prediction CSV ({gt_path}, {pred_path})")
            continue

        print(f"\nVariable: {var}")
        print(f"  GT:   {gt_path}")
        print(f"  Pred: {pred_path}")

        gt_df = pd.read_csv(gt_path)
        pred_df = pd.read_csv(pred_path)
        if gt_df.shape != pred_df.shape:
            raise SystemExit(f"Shape mismatch for {var}: GT {gt_df.shape}, PRED {pred_df.shape}")

        mask_af = _africa_mask(gt_df)
        n_af = int(mask_af.sum())
        print(f"  Africa rows: {n_af}")
        if n_af == 0:
            print("  No Africa rows found; skipping shrinkage for this variable.")
            out_df = pred_df.copy()
        else:
            factors = compute_africa_shrinkage_factors(gt_df, pred_df, var)
            out_df = apply_africa_shrinkage(gt_df, pred_df, var, factors)
            all_params[var] = {
                "factors": factors,
                "africa_rows": n_af,
            }

        out_path = out_dir / f"predictions_Y_{var}_bias_corrected_africa_shrinkage.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Wrote Africa-shrunk predictions to {out_path}")

    if all_params:
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        params_path = analysis_dir / "africa_5p_shrinkage_params.json"
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(all_params, f, indent=2)
        print(f"\nSaved Africa shrinkage parameters to {params_path}")


if __name__ == "__main__":
    main()

