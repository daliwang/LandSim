#!/usr/bin/env python3
"""
Merge Amazon-only and Africa-only 5P bias-corrected predictions into one set.

Given a Phase2 run directory with:

  <run-dir>/cnp_inference_entire_dataset/cnp_predictions/
    soil_2d_predictions/                               (raw Phase2 predictions)
    soil_2d_predictions_5P_bias_corrected_amazon/      (Amazon-only correction)
    soil_2d_predictions_5P_bias_corrected_africa/      (Africa-only correction)

this script creates:

  soil_2d_predictions_5P_bias_corrected_amazon_africa/

such that for each 5P variable and grid cell:
  - If (lat, lon) in Amazon box [-30, 10] x [270, 330],
      use the Amazon-corrected value.
  - Else if (lat, lon) in Africa box [-15, 15] x [0, 30],
      use the Africa-corrected value.
  - Else:
      keep the raw Phase2 prediction.

Usage (example):

  python scripts/merge_5p_bias_corrected_amazon_africa.py \\
    --run-dir cnp_results/run_20260305_153217_phase2_pvariable_focus
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

FIVE_P: List[str] = [
    "labilep_vr",
    "occlp_vr",
    "solutionp_vr",
    "secondp_vr",
    "primp_vr",
]

# Boxes: (lat_min, lat_max, lon_min, lon_max)
AMAZON_BOX = (-30.0, 10.0, 270.0, 330.0)
AFRICA_BOX = (-15.0, 15.0, 0.0, 30.0)


def _region_mask(df: pd.DataFrame, box) -> np.ndarray:
    """Return mask for rows inside a given (lat_min, lat_max, lon_min, lon_max) box."""
    lat_min, lat_max, lon_min, lon_max = box
    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("CSV must contain 'Longitude' and 'Latitude' columns.")
    lon = df["Longitude"].to_numpy()
    lat = df["Latitude"].to_numpy()
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def _per_layer_columns(df: pd.DataFrame, var: str) -> List[str]:
    """Return ordered list of layer columns for a given 5P var (Y_<var>_col1_layer*)."""
    prefix = f"Y_{var}_col1_layer"
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.split("layer")[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Amazon-only and Africa-only 5P bias-corrected CSVs into a single "
            "prediction set: Amazon box uses Amazon-corrected values, Africa box uses "
            "Africa-corrected, elsewhere keep raw Phase2 predictions."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Phase2 run directory containing cnp_inference_entire_dataset/cnp_predictions.",
    )
    parser.add_argument(
        "--raw-subdir",
        default="soil_2d_predictions",
        help="Subdir under cnp_predictions with raw Phase2 soil2d predictions.",
    )
    parser.add_argument(
        "--amazon-subdir",
        default="soil_2d_predictions_5P_bias_corrected_amazon",
        help="Subdir with Amazon-only bias-corrected 5P CSVs.",
    )
    parser.add_argument(
        "--africa-subdir",
        default="soil_2d_predictions_5P_bias_corrected_africa",
        help="Subdir with Africa-only bias-corrected 5P CSVs.",
    )
    parser.add_argument(
        "--output-subdir",
        default="soil_2d_predictions_5P_bias_corrected_amazon_africa",
        help="Output subdir for merged 5P CSVs.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    pred_root = run_dir / "cnp_inference_entire_dataset" / "cnp_predictions"
    raw_dir = pred_root / args.raw_subdir
    amazon_dir = pred_root / args.amazon_subdir
    africa_dir = pred_root / args.africa_subdir
    out_dir = pred_root / args.output_subdir

    if not raw_dir.is_dir():
        raise SystemExit(f"Raw predictions dir not found: {raw_dir}")
    if not amazon_dir.is_dir():
        raise SystemExit(f"Amazon-corrected dir not found: {amazon_dir}")
    if not africa_dir.is_dir():
        raise SystemExit(f"Africa-corrected dir not found: {africa_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Raw dir: {raw_dir}")
    print(f"Amazon-corrected dir: {amazon_dir}")
    print(f"Africa-corrected dir: {africa_dir}")
    print(f"Output dir: {out_dir}")

    for var in FIVE_P:
        raw_path = raw_dir / f"predictions_Y_{var}.csv"
        amz_path = amazon_dir / f"predictions_Y_{var}_bias_corrected.csv"
        afr_path = africa_dir / f"predictions_Y_{var}_bias_corrected.csv"
        if not raw_path.is_file() or not amz_path.is_file() or not afr_path.is_file():
            print(f"Skipping {var}: missing CSV ({raw_path}, {amz_path}, {afr_path})")
            continue

        print(f"\nVariable: {var}")
        print(f"  Raw:    {raw_path}")
        print(f"  Amazon: {amz_path}")
        print(f"  Africa: {afr_path}")

        raw_df = pd.read_csv(raw_path)
        amz_df = pd.read_csv(amz_path)
        afr_df = pd.read_csv(afr_path)

        if raw_df.shape != amz_df.shape or raw_df.shape != afr_df.shape:
            raise SystemExit(
                f"Shape mismatch for {var}: raw {raw_df.shape}, amazon {amz_df.shape}, africa {afr_df.shape}"
            )

        # Sanity check coordinates
        for name, df in [("amazon", amz_df), ("africa", afr_df)]:
            if not np.allclose(raw_df["Longitude"], df["Longitude"]) or not np.allclose(
                raw_df["Latitude"], df["Latitude"]
            ):
                raise SystemExit(f"Longitude/Latitude mismatch between raw and {name} for {var}")

        mask_amz = _region_mask(raw_df, AMAZON_BOX)
        mask_afr = _region_mask(raw_df, AFRICA_BOX)

        print(f"  Amazon rows: {int(mask_amz.sum())}")
        print(f"  Africa rows: {int(mask_afr.sum())}")

        out_df = raw_df.copy()
        layer_cols = _per_layer_columns(raw_df, var)

        # Overwrite Amazon region from amazon-corrected
        for col in layer_cols:
            out_df.loc[mask_amz, col] = amz_df.loc[mask_amz, col].to_numpy()
        # Overwrite Africa region from africa-corrected
        for col in layer_cols:
            out_df.loc[mask_afr, col] = afr_df.loc[mask_afr, col].to_numpy()

        out_path = out_dir / f"predictions_Y_{var}_bias_corrected_amazon_africa.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Wrote merged predictions to {out_path}")


if __name__ == "__main__":
    main()

