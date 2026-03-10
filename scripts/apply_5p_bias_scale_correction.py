import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


P_VARIABLES_DEFAULT: List[str] = [
    "labilep_vr",
    "occlp_vr",
    "solutionp_vr",
    "secondp_vr",
    "primp_vr",
]

# Variables with very small values where linear (a*pred+b) often goes negative;
# use multiplicative-only correction (Y = a*pred) so concentrations stay non-negative.
MULTIPLICATIVE_ONLY_VARS: set = {"solutionp_vr"}


@dataclass
class RegionBox:
    """Lat/lon bounds for a rectangular region in degrees (lon in 0–360)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    name: str

    def contains(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)


def load_region_boxes(config_path: str) -> List[RegionBox]:
    """Load region_boxes from a training config JSON.

    Expects:
      "data_filtering_config": {
          "region_boxes": [
              [lat_min, lat_max, lon_min, lon_max],
              ...
          ]
      }
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dfc = cfg.get("data_filtering_config", {})
    raw_boxes = dfc.get("region_boxes")
    if not raw_boxes:
        raise ValueError(f"No data_filtering_config.region_boxes found in {config_path}")

    # Name the first two boxes to match docs/FIVE_P_TWO_REGION_GROUND_TRUTH_ANALYSIS.md
    names = ["amazon", "africa"]
    boxes: List[RegionBox] = []
    for i, box in enumerate(raw_boxes):
        if len(box) != 4:
            raise ValueError(f"Region box at index {i} must have 4 entries [lat_min, lat_max, lon_min, lon_max], got: {box}")
        lat_min, lat_max, lon_min, lon_max = box
        name = names[i] if i < len(names) else f"region_{i}"
        boxes.append(RegionBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, name=name))

    return boxes


def fit_bias_scale(
    pred: np.ndarray, gt: np.ndarray, multiplicative_only: bool = False
) -> Tuple[float, float]:
    """Fit GT ≈ a * pred + b via least squares (or a * pred only if multiplicative_only).

    Both inputs are 1D arrays of the same length.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for regression: pred {pred.shape}, gt {gt.shape}")

    mask = np.isfinite(pred) & np.isfinite(gt)
    mask &= (pred != 0.0) | (gt != 0.0)

    x = pred[mask]
    y = gt[mask]
    if x.size < 2:
        return 1.0, 0.0

    if multiplicative_only:
        # Regression through origin: y = a * x => a = (x'y) / (x'x). Keeps corrected values non-negative.
        xx = np.dot(x, x)
        if xx <= 0:
            a = 1.0
        else:
            a = float(np.dot(x, y) / xx)
        a = max(a, 1e-6)  # avoid negative or zero scale
        return a, 0.0

    # Linear regression y = a * x + b
    X = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeffs
    return float(a), float(b)


def compute_and_apply_corrections(
    run_dir: str,
    region_boxes: List[RegionBox],
    variables: List[str],
    output_subdir: str,
    multiplicative_only_vars: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute per-variable, per-layer, per-region bias/scale corrections and apply them.

    Returns a nested dict:
      params[var_name][region_name][layer_key] = {"a": ..., "b": ...}
    """
    if multiplicative_only_vars is None:
        multiplicative_only_vars = MULTIPLICATIVE_ONLY_VARS
    predictions_root = os.path.join(
        run_dir,
        "cnp_inference_entire_dataset",
        "cnp_predictions",
    )
    gt_dir = os.path.join(predictions_root, "soil_2d_ground_truth")
    pred_dir = os.path.join(predictions_root, "soil_2d_predictions")

    output_dir = os.path.join(predictions_root, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    params: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for var in variables:
        var_key = f"Y_{var}_col1_layer"
        gt_path = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        pred_path = os.path.join(pred_dir, f"predictions_Y_{var}.csv")

        # Fallback to the *_vr naming convention used in this repo
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"predictions_Y_{var}.csv")

        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"predictions_Y_{var}.csv")

        # In this codebase, files are named ground_truth_Y_<var>_vr.csv and predictions_Y_<var>_vr.csv
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"predictions_Y_{var}.csv")

        # For clarity, explicitly handle the _vr suffix
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"predictions_Y_{var}.csv")

        # Final explicit names for this repo
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"predictions_Y_{var}.csv")

        # In practice, for labilep_vr, occlp_vr, etc., we expect:
        #   ground_truth_Y_labilep_vr.csv
        #   predictions_Y_labilep_vr.csv
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"predictions_Y_{var}.csv")

        # And the explicit natveg_improved naming we have already seen:
        explicit_gt = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
        explicit_pred = os.path.join(pred_dir, f"predictions_Y_{var}.csv")
        if os.path.exists(explicit_gt):
            gt_path = explicit_gt
        if os.path.exists(explicit_pred):
            pred_path = explicit_pred

        # As a final guard, ensure the expected _vr suffixed paths exist
        if not os.path.exists(gt_path):
            alt_gt = os.path.join(gt_dir, f"ground_truth_Y_{var}.csv")
            if os.path.exists(alt_gt):
                gt_path = alt_gt
        if not os.path.exists(pred_path):
            alt_pred = os.path.join(pred_dir, f"predictions_Y_{var}.csv")
            if os.path.exists(alt_pred):
                pred_path = alt_pred

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"Could not find ground truth or prediction CSV for variable '{var}'. "
                f"Tried paths like {gt_path} and {pred_path}"
            )

        print(f"Loading GT from {gt_path}")
        print(f"Loading predictions from {pred_path}")

        gt_df = pd.read_csv(gt_path)
        pred_df = pd.read_csv(pred_path)

        if gt_df.shape != pred_df.shape:
            raise ValueError(
                f"Shape mismatch for GT vs predictions for {var}: "
                f"GT {gt_df.shape}, PRED {pred_df.shape}"
            )

        for col in ["Longitude", "Latitude"]:
            if col not in gt_df.columns or col not in pred_df.columns:
                raise ValueError(
                    f"Expected '{col}' column in both GT and prediction CSVs for {var}"
                )

        # Sanity check alignment
        if not np.allclose(gt_df["Longitude"].values, pred_df["Longitude"].values) or not np.allclose(
            gt_df["Latitude"].values, pred_df["Latitude"].values
        ):
            raise ValueError(f"Longitude/Latitude mismatch between GT and predictions for {var}")

        lon = gt_df["Longitude"].values
        lat = gt_df["Latitude"].values

        # Prepare per-layer coefficients for each region
        var_params: Dict[str, Dict[str, Dict[str, float]]] = {}
        num_layers = 0
        for c in gt_df.columns:
            if c.startswith(var_key):
                num_layers += 1

        if num_layers == 0:
            raise ValueError(f"No layer columns found for variable '{var}' (prefix '{var_key}')")

        for region in region_boxes:
            region_mask = region.contains(lat=lat, lon=lon)

            if not region_mask.any():
                print(f"Warning: no cells found in region '{region.name}' for variable '{var}'")
                continue

            print(f"Fitting corrections for {var} in region '{region.name}' using {region_mask.sum()} cells")
            region_layer_params: Dict[str, Dict[str, float]] = {}

            for layer_idx in range(1, num_layers + 1):
                col_name = f"{var_key}{layer_idx}"
                if col_name not in gt_df.columns or col_name not in pred_df.columns:
                    raise ValueError(
                        f"Expected column '{col_name}' in GT and prediction CSVs for {var}"
                    )

                gt_vals = gt_df.loc[region_mask, col_name].values.astype(float)
                pred_vals = pred_df.loc[region_mask, col_name].values.astype(float)

                mult_only = var in multiplicative_only_vars
                a, b = fit_bias_scale(pred_vals, gt_vals, multiplicative_only=mult_only)
                layer_key = f"layer_{layer_idx}"
                region_layer_params[layer_key] = {"a": a, "b": b}

            var_params[region.name] = region_layer_params

        params[var] = var_params

        # Apply corrections to a copy of the prediction DataFrame
        corrected_df = pred_df.copy()
        for region in region_boxes:
            region_mask = region.contains(lat=lat, lon=lon)
            if not region_mask.any():
                continue

            region_param_dict = params[var].get(region.name)
            if not region_param_dict:
                # Nothing to apply for this region/variable
                continue

            for layer_idx in range(1, num_layers + 1):
                col_name = f"{var_key}{layer_idx}"
                layer_key = f"layer_{layer_idx}"
                coeffs = region_param_dict[layer_key]
                a = coeffs["a"]
                b = coeffs["b"]

                raw = a * corrected_df.loc[region_mask, col_name].astype(float) + b
                # Concentrations must be non-negative (multiplicative-only already keeps solutionp_vr safe)
                corrected_df.loc[region_mask, col_name] = np.maximum(raw, 0.0)

        out_path = os.path.join(output_dir, f"predictions_Y_{var}_bias_corrected.csv")
        print(f"Writing bias/scale–corrected predictions for {var} to {out_path}")
        corrected_df.to_csv(out_path, index=False)

    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply bias/scale correction on top of soil 2D 5P predictions "
            "for Amazon + Africa, using GT vs prediction CSVs from a CNP run."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Path to a cnp_results run directory, e.g. "
            "cnp_results/run_20260228_214757_natveg_improved"
        ),
    )
    parser.add_argument(
        "--region-config-json",
        default="config/training_config_two_region_five_p.json",
        help=(
            "Training config JSON that defines data_filtering_config.region_boxes "
            "for Amazon + Africa."
        ),
    )
    parser.add_argument(
        "--variables",
        default=",".join(P_VARIABLES_DEFAULT),
        help=(
            "Comma-separated list of soil2d P variables to correct. "
            f"Default: {','.join(P_VARIABLES_DEFAULT)}"
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default="soil_2d_predictions_5P_bias_corrected",
        help=(
            "Subdirectory (under cnp_inference_entire_dataset/cnp_predictions) where "
            "corrected prediction CSVs will be written."
        ),
    )
    parser.add_argument(
        "--multiplicative-only-vars",
        default=",".join(sorted(MULTIPLICATIVE_ONLY_VARS)),
        help=(
            "Comma-separated variables that use multiplicative-only correction (Y = a*pred) "
            "to avoid negative/zero concentrations. Default: solutionp_vr"
        ),
    )

    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    region_config_json = os.path.abspath(args.region_config_json)
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    multiplicative_only_vars = {
        v.strip() for v in args.multiplicative_only_vars.split(",") if v.strip()
    }

    print(f"Run directory: {run_dir}")
    print(f"Region config JSON: {region_config_json}")
    print(f"Variables: {variables}")
    print(f"Output subdir (relative to cnp_predictions): {args.output_subdir}")

    if not os.path.isdir(run_dir):
        raise SystemExit(f"run-dir does not exist or is not a directory: {run_dir}")
    if not os.path.isfile(region_config_json):
        raise SystemExit(f"region-config-json not found: {region_config_json}")

    region_boxes = load_region_boxes(region_config_json)
    print("Loaded region boxes:")
    for rb in region_boxes:
        print(
            f"  {rb.name}: lat [{rb.lat_min}, {rb.lat_max}], "
            f"lon [{rb.lon_min}, {rb.lon_max}]"
        )

    params = compute_and_apply_corrections(
        run_dir=run_dir,
        region_boxes=region_boxes,
        variables=variables,
        output_subdir=args.output_subdir,
        multiplicative_only_vars=multiplicative_only_vars,
    )

    # Save parameters to analysis/ for documentation and reuse
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    params_path = os.path.join(analysis_dir, "bias_scale_params_5P_two_regions.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Saved bias/scale parameters to {params_path}")


if __name__ == "__main__":
    main()

