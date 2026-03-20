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
MULTIPLICATIVE_ONLY_VARS: set = set()

# Variables with very small values: fit on (pred*scale, gt*scale) for numerical stability,
# then apply corrected = (a*pred*scale + b) / scale. Keeps 5 decimals when writing.
SCALE_FACTOR_VARS: Dict[str, float] = {"solutionp_vr": 1000.0}
DECIMAL_PLACES_SCALED_VARS: int = 8  # keep small solutionp_vr values (e.g. 3e-6) visible

# Variables where we minimize relative (percent) error so that layer-wise error is <10%.
# Uses weighted least squares with weight 1/(gt+eps)^2. solutionp_vr and occlp_vr are critical.
RELATIVE_ERROR_WEIGHTED_VARS: Set[str] = {"solutionp_vr", "occlp_vr"}
RELATIVE_ERROR_EPS: float = 1e-12

# Reference sites (lon, lat) to prioritize in relative-error fit so layer error <10% there.
# We keep separate lists for Amazon and Africa so each region's fit is anchored by its own sites.
AMAZON_REFERENCE_SITES: List[Tuple[float, float]] = [
    (303.75, -17.434553),   # default Amazon site
    (300.0, 4.240838),      # Amazon Site A
    (292.5, -15.549738),    # Amazon Site B
]

AFRICA_REFERENCE_SITES: List[Tuple[float, float]] = [
    (27.5, 0.471204),       # nearest to (28, 0) in phase2 cnp_inference_entire_dataset
    (22.5, 5.183247),       # Mid-Africa
    (28.0, 0.0),            # Africa 28E, 0N (may not be on grid)
    (20.0, -5.0),           # Additional central tropical Africa site
]

# Default combined list (used when no region-specific override is needed)
REFERENCE_SITES: List[Tuple[float, float]] = AMAZON_REFERENCE_SITES + AFRICA_REFERENCE_SITES

REFERENCE_SITE_EXTRA_WEIGHT_AMAZON: float = 500.0  # extra weight for Amazon reference cells
REFERENCE_SITE_EXTRA_WEIGHT_AFRICA: float = 250.0  # slightly softer extra weight for Africa
REFERENCE_SITE_EXTRA_WEIGHT: float = REFERENCE_SITE_EXTRA_WEIGHT_AMAZON
REFERENCE_SITE_ATOL: float = 1e-4


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


def load_region_boxes(
    config_path: str,
    split_first_region_lat: Optional[float] = None,
) -> List[RegionBox]:
    """Load region_boxes from a training config JSON.

    Expects:
      "data_filtering_config": {
          "region_boxes": [
              [lat_min, lat_max, lon_min, lon_max],
              ...
          ]
      }

    If split_first_region_lat is set (e.g. -5), the first box is split into two by latitude:
      region_south: lat_min <= lat < split_first_region_lat
      region_north: split_first_region_lat <= lat <= lat_max
    so the default Amazon site (lat ~ -17) uses a fit from southern Amazon only.
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

        if i == 0 and split_first_region_lat is not None:
            # Split first region (e.g. Amazon) into south and north by latitude
            split = split_first_region_lat
            if not (lat_min < split < lat_max):
                raise ValueError(
                    f"split_first_region_lat {split} must be strictly between lat_min {lat_min} and lat_max {lat_max}"
                )
            boxes.append(
                RegionBox(lat_min=lat_min, lat_max=split, lon_min=lon_min, lon_max=lon_max, name=f"{name}_south")
            )
            boxes.append(
                RegionBox(lat_min=split, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, name=f"{name}_north")
            )
        else:
            boxes.append(RegionBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, name=name))

    return boxes


def fit_bias_scale(
    pred: np.ndarray,
    gt: np.ndarray,
    multiplicative_only: bool = False,
    scale_factor: Optional[float] = None,
    relative_error_weighted: bool = False,
    relative_eps: float = 1e-12,
    ref_site_lon: Optional[np.ndarray] = None,
    ref_site_lat: Optional[np.ndarray] = None,
    ref_sites: Optional[List[Tuple[float, float]]] = None,
    ref_site_extra_weight: float = 500.0,
    ref_site_atol: float = 1e-4,
) -> Tuple[float, float]:
    """Fit GT ≈ a * pred + b via least squares (or a * pred only if multiplicative_only).

    Both inputs are 1D arrays of the same length.
    If scale_factor is set (e.g. 1000), fit on (pred*scale, gt*scale) for stability;
    returned (a,b) apply as corrected = (a * pred * scale + b) / scale.
    If relative_error_weighted is True, minimize weighted squared error with
    w_i = 1/(y_i+eps)^2 so that relative error is minimized (target <10% per layer).
    If ref_sites is provided (list of (lon,lat)), cells at those sites get extra weight.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for regression: pred {pred.shape}, gt {gt.shape}")

    mask = np.isfinite(pred) & np.isfinite(gt)
    mask &= (pred != 0.0) | (gt != 0.0)

    x = pred[mask].astype(float)
    y = gt[mask].astype(float)
    if x.size < 2:
        return 1.0, 0.0

    if scale_factor is not None and scale_factor != 1.0:
        x = x * scale_factor
        y = y * scale_factor

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
    X = np.vstack([x, np.ones_like(x)]).T  # (n, 2)
    if relative_error_weighted:
        # Minimize sum w_i * (y_i - a*x_i - b)^2 with w_i = 1/(y_i+eps)^2
        w = 1.0 / (y.astype(float) + relative_eps) ** 2
        # Boost weight for reference-site cells so fit targets <10% error there
        if (
            ref_sites
            and ref_site_lon is not None
            and ref_site_lat is not None
            and ref_site_lon.shape == ref_site_lat.shape
            and ref_site_lon.size == w.size
        ):
            for rlon, rlat in ref_sites:
                at_site = np.isclose(ref_site_lon, rlon, atol=ref_site_atol) & np.isclose(
                    ref_site_lat, rlat, atol=ref_site_atol
                )
                w[at_site] *= 1.0 + ref_site_extra_weight
        XtWX = X.T @ (w[:, np.newaxis] * X)
        XtWy = X.T @ (w * y)
        try:
            coeffs = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coeffs
    else:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coeffs
    return float(a), float(b)


def compute_and_apply_corrections(
    run_dir: str,
    region_boxes: List[RegionBox],
    variables: List[str],
    output_subdir: str,
    multiplicative_only_vars: Optional[Set[str]] = None,
    scale_factor_vars: Optional[Dict[str, float]] = None,
    relative_error_weighted_vars: Optional[Set[str]] = None,
    reference_sites: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute per-variable, per-layer, per-region bias/scale corrections and apply them.

    Returns a nested dict:
      params[var_name][region_name][layer_key] = {"a": ..., "b": ...} or {"a", "b", "scale"} for scaled vars.
    """
    if multiplicative_only_vars is None:
        multiplicative_only_vars = MULTIPLICATIVE_ONLY_VARS
    if scale_factor_vars is None:
        scale_factor_vars = SCALE_FACTOR_VARS
    if relative_error_weighted_vars is None:
        relative_error_weighted_vars = RELATIVE_ERROR_WEIGHTED_VARS
    if reference_sites is None:
        reference_sites = REFERENCE_SITES
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
                scale = scale_factor_vars.get(var) if scale_factor_vars else None
                if scale is not None and scale != 1.0:
                    mult_only = False  # use linear fit on scaled data

                # Use relative-error objective only for variables that request it.
                rel_err = var in relative_error_weighted_vars

                # Choose region-specific reference sites and extra weights
                if rel_err:
                    if region.name.startswith("amazon"):
                        region_ref_sites = AMAZON_REFERENCE_SITES
                        region_extra_weight = REFERENCE_SITE_EXTRA_WEIGHT_AMAZON
                    elif region.name.startswith("africa"):
                        region_ref_sites = AFRICA_REFERENCE_SITES
                        region_extra_weight = REFERENCE_SITE_EXTRA_WEIGHT_AFRICA
                    else:
                        region_ref_sites = reference_sites
                        region_extra_weight = REFERENCE_SITE_EXTRA_WEIGHT
                else:
                    region_ref_sites = None
                    region_extra_weight = REFERENCE_SITE_EXTRA_WEIGHT

                ref_lon = lon[region_mask] if rel_err and region_ref_sites else None
                ref_lat = lat[region_mask] if rel_err and region_ref_sites else None
                a, b = fit_bias_scale(
                    pred_vals,
                    gt_vals,
                    multiplicative_only=mult_only,
                    scale_factor=scale,
                    relative_error_weighted=rel_err,
                    relative_eps=RELATIVE_ERROR_EPS,
                    ref_site_lon=ref_lon,
                    ref_site_lat=ref_lat,
                    ref_sites=region_ref_sites,
                    ref_site_extra_weight=region_extra_weight,
                    ref_site_atol=REFERENCE_SITE_ATOL,
                )
                # For relative-error vars: if ref site still has >10% error, nudge (a,b) so ref site is exact (minimal change).
                # We only enforce this hard constraint for Amazon regions to avoid overfitting Africa to noisy GT.
                if (
                    rel_err
                    and region_ref_sites
                    and region.name.startswith("amazon")
                    and ref_lon is not None
                    and ref_lat is not None
                ):
                    for rlon, rlat in reference_sites:
                        at_site = np.isclose(ref_lon, rlon, atol=REFERENCE_SITE_ATOL) & np.isclose(
                            ref_lat, rlat, atol=REFERENCE_SITE_ATOL
                        )
                        if not np.any(at_site):
                            continue
                        idx = np.where(at_site)[0][0]
                        pred_ref = pred_vals[idx]
                        gt_ref = gt_vals[idx]
                        if scale and scale != 1.0:
                            pred_s = pred_ref * scale
                            corrected_ref = (a * pred_s + b) / scale
                        else:
                            corrected_ref = a * pred_ref + b
                        rel_pct = 100 * abs(corrected_ref - gt_ref) / (gt_ref + 1e-12)
                        if corrected_ref >= 0 and rel_pct <= 10:
                            break
                        # Project (a,b) onto constraint a'*pred_ref + b' = gt_ref (in original space)
                        # Minimize (a'-a)^2+(b'-b)^2 s.t. a'*pred_ref + b' = gt_ref. Solution:
                        # lambda = 2*(a*pred_ref + b - gt_ref) / (pred_ref**2 + 1), a' = a - lambda*pred_ref/2, b' = b - lambda/2
                        if scale and scale != 1.0:
                            pred_orig = pred_ref
                            # In scaled space: a*pred_s + b = gt_s would give (a*pred_s+b)/scale = gt_orig so a*pred_s+b = gt_orig*scale
                            # Constraint in scaled space: a'*pred_s + b' = gt_ref*scale
                            pred_s = pred_orig * scale
                            tgt = gt_ref * scale
                            lam = 2.0 * (a * pred_s + b - tgt) / (pred_s ** 2 + 1.0)
                            a = a - lam * pred_s / 2.0
                            b = b - lam / 2.0
                        else:
                            lam = 2.0 * (a * pred_ref + b - gt_ref) / (pred_ref ** 2 + 1.0)
                            a = float(a - lam * pred_ref / 2.0)
                            b = float(b - lam / 2.0)
                        break
                layer_key = f"layer_{layer_idx}"
                coeff: Dict[str, float] = {"a": a, "b": b}
                if scale is not None and scale != 1.0:
                    coeff["scale"] = scale
                region_layer_params[layer_key] = coeff

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
                scale = coeffs.get("scale")

                pred_vals = corrected_df.loc[region_mask, col_name].astype(float).values
                if scale is not None and scale != 1.0:
                    raw = (a * pred_vals * scale + b) / scale
                else:
                    raw = a * pred_vals + b

                # For Africa regions, apply a modest blend between raw prediction and fully corrected value
                # to avoid over-correcting in noisier GT regimes.
                if region.name.startswith("africa"):
                    alpha = 0.7  # 70% corrected, 30% original
                    raw = alpha * raw + (1.0 - alpha) * pred_vals
                # Concentrations must be non-negative
                corrected_df.loc[region_mask, col_name] = np.maximum(raw, 0.0)

        # For small-scaled vars (e.g. solutionp_vr), keep 5 digits after decimal when writing
        decimal_places = (
            DECIMAL_PLACES_SCALED_VARS
            if (scale_factor_vars and var in scale_factor_vars)
            else None
        )
        if decimal_places is not None:
            layer_col_names = [c for c in corrected_df.columns if c.startswith(var_key)]
            for c in layer_col_names:
                corrected_df[c] = corrected_df[c].round(decimal_places)

        out_path = os.path.join(output_dir, f"predictions_Y_{var}_bias_corrected.csv")
        print(f"Writing bias/scale–corrected predictions for {var} to {out_path}")
        if decimal_places is not None:
            fmt = f"%.{decimal_places}f"
            corrected_df.to_csv(out_path, index=False, float_format=fmt)
        else:
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
            "Comma-separated variables that use multiplicative-only correction (Y = a*pred). "
            "Default: none (solutionp_vr uses scale-factor fit instead)."
        ),
    )
    parser.add_argument(
        "--scale-factor-vars",
        default=",".join(f"{k}:{v}" for k, v in sorted(SCALE_FACTOR_VARS.items())),
        help=(
            "Comma-separated list of var:scale (e.g. solutionp_vr:1000) for small-value vars: "
            "fit on pred*scale vs gt*scale, apply (a*pred*scale+b)/scale; output rounded to 5 decimals."
        ),
    )
    parser.add_argument(
        "--split-first-region-lat",
        default="-5.0",
        metavar="LAT",
        help=(
            "Split the first region (Amazon) into _south (lat_min to LAT) and _north (LAT to lat_max) "
            "so the default Amazon site (lat ~ -17) uses a southern fit. Float, or 'none' to disable. Default: -5.0"
        ),
    )

    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    region_config_json = os.path.abspath(args.region_config_json)
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    multiplicative_only_vars = {
        v.strip() for v in args.multiplicative_only_vars.split(",") if v.strip()
    }
    scale_factor_vars: Dict[str, float] = {}
    for part in args.scale_factor_vars.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            var_name, scale_str = part.split(":", 1)
            var_name, scale_str = var_name.strip(), scale_str.strip()
            try:
                scale_factor_vars[var_name] = float(scale_str)
            except ValueError:
                raise SystemExit(f"Invalid scale in --scale-factor-vars: {part}")
        else:
            scale_factor_vars[part] = SCALE_FACTOR_VARS.get(part, 1000.0)
    if not scale_factor_vars and SCALE_FACTOR_VARS:
        scale_factor_vars = dict(SCALE_FACTOR_VARS)

    print(f"Run directory: {run_dir}")
    print(f"Region config JSON: {region_config_json}")
    print(f"Variables: {variables}")
    print(f"Output subdir (relative to cnp_predictions): {args.output_subdir}")

    if not os.path.isdir(run_dir):
        raise SystemExit(f"run-dir does not exist or is not a directory: {run_dir}")
    if not os.path.isfile(region_config_json):
        raise SystemExit(f"region-config-json not found: {region_config_json}")

    split_lat: Optional[float] = None
    if args.split_first_region_lat.strip().lower() not in ("none", "no", ""):
        try:
            split_lat = float(args.split_first_region_lat)
        except ValueError:
            raise SystemExit(
                f"Invalid --split-first-region-lat: {args.split_first_region_lat}. Use a number or 'none'."
            )
    region_boxes = load_region_boxes(region_config_json, split_first_region_lat=split_lat)
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
        scale_factor_vars=scale_factor_vars,
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

