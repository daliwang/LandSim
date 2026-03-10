import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


P_VARIABLES: List[str] = [
    "labilep_vr",
    "occlp_vr",
    "solutionp_vr",
    "secondp_vr",
    "primp_vr",
]

# Default Amazon site from docs/AMAZON_5P_QUALITY_SUMMARY.md
DEFAULT_AMAZON_LON = 303.75
DEFAULT_AMAZON_LAT = -17.434553


@dataclass
class ModelSpec:
    name: str
    run_dir: str
    predictions_rel_dir: str
    filename_template: str


def _find_row_for_site(
    df: pd.DataFrame, lon: float, lat: float, atol: float = 1e-4
) -> int:
    """Return integer row index in df matching the given lon/lat (within atol)."""
    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("DataFrame must contain 'Longitude' and 'Latitude' columns.")

    lon_vals = df["Longitude"].values
    lat_vals = df["Latitude"].values
    mask = np.isclose(lon_vals, lon, atol=atol) & np.isclose(lat_vals, lat, atol=atol)

    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(
            f"No row found at lon={lon}, lat={lat} (within atol={atol}). "
            "Check that the site exists in these CSVs."
        )
    if idx.size > 1:
        raise ValueError(
            f"Multiple rows found at lon={lon}, lat={lat}; "
            "expected exactly one grid cell."
        )
    return int(idx[0])


def _extract_profile(
    df: pd.DataFrame,
    var: str,
    row_idx: int,
) -> Tuple[np.ndarray, List[str]]:
    """Extract the vertical profile (all layers) for one variable at row_idx."""
    prefix = f"Y_{var}_col1_layer"
    layer_cols = sorted([c for c in df.columns if c.startswith(prefix)])
    if not layer_cols:
        raise ValueError(f"No columns starting with '{prefix}' found.")

    values = df.loc[row_idx, layer_cols].astype(float).values
    return values, layer_cols


def load_profiles_for_site(
    gt_run_dir: str,
    models: List[ModelSpec],
    variables: List[str],
    lon: float,
    lat: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], List[str]]:
    """Load GT and model profiles for all variables at a single site."""
    gt_root = os.path.join(
        gt_run_dir,
        "cnp_inference_entire_dataset",
        "cnp_predictions",
        "soil_2d_ground_truth",
    )

    gt_profiles: Dict[str, np.ndarray] = {}
    model_profiles: Dict[str, Dict[str, np.ndarray]] = {m.name: {} for m in models}
    layer_labels: List[str] = []

    for var in variables:
        gt_path = os.path.join(gt_root, f"ground_truth_Y_{var}.csv")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        gt_df = pd.read_csv(gt_path)
        row_idx = _find_row_for_site(gt_df, lon=lon, lat=lat)
        gt_vals, layer_cols = _extract_profile(gt_df, var=var, row_idx=row_idx)

        gt_profiles[var] = gt_vals
        if not layer_labels:
            layer_labels = layer_cols

        for model in models:
            pred_dir = os.path.join(model.run_dir, model.predictions_rel_dir)
            pred_path = os.path.join(
                pred_dir, model.filename_template.format(var=var)
            )
            if not os.path.exists(pred_path):
                # e.g. phase2 bias-corrected only has 4 P vars; skip missing vars
                model_profiles[model.name][var] = np.full_like(gt_vals, np.nan)
                continue

            pred_df = pd.read_csv(pred_path)
            if pred_df.shape != gt_df.shape:
                raise ValueError(
                    f"Shape mismatch GT vs predictions for model '{model.name}', var '{var}': "
                    f"GT {gt_df.shape}, PRED {pred_df.shape}"
                )

            # Ensure same lon/lat ordering
            if not np.allclose(
                pred_df["Longitude"].values, gt_df["Longitude"].values
            ) or not np.allclose(
                pred_df["Latitude"].values, gt_df["Latitude"].values
            ):
                raise ValueError(
                    f"Longitude/Latitude mismatch between GT and model '{model.name}' for var '{var}'"
                )

            pred_vals, _ = _extract_profile(pred_df, var=var, row_idx=row_idx)
            model_profiles[model.name][var] = pred_vals

    return gt_profiles, model_profiles, layer_labels


def make_profile_line_plots(
    output_dir: str,
    gt_profiles: Dict[str, np.ndarray],
    model_profiles: Dict[str, Dict[str, np.ndarray]],
    layer_labels: List[str],
    lon: float,
    lat: float,
) -> None:
    """Plot vertical profiles (one line per model + GT) across layers."""
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(model_profiles.keys())

    for var, gt_vals in gt_profiles.items():
        num_layers = len(gt_vals)
        layers = np.arange(1, num_layers + 1)

        plt.figure(figsize=(6, 6))

        # Ground truth profile
        plt.plot(
            layers,
            gt_vals,
            marker="o",
            linestyle="-",
            color="black",
            label="ground_truth",
        )

        # Each model's profile (skip if all NaN, e.g. missing variable for that model)
        for model_name in model_names:
            preds = model_profiles[model_name][var]
            if np.all(np.isnan(preds)):
                continue
            plt.plot(
                layers,
                preds,
                marker="o",
                linestyle="--",
                label=model_name,
                alpha=0.9,
            )

        plt.xlabel("Soil layer index")
        plt.ylabel(f"{var} value")
        plt.title(
            f"{var} vertical profile at Amazon site\n"
            f"(lon={lon}, lat={lat}, {num_layers} layers)"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"amazon_site_profile_{var}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()


def write_profiles_csv(
    output_dir: str,
    gt_profiles: Dict[str, np.ndarray],
    model_profiles: Dict[str, Dict[str, np.ndarray]],
    layer_labels: List[str],
    lon: float,
    lat: float,
) -> None:
    rows: List[Dict[str, object]] = []

    for var, gt_vals in gt_profiles.items():
        num_layers = len(gt_vals)
        for layer_idx in range(num_layers):
            layer_name = layer_labels[layer_idx] if layer_idx < len(layer_labels) else f"layer_{layer_idx+1}"
            row: Dict[str, object] = {
                "variable": var,
                "layer_index": layer_idx + 1,
                "layer_col": layer_name,
                "lon": lon,
                "lat": lat,
                "gt": float(gt_vals[layer_idx]),
            }
            for model_name, profiles in model_profiles.items():
                row[f"pred_{model_name}"] = float(profiles[var][layer_idx])
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "amazon_site_5p_profiles.csv")
    df.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 5 P soil2d variables at the Amazon site across multiple runs "
            "and produce vertical profile plots vs ground truth."
        )
    )
    parser.add_argument(
        "--gt-run-dir",
        default="cnp_results/run_20260228_214757_natveg_improved",
        help="Run directory providing soil_2d_ground_truth CSVs.",
    )
    parser.add_argument(
        "--natveg-run-dir",
        default="cnp_results/run_20260228_214757_natveg_improved",
        help="Baseline natveg_improved run directory.",
    )
    parser.add_argument(
        "--natveg-bias-corrected-subdir",
        default="cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_5P_bias_corrected",
        help=(
            "Relative path (from natveg-run-dir) to the bias/scale–corrected "
            "prediction CSVs output by apply_5p_bias_scale_correction.py."
        ),
    )
    parser.add_argument(
        "--phase2-run-dir",
        default="cnp_results/run_20260305_153217_phase2_pvariable_focus",
        help="Run directory for phase2_pvariable_focus.",
    )
    parser.add_argument(
        "--phase2-bias-corrected-subdir",
        default="cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_4P_bias_corrected_phase2",
        help=(
            "Relative path (from phase2-run-dir) to bias/scale–corrected Phase2 "
            "prediction CSVs, if available (produced by apply_5p_bias_scale_correction.py)."
        ),
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=DEFAULT_AMAZON_LON,
        help="Amazon site longitude (degrees, 0–360).",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=DEFAULT_AMAZON_LAT,
        help="Amazon site latitude (degrees).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory for plots and CSV. "
            "Default: <natveg-run-dir>/analysis/amazon_5p_comparison_bias_correction"
        ),
    )

    args = parser.parse_args()

    gt_run_dir = os.path.abspath(args.gt_run_dir)
    natveg_run_dir = os.path.abspath(args.natveg_run_dir)
    phase2_run_dir = os.path.abspath(args.phase2_run_dir)
    lon = args.lon
    lat = args.lat

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(
            natveg_run_dir, "analysis", "amazon_5p_comparison_bias_correction"
        )

    models: List[ModelSpec] = [
        # Baseline natveg_improved (for global comparison)
        ModelSpec(
            name="natveg_improved",
            run_dir=natveg_run_dir,
            predictions_rel_dir=(
                "cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions"
            ),
            filename_template="predictions_Y_{var}.csv",
        ),
        # Bias/scale–corrected natveg_improved (when using natveg as baseline)
        ModelSpec(
            name="natveg_improved_bias_corrected",
            run_dir=natveg_run_dir,
            predictions_rel_dir=args.natveg_bias_corrected_subdir,
            filename_template="predictions_Y_{var}_bias_corrected.csv",
        ),
        # Original Phase2 predictions
        ModelSpec(
            name="phase2_pvariable_focus",
            run_dir=phase2_run_dir,
            predictions_rel_dir=(
                "cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions"
            ),
            filename_template="predictions_Y_{var}.csv",
        ),
        # Optional bias/scale–corrected Phase2 predictions (4P in tropics)
        ModelSpec(
            name="phase2_pvariable_focus_bias_corrected",
            run_dir=phase2_run_dir,
            predictions_rel_dir=args.phase2_bias_corrected_subdir,
            filename_template="predictions_Y_{var}_bias_corrected.csv",
        ),
    ]

    for model in models:
        if not os.path.isdir(model.run_dir):
            raise SystemExit(f"Run directory for model '{model.name}' not found: {model.run_dir}")

    if not os.path.isdir(gt_run_dir):
        raise SystemExit(f"gt-run-dir not found: {gt_run_dir}")

    print(f"Using GT from: {gt_run_dir}")
    print("Models:")
    for m in models:
        print(f"  {m.name}: run_dir={m.run_dir}, rel_dir={m.predictions_rel_dir}")
    print(f"Amazon site: lon={lon}, lat={lat}")
    print(f"Output directory: {output_dir}")

    gt_profiles, model_profiles, layer_labels = load_profiles_for_site(
        gt_run_dir=gt_run_dir,
        models=models,
        variables=P_VARIABLES,
        lon=lon,
        lat=lat,
    )

    make_profile_line_plots(
        output_dir=output_dir,
        gt_profiles=gt_profiles,
        model_profiles=model_profiles,
        layer_labels=layer_labels,
        lon=lon,
        lat=lat,
    )

    write_profiles_csv(
        output_dir=output_dir,
        gt_profiles=gt_profiles,
        model_profiles=model_profiles,
        layer_labels=layer_labels,
        lon=lon,
        lat=lat,
    )


if __name__ == "__main__":
    main()

