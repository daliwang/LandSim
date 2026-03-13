import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message=".*multiple fill values.*", category=UserWarning, module="xarray")

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
    # Sort by numeric layer index so that layer1, layer2, ..., layer10 are in
    # the correct physical order instead of the lexicographic order
    # (layer1, layer10, layer2, ...).
    layer_cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.split("layer")[-1]),
    )
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
        gt_row_idx = _find_row_for_site(gt_df, lon=lon, lat=lat)
        gt_vals, layer_cols = _extract_profile(gt_df, var=var, row_idx=gt_row_idx)

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

            # Find the matching site row independently in each prediction DataFrame.
            # This avoids assumptions about global shape or ordering and supports
            # region-only prediction CSVs (e.g., Amazon/Africa subsets).
            pred_row_idx = _find_row_for_site(pred_df, lon=lon, lat=lat)
            pred_vals, _ = _extract_profile(pred_df, var=var, row_idx=pred_row_idx)
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


def _load_5p_from_restart(path: str) -> Dict[str, np.ndarray]:
    """Load 5P variables from a restart NetCDF. Returns dict of (n_col, n_lev) arrays.
    Handles (column, levgrnd) or (gridcell, column, levgrnd) by taking gridcell=0 if needed."""
    out: Dict[str, np.ndarray] = {}
    with xr.open_dataset(path, decode_times=False) as ds:
        for var in P_VARIABLES:
            if var not in ds:
                continue
            arr = np.asarray(ds[var].values, dtype=float)
            if arr.ndim == 2:
                out[var] = arr  # (column, levgrnd)
            elif arr.ndim == 3:
                out[var] = arr[0, :, :]  # (gridcell, column, levgrnd) -> take first
            else:
                out[var] = arr.reshape(-1, arr.shape[-1])
    return out


def _profile_from_2d(data: Dict[str, np.ndarray], use_column: int = 0) -> Dict[str, np.ndarray]:
    """Extract 1D profile (first column) per variable for line plots."""
    return {v: arr[use_column, :].copy() for v, arr in data.items()}


def make_restart_profile_plots(
    output_dir: str,
    name_a: str,
    name_b: str,
    profiles_a: Dict[str, np.ndarray],
    profiles_b: Dict[str, np.ndarray],
    lon: float,
    lat: float,
) -> None:
    """Plot vertical profiles (one line per restart) for each 5P variable."""
    os.makedirs(output_dir, exist_ok=True)
    for var in P_VARIABLES:
        if var not in profiles_a or var not in profiles_b:
            continue
        a_vals = profiles_a[var]
        b_vals = profiles_b[var]
        n_layers = len(a_vals)
        if len(b_vals) != n_layers:
            continue
        layers = np.arange(1, n_layers + 1)
        plt.figure(figsize=(6, 6))
        plt.plot(layers, a_vals, marker="o", linestyle="-", label=name_a, alpha=0.9)
        plt.plot(layers, b_vals, marker="s", linestyle="--", label=name_b, alpha=0.9)
        plt.xlabel("Soil layer index")
        plt.ylabel(f"{var} value")
        plt.title(f"{var} at Amazon site (lon={lon}, lat={lat})\nrestart comparison (col 0)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"amazon_site_profile_{var}_restart_compare.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Wrote {out_path}")


def make_restart_scatter_plots(
    output_dir: str,
    name_a: str,
    name_b: str,
    data_a: Dict[str, np.ndarray],
    data_b: Dict[str, np.ndarray],
) -> None:
    """Scatter plot: value in restart A vs value in restart B for each 5P variable (all columns and layers)."""
    os.makedirs(output_dir, exist_ok=True)
    for var in P_VARIABLES:
        if var not in data_a or var not in data_b:
            continue
        a_flat = data_a[var].ravel()
        b_flat = data_b[var].ravel()
        if a_flat.size != b_flat.size:
            continue
        plt.figure(figsize=(6, 6))
        plt.scatter(a_flat, b_flat, alpha=0.5, s=10)
        lims = [min(a_flat.min(), b_flat.min()), max(a_flat.max(), b_flat.max())]
        plt.plot(lims, lims, "k--", label="1:1")
        plt.xlabel(f"{var} ({name_a})")
        plt.ylabel(f"{var} ({name_b})")
        plt.title(f"{var}: restart A vs B (all columns × layers)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"amazon_site_scatter_{var}_restart_compare.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Wrote {out_path}")


def write_restart_profiles_csv(
    output_dir: str,
    name_a: str,
    name_b: str,
    profiles_a: Dict[str, np.ndarray],
    profiles_b: Dict[str, np.ndarray],
    lon: float,
    lat: float,
) -> None:
    """Write CSV of layer-wise values for both restarts (col 0 profile)."""
    rows: List[Dict[str, object]] = []
    for var in P_VARIABLES:
        if var not in profiles_a or var not in profiles_b:
            continue
        a_vals = profiles_a[var]
        b_vals = profiles_b[var]
        for layer_idx in range(min(len(a_vals), len(b_vals))):
            rows.append({
                "variable": var,
                "layer_index": layer_idx + 1,
                "lon": lon,
                "lat": lat,
                name_a: float(a_vals[layer_idx]),
                name_b: float(b_vals[layer_idx]),
                "abs_diff": float(abs(a_vals[layer_idx] - b_vals[layer_idx])),
            })
    if rows:
        df = pd.DataFrame(rows)
        out_csv = os.path.join(output_dir, "amazon_site_5p_restart_compare_profiles.csv")
        df.to_csv(out_csv, index=False)
        print(f"  Wrote {out_csv}")


def run_restart_compare(
    restart_a: str,
    restart_b: str,
    name_a: str,
    name_b: str,
    output_dir: str,
    lon: float = DEFAULT_AMAZON_LON,
    lat: float = DEFAULT_AMAZON_LAT,
) -> None:
    """Load two restart NetCDFs, plot 5P profiles and scatter, write CSV."""
    if not os.path.isfile(restart_a):
        raise SystemExit(f"Restart A not found: {restart_a}")
    if not os.path.isfile(restart_b):
        raise SystemExit(f"Restart B not found: {restart_b}")
    print(f"Loading restart A: {restart_a}")
    data_a = _load_5p_from_restart(restart_a)
    print(f"Loading restart B: {restart_b}")
    data_b = _load_5p_from_restart(restart_b)
    if not data_a or not data_b:
        raise SystemExit("No 5P variables found in one or both restarts.")
    profiles_a = _profile_from_2d(data_a)
    profiles_b = _profile_from_2d(data_b)
    os.makedirs(output_dir, exist_ok=True)
    print("Profile plots (col 0)...")
    make_restart_profile_plots(
        output_dir=output_dir,
        name_a=name_a,
        name_b=name_b,
        profiles_a=profiles_a,
        profiles_b=profiles_b,
        lon=lon,
        lat=lat,
    )
    print("Scatter plots (all columns × layers)...")
    make_restart_scatter_plots(output_dir=output_dir, name_a=name_a, name_b=name_b, data_a=data_a, data_b=data_b)
    write_restart_profiles_csv(
        output_dir=output_dir,
        name_a=name_a,
        name_b=name_b,
        profiles_a=profiles_a,
        profiles_b=profiles_b,
        lon=lon,
        lat=lat,
    )


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
    # Restart-vs-restart mode: compare 5P between two Amazon-site restart NetCDFs
    parser.add_argument(
        "--restart-a",
        default="",
        help="Path to first Amazon-site restart NetCDF (e.g. phase3 extract). If set with --restart-b, run restart comparison only.",
    )
    parser.add_argument(
        "--restart-b",
        default="",
        help="Path to second Amazon-site restart NetCDF (e.g. phase2 Amazon 5P bias-corrected).",
    )
    parser.add_argument(
        "--restart-name-a",
        default="phase3_amazon",
        help="Label for restart A in plots (used with --restart-a/--restart-b).",
    )
    parser.add_argument(
        "--restart-name-b",
        default="phase2_amazon_5P_bias_corrected",
        help="Label for restart B in plots.",
    )

    args = parser.parse_args()

    # Restart-vs-restart mode
    if args.restart_a and args.restart_b:
        output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(
            os.path.dirname(os.path.abspath(args.restart_a)), "analysis", "amazon_5p_restart_compare"
        )
        run_restart_compare(
            restart_a=os.path.abspath(args.restart_a),
            restart_b=os.path.abspath(args.restart_b),
            name_a=args.restart_name_a,
            name_b=args.restart_name_b,
            output_dir=output_dir,
            lon=args.lon,
            lat=args.lat,
        )
        return

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

