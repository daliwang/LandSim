#!/usr/bin/env python3
"""
Compare the 5 P variables (labilep_vr, occlp_vr, solutionp_vr, secondp_vr, primp_vr) at the Amazon site
between ground truth and multiple runs. Runs can be:
- Restart NetCDF: path to an Amazon-site restart .nc
- Inference dir: path to cnp_inference_entire_dataset (contains cnp_predictions/ with soil_2d CSVs).
Uses first 10 soil layers, column 0. Outputs value tables (layer x source) and comparison plots.
"""
import argparse
import numpy as np
import sys
from pathlib import Path

# 5 P variables to compare (first 10 layers, first column)
FOUR_P_VARS = ["labilep_vr", "occlp_vr", "solutionp_vr", "secondp_vr", "primp_vr"]
SOIL_N_LAYERS = 10
SOIL_COLUMN_INDEX = 0
AMAZON_LON, AMAZON_LAT = 303.75, -17.434553


def extract_soil_10(ds, var_name):
    """Extract first column, first 10 layers for a 2D soil variable from xarray Dataset."""
    if var_name not in ds.variables:
        return None
    var = ds[var_name]
    if "column" not in var.dims or "levgrnd" not in var.dims:
        return None
    slab = var.isel(column=SOIL_COLUMN_INDEX, levgrnd=slice(0, SOIL_N_LAYERS))
    return np.asarray(slab.values, dtype=float).ravel()


def _layer_columns(df):
    """Return ordered layer column names (col1_layer1..col1_layer10)."""
    import re
    layer_cols = [c for c in df.columns if "layer" in c.lower()]
    def layer_num(c):
        m = re.search(r"layer(\d+)", c, re.I)
        return int(m.group(1)) if m else 0
    layer_cols = sorted(layer_cols, key=layer_num)[:SOIL_N_LAYERS]
    return layer_cols


def _find_amazon_row_index(df, tol_deg=1.5):
    """Return row index of the gridcell closest to (AMAZON_LON, AMAZON_LAT)."""
    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        return None
    lon = np.asarray(df["Longitude"], dtype=float)
    lat = np.asarray(df["Latitude"], dtype=float)
    dist = (lon - AMAZON_LON) ** 2 + (lat - AMAZON_LAT) ** 2
    idx = np.nanargmin(dist)
    if np.sqrt(dist[idx]) > tol_deg:
        return None
    return int(idx)


def extract_soil_10_from_inference(inference_dir: Path, var_name: str, use_predictions: bool = True):
    """
    Read 4 P variable from inference CSV at Amazon site.
    inference_dir: path to cnp_inference_entire_dataset (or cnp_predictions).
    use_predictions: True -> predictions_Y_*.csv, False -> ground_truth_Y_*.csv.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    base = inference_dir / "cnp_predictions" if (inference_dir / "cnp_predictions").exists() else inference_dir
    sub = "soil_2d_predictions" if use_predictions else "soil_2d_ground_truth"
    prefix = "predictions_Y_" if use_predictions else "ground_truth_Y_"
    path = base / sub / (prefix + var_name + ".csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row_idx = _find_amazon_row_index(df)
    if row_idx is None:
        return None
    lcols = _layer_columns(df)
    if len(lcols) < SOIL_N_LAYERS:
        return None
    return df.iloc[row_idx][lcols].astype(float).values[:SOIL_N_LAYERS]


def _load_run_data(run_name, path, xr, run_data, log):
    """Load 4 P values for one run from restart NetCDF or inference directory."""
    path = Path(path)
    run_data[run_name] = {}
    if path.suffix.lower() == ".nc" and path.is_file():
        try:
            ds = xr.open_dataset(path)
        except Exception as e:
            log("Error opening {}: {}".format(path, e))
            return
        try:
            for v in FOUR_P_VARS:
                pred = extract_soil_10(ds, v)
                run_data[run_name][v] = pred if pred is not None and len(pred) == SOIL_N_LAYERS else None
        finally:
            ds.close()
    elif path.is_dir():
        for v in FOUR_P_VARS:
            arr = extract_soil_10_from_inference(path, v, use_predictions=True)
            run_data[run_name][v] = arr if arr is not None and len(arr) == SOIL_N_LAYERS else None
    else:
        log("Warning: {} is not a .nc file or directory, skipping run '{}'".format(path, run_name))


def main():
    parser = argparse.ArgumentParser(
        description="Compare 5 P variables (labilep_vr, occlp_vr, solutionp_vr, secondp_vr, primp_vr) at Amazon vs ground truth across runs."
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Ground truth Amazon-site restart NetCDF",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help="Run name and path: restart .nc or cnp_inference_entire_dataset dir",
    )
    parser.add_argument("--output", default=None, help="Write report to this file (default: stdout)")
    parser.add_argument("--plot-dir", default=None, help="Save one plot per variable (GT vs runs) here")
    args = parser.parse_args()

    # Parse NAME:PATH
    run_list = []
    for s in args.runs:
        if ":" not in s:
            print("Each run must be NAME:PATH (e.g. run1:path/to/file.nc)", file=sys.stderr)
            sys.exit(1)
        name, path = s.split(":", 1)
        path = Path(path)
        if not path.exists():
            print("Warning: {} does not exist, skipping run '{}'".format(path, name), file=sys.stderr)
            continue
        run_list.append((name.strip(), path))

    if not run_list:
        print("No valid run paths.", file=sys.stderr)
        sys.exit(1)

    try:
        import xarray as xr
    except ImportError:
        print("Need xarray: pip install xarray", file=sys.stderr)
        sys.exit(1)

    out = open(args.output, "w") if args.output else sys.stdout

    def log(s=""):
        print(s, file=out)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print("Ground truth file not found: {}".format(gt_path), file=sys.stderr)
        sys.exit(1)

        log("5 P variables comparison at Amazon site (first 10 layers, column 0)")
    log("  Ground truth: {}".format(gt_path))
    log("  Variables: {}".format(", ".join(FOUR_P_VARS)))
    log("")

    ds_gt = xr.open_dataset(gt_path)
    try:
        gt_vals = {}
        for v in FOUR_P_VARS:
            gt_vals[v] = extract_soil_10(ds_gt, v)
            if gt_vals[v] is None:
                log("Warning: {} missing or not 2D in ground truth".format(v))
        if not any(gt_vals[v] is not None for v in FOUR_P_VARS):
            log("No P variables found in ground truth.")
            return

        run_data = {}
        for run_name, path in run_list:
            _load_run_data(run_name, path, xr, run_data, log)

        # Value tables: for each variable, table Layer | Ground_truth | run1 | run2 | ...
        col_w = 14
        for v in FOUR_P_VARS:
            if gt_vals[v] is None:
                continue
            log("# {}".format(v))
            headers = ["Layer", "Ground_truth"] + [r[0] for r in run_list]
            log("  " + "  ".join(h[:col_w].ljust(col_w) for h in headers))
            log("  " + "-" * (len(headers) * (col_w + 2)))
            for layer in range(SOIL_N_LAYERS):
                row = [str(layer), "{:.6g}".format(gt_vals[v][layer])]
                for run_name, _ in run_list:
                    arr = run_data.get(run_name, {}).get(v)
                    row.append("{:.6g}".format(arr[layer]) if arr is not None else "-")
                log("  " + "  ".join(s[:col_w].ljust(col_w) for s in row))
            log("")

        # Plots: one per variable, GT vs each run
        if args.plot_dir:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib required for --plot-dir; skipping plots.", file=sys.stderr)
            else:
                plot_dir = Path(args.plot_dir)
                plot_dir.mkdir(parents=True, exist_ok=True)
                x = np.arange(0, SOIL_N_LAYERS, dtype=float)
                for v in FOUR_P_VARS:
                    if gt_vals[v] is None:
                        continue
                    fig, ax = plt.subplots()
                    ax.plot(x, gt_vals[v], "k-o", label="Ground truth", markersize=5, linewidth=2)
                    for run_name, _ in run_list:
                        arr = run_data.get(run_name, {}).get(v)
                        if arr is not None:
                            ax.plot(x, arr, "-s", label=run_name, markersize=4)
                    ax.set_xlabel("Soil layer (0-based)")
                    ax.set_ylabel(v)
                    ax.set_title("Amazon site: {} (first 10 layers)".format(v))
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.savefig(plot_dir / "{}.png".format(v), dpi=120, bbox_inches="tight")
                    plt.close(fig)
                log("Plots saved to: {}".format(plot_dir))
    finally:
        ds_gt.close()
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
