#!/usr/bin/env python3
"""
Generate validation scatter plots and highlight a site-specific sample
based on a given longitude/latitude.
"""

import os
import re
import json
from glob import glob
import argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _find_lat_lon_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Return (lon_col, lat_col) if found, else None."""
    lower_map = {c.lower(): c for c in df.columns}
    # Prefer full names
    if "longitude" in lower_map and "latitude" in lower_map:
        return lower_map["longitude"], lower_map["latitude"]
    # Common short names
    if "lon" in lower_map and "lat" in lower_map:
        return lower_map["lon"], lower_map["lat"]
    if "long" in lower_map and "lat" in lower_map:
        return lower_map["long"], lower_map["lat"]
    # Fallback: any column containing 'lon' and 'lat'
    lon_candidates = [c for c in df.columns if "lon" in c.lower()]
    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    if lon_candidates and lat_candidates:
        return lon_candidates[0], lat_candidates[0]
    return None


def _find_site_indices_in_df(df: pd.DataFrame, lon: float, lat: float, tol: float) -> List[int]:
    cols = _find_lat_lon_columns(df)
    if cols is None:
        return []
    lon_col, lat_col = cols
    lon_vals = pd.to_numeric(df[lon_col], errors="coerce")
    lat_vals = pd.to_numeric(df[lat_col], errors="coerce")
    mask = (np.abs(lon_vals - lon) <= tol) & (np.abs(lat_vals - lat) <= tol)
    return np.where(mask)[0].tolist()


def _nearest_coord_in_df(df: pd.DataFrame, lon: float, lat: float):
    """Return (nearest_lon, nearest_lat, distance) or None if no lon/lat columns."""
    cols = _find_lat_lon_columns(df)
    if not cols:
        return None
    lon_col, lat_col = cols
    lon_vals = pd.to_numeric(df[lon_col], errors="coerce").values
    lat_vals = pd.to_numeric(df[lat_col], errors="coerce").values
    valid = np.isfinite(lon_vals) & np.isfinite(lat_vals)
    if not np.any(valid):
        return None
    dist = np.sqrt((lon_vals - lon) ** 2 + (lat_vals - lat) ** 2)
    dist[~valid] = np.inf
    idx = np.argmin(dist)
    return float(lon_vals[idx]), float(lat_vals[idx]), float(dist[idx])


def _find_site_indices(results_dir: str, lon: float, lat: float, tol: float, coord_csv: Optional[str]) -> List[int]:
    candidates = []
    if coord_csv:
        candidates.append(coord_csv)
    else:
        candidates.append(os.path.join(results_dir, "cnp_predictions", "test_static_inverse.csv"))
        candidates.append(os.path.join(results_dir, "cnp_predictions", "ground_truth_scalar.csv"))
        candidates.append(os.path.join(results_dir, "cnp_predictions", "predictions_scalar.csv"))
        # Fallback to any 1D/2D file if needed
        pft_dir = os.path.join(results_dir, "cnp_predictions", "pft_1d_ground_truth")
        soil_dir = os.path.join(results_dir, "cnp_predictions", "soil_2d_ground_truth")
        if os.path.isdir(pft_dir):
            pft_files = glob(os.path.join(pft_dir, "ground_truth_Y_*.csv"))
            if pft_files:
                candidates.append(pft_files[0])
        if os.path.isdir(soil_dir):
            soil_files = glob(os.path.join(soil_dir, "ground_truth_Y_*.csv"))
            if soil_files:
                candidates.append(soil_files[0])

    for path in candidates:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                indices = _find_site_indices_in_df(df, lon, lat, tol)
                if indices:
                    print(f"Found {len(indices)} matching samples using: {path}")
                    return indices
            except Exception as e:
                print(f"Warning: failed to read {path}: {e}")

    # No match: try to report nearest point to help the user
    for path in candidates:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                near = _nearest_coord_in_df(df, lon, lat)
                if near is not None:
                    print(f"No row with (lon, lat) within tolerance {tol}. Nearest point in data: lon={near[0]:.4f}, lat={near[1]:.4f} (distance ~{near[2]:.3f}°). Try --tolerance {max(0.5, min(5, near[2] + 0.1)):.1f} or use that coordinate.")
                    break
            except Exception:
                pass
    return []


def _extract_site_points(gt_col: np.ndarray, pred_col: np.ndarray, site_indices: List[int]) -> Optional[np.ndarray]:
    if not site_indices:
        return None
    valid_indices = [i for i in site_indices if 0 <= i < len(gt_col)]
    if not valid_indices:
        return None
    site_gt = gt_col[valid_indices]
    site_pred = pred_col[valid_indices]
    valid_mask = ~(np.isnan(site_gt) | np.isnan(site_pred))
    if not np.any(valid_mask):
        return None
    return np.column_stack([site_gt[valid_mask], site_pred[valid_mask]])


def plot_gt_vs_pred(gt, pred, title, save_path, site_points=None, site_label=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(gt, pred, alpha=0.5, s=20, color="#1f77b4")
    plt.plot([gt.min(), gt.max()], [gt.min(), gt.max()], "r--")

    if site_points is not None and len(site_points) > 0:
        plt.scatter(
            site_points[:, 0],
            site_points[:, 1],
            marker="*",
            s=140,
            c="red",
            edgecolors="black",
            linewidths=0.6,
            label=site_label or "site",
            zorder=5,
        )
        plt.legend()

    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def analyze_pair(gt_path, pred_path, label, out_dir, stats_data, site_indices, site_label, per_column=False, plot_scatter=True, selection=None):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    if per_column:
        for col in gt.columns:
            col_norm = col[2:] if isinstance(col, str) and col.startswith("Y_") else col
            if selection is not None and str(col_norm) in ("Latitude", "Longitude"):
                continue
            if selection is not None and col_norm not in selection:
                continue
            if col in pred.columns:
                print(f"Analyzing variable: {col}")
                gt_col = gt[col].values.flatten()
                pred_col = pred[col].values.flatten()

                gt_stats = {"min": np.nanmin(gt_col), "max": np.nanmax(gt_col), "sum": np.nansum(gt_col)}
                pred_stats = {"min": np.nanmin(pred_col), "max": np.nanmax(pred_col), "sum": np.nansum(pred_col)}

                rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
                mae = mean_absolute_error(gt_col, pred_col)
                r2 = r2_score(gt_col, pred_col)
                print(f"  {label} - {col}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

                if plot_scatter:
                    site_points = _extract_site_points(gt_col, pred_col, site_indices)
                    plot_gt_vs_pred(
                        gt_col,
                        pred_col,
                        f"{label} {col} GT vs Pred",
                        os.path.join(out_dir, f"{label}_{col}_gt_vs_pred.png"),
                        site_points=site_points,
                        site_label=site_label,
                    )

                stats_data.append(
                    {
                        "type": label,
                        "variable": col,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "gt_min": gt_stats["min"],
                        "gt_max": gt_stats["max"],
                        "gt_sum": gt_stats["sum"],
                        "pred_min": pred_stats["min"],
                        "pred_max": pred_stats["max"],
                        "pred_sum": pred_stats["sum"],
                    }
                )
            else:
                print(f"Column {col} missing in predictions for {label}")
        return None

    gt_flat = gt.values.flatten()
    pred_flat = pred.values.flatten()
    gt_stats = {"min": np.nanmin(gt_flat), "max": np.nanmax(gt_flat), "sum": np.nansum(gt_flat)}
    pred_stats = {"min": np.nanmin(pred_flat), "max": np.nanmax(pred_flat), "sum": np.nansum(pred_flat)}
    rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))
    mae = mean_absolute_error(gt_flat, pred_flat)
    r2 = r2_score(gt_flat, pred_flat)
    print(f"{label} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    if plot_scatter:
        site_points = _extract_site_points(gt_flat, pred_flat, site_indices)
        plot_gt_vs_pred(
            gt_flat,
            pred_flat,
            f"{label} GT vs Pred",
            os.path.join(out_dir, f"{label}_gt_vs_pred.png"),
            site_points=site_points,
            site_label=site_label,
        )

    stats_data.append(
        {
            "type": label,
            "variable": "all",
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "gt_min": gt_stats["min"],
            "gt_max": gt_stats["max"],
            "gt_sum": gt_stats["sum"],
            "pred_min": pred_stats["min"],
            "pred_max": pred_stats["max"],
            "pred_sum": pred_stats["sum"],
        }
    )
    return {"rmse": rmse, "mae": mae, "r2": r2}


def analyze_1d_new_structure(results_dir, label, out_dir, stats_data, site_indices, site_label, plot_scatter=True, selection=None):
    gt_dir = os.path.join(results_dir, "cnp_predictions", "pft_1d_ground_truth")
    pred_dir = os.path.join(results_dir, "cnp_predictions", "pft_1d_predictions")
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Missing 1D directories: {gt_dir} or {pred_dir}")
        return
    gt_files = glob(os.path.join(gt_dir, "ground_truth_Y_*.csv"))
    if not gt_files:
        print(f"No ground truth files found in {gt_dir}")
        return
    print(f"Found {len(gt_files)} 1D variables to analyze")

    for gt_file in gt_files:
        var_name = os.path.basename(gt_file).replace("ground_truth_Y_", "").replace(".csv", "")
        pred_file = os.path.join(pred_dir, f"predictions_Y_{var_name}.csv")
        if not os.path.exists(pred_file):
            print(f"Missing prediction file for {var_name}: {pred_file}")
            continue
        if selection is not None and var_name not in selection:
            continue

        print(f"Analyzing variable: {var_name}")
        gt_data = pd.read_csv(gt_file)
        pred_data = pd.read_csv(pred_file)
        for col in ["long", "lat", "Long", "Lat", "Longitude", "Latitude"]:
            if col in gt_data.columns:
                gt_data = gt_data.drop(columns=[col])
            if col in pred_data.columns:
                pred_data = pred_data.drop(columns=[col])
        if gt_data.shape != pred_data.shape:
            print(f"Shape mismatch for {var_name}: GT {gt_data.shape} vs Pred {pred_data.shape}")
            continue

        num_pfts = gt_data.shape[1]
        print(f"  {var_name}: {num_pfts} PFT columns")

        for col_name in gt_data.columns:
            if selection is not None:
                sel = selection.get(var_name, None)
                if sel is not None and sel["pfts"]:
                    pft_match = re.search(r"pft(\d+)$", col_name)
                    if pft_match:
                        try:
                            pft_num = int(pft_match.group(1))
                            if pft_num not in sel["pfts"]:
                                continue
                        except Exception:
                            pass

            gt_col = gt_data[col_name].values
            pred_col = pred_data[col_name].values
            if np.all(np.isnan(gt_col)) or np.all(np.isnan(pred_col)):
                continue
            valid_mask = ~(np.isnan(gt_col) | np.isnan(pred_col))
            if np.sum(valid_mask) < 3:
                continue
            gt_valid = gt_col[valid_mask]
            pred_valid = pred_col[valid_mask]

            gt_stats = {"min": np.nanmin(gt_valid), "max": np.nanmax(gt_valid), "sum": np.nansum(gt_valid)}
            pred_stats = {"min": np.nanmin(pred_valid), "max": np.nanmax(pred_valid), "sum": np.nansum(pred_valid)}
            rmse = np.sqrt(mean_squared_error(gt_valid, pred_valid))
            mae = mean_absolute_error(gt_valid, pred_valid)
            r2 = r2_score(gt_valid, pred_valid)

            print(f"    {col_name}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

            if plot_scatter:
                site_points = _extract_site_points(gt_col, pred_col, site_indices)
                plot_gt_vs_pred(
                    gt_valid,
                    pred_valid,
                    f"{label} {col_name} GT vs Pred",
                    os.path.join(out_dir, f"{label}_{col_name}_gt_vs_pred.png"),
                    site_points=site_points,
                    site_label=site_label,
                )

            stats_data.append(
                {
                    "type": "1D",
                    "variable": var_name,
                    "pft": col_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "gt_min": gt_stats["min"],
                    "gt_max": gt_stats["max"],
                    "gt_sum": gt_stats["sum"],
                    "pred_min": pred_stats["min"],
                    "pred_max": pred_stats["max"],
                    "pred_sum": pred_stats["sum"],
                }
            )


def analyze_1d(gt_path, pred_path, label, out_dir, results_dir, stats_data, site_indices, site_label, plot_scatter=True, selection=None):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    for col in ["long", "lat", "Long", "Lat", "Longitude", "Latitude"]:
        if col in gt.columns:
            gt = gt.drop(columns=[col])
        if col in pred.columns:
            pred = pred.drop(columns=[col])
    config_path = os.path.join(results_dir, "cnp_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        variable_names = config.get("data_info", {}).get("variables_1d_pft", None)
        if variable_names is None:
            print("[ERROR] Could not find variables_1d_pft in cnp_config.json!")
            return
    else:
        print(f"[ERROR] cnp_config.json not found in {results_dir}!")
        return
    num_vars = len(variable_names)
    num_pfts = 16
    num_samples = gt.shape[0]
    expected_cols = num_vars * num_pfts
    print(f"[DEBUG] 1D: num_samples={num_samples}, num_vars={num_vars}, num_pfts={num_pfts}, expected_cols={expected_cols}, actual_cols={gt.shape[1]}")
    if gt.shape[1] != expected_cols:
        print("[ERROR] Unexpected number of columns in 1D data!")
        print("Column names:", list(gt.columns))
        return
    gt_reshaped = gt.values.reshape(num_samples, num_vars, num_pfts)
    pred_reshaped = pred.values.reshape(num_samples, num_vars, num_pfts)
    for i, var in enumerate(variable_names):
        if selection is not None and var not in selection:
            continue
        print(f"Analyzing variable: {var}")
        for j in range(num_pfts):
            gt_col = gt_reshaped[:, i, j]
            pred_col = pred_reshaped[:, i, j]
            col_name = gt.columns[i * num_pfts + j] if (i * num_pfts + j) < len(gt.columns) else f"{var}_pft{j+1}"
            gt_stats = {"min": np.nanmin(gt_col), "max": np.nanmax(gt_col), "sum": np.nansum(gt_col)}
            pred_stats = {"min": np.nanmin(pred_col), "max": np.nanmax(pred_col), "sum": np.nansum(pred_col)}
            rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
            mae = mean_absolute_error(gt_col, pred_col)
            r2 = r2_score(gt_col, pred_col)
            print(f"{label} - {col_name}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

            if plot_scatter:
                site_points = _extract_site_points(gt_col, pred_col, site_indices)
                plot_gt_vs_pred(
                    gt_col,
                    pred_col,
                    f"{label} {col_name} GT vs Pred",
                    os.path.join(out_dir, f"{label}_{col_name}_gt_vs_pred.png"),
                    site_points=site_points,
                    site_label=site_label,
                )

            stats_data.append(
                {
                    "type": "1D",
                    "variable": var,
                    "pft": col_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "gt_min": gt_stats["min"],
                    "gt_max": gt_stats["max"],
                    "gt_sum": gt_stats["sum"],
                    "pred_min": pred_stats["min"],
                    "pred_max": pred_stats["max"],
                    "pred_sum": pred_stats["sum"],
                }
            )


def analyze_2d_new_structure(results_dir, label, out_dir, stats_data, site_indices, site_label, plot_scatter=True, selection=None):
    gt_dir = os.path.join(results_dir, "cnp_predictions", "soil_2d_ground_truth")
    pred_dir = os.path.join(results_dir, "cnp_predictions", "soil_2d_predictions")
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Missing 2D directories: {gt_dir} or {pred_dir}")
        return
    gt_files = glob(os.path.join(gt_dir, "ground_truth_Y_*.csv"))
    if not gt_files:
        print(f"No 2D ground truth files found in {gt_dir}")
        return
    print(f"Found {len(gt_files)} 2D variables to analyze")

    for gt_file in gt_files:
        var_name = os.path.basename(gt_file).replace("ground_truth_Y_", "").replace(".csv", "")
        pred_file = os.path.join(pred_dir, f"predictions_Y_{var_name}.csv")
        if not os.path.exists(pred_file):
            print(f"Missing prediction file for {var_name}: {pred_file}")
            continue
        if selection is not None and var_name not in selection:
            continue

        print(f"Analyzing 2D variable: {var_name}")
        gt_data = pd.read_csv(gt_file)
        pred_data = pd.read_csv(pred_file)
        for col in ["long", "lat", "Long", "Lat", "Longitude", "Latitude"]:
            if col in gt_data.columns:
                gt_data = gt_data.drop(columns=[col])
            if col in pred_data.columns:
                pred_data = pred_data.drop(columns=[col])
        if gt_data.shape != pred_data.shape:
            print(f"Shape mismatch for {var_name}: GT {gt_data.shape} vs Pred {pred_data.shape}")
            continue

        total_columns = gt_data.shape[1]
        expected_columns = 1 * 10
        if total_columns != expected_columns:
            print(f"  Warning: Expected {expected_columns} columns for 2D data, but found {total_columns}")
            if total_columns % 10 != 0:
                print(f"  Error: Number of columns ({total_columns}) is not divisible by 10")
                continue
            num_columns = total_columns // 10
            print(f"  Assuming {num_columns} columns with 10 layers each")
        else:
            num_columns = 1
            print(f"  {var_name}: {num_columns} columns, each with 10 layers ({total_columns} total columns)")

        first_column_idx = 0
        layers_to_analyze = 10

        for layer_idx in range(layers_to_analyze):
            if selection is not None:
                sel = selection.get(var_name, None)
                if sel is not None and sel["layers"] and (layer_idx + 1) not in sel["layers"]:
                    continue
            col_idx = layer_idx if num_columns == 1 else (first_column_idx * 10 + layer_idx)
            if col_idx >= total_columns:
                print(f"    Warning: Column index {col_idx} out of range for {total_columns} columns")
                continue

            gt_col = gt_data.iloc[:, col_idx].values
            pred_col = pred_data.iloc[:, col_idx].values
            if np.all(np.isnan(gt_col)) or np.all(np.isnan(pred_col)):
                continue
            valid_mask = ~(np.isnan(gt_col) | np.isnan(pred_col))
            if np.sum(valid_mask) < 3:
                continue

            gt_valid = gt_col[valid_mask]
            pred_valid = pred_col[valid_mask]
            gt_stats = {"min": np.nanmin(gt_valid), "max": np.nanmax(gt_valid), "sum": np.nansum(gt_valid)}
            pred_stats = {"min": np.nanmin(pred_valid), "max": np.nanmax(pred_valid), "sum": np.nansum(pred_valid)}
            rmse = np.sqrt(mean_squared_error(gt_valid, pred_valid))
            mae = mean_absolute_error(gt_valid, pred_valid)
            r2 = r2_score(gt_valid, pred_valid)

            print(f"    Layer {layer_idx+1}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

            if plot_scatter:
                site_points = _extract_site_points(gt_col, pred_col, site_indices)
                plot_gt_vs_pred(
                    gt_valid,
                    pred_valid,
                    f"{label} {var_name} Layer{layer_idx+1} GT vs Pred",
                    os.path.join(out_dir, f"{label}_{var_name}_Layer{layer_idx+1}_gt_vs_pred.png"),
                    site_points=site_points,
                    site_label=site_label,
                )

            stats_data.append(
                {
                    "type": "2D",
                    "variable": var_name,
                    "layer": layer_idx + 1,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "gt_min": gt_stats["min"],
                    "gt_max": gt_stats["max"],
                    "gt_sum": gt_stats["sum"],
                    "pred_min": pred_stats["min"],
                    "pred_max": pred_stats["max"],
                    "pred_sum": pred_stats["sum"],
                }
            )

        try:
            gt_firstcol = gt_data.iloc[:, 0:10].values
            pred_firstcol = pred_data.iloc[:, 0:10].values
            valid_mask = ~(np.isnan(gt_firstcol) | np.isnan(pred_firstcol))
            if np.sum(valid_mask) >= 3 and plot_scatter:
                gt_flat = gt_firstcol.flatten()
                pred_flat = pred_firstcol.flatten()
                site_vals = []
                for idx in site_indices:
                    if 0 <= idx < gt_firstcol.shape[0]:
                        site_vals.append((gt_firstcol[idx, :], pred_firstcol[idx, :]))
                if site_vals:
                    site_gt = np.concatenate([v[0] for v in site_vals])
                    site_pred = np.concatenate([v[1] for v in site_vals])
                    site_points = _extract_site_points(site_gt, site_pred, list(range(len(site_gt))))
                else:
                    site_points = None

                plot_gt_vs_pred(
                    gt_flat[valid_mask.flatten()],
                    pred_flat[valid_mask.flatten()],
                    f"{label} {var_name} FirstCol(10 layers) GT vs Pred",
                    os.path.join(out_dir, f"{label}_{var_name}_FirstCol_AllLayers_gt_vs_pred.png"),
                    site_points=site_points,
                    site_label=site_label,
                )
        except Exception as e:
            print(f"  Skipped overall plot for {var_name}: {e}")


def analyze_2d(gt_path, pred_path, label, out_dir, results_dir, stats_data, site_indices, site_label, plot_scatter=True, selection=None):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    config_path = os.path.join(results_dir, "cnp_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        variable_names = config.get("data_info", {}).get("variables_2d_soil", None)
        if variable_names is None:
            print("[ERROR] Could not find variables_2d_soil in cnp_config.json!")
            return
    else:
        print(f"[ERROR] cnp_config.json not found in {results_dir}!")
        return
    num_vars = len(variable_names)
    num_columns = 18
    num_layers_per_column = 10
    num_samples = gt.shape[0]
    expected_cols = num_vars * num_columns * num_layers_per_column
    print(f"[DEBUG] 2D: num_samples={num_samples}, num_vars={num_vars}, num_columns={num_columns}, layers_per_column={num_layers_per_column}, expected_cols={expected_cols}, actual_cols={gt.shape[1]}")
    if gt.shape[1] != expected_cols:
        print("[ERROR] Unexpected number of columns in 2D data!")
        print("Column names:", list(gt.columns))
        return
    gt_reshaped = gt.values.reshape(num_samples, num_vars, num_columns, num_layers_per_column)
    pred_reshaped = pred.values.reshape(num_samples, num_vars, num_columns, num_layers_per_column)
    for i, var in enumerate(variable_names):
        if selection is not None and var not in selection:
            continue
        print(f"Analyzing 2D variable: {var}")
        for j in range(10):
            if selection is not None:
                sel = selection.get(var, None)
                if sel is not None and sel["layers"] and (j + 1) not in sel["layers"]:
                    continue
            gt_col = gt_reshaped[:, i, 0, j]
            pred_col = pred_reshaped[:, i, 0, j]

            gt_stats = {"min": np.nanmin(gt_col), "max": np.nanmax(gt_col), "sum": np.nansum(gt_col)}
            pred_stats = {"min": np.nanmin(pred_col), "max": np.nanmax(pred_col), "sum": np.nansum(pred_col)}
            rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
            mae = mean_absolute_error(gt_col, pred_col)
            r2 = r2_score(gt_col, pred_col)
            print(f"{label} - {var} (Layer {j+1}): RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

            if plot_scatter:
                site_points = _extract_site_points(gt_col, pred_col, site_indices)
                plot_gt_vs_pred(
                    gt_col,
                    pred_col,
                    f"{label} {var} Layer{j+1} GT vs Pred",
                    os.path.join(out_dir, f"{label}_{var}_Layer{j+1}_gt_vs_pred.png"),
                    site_points=site_points,
                    site_label=site_label,
                )

            stats_data.append(
                {
                    "type": "2D",
                    "variable": var,
                    "layer": j + 1,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "gt_min": gt_stats["min"],
                    "gt_max": gt_stats["max"],
                    "gt_sum": gt_stats["sum"],
                    "pred_min": pred_stats["min"],
                    "pred_max": pred_stats["max"],
                    "pred_sum": pred_stats["sum"],
                }
            )


def plot_train_val_accuracy(loss_csv, out_dir):
    df = pd.read_csv(loss_csv)
    plt.figure()
    if "Train Loss" in df.columns and "Validation Loss" in df.columns:
        plt.plot(df["Train Loss"], label="Train Loss")
        plt.plot(df["Validation Loss"], label="Validation Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Train/Validation Loss")
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "train_val_loss.png"))
        plt.close()
    else:
        print("train_loss or val_loss columns not found in loss CSV.")


def _parse_top_bad_report(report_path):
    selection = {}
    if not os.path.exists(report_path):
        print(f"Top-bad report not found: {report_path}")
        return selection
    in_section = False
    try:
        with open(report_path, "r") as f:
            for line in f:
                stripped = line.strip("\n")
                header = stripped.strip()
                if header.startswith("Top variables by bad-count"):
                    in_section = True
                    continue
                if in_section and header.startswith("## "):
                    break
                if in_section and stripped.startswith("  "):
                    m = re.match(r"\s+([A-Za-z0-9_]+):\s*([0-9]+)(;.*)?$", stripped)
                    if not m:
                        continue
                    var = m.group(1)
                    details = m.group(3) or ""
                    pfts = set()
                    layers = set()
                    if "pfts:" in details:
                        m_p = re.search(r"pfts:\s*([0-9,\s]+)", details)
                        if m_p:
                            nums = [n.strip() for n in m_p.group(1).split(",") if n.strip()]
                            for n in nums:
                                try:
                                    pfts.add(int(n))
                                except Exception:
                                    pass
                    if "layers:" in details:
                        m_l = re.search(r"layers:\s*([0-9,\s]+)", details)
                        if m_l:
                            nums = [n.strip() for n in m_l.group(1).split(",") if n.strip()]
                            for n in nums:
                                try:
                                    layers.add(int(n))
                                except Exception:
                                    pass
                    selection[var] = {"pfts": pfts, "layers": layers}
    except Exception as e:
        print(f"Failed to parse top-bad report {report_path}: {e}")
        return {}
    return selection


def _parse_worst_vars_report(report_path):
    selection = {}
    if not os.path.exists(report_path):
        print(f"Worst-variables report not found: {report_path}")
        return selection
    in_section = False
    try:
        with open(report_path, "r") as f:
            for line in f:
                stripped = line.strip("\n")
                header = stripped.strip()
                if header.startswith("## Variables with Worst Predictions"):
                    in_section = True
                    continue
                if in_section and header.startswith("## "):
                    break
                if in_section and stripped and not stripped.startswith("#"):
                    m = re.match(r"\s*([A-Za-z0-9_]+):\s*", stripped)
                    if not m:
                        continue
                    var = m.group(1)
                    selection[var] = {"pfts": set(), "layers": set()}
    except Exception as e:
        print(f"Failed to parse worst variables section {report_path}: {e}")
        return {}
    return selection


def main_with_site(results_dir, plot_scatter, plot_loss, lon, lat, tol, coord_csv, top_bad_only=False, top_bad_report=None, worst_only=False):
    plots_dir = os.path.join(results_dir, "plots_site")
    os.makedirs(plots_dir, exist_ok=True)
    stats_path = os.path.join(results_dir, "validation_stats.csv")
    stats_data = []

    site_indices = _find_site_indices(results_dir, lon, lat, tol, coord_csv)
    if not site_indices:
        raise ValueError(
            "No matching samples found for the given lon/lat. "
            "The default --tolerance is 0.01 (degrees). Grid data is often coarser (e.g. 1–2°); "
            "try e.g. --tolerance 2.0 or check the message above for the nearest point."
        )
    site_label = f"Site ({lon}, {lat})"

    selection = None
    if top_bad_only or worst_only:
        report_path = top_bad_report or os.path.join(results_dir, "analysis", "quality_summary_report.txt")
        if worst_only:
            selection = _parse_worst_vars_report(report_path)
            if selection:
                print(f"Plotting restricted to worst variables from: {report_path}")
        if (not selection) and top_bad_only:
            selection = _parse_top_bad_report(report_path)
            if selection:
                print(f"Plotting restricted to top-bad variables from: {report_path}")
        if not selection:
            print("No selections parsed from report; proceeding without restriction.")
            selection = None

    pft_gt_dir = os.path.join(results_dir, "cnp_predictions", "pft_1d_ground_truth")
    pft_pred_dir = os.path.join(results_dir, "cnp_predictions", "pft_1d_predictions")
    if os.path.exists(pft_gt_dir) and os.path.exists(pft_pred_dir):
        print("Using new 1D directory structure")
        analyze_1d_new_structure(results_dir, "1D", plots_dir, stats_data, site_indices, site_label, plot_scatter, selection)
    else:
        print("Using legacy 1D single-file format")
        gt_path = os.path.join(results_dir, "cnp_predictions", "ground_truth_1d.csv")
        pred_path = os.path.join(results_dir, "cnp_predictions", "predictions_1d.csv")
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            analyze_1d(gt_path, pred_path, "1D", plots_dir, results_dir, stats_data, site_indices, site_label, plot_scatter, selection)
        else:
            print("No 1D data found in either format")

    scalar_gt = os.path.join(results_dir, "cnp_predictions", "ground_truth_scalar.csv")
    scalar_pred = os.path.join(results_dir, "cnp_predictions", "predictions_scalar.csv")
    if os.path.exists(scalar_gt) and os.path.exists(scalar_pred):
        print("Analyzing scalar data...")
        analyze_pair(scalar_gt, scalar_pred, "Scalar", plots_dir, stats_data, site_indices, site_label, per_column=True, plot_scatter=plot_scatter, selection=selection)
    else:
        print("Scalar data files not found")

    soil_gt_dir = os.path.join(results_dir, "cnp_predictions", "soil_2d_ground_truth")
    soil_pred_dir = os.path.join(results_dir, "cnp_predictions", "soil_2d_predictions")
    if os.path.exists(soil_gt_dir) and os.path.exists(soil_pred_dir):
        print("Using new 2D directory structure")
        analyze_2d_new_structure(results_dir, "2D", plots_dir, stats_data, site_indices, site_label, plot_scatter, selection)
    else:
        print("Using legacy 2D single-file format")
        soil_gt = os.path.join(results_dir, "cnp_predictions", "ground_truth_2d.csv")
        soil_pred = os.path.join(results_dir, "cnp_predictions", "predictions_2d.csv")
        if os.path.exists(soil_gt) and os.path.exists(soil_pred):
            analyze_2d(soil_gt, soil_pred, "2D", plots_dir, results_dir, stats_data, site_indices, site_label, plot_scatter, selection)
        else:
            print("No 2D data found in either format")

    if plot_loss:
        loss_csv = os.path.join(results_dir, "cnp_training_losses.csv")
        if os.path.exists(loss_csv):
            plot_train_val_accuracy(loss_csv, plots_dir)
        else:
            print("Loss CSV not found for train/val accuracy plot.")

    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved validation stats to: {stats_path}")

    test_metrics_path = os.path.join(results_dir, "cnp_predictions", "test_metrics.csv")
    if os.path.exists(test_metrics_path):
        print("\nTest Metrics:")
        print(pd.read_csv(test_metrics_path))
    else:
        print("test_metrics.csv not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess CNP model results with a site-specific highlight")
    parser.add_argument("results_dir", nargs="?", default=".", help="Results directory (run_xxxxx). Default is current directory.")
    parser.add_argument("--lon", type=float, required=True, help="Site longitude")
    parser.add_argument("--lat", type=float, required=True, help="Site latitude")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Coordinate matching tolerance (default: 0.01)")
    parser.add_argument("--coord-csv", type=str, default=None, help="CSV file containing Longitude/Latitude for matching indices")

    parser.add_argument("--no-scatter", action="store_false", dest="plot_scatter", help="Do not generate scatter plots")
    parser.add_argument("--no-plot-loss", action="store_false", dest="plot_loss", help="Do not plot train/val loss curve")
    parser.add_argument("--stats-only", action="store_true", help="Only compute and save statistics CSV; do not generate any plots")
    parser.add_argument("--top-bad-only", action="store_true", help="Plot only variables listed in the quality summary top-bad section")
    parser.add_argument("--worst-only", action="store_true", help='Plot only variables listed under "Variables with Worst Predictions"')
    parser.add_argument("--top-bad-report", type=str, default=None, help="Path to quality_summary_report.txt")

    parser.set_defaults(plot_scatter=True, plot_loss=True)
    args = parser.parse_args()

    if getattr(args, "stats_only", False):
        args.plot_scatter = False
        args.plot_loss = False

    if len(os.sys.argv) < 2:
        print("Using current directory as results directory")

    main_with_site(
        args.results_dir,
        args.plot_scatter,
        args.plot_loss,
        args.lon,
        args.lat,
        args.tolerance,
        args.coord_csv,
        args.top_bad_only,
        args.top_bad_report,
        worst_only=getattr(args, "worst_only", False),
    )
