#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from typing import Dict, List, Tuple

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import netCDF4 as nc

VARS_TO_CLEAN = {'H2OSOI_LIQ', 'H2OSOI_ICE'}
FILL_VALUE_THRESHOLD = 1e35

def build_restart_kdtree(ds_restart: nc.Dataset) -> Tuple[cKDTree, np.ndarray]:
    gridcell_lat = ds_restart.variables["grid1d_lat"][:]
    gridcell_lon = ds_restart.variables["grid1d_lon"][:]
    coords = np.vstack((gridcell_lat, gridcell_lon)).T
    tree = cKDTree(coords)
    return tree, coords

def build_column_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    cols1d_gridcell_index = ds_restart.variables["cols1d_gridcell_index"][:]
    unique_ids = np.unique(cols1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(cols1d_gridcell_index == grid_id)[0]
    return mapping

def ensure_vars_exist(ds: nc.Dataset, var_names: List[str]) -> List[str]:
    existing = []
    for name in var_names:
        if name in ds.variables:
            existing.append(name)
    return existing

def build_pft_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    pfts1d_gridcell_index = ds_restart.variables["pfts1d_gridcell_index"][:]
    unique_ids = np.unique(pfts1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(pfts1d_gridcell_index == grid_id)[0]
    return mapping

def extract_col1d_x(ds_restart: nc.Dataset, var_name: str, col_indices: np.ndarray) -> List[float]:
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices]
    return values.astype(float).tolist()

def extract_col1d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[float]:
    if col_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][col_indices]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()

def extract_col2d_x(ds_restart: nc.Dataset, var_name: str, col_indices: np.ndarray) -> List[List[float]]:
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices, :]
    values_np = np.asarray(values, dtype=float)
    if var_name in VARS_TO_CLEAN:
        values_np[values_np >= FILL_VALUE_THRESHOLD] = 0.0
    return values_np.tolist()

def extract_col2d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[List[float]]:
    if col_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][col_indices, :]
        values_np = np.asarray(values, dtype=float)
        if var_name in VARS_TO_CLEAN:
            values_np[values_np >= FILL_VALUE_THRESHOLD] = 0.0
        slices.append(values_np)
    stacked = np.stack(slices, axis=0)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()

def extract_pft1d_x(ds_restart: nc.Dataset, var_name: str, pft_indices: np.ndarray) -> List[float]:
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices]
    return np.asarray(values, dtype=float).tolist()

def extract_pft1d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[float]:
    if pft_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][pft_indices]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()

def extract_pft2d_x(ds_restart: nc.Dataset, var_name: str, pft_indices: np.ndarray) -> List[List[float]]:
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices, :]
    return np.asarray(values, dtype=float).tolist()

def extract_pft2d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[List[float]]:
    if pft_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][pft_indices, :]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()

def augment_dataframe_with_pools(
    df: pd.DataFrame,
    ds_restart: nc.Dataset,
    ds_r_list: List[nc.Dataset],
    restart_tree: cKDTree,
    restart_coords: np.ndarray,
    col_index_map: Dict[int, np.ndarray],
    pool_vars: List[str],
) -> pd.DataFrame:
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("DataFrame is missing Latitude/Longitude columns for mapping.")

    pool_vars_existing = ensure_vars_exist(ds_restart, pool_vars)
    if not pool_vars_existing:
        raise ValueError(f"None of the target variables found in restart file: {pool_vars}")

    pool_vars_final: List[str] = [
        v for v in pool_vars_existing if all(v in ds_r.variables for ds_r in ds_r_list)
    ]
    if not pool_vars_final:
        raise ValueError("Target variables do not exist in the set of Y files.")

    latitudes = df["Latitude"].to_numpy()
    longitudes = df["Longitude"].to_numpy()
    query_coords = np.vstack((latitudes, longitudes)).T
    _, nearest_restart_indices = restart_tree.query(query_coords, k=1)

    results_x: Dict[str, List[List[float]]] = {v: [] for v in pool_vars_final}
    results_y: Dict[str, List[List[float]]] = {f"Y_{v}": [] for v in pool_vars_final}

    for row_idx, restart_idx in enumerate(nearest_restart_indices):
        gridcell_id = int(restart_idx) + 1
        col_indices = col_index_map.get(gridcell_id, np.array([], dtype=int))
        for v in pool_vars_final:
            x_vals = extract_col1d_x(ds_restart, v, col_indices)
            y_vals = extract_col1d_y(ds_r_list, v, col_indices)
            results_x[v].append(x_vals)
            results_y[f"Y_{v}"].append(y_vals)

    for v in pool_vars_final:
        df[v] = results_x[v]
        df[f"Y_{v}"] = results_y[f"Y_{v}"]

    return df

def augment_dataframe_with_vars(
    df: pd.DataFrame,
    ds_restart: nc.Dataset,
    ds_special_p_restart: nc.Dataset,
    ds_r_list: List[nc.Dataset],
    restart_tree: cKDTree,
    restart_coords: np.ndarray,
    col_index_map: Dict[int, np.ndarray],
    pft_index_map: Dict[int, np.ndarray],
    vars_1d: List[str],
    vars_2d: List[str],
    special_p_vars: List[str],
) -> pd.DataFrame:
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("DataFrame is missing Latitude/Longitude columns for mapping.")

    vars_1d_existing = ensure_vars_exist(ds_restart, vars_1d)
    vars_2d_existing = ensure_vars_exist(ds_restart, vars_2d)
    
    final_1d: List[str] = [v for v in vars_1d_existing if all(v in ds_r.variables for ds_r in ds_r_list)]
    final_2d: List[str] = [v for v in vars_2d_existing if all(v in ds_r.variables for ds_r in ds_r_list)]

    if not final_1d and not final_2d:
        return df

    latitudes = df["Latitude"].to_numpy()
    longitudes = df["Longitude"].to_numpy()
    query_coords = np.vstack((latitudes, longitudes)).T
    _, nearest_restart_indices = restart_tree.query(query_coords, k=1)

    results_x_1d: Dict[str, List[List[float]]] = {v: [] for v in final_1d}
    results_y_1d: Dict[str, List[List[float]]] = {f"Y_{v}": [] for v in final_1d}
    results_x_2d: Dict[str, List[List[List[float]]]] = {v: [] for v in final_2d}
    results_y_2d: Dict[str, List[List[List[float]]]] = {f"Y_{v}": [] for v in final_2d}

    for row_idx, restart_idx in enumerate(nearest_restart_indices):
        gridcell_id = int(restart_idx) + 1
        col_indices = col_index_map.get(gridcell_id, np.array([], dtype=int))
        pft_indices = pft_index_map.get(gridcell_id, np.array([], dtype=int))

        for v in final_1d:
            var_obj = ds_restart.variables[v]
            dims = tuple(var_obj.dimensions)
            if "pft" in dims:
                x_vals = extract_pft1d_x(ds_restart, v, pft_indices)
                y_vals = extract_pft1d_y(ds_r_list, v, pft_indices)
            else:
                x_vals = extract_col1d_x(ds_restart, v, col_indices)
                y_vals = extract_col1d_y(ds_r_list, v, col_indices)
            results_x_1d[v].append(x_vals)
            results_y_1d[f"Y_{v}"].append(y_vals)

        for v in final_2d:
            ds_for_x = ds_special_p_restart if v in special_p_vars else ds_restart
            var_obj = ds_for_x.variables[v]
            dims = tuple(var_obj.dimensions)
            
            if "pft" in dims:
                x_vals_2d = extract_pft2d_x(ds_for_x, v, pft_indices)
                y_vals_2d = extract_pft2d_y(ds_r_list, v, pft_indices)
            else:
                x_vals_2d = extract_col2d_x(ds_for_x, v, col_indices)
                y_vals_2d = extract_col2d_y(ds_r_list, v, col_indices)
            
            results_x_2d[v].append(x_vals_2d)
            results_y_2d[f"Y_{v}"].append(y_vals_2d)

    for v in final_1d:
        df[v] = results_x_1d[v]
        df[f"Y_{v}"] = results_y_1d[f"Y_{v}"]
    for v in final_2d:
        df[v] = results_x_2d[v]
        df[f"Y_{v}"] = results_y_2d[f"Y_{v}"]

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Augment batched training data with variables from RESTART files."
    )
    parser.add_argument(
        "--input_glob",
        default=Config.INPUT_GLOB,
        help="Glob pattern for input pkl batch files."
    )
    parser.add_argument(
        "--output_dir",
        default=Config.OUTPUT_DIR,
        help="Output directory for augmented data."
    )
    args = parser.parse_args()

    file_path10 = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_adspinref.elm.r.0021-01-01-00000.nc"
    file_path17 = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc"
    file_path18 = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc"
    file_path19 = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc"
    file_path20 = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc"
    file_path21 = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc"

    restart_pft_vars = list(dict.fromkeys(Config.dataset_new_1D_PFT_VARIABLES))
    restart_col_1d_vars = list(dict.fromkeys(Config.dataset_new_RESTART_COL_1D_VARS))
    restart_col_2d_vars = list(dict.fromkeys(list(Config.dataset_new_Water_variables) + list(Config.dataset_new_2D_VARIABLES)))
    
    configured_vars_1d = list(dict.fromkeys(restart_pft_vars + restart_col_1d_vars))
    configured_vars_2d = list(dict.fromkeys(restart_col_2d_vars + Config.SPECIAL_P_VARS))

    input_files = sorted(glob.glob(args.input_glob))
    if not input_files:
        sys.exit(f"No input files found matching pattern: {args.input_glob}")

    os.makedirs(args.output_dir, exist_ok=True)

    ds_restart = nc.Dataset(file_path10)
    ds_special_p_restart = nc.Dataset(Config.SPECIAL_P_INPUT_NC) 
    ds_r_list = [nc.Dataset(fp) for fp in [file_path17, file_path18, file_path19, file_path20, file_path21]]

    try:
        restart_tree, restart_coords = build_restart_kdtree(ds_restart)
        col_index_map = build_column_index_map(ds_restart)
        pft_index_map = build_pft_index_map(ds_restart)
        
        for fp in input_files:
            df = pd.read_pickle(fp)

            force_replace_vars = set(Config.SPECIAL_P_VARS)
            missing_other_vars_1d = [
                v for v in configured_vars_1d 
                if v not in force_replace_vars and not (v in df.columns and f"Y_{v}" in df.columns)
            ]
            missing_other_vars_2d = [
                v for v in configured_vars_2d 
                if v not in force_replace_vars and not (v in df.columns and f"Y_{v}" in df.columns)
            ]
            vars_to_process_1d = missing_other_vars_1d
            vars_to_process_2d = list(set(missing_other_vars_2d).union(force_replace_vars))

            if not vars_to_process_1d and not vars_to_process_2d:
                continue

            df_aug = augment_dataframe_with_vars(
                df=df,
                ds_restart=ds_restart,
                ds_special_p_restart=ds_special_p_restart,
                ds_r_list=ds_r_list,
                restart_tree=restart_tree,
                restart_coords=restart_coords,
                col_index_map=col_index_map,
                pft_index_map=pft_index_map,
                vars_1d=vars_to_process_1d,
                vars_2d=vars_to_process_2d,
                special_p_vars=Config.SPECIAL_P_VARS
            )

            base_name = os.path.basename(fp)
            out_name = f"{Config.ENHANCED_PREFIX}{base_name}"
            out_path = os.path.join(args.output_dir, out_name)
            df_aug.to_pickle(out_path)
            
    finally:
        ds_restart.close()
        ds_special_p_restart.close()
        for ds in ds_r_list:
            try:
                ds.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()