#!/usr/bin/env python3
"""
AI Predictions to Restart File Updater

This script directly updates model restart files with AI predictions for PFT1D and soil2D variables.
It uses netCDF4 for direct file manipulation to avoid xarray encoding issues.

Usage:
    python ai_predictions_to_restart.py --variable-list CNP_IO_demo1.txt

The script will:
1. Load AI predictions and model restart file
2. Create spatial mapping between AI and model gridcells
3. Update only CNP_IO variables (PFT1D and soil2D)
4. Save updated restart file with original attributes preserved
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import netCDF4 as nc
import sys
import json
import re
import math

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list


def _safe_get(ds, name):
    """Safely get a variable from dataset, with error handling."""
    if name not in ds:
        raise KeyError(f"Missing required variable/coordinate: {name}")
    return ds[name]

def _to_zero_based_index(idx_raw, n_grid):
    """Convert one-based indices to zero-based indices."""
    idx = np.asarray(idx_raw, dtype=np.int64).copy()
    if idx.size == 0:
        return np.full_like(idx, -1)
    is_one_based = (np.any(idx == n_grid) or (np.nanmin(idx) == 1))
    if is_one_based:
        idx = idx - 1
    idx[(idx < 0) | (idx >= n_grid)] = -1
    return idx

def _build_gridcell_groups(one_d_to_grid, n_grid):
    """Build mapping from gridcell to indices."""
    groups = [[] for _ in range(n_grid)]
    for idx, g in enumerate(one_d_to_grid):
        if 0 <= g < n_grid:
            groups[g].append(idx)
    return groups


def _parse_lat_range(lat_range_text: str) -> tuple[float, float]:
    """Parse latitude range string in 'min,max' format."""
    parts = [p.strip() for p in str(lat_range_text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid latitude range '{lat_range_text}'. Expected format: min,max")
    lat_min, lat_max = float(parts[0]), float(parts[1])
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    return lat_min, lat_max


def _wrap_lon(lon: float) -> float:
    """Wrap longitude to [-180, 180) for stable coordinate matching."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _coord_key(lon: float, lat: float, decimals: int) -> tuple[float, float]:
    return (round(_wrap_lon(lon), decimals), round(float(lat), decimals))


def build_update_grid_mask(
    variable_mapping: Dict[str, Any],
    merge_scope: str,
    tropical_lat_range: tuple[float, float],
    coord_tol: float
) -> np.ndarray:
    """Build per-gridcell update mask for restart overwrite."""
    n_grid = int(variable_mapping["n_grid"])
    mask_all = np.ones(n_grid, dtype=bool)
    if merge_scope == "all":
        print("Merge scope: all model gridcells (backward-compatible behavior)")
        return mask_all

    model_lon = np.asarray(variable_mapping["model_lon"], dtype=float)
    model_lat = np.asarray(variable_mapping["model_lat"], dtype=float)
    ai_lon = np.asarray(variable_mapping["ai_lon"], dtype=float)
    ai_lat = np.asarray(variable_mapping["ai_lat"], dtype=float)

    lat_min, lat_max = tropical_lat_range
    in_tropical = np.isfinite(model_lat) & (model_lat >= lat_min) & (model_lat <= lat_max)

    tol = float(coord_tol)
    if not np.isfinite(tol) or tol <= 0:
        tol = 1e-6
    decimals = max(0, int(math.ceil(-math.log10(tol))))

    ai_coord_keys = set()
    for lon, lat in zip(ai_lon, ai_lat):
        if np.isfinite(lon) and np.isfinite(lat):
            ai_coord_keys.add(_coord_key(lon, lat, decimals))

    has_ai_match = np.zeros(n_grid, dtype=bool)
    for g in range(n_grid):
        if not (np.isfinite(model_lon[g]) and np.isfinite(model_lat[g])):
            continue
        has_ai_match[g] = _coord_key(model_lon[g], model_lat[g], decimals) in ai_coord_keys

    update_mask = in_tropical & has_ai_match
    print(f"Merge scope: tropical-only (lat in [{lat_min}, {lat_max}])")
    print(f"Coordinate tolerance for matching: {tol} (rounded decimals: {decimals})")
    print(f"  Tropical model gridcells: {int(in_tropical.sum())}/{n_grid}")
    print(f"  Model gridcells matched in AI coords: {int(has_ai_match.sum())}/{n_grid}")
    print(f"  Gridcells eligible for overwrite: {int(update_mask.sum())}/{n_grid}")
    return update_mask


def load_datasets(ai_predictions_path: Path, restart_file_path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Load AI predictions and model restart datasets."""
    print("Loading datasets...")
    
    # Load AI predictions
    print(f"  Loading AI predictions: {ai_predictions_path}")
    ds_ai = xr.open_dataset(ai_predictions_path)
    print(f"    AI dataset shape: {dict(ds_ai.sizes)}")
    
    # Load model restart file
    print(f"  Loading model restart file: {restart_file_path}")
    ds_model = xr.open_dataset(restart_file_path)
    print(f"    Model dataset shape: {dict(ds_model.sizes)}")
    
    return ds_ai, ds_model


def create_spatial_mapping(ds_ai: xr.Dataset, ds_model: xr.Dataset) -> tuple[np.ndarray, Dict[str, Any]]:
    """Create spatial mapping between AI and model gridcells."""
    print("Creating spatial mapping...")
    
    # Get coordinates
    ai_lon = ds_ai['grid1d_lon'].values
    ai_lat = ds_ai['grid1d_lat'].values
    model_lon = ds_model['grid1d_lon'].values
    model_lat = ds_model['grid1d_lat'].values
    
    print(f"  AI coordinates: {len(ai_lon)} gridcells")
    print(f"  Model coordinates: {len(model_lon)} gridcells")
    
    # Create spatial mapping using nearest neighbor (exact same as working script)
    from scipy.spatial.distance import cdist
    ai_coords = np.column_stack([ai_lon, ai_lat])
    model_coords = np.column_stack([model_lon, model_lat])
    distances = cdist(model_coords, ai_coords)
    model_to_ai_mapping = np.argmin(distances, axis=1) # 索引是模型格点, 值是最近的 AI 格点
    print(f"  Spatial mapping created: {len(model_to_ai_mapping)} Model -> {len(set(model_to_ai_mapping))} AI")
    # Get grid information from the MODEL file as the master coordinate system
    n_grid = ds_model.sizes["gridcell"]
    print(f"  Using MODEL gridcell count: {n_grid}")
    
    # Build mappings from the model file for extracting model data (exact same as working script)
    col2grid = _to_zero_based_index(_safe_get(ds_model, "cols1d_gridcell_index").values, n_grid)
    pft2grid = _to_zero_based_index(_safe_get(ds_model, "pfts1d_gridcell_index").values, n_grid)
    
    grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
    grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)
    
    print(f"  Model mappings: total columns: {col2grid.size} | total pfts: {pft2grid.size}")
    print(f"  Example: gridcell 0 -> columns {grid_to_cols[0][:5]}, pfts {grid_to_pfts[0][:5]}")
    
    # Create variable mapping for PFT and column indices
    variable_mapping = {
        'grid_to_cols': grid_to_cols,
        'grid_to_pfts': grid_to_pfts,
        'n_grid': n_grid,
        'ai_lon': ai_lon,
        'ai_lat': ai_lat,
        'model_lon': model_lon,
        'model_lat': model_lat
    }
    
    return model_to_ai_mapping, variable_mapping


def auto_detect_variable_list(ai_predictions_path: Path) -> list:
    # Search for cnp_config.json in ai_predictions_path and its parents
    for parent in [ai_predictions_path.parent] + list(ai_predictions_path.parents):
        config_path = parent / 'cnp_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                data_info = config.get('data_info', {})
                vars_1d = data_info.get('variables_1d_pft', [])
                # Patch: check both possible keys for soil2D variables
                vars_2d = data_info.get('variables_2d_soil', [])
                if not vars_2d:
                    vars_2d = data_info.get('x_list_columns_2d', [])
                print(f"Auto-detected variables from {config_path}")
                print(f"  1D PFT variables: {vars_1d}")
                print(f"  2D soil variables: {vars_2d}")
                return list(vars_1d) + list(vars_2d)
            except Exception as e:
                print(f"Warning: Failed to parse {config_path}: {e}")
    print("Warning: Could not auto-detect variable list. No variables will be updated.")
    return []


def create_updated_restart_file(restart_file_path: Path, output_path: Path, 
                               ai_predictions_path: Path, cnp_io_variables: List[str],
                               model_to_ai_mapping: np.ndarray, variable_mapping: Dict[str, Any],
                               strict_dims: bool = False,
                               tropical_lat_range: Optional[tuple] = None) -> None:
    """
    tropical_lat_range: If (min_lat, max_lat), only update gridcells with lat in [min_lat, max_lat];
        others are left unchanged (for merging tropical-model into global restart).
    """
    print(f"Saving updated restart file to: {output_path}")
    if tropical_lat_range is not None:
        print(f"  Tropical-only update: lat in [{tropical_lat_range[0]}, {tropical_lat_range[1]}]")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the original restart file
    shutil.copy2(restart_file_path, output_path)
    print(f"Copied original restart file to: {output_path}")
    
    # Open the output file for direct modification
    with nc.Dataset(output_path, 'r+') as ds_out:
        # Build mask of gridcells to update (all, or only those in tropical_lat_range)
        n_grid = variable_mapping.get("n_grid", 0)
        update_mask = np.ones(n_grid, dtype=bool)
        if tropical_lat_range is not None and n_grid > 0:
            if "grid1d_lat" in ds_out.variables:
                grid_lat = np.asarray(ds_out.variables["grid1d_lat"][:]).ravel()
                min_lat, max_lat = float(tropical_lat_range[0]), float(tropical_lat_range[1])
                update_mask = (grid_lat >= min_lat) & (grid_lat <= max_lat)
                print(f"  Gridcells in tropical band: {update_mask.sum()} / {n_grid}")
            else:
                print("  Warning: grid1d_lat not found; applying tropical filter to all gridcells")
        # Verify and adjust spinup_state
        try:
            if 'spinup_state' in ds_out.variables:
                spin_var = ds_out.variables['spinup_state']
                try:
                    orig_val = np.array(spin_var[:]).item() if spin_var.size == 1 else None
                except Exception:
                    orig_val = None
                if orig_val is not None:
                    print(f"spinup_state in original restart (copied): {orig_val}")
                    if orig_val != 1:
                        print("Warning: Expected spinup_state==1 for adspinup; proceeding to set final_spinup (0) anyway")
                else:
                    print("Warning: Could not read scalar value of spinup_state; proceeding to set to 0")
                # Set to final_spinup mode (0)
                try:
                    spin_var[...] = 0
                    print("Set spinup_state to 0 (final_spinup) in updated restart")
                except Exception as e:
                    print(f"Warning: Failed to set spinup_state to 0: {e}")
            else:
                print("Warning: 'spinup_state' variable not found in restart; skipping spinup flag update")
        except Exception as e:
            print(f"Warning: spinup_state check/update failed: {e}")
        # Load AI predictions
        with nc.Dataset(ai_predictions_path, 'r') as ds_ai:
            # Helpers for shape/dimension checks
            def _fail_or_warn(msg: str) -> bool:
                if strict_dims:
                    raise ValueError(msg)
                print(f"Warning: {msg} — skipping this variable")
                return False
            
            def _check_pft_compat(ai_var: nc.Variable, model_var: nc.Variable) -> bool:
                # Expect AI dims to include pft and gridcell
                ai_dims = list(ai_var.dimensions)
                if not ('pft' in ai_dims and 'gridcell' in ai_dims):
                    return _fail_or_warn(f"PFT var '{ai_var.name}' missing required dims (has {ai_dims}, need ['pft','gridcell'])")
                # Model var should be 1D over pfts1d (or equivalent)
                if len(model_var.shape) < 1:
                    return _fail_or_warn(f"Model PFT var '{model_var.name}' has invalid shape {model_var.shape}")
                # Require at least 16 PFT slots (PFT1..PFT16). We skip PFT0 by design.
                if model_var.shape[0] < 16:
                    return _fail_or_warn(f"Model PFT var '{model_var.name}' has insufficient length {model_var.shape[0]} (<16)")
                return True
            
            def _check_soil_compat(ai_var: nc.Variable, model_var: nc.Variable) -> bool:
                # Expect AI dims: (column, levgrnd, gridcell)
                ai_dims = list(ai_var.dimensions)
                required = {'column','levgrnd','gridcell'}
                if not required.issubset(set(ai_dims)):
                    return _fail_or_warn(f"Soil var '{ai_var.name}' missing required dims (has {ai_dims}, need {sorted(required)})")
                if len(model_var.shape) < 2:
                    return _fail_or_warn(f"Model soil var '{model_var.name}' has invalid shape {model_var.shape}")
                # Need at least 10 layers in model to write top 10
                if model_var.shape[1] < 10:
                    return _fail_or_warn(f"Model soil var '{model_var.name}' has insufficient levgrnd={model_var.shape[1]} (<10)")
                return True
            
            # Update PFT variables
            for var_name in ds_ai.variables:
                if (var_name in ds_out.variables and 
                    'pft' in ds_ai.variables[var_name].dimensions and
                    var_name in cnp_io_variables):
                    print(f"  Updating PFT variable: {var_name}")
                    ai_data = ds_ai.variables[var_name][:]  # (pft, gridcell)
                    model_var = ds_out.variables[var_name]
                    print(f"    AI data shape: {ai_data.shape}")
                    print(f"    Model variable shape: {model_var.shape}")
                    
                    # Dimension compatibility check
                    if not _check_pft_compat(ds_ai.variables[var_name], model_var):
                        continue
                    
                    # Get the grid-to-pfts mapping
                    if 'grid_to_pfts' in variable_mapping:
                        grid_to_pfts = variable_mapping['grid_to_pfts']
                        
                        # For each model gridcell, update PFT data
                        for g in range(variable_mapping['n_grid']):
                            if not update_mask[g]:
                                continue
                            if g < len(grid_to_pfts) and len(grid_to_pfts[g]) > 0:
                                # Get PFTs in this gridcell
                                gridcell_pfts = grid_to_pfts[g]
                                

                                ai_gridcell_idx = model_to_ai_mapping[g]
                                gridcell_pfts = gridcell_pfts[:16] 
                                for pft_idx, model_pft_idx in enumerate(gridcell_pfts):
                                    # Skip PFT0 (index 0), start from PFT1 (index 1)
                                    if 1 <= pft_idx <= 16 and model_pft_idx < len(model_var):
                                        adjusted_k = pft_idx - 1  
                                        if adjusted_k < ai_data.shape[0]:
                                            model_var[model_pft_idx] = ai_data[adjusted_k, ai_gridcell_idx]
            
            # Update soil variables
            for var_name in ds_ai.variables:
                if (var_name in ds_out.variables and 
                    'column' in ds_ai.variables[var_name].dimensions and 
                    'levgrnd' in ds_ai.variables[var_name].dimensions and
                    var_name in cnp_io_variables):
                    print(f"  Updating soil variable: {var_name}")
                    ai_data = ds_ai.variables[var_name][:]  # (column, levgrnd, gridcell)
                    model_var = ds_out.variables[var_name]
                    print(f"    AI data shape: {ai_data.shape}")
                    print(f"    Model variable shape: {model_var.shape}")
                    
                    # Dimension compatibility check
                    if not _check_soil_compat(ds_ai.variables[var_name], model_var):
                        continue
                    
                    # Get the grid-to-cols mapping
                    if 'grid_to_cols' in variable_mapping:
                        grid_to_cols = variable_mapping['grid_to_cols']
                        
                        # For each model gridcell, update column data
                        for g in range(variable_mapping['n_grid']):
                            if not update_mask[g]:
                                continue
                            if g < len(grid_to_cols) and len(grid_to_cols[g]) > 0:
                                # Get columns in this gridcell
                                gridcell_cols = grid_to_cols[g]
                                

                                ai_gridcell_idx = model_to_ai_mapping[g]

                                if len(gridcell_cols) > 0:
                                    model_col_idx = gridcell_cols[0]  
                                    if model_col_idx < model_var.shape[0]:
                                        layers_to_update = min(10, ai_data.shape[1])
                                        for layer_idx in range(layers_to_update):
                                            if ai_data.ndim == 3:
                                                model_var[model_col_idx, layer_idx] = ai_data[0, layer_idx, ai_gridcell_idx]
                                            else:
                                                model_var[model_col_idx, layer_idx] = ai_data[0, layer_idx]
    
    print(f"Updated restart file saved successfully!")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

def get_varlist_name_from_config(ai_predictions_path):
    for parent in [ai_predictions_path.parent] + list(ai_predictions_path.parents):
        config_path = parent / 'cnp_config.json'
        if config_path.exists():
            return config_path.stem
    return 'auto'


def get_varlist_name_from_log_or_config(ai_predictions_path):
    # 1. Search for most recent cnp_training_*.log in run dir or parents
    run_dir = ai_predictions_path.parent
    log_file = None
    for parent in [run_dir] + list(run_dir.parents):
        logs = sorted(parent.glob('cnp_training_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            log_file = logs[0]
            break
    if log_file:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    m = re.search(r'variable list file: (\S+)', line)
                    if m:
                        return Path(m.group(1)).stem
        except Exception as e:
            print(f"Warning: Failed to parse {log_file}: {e}")
    # 2. Fallback to config.json
    for parent in [run_dir] + list(run_dir.parents):
        config_path = parent / 'cnp_config.json'
        if config_path.exists():
            return config_path.stem
    # 3. Fallback
    return 'auto'


def main():
    parser = argparse.ArgumentParser(
        description='Insert AI predictions (PFT1D and soil2D variables only) into model restart file to create updated restart file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update restart file with AI predictions
  python ai_predictions_to_restart.py \
    --ai-predictions ai_predictions_for_plotting.nc \
    --restart-file model_restart.nc \
    --output updated_restart.nc

  # Use with specific variable list
  python ai_predictions_to_restart.py \
    --ai-predictions ai_predictions_for_plotting.nc \
    --restart-file model_restart.nc \
    --output updated_restart.nc \
    --variable-list CNP_IO_demo1.txt

  # Preview changes without saving
  python ai_predictions_to_restart.py \
    --ai-predictions ai_predictions_for_plotting.nc \
    --restart-file model_restart.nc \
    --output updated_restart.nc \
    --preview-only

  # Phase 2: update only P variables in tropical cells (merge Phase 2 into Phase 1 restart)
  python ai_predictions_to_restart.py \
    --ai-predictions phase2_predictions.nc \
    --restart-file phase1_global_restart.nc \
    --output merged_restart.nc \
    --variable-list CNP_IO_updated9_dev_dw.txt \
    --tropical-lat-range -30,30 \
    --variables-to-update occlp_vr,labilep_vr,solutionp_vr,primp_vr,secondp_vr,soil1p_vr,soil2p_vr,soil3p_vr,soil4p_vr,litr2p_vr,litr3p_vr,cwdp_vr
        """
    )
    
    parser.add_argument('--ai-predictions', default='./comparison_results/ai_predictions_for_plotting.nc',
                       help='Path to AI predictions NetCDF file (ai_predictions_for_plotting.nc)')
    parser.add_argument('--restart-file', default='/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc',
                       help='Path to model restart file to update')
    parser.add_argument('--output', default=None,
                       help='Output path for updated restart file [default: auto-generated based on variable list]')
    parser.add_argument('--variable-list', type=str, required=False,
                       help='Path to CNP_IO_list file to specify which variables to update (optional)')
    parser.add_argument('--preview-only', action='store_true',
                       help='Preview changes without saving updated restart file')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of original restart file before updating')
    parser.add_argument('--strict-dims', action='store_true',
                       help='Abort on any dimension mismatch instead of skipping')
    parser.add_argument('--tropical-lat-range', type=str, metavar='MIN,MAX', default=None,
                       help='Only update gridcells with lat in [MIN, MAX] (e.g. "-30,30"). Use when merging tropical-model predictions into a global restart.')
    parser.add_argument('--variables-to-update', type=str, default=None, metavar='VAR1,VAR2,...|@file.txt',
                       help='Only update these variables (subset of CNP_IO list). Comma-separated names (e.g. occlp_vr,labilep_vr,solutionp_vr) or path to a file with one variable per line (e.g. @phase2_p_vars.txt). If not set, all variables from the variable list are updated. Use with Phase 2 to update only P variables in tropical cells.')
    
    args = parser.parse_args()
    
    # Validate input files
    ai_predictions_path = Path(args.ai_predictions)
    restart_file_path = Path(args.restart_file)
    
    if not ai_predictions_path.exists():
        parser.error(f'AI predictions file not found: {ai_predictions_path}')
    
    if not restart_file_path.exists():
        parser.error(f'Restart file not found: {restart_file_path}')
    
    # Generate output path based on log/config name if --output is not specified
    if args.output is None:
        varlist_name = get_varlist_name_from_log_or_config(ai_predictions_path)
        restart_name = Path(args.restart_file).stem
        output_path = Path(f"updated_restart_{varlist_name}_{restart_name}.nc")
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("AI Predictions to Restart File Updater")
    print("=" * 60)
    print(f"AI predictions: {ai_predictions_path}")
    print(f"Restart file: {restart_file_path}")
    print(f"Output: {output_path}")
    print(f"Preview only: {args.preview_only}")
    print(f"Create backup: {args.backup}")
    print(f"Merge scope: {args.merge_scope}")
    print("=" * 60)
    
    # Load datasets
    ds_ai, ds_model = load_datasets(ai_predictions_path, restart_file_path)
    
    # Create spatial mapping
    ai_to_model_mapping, variable_mapping = create_spatial_mapping(ds_ai, ds_model)
    
    tropical_lat_range = (-23.5, 23.5)
    if args.merge_scope == 'tropical-only':
        try:
            tropical_lat_range = _parse_lat_range(args.tropical_lat_range)
        except Exception as e:
            parser.error(f"Invalid --tropical-lat-range: {e}")
    update_grid_mask = build_update_grid_mask(
        variable_mapping=variable_mapping,
        merge_scope=args.merge_scope,
        tropical_lat_range=tropical_lat_range,
        coord_tol=args.coord_tol
    )
    
    # Parse CNP_IO list if provided, else auto-detect
    cnp_io_variables = []
    if args.variable_list:
        var_list_path = Path(args.variable_list)
        if var_list_path.exists():
            try:
                parsed_vars = parse_cnp_io_list(var_list_path)
                if 'pft_1d_variables' in parsed_vars:
                    cnp_io_variables.extend(parsed_vars['pft_1d_variables'])
                # Fix: include both possible keys for soil2D variables
                if 'variables_2d_soil' in parsed_vars:
                    cnp_io_variables.extend(parsed_vars['variables_2d_soil'])
                elif 'x_list_columns_2d' in parsed_vars:
                    cnp_io_variables.extend(parsed_vars['x_list_columns_2d'])
                print(f"  Variables to update: {cnp_io_variables}")
            except Exception as e:
                print(f"  Warning: Could not parse variable list: {e}")
                parsed_vars = None
                cnp_io_variables = []
        else:
            print(f"  Warning: Variable list file not found: {var_list_path}")
            parsed_vars = None
            cnp_io_variables = []
    else:
        # Auto-detect from config.json
        cnp_io_variables = auto_detect_variable_list(Path(args.ai_predictions))
    
    # Optionally restrict to a subset of variables (e.g. P-only for Phase 2 merge)
    effective_update_variables = list(cnp_io_variables)
    if getattr(args, 'variables_to_update', None) and args.variables_to_update.strip():
        raw = args.variables_to_update.strip()
        if raw.startswith('@'):
            path = Path(raw[1:].strip())
            if path.exists():
                with open(path, 'r') as f:
                    requested = {line.strip() for line in f if line.strip() and not line.strip().startswith('#')}
            else:
                parser.error(f'Variables-to-update file not found: {path}')
        else:
            requested = {v.strip() for v in raw.split(',') if v.strip()}
        effective_update_variables = [v for v in cnp_io_variables if v in requested]
        not_in_cnp = requested - set(cnp_io_variables)
        if not_in_cnp:
            print(f"  Note: --variables-to-update names not in CNP_IO list (skipped): {sorted(not_in_cnp)}")
        if not effective_update_variables:
            parser.error('--variables-to-update resulted in no variables to update (none matched CNP_IO list)')
        print(f"  Restricting to {len(effective_update_variables)} variables: {effective_update_variables}")
    else:
        effective_update_variables = list(cnp_io_variables)
    
    # Note: We only update variables in the effective list (CNP_IO list, optionally filtered)
    print("Note: Only variables in the update list will be modified in the restart")
    print("All other variables and attributes remain unchanged")
    
    # Print summary of changes
    print("\n" + "=" * 60)
    print("UPDATE SUMMARY")
    print("=" * 60)
    
    # Count variables that will be updated (only PFT1D and soil2D from effective list)
    updated_vars = []
    for var_name in ds_ai.data_vars:
        if var_name in ds_model.data_vars and var_name in effective_update_variables:
            # Only count PFT1D and soil2D variables that are in the update list
            if ('pft' in ds_ai[var_name].dims) or ('column' in ds_ai[var_name].dims and 'levgrnd' in ds_ai[var_name].dims):
                updated_vars.append(var_name)
    
    print(f"Variables to update (PFT1D and soil2D): {len(updated_vars)}")
    for var_name in updated_vars:
        ai_shape = ds_ai[var_name].shape
        model_shape = ds_model[var_name].shape
        var_type = "PFT1D" if 'pft' in ds_ai[var_name].dims else "Soil2D"
        print(f"  {var_name} ({var_type}): AI {ai_shape} -> Model {model_shape}")
    
    # Show which variables were skipped (in CNP_IO but not in effective update list)
    skipped_vars = []
    for var_name in ds_ai.data_vars:
        if (var_name in ds_model.data_vars and 
            ('pft' in ds_ai[var_name].dims or ('column' in ds_ai[var_name].dims and 'levgrnd' in ds_ai[var_name].dims)) and
            var_name in cnp_io_variables and var_name not in effective_update_variables):
            skipped_vars.append(var_name)
    
    if skipped_vars:
        print(f"\nVariables skipped (not in --variables-to-update): {len(skipped_vars)}")
        for var_name in skipped_vars:
            print(f"  {var_name}")
    
    # Print coordinate information
    print(f"\nCoordinate mapping:")
    print(f"  AI gridcells: {ds_ai.sizes.get('gridcell', 'N/A')}")
    print(f"  Model gridcells: {ds_model.sizes.get('gridcell', 'N/A')}")
    print(f"  Spatial mapping: {len(ai_to_model_mapping)} AI -> {len(set(ai_to_model_mapping))} Model")
    print(f"  Effective overwrite gridcells: {int(np.count_nonzero(update_grid_mask))}")
    
    if not args.preview_only:
        # Create backup if requested
        if args.backup:
            backup_path = restart_file_path.with_suffix('.backup.nc')
            print(f"\nCreating backup: {backup_path}")
            shutil.copy2(restart_file_path, backup_path)
            print(f"Backup created: {backup_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Save updated restart file using direct NetCDF manipulation
        tropical_lat_range = None
        if getattr(args, 'tropical_lat_range', None):
            parts = [p.strip() for p in args.tropical_lat_range.split(',')]
            if len(parts) >= 2:
                tropical_lat_range = (float(parts[0]), float(parts[1]))
        create_updated_restart_file(restart_file_path, output_path, ai_predictions_path, 
                                   effective_update_variables, ai_to_model_mapping, variable_mapping,
                                   strict_dims=args.strict_dims,
                                   tropical_lat_range=tropical_lat_range)
        
        print(f"\nRestart file updated successfully!")
        print(f"Original: {restart_file_path}")
        print(f"Updated: {output_path}")
        if args.backup:
            print(f"Backup: {restart_file_path.with_suffix('.backup.nc')}")
        
        print(f"\nUpdate Summary:")
        print(f"  PFT1D variables: Updated PFT1-PFT16 (skipped PFT0) in first column of each gridcell")
        print(f"  Soil2D variables: Updated first column and first 10 layers in each gridcell")
        print(f"  Model layers 11-15: Preserved (not modified)")
        print(f"  Other columns: Preserved (not modified)")
        print(f"  Spatial mapping: Used geographic coordinates to map AI gridcells to model gridcells")
        print(f"  Coordinate system: Model coordinates used as master reference for alignment")
        print(f"  Overwrite scope: {args.merge_scope} (eligible gridcells: {int(np.count_nonzero(update_grid_mask))})")
        print(f"  Important: Only CNP_IO variables were modified - all other variables and attributes unchanged")
        
        print(f"\nYou can now use the updated restart file for model simulations!")
    else:
        print(f"\nPreview mode - no files were modified")
        print(f"To apply changes, run without --preview-only flag")
    
    # Clean up
    ds_ai.close()
    ds_model.close()


if __name__ == '__main__':
    main()