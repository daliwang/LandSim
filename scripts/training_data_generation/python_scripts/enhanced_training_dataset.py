#!/usr/bin/env python3
"""
Enhanced Training Dataset Generation Script
Combines 72_dataset_construction.py and 37_dataset.py functionality
Dynamically extracts variables from CNP_IO file instead of using hardcoded lists
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os
import sys
import time
import glob
import argparse
from typing import Dict, List, Tuple

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from python_scripts.cnp_io_parse import parse_cnp_io_list

print("="*80)
print("Enhanced Training Dataset Generation")
print("Combining 72_dataset_construction.py + 37_dataset.py")
print("Dynamic variable extraction from CNP_IO file")
print("="*80)

# Constants
VARS_TO_CLEAN = {'H2OSOI_LIQ', 'H2OSOI_ICE'}
FILL_VALUE_THRESHOLD = 1e35

def parse_cnp_io_variables():
    """Parse CNP_IO file to get all variable definitions dynamically"""
    cnp_io_file = config.Config.CNP_IO_FILE
    if not os.path.exists(cnp_io_file):
        raise FileNotFoundError(f"CNP_IO file not found: {cnp_io_file}")
    
    parsed = parse_cnp_io_list(cnp_io_file)
    print(f"✅ CNP_IO file parsed: {cnp_io_file}")
    
    # Extract variable categories
    time_series_vars = parsed.get('time_series_variables', [])
    surface_vars = parsed.get('surface_properties', [])
    scalar_vars = parsed.get('scalar_variables', [])
    pft_1d_vars = parsed.get('pft_1d_variables', [])
    variables_2d_soil = parsed.get('variables_2d_soil', [])
    water_vars = parsed.get('water_variables', [])
    pool_vars = parsed.get('pool_variables', ['cpool', 'npool', 'ppool', 'xsmrpool'])
    pft_parameters = parsed.get('pft_parameters', [])
    
    print(f"Variable categories from CNP_IO:")
    print(f"  Time series variables: {len(time_series_vars)}")
    print(f"  Surface properties: {len(surface_vars)}")
    print(f"  Scalar variables: {len(scalar_vars)}")
    print(f"  PFT 1D variables: {len(pft_1d_vars)}")
    print(f"  2D soil variables: {len(variables_2d_soil)}")
    print(f"  Water variables: {len(water_vars)}")
    print(f"  Pool variables: {len(pool_vars)}")
    print(f"  PFT parameters: {len(pft_parameters)}")
    
    return {
        'time_series_vars': time_series_vars,
        'surface_vars': surface_vars,
        'scalar_vars': scalar_vars,
        'pft_1d_vars': pft_1d_vars,
        'variables_2d_soil': variables_2d_soil,
        'water_vars': water_vars,
        'pool_vars': pool_vars,
        'pft_parameters': pft_parameters
    }

def build_restart_kdtree(ds_restart: nc.Dataset) -> Tuple[cKDTree, np.ndarray]:
    """Build KDTree for restart file coordinates"""
    gridcell_lat = ds_restart.variables["grid1d_lat"][:]
    gridcell_lon = ds_restart.variables["grid1d_lon"][:]
    coords = np.vstack((gridcell_lat, gridcell_lon)).T
    tree = cKDTree(coords)
    return tree, coords

def build_column_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    """Build mapping from gridcell ID to column indices"""
    cols1d_gridcell_index = ds_restart.variables["cols1d_gridcell_index"][:]
    unique_ids = np.unique(cols1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(cols1d_gridcell_index == grid_id)[0]
    return mapping

def build_pft_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    """Build mapping from gridcell ID to PFT indices"""
    pfts1d_gridcell_index = ds_restart.variables["pfts1d_gridcell_index"][:]
    unique_ids = np.unique(pfts1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(pfts1d_gridcell_index == grid_id)[0]
    return mapping

def ensure_vars_exist(ds: nc.Dataset, var_names: List[str]) -> List[str]:
    """Check which variables exist in the dataset"""
    existing = []
    for name in var_names:
        if name in ds.variables:
            existing.append(name)
    return existing

def extract_col1d_x(ds_restart: nc.Dataset, var_name: str, col_indices: np.ndarray) -> List[float]:
    """Extract 1D column variables from restart file"""
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices]
    return values.astype(float).tolist()

def extract_col1d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[float]:
    """Extract 1D column variables from Y files (future restart files)"""
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
    """Extract 2D column variables from restart file"""
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices, :]
    values_np = np.asarray(values, dtype=float)
    if var_name in VARS_TO_CLEAN:
        values_np[values_np >= FILL_VALUE_THRESHOLD] = 0.0
    return values_np.tolist()

def extract_col2d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[List[float]]:
    """Extract 2D column variables from Y files"""
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
    """Extract 1D PFT variables from restart file"""
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices]
    return np.asarray(values, dtype=float).tolist()

def extract_pft1d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[float]:
    """Extract 1D PFT variables from Y files"""
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
    """Extract 2D PFT variables from restart file"""
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices, :]
    return np.asarray(values, dtype=float).tolist()

def extract_pft2d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[List[float]]:
    """Extract 2D PFT variables from Y files"""
    if pft_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][pft_indices, :]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()

def calculate_monthly_avg(time_series, time_series_length=58400):
    """Calculate monthly averages from high-resolution time series"""
    if not isinstance(time_series, (list, np.ndarray)):
        return []
    
    if len(time_series) != time_series_length:
        return []

    # TVA data parameters (20 years, 3-hour interval)
    steps_per_day = 8           # 3-hour interval = 8 steps/day
    days_per_year = 365
    years_in_data = 20          # 1980-1999
    months_per_year = 12
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    monthly_averages = []
    start_idx = 0
    
    for year in range(years_in_data):
        for month_idx, month_days in enumerate(days_per_month):
            # Handle leap year February
            if year % 4 == 0 and month_idx == 1:  # Leap year February
                month_days = 29
            
            end_idx = start_idx + month_days * steps_per_day
            monthly_avg = np.mean(time_series[start_idx:end_idx])
            # Ensure float64 precision to match reference file
            monthly_averages.append(float(monthly_avg))
            start_idx = end_idx
    
    return monthly_averages

def detect_grid_format(ds):
    """Detect if dataset uses 1D (gridcell-based) or 2D (lat/lon grid) format"""
    if 'landfrac' not in ds.variables:
        return None, None, None
    
    landfrac_var = ds.variables['landfrac']
    landfrac_dims = landfrac_var.dimensions
    
    if len(landfrac_dims) == 1:
        return '1d', landfrac_dims[0], None
    elif len(landfrac_dims) == 2:
        return '2d', landfrac_dims[0], landfrac_dims[1]
    else:
        return None, None, None

def get_gridcell_value(var, var_dims, gridcell_idx, grid_format, grid_info, dim1=None, dim2=None, lat_idx=None, lon_idx=None):
    """Get a scalar value from a variable, handling both 1D and 2D grid formats"""
    val = None
    
    if grid_format == '2d' and len(var_dims) == 2 and var_dims[0] == dim1 and var_dims[1] == dim2:
        # Variable is 2D (lat, lon) format
        if lat_idx is not None and lon_idx is not None:
            val = var[lat_idx, lon_idx]
        else:
            # Fallback: flatten and index
            val = var[:].flatten()[gridcell_idx]
    elif len(var_dims) == 1:
        # Variable is 1D, use direct indexing
        val = var[gridcell_idx]
    else:
        # Try direct indexing as fallback
        try:
            val = var[gridcell_idx]
        except:
            # Last resort: flatten and index
            val = var[:].flatten()[gridcell_idx]
    
    # Convert MaskedArray to regular array if needed
    if hasattr(val, 'data'):  # MaskedArray
        val = val.data
    
    # Ensure scalar value
    if isinstance(val, np.ndarray):
        if val.size == 1:
            val = val.item()
        elif val.size > 1:
            # Take first element if array
            val = val.flatten()[0]
    
    return val

def generate_base_dataset(variable_definitions, use_monthly_forcing=False, forcing_year_range="1980-1999"):
    """Generate base training dataset (72_dataset_construction.py logic)"""
    print(f"\n{'='*80}")
    print("STEP 1: Base Dataset Generation (72_dataset_construction.py)")
    print(f"{'='*80}")
    
    if use_monthly_forcing:
        print("Using pre-computed monthly average forcing data")
        print(f"   Year range: {forcing_year_range}")
    
    # File paths from config
    surface_data_files = config.surface_data_files
    ad_spinup_history_files = config.ad_spinup_history_files
    ad_spinup_restart_files = config.ad_spinup_restart_files
    final_spinup_history_files = config.final_spinup_history_files
    final_spinup_restart_files = config.final_spinup_restart_files
    
    # Forcing data files - dynamically find files containing variable names
    forcing_files = {}
    for var_name in variable_definitions['time_series_vars']:
        # Look for files containing the variable name in forcing_netcdf directory
        # Support both old format (*VAR*1980-1999.nc) and new format (*VAR*2004-2023.nc)
        pattern = os.path.join(config.forcing_netcdf_output_dir, f'*{var_name}*{forcing_year_range}.nc')
        matching_files = glob.glob(pattern)
        if not matching_files:
            # Try alternative pattern without year range in filename
            pattern_alt = os.path.join(config.forcing_netcdf_output_dir, f'{var_name}_*.nc')
            matching_files = glob.glob(pattern_alt)
        if matching_files:
            forcing_files[var_name] = matching_files[0]  # Use first match
            print(f"Found forcing file: {os.path.basename(matching_files[0])}")
        else:
            print(f"Forcing file not found for {var_name}: {pattern}")
    
    print(f"Found {len(forcing_files)} forcing files")
    
    # Output directory
    output_dir = os.path.join(config.output_dir, 'training_dataset_pkl')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading NetCDF files...")
    start_time = time.time()
    
    # Load all required files
    ds1 = nc.Dataset(surface_data_files[0])  # Surface data
    ds2 = nc.Dataset(ad_spinup_history_files[0])  # AD-SPINUP history
    ds10 = nc.Dataset(ad_spinup_restart_files[0])  # AD-SPINUP restart
    
    # Load forcing data
    ds_forcing = {}
    for var_name, file_path in forcing_files.items():
        ds_forcing[var_name] = nc.Dataset(file_path)
        print(f"Forcing data loaded: {var_name}")
    
    # Load future files for Y variables
    ds_h0_list = [nc.Dataset(fp) for fp in final_spinup_history_files]
    ds_r_list = [nc.Dataset(fp) for fp in final_spinup_restart_files]
    
    print(f"All files loaded in {time.time() - start_time:.2f} seconds")
    
    # Detect grid format for history file (ds2)
    # Detect grid format for both ds1 (surface) and ds2 (history) files
    grid_format_ds1, dim1_ds1, dim2_ds1 = detect_grid_format(ds1)
    grid_format_ds2, dim1_ds2, dim2_ds2 = detect_grid_format(ds2)
    print(f"Surface file (ds1) grid format: {grid_format_ds1} (dims: {dim1_ds1}, {dim2_ds1})")
    print(f"History file (ds2) grid format: {grid_format_ds2} (dims: {dim1_ds2}, {dim2_ds2})")
    
    # For backward compatibility, grid_format refers to ds2 format (used for history variables)
    grid_format = grid_format_ds2
    dim1 = dim1_ds2
    dim2 = dim2_ds2
    
    # Get coordinates and build spatial filtering
    landmask_var = ds2.variables['landfrac']
    landmask = landmask_var[:]
    
    # Store grid format info and conversion mappings for both ds1 and ds2
    grid_info = {
        'format': grid_format,  # ds2 format (for backward compatibility)
        'format_ds1': grid_format_ds1,  # ds1 format
        'format_ds2': grid_format_ds2,  # ds2 format
        'dim1_ds1': dim1_ds1,
        'dim2_ds1': dim2_ds1,
        'dim1_ds2': dim1_ds2,
        'dim2_ds2': dim2_ds2,
        'lat_idx_map': {},  # flat_idx -> lat_idx for 2D format (ds2)
        'lon_idx_map': {},  # flat_idx -> lon_idx for 2D format (ds2)
        'lat_idx_map_ds1': {},  # flat_idx -> lat_idx for 2D format (ds1)
        'lon_idx_map_ds1': {},  # flat_idx -> lon_idx for 2D format (ds1)
        'n_lon': None,
        'n_lon_ds1': None
    }
    
    # Build mappings for ds2 (history file)
    if grid_format_ds2 == '1d':
        # 1D format: (gridcell) or (lndgrid)
        lats = ds2.variables['lat'][:]
        lons = ds2.variables['lon'][:]
        
        # Filter for land gridcells
        valid_mask = (landmask > 0)
        valid_gridcells = np.where(valid_mask)[0]
        
        # Get coordinates for valid gridcells (direct indexing)
        query_coords = np.array([(lats[i], lons[i]) for i in valid_gridcells])
        
    elif grid_format_ds2 == '2d':
        # 2D format: (lat, lon)
        lat_coords = ds2.variables['lat'][:]
        lon_coords = ds2.variables['lon'][:]
        grid_info['n_lon'] = len(lon_coords)
        
        # Filter for land gridcells (2D mask)
        valid_mask = (landmask > 0)
        valid_lat_indices, valid_lon_indices = np.where(valid_mask)
        
        # Convert 2D indices to flat gridcell indices
        valid_gridcells_2d = valid_lat_indices * grid_info['n_lon'] + valid_lon_indices
        
        # Get coordinates for valid gridcells
        query_coords_2d = np.array([(lat_coords[i], lon_coords[j]) 
                                    for i, j in zip(valid_lat_indices, valid_lon_indices)])
        
        # Create mapping from flat index to 2D indices for ds2
        for i, flat_idx in enumerate(valid_gridcells_2d):
            grid_info['lat_idx_map'][i] = valid_lat_indices[i]
            grid_info['lon_idx_map'][i] = valid_lon_indices[i]
    
        # For compatibility with later code
        lats = np.array([lat_coords[i] for i in valid_lat_indices])
        lons = np.array([lon_coords[j] for j in valid_lon_indices])
        query_coords = query_coords_2d
        valid_gridcells = valid_gridcells_2d
    else:
        raise ValueError(f"Unsupported grid format: {grid_format}")
    
    # Build mappings for ds1 (surface file) if it's 2D format
    # This needs to be done after we have gridcell coordinates from ds10
    if grid_format_ds1 == '2d' and 'landfrac' in ds1.variables:
        print("Building ds1 (surface file) 2D index mappings...")
        landmask_ds1 = ds1.variables['landfrac'][:]
        lat_coords_ds1 = ds1.variables[dim1_ds1][:]
        lon_coords_ds1 = ds1.variables[dim2_ds1][:]
        grid_info['n_lon_ds1'] = len(lon_coords_ds1)
        
        # Filter for land gridcells in ds1 (2D mask)
        valid_mask_ds1 = (landmask_ds1 > 0)
        valid_lat_indices_ds1, valid_lon_indices_ds1 = np.where(valid_mask_ds1)
        
        # Get restart file coordinates (these are defined later, so we'll build mapping in the main loop if needed)
        # For now, we'll build it after restart coordinates are available
        print("  ds1 2D mapping will be built dynamically during processing")
    
    print(f"Total land gridcells (before deduplication): {len(valid_gridcells)}")
    print(f"Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")
    
    # Build KDTree indices
    print("Building KDTree indices...")
    
    # Restart file coordinates (these are the unique gridcells we want - 20975 total)
    gridcell_lat = ds10.variables['grid1d_lat'][:]
    gridcell_lon = ds10.variables['grid1d_lon'][:]
    restart_grid_coords = np.vstack((gridcell_lat, gridcell_lon)).T
    restart_tree = cKDTree(restart_grid_coords)
    
    # Build mappings for ds1 (surface file) if it's 2D format
    # Now that we have restart coordinates, we can build the mapping
    if grid_format_ds1 == '2d' and 'landfrac' in ds1.variables and len(grid_info['lat_idx_map_ds1']) == 0:
        print("Building ds1 (surface file) 2D index mappings from restart coordinates...")
        landmask_ds1 = ds1.variables['landfrac'][:]
        lat_coords_ds1 = ds1.variables[dim1_ds1][:]
        lon_coords_ds1 = ds1.variables[dim2_ds1][:]
        grid_info['n_lon_ds1'] = len(lon_coords_ds1)
        
        # Filter for land gridcells in ds1 (2D mask)
        valid_mask_ds1 = (landmask_ds1 > 0)
        valid_lat_indices_ds1, valid_lon_indices_ds1 = np.where(valid_mask_ds1)
        
        # Build KDTree for ds1 coordinates
        ds1_coords = np.vstack((lat_coords_ds1[valid_lat_indices_ds1], 
                                 lon_coords_ds1[valid_lon_indices_ds1])).T
        ds1_tree = cKDTree(ds1_coords)
        
        # Query restart coordinates to find closest ds1 gridcell
        _, ds1_matched_indices = ds1_tree.query(restart_grid_coords, k=1)
        
        # Create mapping: restart_idx -> (lat_idx_ds1, lon_idx_ds1)
        for restart_idx in range(len(gridcell_lat)):
            if ds1_matched_indices[restart_idx] < len(valid_lat_indices_ds1):
                matched_flat_idx = ds1_matched_indices[restart_idx]
                grid_info['lat_idx_map_ds1'][restart_idx] = valid_lat_indices_ds1[matched_flat_idx]
                grid_info['lon_idx_map_ds1'][restart_idx] = valid_lon_indices_ds1[matched_flat_idx]
        print(f"  Built {len(grid_info['lat_idx_map_ds1'])} ds1 2D mappings")
    
    # Forcing file coordinates
    forcing_lats = list(ds_forcing.values())[0].variables['LATIXY'][0, :]
    forcing_lons = list(ds_forcing.values())[0].variables['LONGXY'][0, :]
    forcing_grid_coords = np.vstack((forcing_lats, forcing_lons)).T
    forcing_tree = cKDTree(forcing_grid_coords)
    
    # Query coordinates for valid gridcells
    _, all_restart_indices = restart_tree.query(query_coords, k=1)
    _, all_forcing_indices = forcing_tree.query(query_coords, k=1)
    
    # Deduplicate: ensure each unique restart gridcell is only processed once
    # Multiple (lat,lon) pairs may map to the same restart gridcell
    seen_restart_indices = {}
    unique_indices = []
    for i, restart_idx in enumerate(all_restart_indices):
        if restart_idx not in seen_restart_indices:
            unique_indices.append(i)
            seen_restart_indices[restart_idx] = i
    
    # Filter to unique gridcells
    valid_gridcells = valid_gridcells[unique_indices]
    all_restart_indices = all_restart_indices[unique_indices]
    all_forcing_indices = all_forcing_indices[unique_indices]
    lats = lats[unique_indices]
    lons = lons[unique_indices]
    
    # Update grid_info mappings to reflect deduplication
    if grid_format == '2d':
        old_lat_idx_map = grid_info['lat_idx_map'].copy()
        old_lon_idx_map = grid_info['lon_idx_map'].copy()
        grid_info['lat_idx_map'] = {new_idx: old_lat_idx_map[old_idx] 
                                    for new_idx, old_idx in enumerate(unique_indices)}
        grid_info['lon_idx_map'] = {new_idx: old_lon_idx_map[old_idx] 
                                    for new_idx, old_idx in enumerate(unique_indices)}
    
    print(f"Total unique gridcells (after deduplication): {len(valid_gridcells)}")
    print(f"Expected: 20975 gridcells")
    if len(valid_gridcells) != 20975:
        print(f"⚠️  Warning: Expected 20975 gridcells but got {len(valid_gridcells)}")
    
    print("KDTree indices built and deduplication completed")
    
    # Pre-load forcing data into memory for optimization
    print("Pre-loading forcing data into memory...")
    forcing_data = {}
    for var_name, ds in ds_forcing.items():
        forcing_data[var_name] = ds.variables[var_name][:, 0, :]  # (time, 1, grid_cells)
        print(f"  Loaded {var_name}: {forcing_data[var_name].shape}")
    
    # Close forcing NetCDF files (data is now in memory)
    for ds in ds_forcing.values():
        ds.close()
    
    # Build index mappings
    pft_gridcell_index = ds10.variables['pfts1d_gridcell_index'][:]
    column_gridcell_index = ds10.variables['cols1d_gridcell_index'][:]
    
    pft_map = {}
    column_map = {}
    unique_gridcell_ids = np.unique(pft_gridcell_index)
    for grid_id in unique_gridcell_ids:
        pft_map[grid_id] = np.where(pft_gridcell_index == grid_id)[0]
        column_map[grid_id] = np.where(column_gridcell_index == grid_id)[0]
    
    print("Index mappings built")
    
    # Process data in batches
    batch_size = 1000
    batch_number = 1
    batch_files = []  # Track all generated files for post-processing
    
    print(f"\nProcessing {len(valid_gridcells)} gridcells in batches of {batch_size}...")
    
    for start_idx in range(0, len(valid_gridcells), batch_size):
        end_idx = min(start_idx + batch_size, len(valid_gridcells))
        batch_gridcells = valid_gridcells[start_idx:end_idx]
        batch_restart_indices = all_restart_indices[start_idx:end_idx]
        batch_forcing_indices = all_forcing_indices[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_number}: gridcells {start_idx+1}-{end_idx}")
        print(f"  Batch size: {len(batch_gridcells)} gridcells")
        print(f"  batch_gridcells range: {batch_gridcells[0]} to {batch_gridcells[-1]}")
        print(f"  Unique gridcells in batch: {len(set(batch_gridcells))}")
        if len(set(batch_gridcells)) != len(batch_gridcells):
            print(f"WARNING: Duplicate gridcells detected in batch!")
        batch_start_time = time.time()
        
        # Initialize data dictionary dynamically - completely from CNP_IO file
        data_dict = {}
        
        # Add time series variables (forcing data)
        for var_name in variable_definitions['time_series_vars']:
            data_dict[var_name] = []
        
        # Add surface properties
        for var_name in variable_definitions['surface_vars']:
            data_dict[var_name] = []
        
        # Add scalar variables
        for var_name in variable_definitions['scalar_vars']:
            data_dict[var_name] = []
            data_dict[f'Y_{var_name}'] = []
        
        # Add PFT variables
        for var_name in variable_definitions['pft_1d_vars']:
            data_dict[var_name] = []
            data_dict[f'Y_{var_name}'] = []
        
        # Add 2D soil variables
        for var_name in variable_definitions['variables_2d_soil']:
            data_dict[var_name] = []
            data_dict[f'Y_{var_name}'] = []
        
        # Add water variables
        for var_name in variable_definitions['water_vars']:
            data_dict[var_name] = []
            data_dict[f'Y_{var_name}'] = []
        
        # Add pool variables
        for var_name in variable_definitions['pool_vars']:
            data_dict[var_name] = []
            data_dict[f'Y_{var_name}'] = []
        
        # Process each gridcell in the batch
        for k in range(len(batch_gridcells)):
            if k % 100 == 0:
                print(f"  Processing gridcell {k}/{len(batch_gridcells)}")
            
            # Get indices
            restart_idx = int(batch_restart_indices[k])  # Index for ds1 (surface) and ds10 (restart) files
            forcing_idx = int(batch_forcing_indices[k])  # Index for forcing files
            gridcell_idx_history = batch_gridcells[k]  # Index for ds2 (history) file - may be 2D flattened
            gridcell_id = restart_idx + 1
            
            pft_indices_for_cell = pft_map.get(gridcell_id, [])
            column_indices_for_cell = column_map.get(gridcell_id, [])
            
            # All variables are now processed dynamically from CNP_IO file
            
            # Debug: Check data_dict length after each gridcell
            if k == 0:
                print(f"    After first gridcell: Latitude={len(data_dict['Latitude'])}, FLDS={len(data_dict.get('FLDS', []))}")
                print(f"    pft_indices_for_cell length: {len(pft_indices_for_cell)}")
                print(f"    column_indices_for_cell length: {len(column_indices_for_cell)}")
            if k == 999:
                print(f"    After 1000th gridcell: Latitude={len(data_dict['Latitude'])}, FLDS={len(data_dict.get('FLDS', []))}")
            
            # Process forcing data (time series variables)
            for var_name in variable_definitions['time_series_vars']:
                if var_name in forcing_data:
                    time_series = forcing_data[var_name][:, forcing_idx]
                    # If using monthly forcing, data is already monthly average, store as-is
                    # Otherwise, store raw time series (will be processed later in post-processing)
                    if use_monthly_forcing:
                        # Data is already monthly average, convert to list of floats
                        data_dict[var_name].append(time_series.tolist() if isinstance(time_series, np.ndarray) else list(time_series))
                    else:
                        # Store raw time series (will be processed later)
                        data_dict[var_name].append(time_series)
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
            
            # Process surface properties
            for var_name in variable_definitions['surface_vars']:
                if var_name == 'Latitude':
                    # Use global index (start_idx + k) since lats array is already filtered for valid gridcells
                    global_idx = start_idx + k
                    data_dict[var_name].append(lats[global_idx])
                elif var_name == 'Longitude':
                    # Use global index (start_idx + k) since lons array is already filtered for valid gridcells
                    global_idx = start_idx + k
                    data_dict[var_name].append(lons[global_idx])
                elif var_name == 'landfrac':
                    # landfrac comes from history file (ds2), not surface file (ds1)
                    # Convert to float64 to match reference file
                    global_idx = start_idx + k
                    if grid_format == '1d':
                        landfrac_val = ds2.variables['landfrac'][gridcell_idx_history]
                    elif grid_format == '2d':
                        lat_idx = grid_info['lat_idx_map'].get(global_idx, None)
                        lon_idx = grid_info['lon_idx_map'].get(global_idx, None)
                        if lat_idx is not None and lon_idx is not None:
                            landfrac_val = ds2.variables['landfrac'][lat_idx, lon_idx]
                        else:
                            # Fallback: use flat index
                            landfrac_val = ds2.variables['landfrac'].flatten()[gridcell_idx_history]
                    else:
                        landfrac_val = ds2.variables['landfrac'][gridcell_idx_history]
                    
                    if hasattr(landfrac_val, 'data'):  # MaskedArray
                        landfrac_val = landfrac_val.data
                    # Ensure scalar value
                    if isinstance(landfrac_val, np.ndarray) and landfrac_val.size > 1:
                        landfrac_val = landfrac_val.item() if landfrac_val.size == 1 else landfrac_val[0]
                    data_dict[var_name].append(float(landfrac_val))
                elif var_name == 'PCT_CLAY':
                    # Store PCT_CLAY as a list (all levels)
                    # PCT_CLAY is from ds1, check ds1's grid format and variable dimensions
                    var_obj = ds1.variables['PCT_CLAY']
                    var_dims = var_obj.dimensions
                    grid_format_ds1 = grid_info['format_ds1']
                    dim1_ds1 = grid_info['dim1_ds1']
                    dim2_ds1 = grid_info['dim2_ds1']
                    
                    try:
                        if grid_format_ds1 == '2d' and len(var_dims) == 3 and var_dims[1] == dim1_ds1 and var_dims[2] == dim2_ds1:
                            # Variable is (level, lat, lon) in ds1 - use 2D indexing
                            lat_idx = grid_info['lat_idx_map_ds1'].get(restart_idx, None)
                            lon_idx = grid_info['lon_idx_map_ds1'].get(restart_idx, None)
                            if lat_idx is not None and lon_idx is not None:
                                pct_clay_data = var_obj[:, lat_idx, lon_idx]
                            else:
                                # Fallback: use 1D indexing if mapping not available
                                pct_clay_data = var_obj[:, restart_idx] if restart_idx < var_obj.shape[-1] else var_obj[:, 0]
                        else:
                            # Variable is (level, gridcell) or 1D format - use 1D indexing with restart_idx
                            if len(var_dims) >= 2 and restart_idx < var_obj.shape[-1]:
                                pct_clay_data = var_obj[:, restart_idx]
                            else:
                                print(f"  Warning: restart_idx {restart_idx} out of bounds for PCT_CLAY (shape: {var_obj.shape})")
                                pct_clay_data = np.zeros(var_obj.shape[0])
                        
                        # Convert to numpy array
                        pct_clay_data = np.asarray(pct_clay_data)
                        # Convert MaskedArray to regular array and ensure float64
                        if hasattr(pct_clay_data, 'data'):  # MaskedArray
                            pct_clay_data = pct_clay_data.data
                        data_dict[var_name].append(pct_clay_data.astype(np.float64).tolist())
                    except (IndexError, ValueError, TypeError) as e:
                        print(f"  Error processing PCT_CLAY: {e}")
                        data_dict[var_name].append([])
                elif var_name == 'PCT_SAND':
                    # Store PCT_SAND as a list (all levels)
                    # PCT_SAND is from ds1, check ds1's grid format and variable dimensions
                    var_obj = ds1.variables['PCT_SAND']
                    var_dims = var_obj.dimensions
                    grid_format_ds1 = grid_info['format_ds1']
                    dim1_ds1 = grid_info['dim1_ds1']
                    dim2_ds1 = grid_info['dim2_ds1']
                    
                    try:
                        if grid_format_ds1 == '2d' and len(var_dims) == 3 and var_dims[1] == dim1_ds1 and var_dims[2] == dim2_ds1:
                            # Variable is (level, lat, lon) in ds1 - use 2D indexing
                            lat_idx = grid_info['lat_idx_map_ds1'].get(restart_idx, None)
                            lon_idx = grid_info['lon_idx_map_ds1'].get(restart_idx, None)
                            if lat_idx is not None and lon_idx is not None:
                                pct_sand_data = var_obj[:, lat_idx, lon_idx]
                            else:
                                # Fallback: use 1D indexing if mapping not available
                                pct_sand_data = var_obj[:, restart_idx] if restart_idx < var_obj.shape[-1] else var_obj[:, 0]
                        else:
                            # Variable is (level, gridcell) or 1D format - use 1D indexing with restart_idx
                            if len(var_dims) >= 2 and restart_idx < var_obj.shape[-1]:
                                pct_sand_data = var_obj[:, restart_idx]
                            else:
                                print(f"  Warning: restart_idx {restart_idx} out of bounds for PCT_SAND (shape: {var_obj.shape})")
                                pct_sand_data = np.zeros(var_obj.shape[0])
                        
                        # Convert to numpy array
                        pct_sand_data = np.asarray(pct_sand_data)
                        # Convert MaskedArray to regular array and ensure float64
                        if hasattr(pct_sand_data, 'data'):  # MaskedArray
                            pct_sand_data = pct_sand_data.data
                        data_dict[var_name].append(pct_sand_data.astype(np.float64).tolist())
                    except (IndexError, ValueError, TypeError) as e:
                        print(f"  Error processing PCT_SAND: {e}")
                        data_dict[var_name].append([])
                elif var_name.startswith('PCT_NAT_PFT_') or var_name.startswith('PCT_CLAY_') or var_name.startswith('PCT_SAND_'):
                    # Handle 2D variables with level indices
                    if '_' in var_name:
                        level_idx = int(var_name.split('_')[-1])
                        base_var = '_'.join(var_name.split('_')[:-1])  # e.g., 'PCT_CLAY'
                        if base_var in ds1.variables:
                            var_obj = ds1.variables[base_var]
                            var_dims = var_obj.dimensions
                            var_shape = var_obj.shape
                            grid_format_ds1 = grid_info['format_ds1']
                            dim1_ds1 = grid_info['dim1_ds1']
                            dim2_ds1 = grid_info['dim2_ds1']
                            
                            try:
                                if grid_format_ds1 == '2d' and len(var_dims) == 3 and var_dims[1] == dim1_ds1 and var_dims[2] == dim2_ds1:
                                    # Variable is (level, lat, lon) in ds1 - use 2D indexing
                                    lat_idx = grid_info['lat_idx_map_ds1'].get(restart_idx, None)
                                    lon_idx = grid_info['lon_idx_map_ds1'].get(restart_idx, None)
                                    if lat_idx is not None and lon_idx is not None and level_idx < var_shape[0]:
                                        pct_val_raw = var_obj[level_idx, lat_idx, lon_idx]
                                        pct_val = np.asarray(pct_val_raw).item()
                                    else:
                                        # Fallback: use 1D indexing if mapping not available
                                        if level_idx < var_shape[0] and restart_idx < var_shape[1]:
                                            pct_val_raw = var_obj[level_idx, restart_idx]
                                            pct_val = np.asarray(pct_val_raw).item()
                                        else:
                                            pct_val = 0.0
                                else:
                                    # Variable is (level, gridcell) or 1D format - use 1D indexing with restart_idx
                                    if level_idx < var_shape[0] and restart_idx < var_shape[1]:
                                        pct_val_raw = var_obj[level_idx, restart_idx]
                                        pct_val = np.asarray(pct_val_raw).item()
                                    else:
                                        print(f"  Warning: Index out of bounds for {var_name} (level_idx={level_idx}, restart_idx={restart_idx}, shape={var_shape})")
                                        pct_val = 0.0
                                
                                # Final check: ensure we have a numeric value
                                if not isinstance(pct_val, (int, float, np.integer, np.floating)):
                                    pct_val = float(pct_val)
                                
                                data_dict[var_name].append(float(pct_val))
                            except (IndexError, ValueError, TypeError, AttributeError) as e:
                                print(f"  Error processing {var_name}: {e}")
                                data_dict[var_name].append(0.0)
                        else:
                            data_dict[var_name].append(0.0)
                    else:
                        data_dict[var_name].append(0.0)
                elif var_name in ds1.variables:
                    # ds1 variables are indexed by restart_idx (gridcell index), not gridcell_idx_history
                    var_obj = ds1.variables[var_name]
                    var_dims = var_obj.dimensions
                    var_shape = var_obj.shape
                    
                    try:
                        if len(var_dims) == 1:
                            # Variable is 1D (gridcell) - typical for surface scalar variables
                            if restart_idx < var_shape[0]:
                                # Read value and immediately convert to scalar using np.asarray().item()
                                # This handles memoryview, MaskedArray, and other netCDF types
                                val = np.asarray(var_obj[restart_idx]).item()
                            else:
                                print(f"  Error: restart_idx {restart_idx} out of bounds for {var_name} (shape: {var_shape})")
                                val = 0.0
                        else:
                            # Multi-dimensional variable - surface variables should typically be 1D
                            # This might be incorrectly classified, but try to handle it
                            print(f"  Warning: {var_name} has {len(var_dims)} dimensions but is in surface_vars")
                            # For multi-dim, assume last dimension is gridcell, take first element of other dims
                            if restart_idx < var_shape[-1]:
                                indices = [0] * (len(var_shape) - 1) + [restart_idx]
                                val = np.asarray(var_obj[tuple(indices)]).item()
                            else:
                                val = 0.0
                    except (IndexError, ValueError, TypeError, AttributeError) as e:
                        print(f"  Error indexing {var_name} (dims: {var_dims}, shape: {var_shape}, restart_idx: {restart_idx}): {e}")
                        val = 0.0
                    
                    # Final check: ensure we have a numeric value
                    if not isinstance(val, (int, float, np.integer, np.floating)):
                        try:
                            val = float(val)
                        except (ValueError, TypeError) as e:
                            print(f"  Error: Could not convert {var_name} to float: {val}, type: {type(val)}, error: {e}")
                            val = 0.0
                    
                    # Handle integer variables
                    if var_name in ['SOIL_COLOR', 'SOIL_ORDER']:
                        data_dict[var_name].append(int(val))
                    else:
                        data_dict[var_name].append(float(val))
                else:
                    data_dict[var_name].append(0.0)  # Add default value if variable not found
            
            # Process scalar variables from history file
            for var_name in variable_definitions['scalar_vars']:
                if var_name in ds2.variables:
                    var_obj = ds2.variables[var_name]
                    var_dims = var_obj.dimensions
                    
                    try:
                        # Check if variable has time dimension
                        if len(var_dims) >= 2 and var_dims[0] == 'time':
                            # Variable has (time, ...) dimensions
                            if grid_format == '2d' and len(var_dims) == 3 and var_dims[1] == dim1 and var_dims[2] == dim2:
                                # Variable is (time, lat, lon)
                                global_idx = start_idx + k
                                lat_idx = grid_info['lat_idx_map'].get(global_idx, None)
                                lon_idx = grid_info['lon_idx_map'].get(global_idx, None)
                                if lat_idx is not None and lon_idx is not None:
                                    val_raw = var_obj[0, lat_idx, lon_idx]
                                    val = np.asarray(val_raw).item()
                                else:
                                    val_raw = var_obj[0, :, :].flatten()[gridcell_idx_history]
                                    val = np.asarray(val_raw).item()
                            else:
                                # Variable is (time, gridcell) or similar - use gridcell_idx_history for ds2
                                if grid_format == '1d' and gridcell_idx_history < var_obj.shape[1]:
                                    val_raw = var_obj[0, gridcell_idx_history]
                                    val = np.asarray(val_raw).item()
                                else:
                                    # For 2D format but variable is not 2D, try using gridcell_idx_history
                                    if gridcell_idx_history < var_obj.shape[1]:
                                        val_raw = var_obj[0, gridcell_idx_history]
                                    else:
                                        val_raw = var_obj[0, 0]
                                    val = np.asarray(val_raw).item()
                        else:
                            # Variable doesn't have time dimension
                            if grid_format == '2d' and len(var_dims) == 2 and var_dims[0] == dim1 and var_dims[1] == dim2:
                                global_idx = start_idx + k
                                lat_idx = grid_info['lat_idx_map'].get(global_idx, None)
                                lon_idx = grid_info['lon_idx_map'].get(global_idx, None)
                                if lat_idx is not None and lon_idx is not None:
                                    val_raw = var_obj[lat_idx, lon_idx]
                                    val = np.asarray(val_raw).item()
                                else:
                                    val_raw = var_obj[:].flatten()[gridcell_idx_history]
                                    val = np.asarray(val_raw).item()
                            else:
                                # Variable is 1D or other format - use gridcell_idx_history for ds2
                                if grid_format == '1d' and gridcell_idx_history < var_obj.shape[0]:
                                    val_raw = var_obj[gridcell_idx_history]
                                    val = np.asarray(val_raw).item()
                                else:
                                    # Try to handle gracefully
                                    if gridcell_idx_history < var_obj.shape[0]:
                                        val_raw = var_obj[gridcell_idx_history]
                                    else:
                                        val_raw = var_obj[0]
                                    val = np.asarray(val_raw).item()
                        
                        # Final check: ensure we have a numeric value (val should already be scalar from np.asarray().item())
                        if not isinstance(val, (int, float, np.integer, np.floating)):
                            try:
                                val = float(val)
                            except (ValueError, TypeError) as e:
                                print(f"  Error: Could not convert {var_name} to float: {val}, type: {type(val)}, error: {e}")
                                val = 0.0
                    except (IndexError, ValueError, TypeError, AttributeError) as e:
                        print(f"  Error processing {var_name} (dims: {var_dims}): {e}")
                        val = 0.0
                    
                    # Handle integer variables
                    if var_name in ['SOIL_COLOR', 'SOIL_ORDER']:
                        data_dict[var_name].append(int(val))
                    else:
                        data_dict[var_name].append(float(val))
                    
                    # Add Y_ version for scalar variables (from final_spinup history files)
                    y_vals = []
                    for ds_h0 in ds_h0_list:
                        if var_name in ds_h0.variables:
                            # Y variables are from final_spinup history files (same format as ds2)
                            if grid_format == '2d':
                                global_idx = start_idx + k
                                lat_idx = grid_info['lat_idx_map'].get(global_idx, None)
                                lon_idx = grid_info['lon_idx_map'].get(global_idx, None)
                                if lat_idx is not None and lon_idx is not None:
                                    y_val = ds_h0.variables[var_name][0, lat_idx, lon_idx]
                                else:
                                    y_val = ds_h0.variables[var_name][0, :, :].flatten()[gridcell_idx_history]
                            else:
                                y_val = ds_h0.variables[var_name][0, gridcell_idx_history]
                            # Convert MaskedArray to regular array
                            if hasattr(y_val, 'data'):  # MaskedArray
                                y_val = y_val.data
                            y_vals.append(y_val)
                    if y_vals:
                        avg_y_val = np.mean(y_vals)
                        data_dict[f'Y_{var_name}'].append(float(avg_y_val))
                    else:
                        data_dict[f'Y_{var_name}'].append(0.0)
                else:
                    data_dict[var_name].append(0.0)  # Add default value if variable not found
                    data_dict[f'Y_{var_name}'].append(0.0)
            
            # Process PFT variables
            for var_name in variable_definitions['pft_1d_vars']:
                if var_name in ds10.variables:
                    # X values from restart file
                    x_val = ds10.variables[var_name][pft_indices_for_cell]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                    
                    # Y values from future restart files
                    y_vals = []
                    for ds_r in ds_r_list:
                        if var_name in ds_r.variables:
                            y_val = ds_r.variables[var_name][pft_indices_for_cell]
                            # Convert MaskedArray to regular array
                            if hasattr(y_val, 'data'):  # MaskedArray
                                y_val = y_val.data
                            y_vals.append(y_val)
                    if y_vals:
                        avg_y_val = np.mean(y_vals, axis=0)
                        data_dict[f'Y_{var_name}'].append(avg_y_val.astype(np.float64).tolist())
                    else:
                        data_dict[f'Y_{var_name}'].append([])
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
                    data_dict[f'Y_{var_name}'].append([])
            
            # Process 2D soil variables
            for var_name in variable_definitions['variables_2d_soil']:
                if var_name in ds10.variables:
                    # X values
                    x_val = ds10.variables[var_name][column_indices_for_cell, :]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                    
                    # Y values
                    y_vals = []
                    for ds_r in ds_r_list:
                        if var_name in ds_r.variables:
                            y_val = ds_r.variables[var_name][column_indices_for_cell, :]
                            # Convert MaskedArray to regular array
                            if hasattr(y_val, 'data'):  # MaskedArray
                                y_val = y_val.data
                            y_vals.append(y_val)
                    if y_vals:
                        avg_y_val = np.mean(y_vals, axis=0)
                        data_dict[f'Y_{var_name}'].append(avg_y_val.astype(np.float64).tolist())
                    else:
                        data_dict[f'Y_{var_name}'].append([])
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
                    data_dict[f'Y_{var_name}'].append([])
            
            # Process water variables
            for var_name in variable_definitions['water_vars']:
                if var_name in ds10.variables:
                    # X values
                    x_val = ds10.variables[var_name][column_indices_for_cell, :]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                    
                    # Y values
                    y_vals = []
                    for ds_r in ds_r_list:
                        if var_name in ds_r.variables:
                            y_val = ds_r.variables[var_name][column_indices_for_cell, :]
                            # Convert MaskedArray to regular array
                            if hasattr(y_val, 'data'):  # MaskedArray
                                y_val = y_val.data
                            y_vals.append(y_val)
                    if y_vals:
                        avg_y_val = np.mean(y_vals, axis=0)
                        data_dict[f'Y_{var_name}'].append(avg_y_val.astype(np.float64).tolist())
                    else:
                        data_dict[f'Y_{var_name}'].append([])
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
                    data_dict[f'Y_{var_name}'].append([])
            
            # Process pool variables (only if not already processed as PFT variables)
            for var_name in variable_definitions['pool_vars']:
                # Skip if already processed as PFT variable
                if var_name in variable_definitions['pft_1d_vars']:
                    continue
                    
                if var_name in ds10.variables:
                    # X values
                    x_val = ds10.variables[var_name][column_indices_for_cell]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                    
                    # Y values
                    y_vals = []
                    for ds_r in ds_r_list:
                        if var_name in ds_r.variables:
                            y_val = ds_r.variables[var_name][column_indices_for_cell]
                            # Convert MaskedArray to regular array
                            if hasattr(y_val, 'data'):  # MaskedArray
                                y_val = y_val.data
                            y_vals.append(y_val)
                    if y_vals:
                        avg_y_val = np.mean(y_vals, axis=0)
                        data_dict[f'Y_{var_name}'].append(avg_y_val.astype(np.float64).tolist())
                    else:
                        data_dict[f'Y_{var_name}'].append([])
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
                    data_dict[f'Y_{var_name}'].append([])
        
        # Create DataFrame and save
        print(f"  Creating DataFrame...")
        
        # Debug: Check array lengths
        print(f"  Debug: Checking array lengths...")
        for key, values in data_dict.items():
            print(f"    {key}: {len(values)} items")
        
        # Check for length mismatches
        lengths = {key: len(values) for key, values in data_dict.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            print(f"Length mismatch detected!")
            for length in unique_lengths:
                vars_with_length = [k for k, v in lengths.items() if v == length]
                print(f"    Length {length}: {len(vars_with_length)} variables")
                if length != max(unique_lengths):
                    print(f"      Variables: {vars_with_length[:5]}{'...' if len(vars_with_length) > 5 else ''}")
        
        df_batch = pd.DataFrame(data_dict)
        
        print(f"  Saving to disk...")
        batch_save_path = f"{output_dir}/training_data_batch_{batch_number:02d}.pkl"
        df_batch.to_pickle(batch_save_path)
        batch_files.append(batch_save_path)  # Add to list for post-processing
        
        batch_time = time.time() - batch_start_time
        print(f"    Batch {batch_number} completed: {batch_time:.2f}s")
        print(f"    Path: {batch_save_path}")
        print(f"    Shape: {df_batch.shape}")
        print(f"    Columns: {len(df_batch.columns)}")
        
        batch_number += 1
    
    # Cleanup NetCDF files
    print(f"\nCleaning up NetCDF files...")
    ds1.close()
    ds2.close()
    ds10.close()
    for ds_h0 in ds_h0_list:
        ds_h0.close()
    for ds_r in ds_r_list:
        ds_r.close()
    
    print("All NetCDF files closed")
    print(f"Base dataset generation completed!")
    print(f"Total batches: {batch_number - 1}")
    print(f"Output directory: {output_dir}")
    
    # Post-processing like original script
    print(f"\n{'='*80}")
    if use_monthly_forcing:
        print("POST-PROCESSING: Expanding variables (skipping monthly average conversion - data already monthly)")
    else:
        print("POST-PROCESSING: Converting to monthly averages and expanding variables")
    print(f"{'='*80}")
    
    # Process all generated files
    for file_path in batch_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Load the file
        df = pd.read_pickle(file_path)
        
        # Process time series columns (convert to monthly averages only if not using pre-computed monthly data)
        if not use_monthly_forcing:
            print(f"  Processing time series columns (converting to monthly averages)...")
            for col in variable_definitions['time_series_vars']:
                if col in df.columns:
                    print(f"    Processing {col}...")
                    df[col] = df[col].apply(calculate_monthly_avg)
        else:
            print(f"  Skipping monthly average conversion (data already monthly)")
        
        # Process list columns (expand PCT variables)
        print(f"  Processing list columns...")
        list_like_columns = ['PCT_CLAY', 'PCT_SAND', 'PCT_NAT_PFT']
        for col in list_like_columns:
            if col in df.columns:
                print(f"    Expanding {col}...")
                expanded_cols = df[col].apply(pd.Series).fillna(0)
                # Convert to float64 to match reference file data types
                expanded_cols = expanded_cols.astype(np.float64)
                expanded_cols = expanded_cols.add_prefix(f"{col}_")
                df = df.drop(col, axis=1).join(expanded_cols)
        
        # Save processed file
        df.to_pickle(file_path)
        print(f"  Post-processing completed for {os.path.basename(file_path)}")
    
    return output_dir

def generate_base_dataset_initial_only(variable_definitions, use_monthly_forcing=False, forcing_year_range="1980-1999"):
    """Generate base training dataset for initial-only mode (excludes Y_ variables from final_spinup files)"""
    print(f"\n{'='*80}")
    print("STEP 1: Base Dataset Generation (Initial-Only Mode)")
    print(f"{'='*80}")
    
    if use_monthly_forcing:
        print("📊 Using pre-computed monthly average forcing data")
        print(f"   Year range: {forcing_year_range}")
    
    # File paths from config
    surface_data_files = config.surface_data_files
    ad_spinup_history_files = config.ad_spinup_history_files
    ad_spinup_restart_files = config.ad_spinup_restart_files
    # NOTE: We do NOT load final_spinup files for initial-only mode
    
    # Forcing data files - dynamically find files containing variable names
    forcing_files = {}
    for var_name in variable_definitions['time_series_vars']:
        # Look for files containing the variable name in forcing_netcdf directory
        # Support both old format (*VAR*1980-1999.nc) and new format (*VAR*2004-2023.nc)
        pattern = os.path.join(config.forcing_netcdf_output_dir, f'*{var_name}*{forcing_year_range}.nc')
        matching_files = glob.glob(pattern)
        if not matching_files:
            # Try alternative pattern without year range in filename
            pattern_alt = os.path.join(config.forcing_netcdf_output_dir, f'{var_name}_*.nc')
            matching_files = glob.glob(pattern_alt)
        if matching_files:
            forcing_files[var_name] = matching_files[0]  # Use first match
            print(f"Found forcing file: {os.path.basename(matching_files[0])}")
        else:
            print(f"Forcing file not found for {var_name}: {pattern}")
    
    print(f"Found {len(forcing_files)} forcing files")
    
    # Output directory
    output_dir = os.path.join(config.output_dir, 'initial_condition_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading NetCDF files...")
    start_time = time.time()
    
    # Load all required files (excluding final_spinup files)
    ds1 = nc.Dataset(surface_data_files[0])  # Surface data
    ds2 = nc.Dataset(ad_spinup_history_files[0])  # AD-SPINUP history
    ds10 = nc.Dataset(ad_spinup_restart_files[0])  # AD-SPINUP restart
    
    # Load forcing data
    ds_forcing = {}
    for var_name, file_path in forcing_files.items():
        ds_forcing[var_name] = nc.Dataset(file_path)
        print(f"Forcing data loaded: {var_name}")
    
    print(f"All files loaded in {time.time() - start_time:.2f} seconds")
    
    # Get coordinates and build spatial filtering
    lats = ds2.variables['lat'][:]
    lons = ds2.variables['lon'][:]
    landmask = ds2.variables['landfrac'][:]
    
    # Filter for land gridcells
    valid_mask = (landmask > 0)
    valid_gridcells = np.where(valid_mask)[0]
    
    print(f"Total land gridcells: {len(valid_gridcells)}")
    print(f"Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")
    
    # Build KDTree indices
    print("Building KDTree indices...")
    
    # Restart file coordinates
    gridcell_lat = ds10.variables['grid1d_lat'][:]
    gridcell_lon = ds10.variables['grid1d_lon'][:]
    restart_grid_coords = np.vstack((gridcell_lat, gridcell_lon)).T
    restart_tree = cKDTree(restart_grid_coords)
    
    # Forcing file coordinates
    forcing_lats = list(ds_forcing.values())[0].variables['LATIXY'][0, :]
    forcing_lons = list(ds_forcing.values())[0].variables['LONGXY'][0, :]
    forcing_grid_coords = np.vstack((forcing_lats, forcing_lons)).T
    forcing_tree = cKDTree(forcing_grid_coords)
    
    # Query coordinates for valid gridcells
    query_coords = np.array([(lats[i], lons[i]) for i in valid_gridcells])
    _, all_restart_indices = restart_tree.query(query_coords, k=1)
    _, all_forcing_indices = forcing_tree.query(query_coords, k=1)
    
    print("KDTree indices built")
    
    # Pre-load forcing data into memory for optimization
    print("Pre-loading forcing data into memory...")
    forcing_data = {}
    for var_name, ds in ds_forcing.items():
        forcing_data[var_name] = ds.variables[var_name][:, 0, :]  # (time, 1, grid_cells)
        print(f"  Loaded {var_name}: {forcing_data[var_name].shape}")
    
    # Close forcing NetCDF files (data is now in memory)
    for ds in ds_forcing.values():
        ds.close()
    
    # Build index mappings
    pft_gridcell_index = ds10.variables['pfts1d_gridcell_index'][:]
    column_gridcell_index = ds10.variables['cols1d_gridcell_index'][:]
    
    pft_map = {}
    column_map = {}
    unique_gridcell_ids = np.unique(pft_gridcell_index)
    for grid_id in unique_gridcell_ids:
        pft_map[grid_id] = np.where(pft_gridcell_index == grid_id)[0]
        column_map[grid_id] = np.where(column_gridcell_index == grid_id)[0]
    
    print("Index mappings built")
    
    # Process data in batches
    batch_size = 1000
    batch_number = 1
    batch_files = []  # Track all generated files for post-processing
    
    print(f"\nProcessing {len(valid_gridcells)} gridcells in batches of {batch_size}...")
    
    for start_idx in range(0, len(valid_gridcells), batch_size):
        end_idx = min(start_idx + batch_size, len(valid_gridcells))
        batch_gridcells = valid_gridcells[start_idx:end_idx]
        batch_restart_indices = all_restart_indices[start_idx:end_idx]
        batch_forcing_indices = all_forcing_indices[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_number}: gridcells {start_idx+1}-{end_idx}")
        print(f"  Batch size: {len(batch_gridcells)} gridcells")
        print(f"  batch_gridcells range: {batch_gridcells[0]} to {batch_gridcells[-1]}")
        print(f"  Unique gridcells in batch: {len(set(batch_gridcells))}")
        if len(set(batch_gridcells)) != len(batch_gridcells):
            print(f"  ⚠️ WARNING: Duplicate gridcells detected in batch!")
        batch_start_time = time.time()
        
        # Initialize data dictionary dynamically - completely from CNP_IO file
        data_dict = {}
        
        # Add time series variables (forcing data)
        for var_name in variable_definitions['time_series_vars']:
            data_dict[var_name] = []
        
        # Add surface properties
        for var_name in variable_definitions['surface_vars']:
            data_dict[var_name] = []
        
        # Add scalar variables (X only, no Y_ variables for initial-only mode)
        for var_name in variable_definitions['scalar_vars']:
            data_dict[var_name] = []
        
        # Add PFT variables (X only, no Y_ variables)
        for var_name in variable_definitions['pft_1d_vars']:
            data_dict[var_name] = []
            # NOTE: We do NOT add Y_ variables for initial-only mode
        
        # Add 2D soil variables (X only, no Y_ variables)
        for var_name in variable_definitions['variables_2d_soil']:
            data_dict[var_name] = []
            # NOTE: We do NOT add Y_ variables for initial-only mode
        
        # Add water variables (X only, no Y_ variables)
        for var_name in variable_definitions['water_vars']:
            data_dict[var_name] = []
            # NOTE: We do NOT add Y_ variables for initial-only mode
        
        # Add pool variables (X only, no Y_ variables)
        for var_name in variable_definitions['pool_vars']:
            data_dict[var_name] = []
            # NOTE: We do NOT add Y_ variables for initial-only mode
        
        # Process each gridcell in the batch
        for k in range(len(batch_gridcells)):
            if k % 100 == 0:
                print(f"  Processing gridcell {k}/{len(batch_gridcells)}")
            
            # Get indices
            restart_idx = int(batch_restart_indices[k])  # Index for ds1 (surface) and ds10 (restart) files
            forcing_idx = int(batch_forcing_indices[k])  # Index for forcing files
            gridcell_idx_history = batch_gridcells[k]  # Index for ds2 (history) file - may be 2D flattened
            gridcell_id = restart_idx + 1
            
            pft_indices_for_cell = pft_map.get(gridcell_id, [])
            column_indices_for_cell = column_map.get(gridcell_id, [])
            
            # Debug: Check data_dict length after each gridcell
            if k == 0:
                print(f"    After first gridcell: Latitude={len(data_dict['Latitude'])}, FLDS={len(data_dict.get('FLDS', []))}")
                print(f"    pft_indices_for_cell length: {len(pft_indices_for_cell)}")
                print(f"    column_indices_for_cell length: {len(column_indices_for_cell)}")
            if k == 999:
                print(f"    After 1000th gridcell: Latitude={len(data_dict['Latitude'])}, FLDS={len(data_dict.get('FLDS', []))}")
            
            # Process forcing data (time series variables)
            for var_name in variable_definitions['time_series_vars']:
                if var_name in forcing_data:
                    time_series = forcing_data[var_name][:, forcing_idx]
                    # If using monthly forcing, data is already monthly average, store as-is
                    # Otherwise, store raw time series (will be processed later in post-processing)
                    if use_monthly_forcing:
                        # Data is already monthly average, convert to list of floats
                        data_dict[var_name].append(time_series.tolist() if isinstance(time_series, np.ndarray) else list(time_series))
                    else:
                        # Store raw time series (will be processed later)
                        data_dict[var_name].append(time_series)
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
            
            # Process surface properties
            for var_name in variable_definitions['surface_vars']:
                if var_name == 'Latitude':
                    # Use global index (start_idx + k) since lats array is already filtered for valid gridcells
                    global_idx = start_idx + k
                    data_dict[var_name].append(lats[global_idx])
                elif var_name == 'Longitude':
                    # Use global index (start_idx + k) since lons array is already filtered for valid gridcells
                    global_idx = start_idx + k
                    data_dict[var_name].append(lons[global_idx])
                elif var_name == 'landfrac':
                    # landfrac comes from history file (ds2), not surface file (ds1)
                    # Convert to float64 to match reference file
                    global_idx = start_idx + k
                    if grid_format == '1d':
                        landfrac_val = ds2.variables['landfrac'][gridcell_idx_history]
                    elif grid_format == '2d':
                        lat_idx = grid_info['lat_idx_map'].get(global_idx, None)
                        lon_idx = grid_info['lon_idx_map'].get(global_idx, None)
                        if lat_idx is not None and lon_idx is not None:
                            landfrac_val = ds2.variables['landfrac'][lat_idx, lon_idx]
                        else:
                            # Fallback: use flat index
                            landfrac_val = ds2.variables['landfrac'].flatten()[gridcell_idx_history]
                    else:
                        landfrac_val = ds2.variables['landfrac'][gridcell_idx_history]
                    
                    if hasattr(landfrac_val, 'data'):  # MaskedArray
                        landfrac_val = landfrac_val.data
                    # Ensure scalar value
                    if isinstance(landfrac_val, np.ndarray) and landfrac_val.size > 1:
                        landfrac_val = landfrac_val.item() if landfrac_val.size == 1 else landfrac_val[0]
                    data_dict[var_name].append(float(landfrac_val))
                elif var_name == 'PCT_CLAY':
                    # Store PCT_CLAY as a list (all levels)
                    # PCT_CLAY is from ds1, check ds1's grid format and variable dimensions
                    var_obj = ds1.variables['PCT_CLAY']
                    var_dims = var_obj.dimensions
                    grid_format_ds1 = grid_info['format_ds1']
                    dim1_ds1 = grid_info['dim1_ds1']
                    dim2_ds1 = grid_info['dim2_ds1']
                    
                    try:
                        if grid_format_ds1 == '2d' and len(var_dims) == 3 and var_dims[1] == dim1_ds1 and var_dims[2] == dim2_ds1:
                            # Variable is (level, lat, lon) in ds1 - use 2D indexing
                            lat_idx = grid_info['lat_idx_map_ds1'].get(restart_idx, None)
                            lon_idx = grid_info['lon_idx_map_ds1'].get(restart_idx, None)
                            if lat_idx is not None and lon_idx is not None:
                                pct_clay_data = var_obj[:, lat_idx, lon_idx]
                            else:
                                # Fallback: use 1D indexing if mapping not available
                                pct_clay_data = var_obj[:, restart_idx] if restart_idx < var_obj.shape[-1] else var_obj[:, 0]
                        else:
                            # Variable is (level, gridcell) or 1D format - use 1D indexing with restart_idx
                            if len(var_dims) >= 2 and restart_idx < var_obj.shape[-1]:
                                pct_clay_data = var_obj[:, restart_idx]
                            else:
                                print(f"  Warning: restart_idx {restart_idx} out of bounds for PCT_CLAY (shape: {var_obj.shape})")
                                pct_clay_data = np.zeros(var_obj.shape[0])
                        
                        # Convert to numpy array
                        pct_clay_data = np.asarray(pct_clay_data)
                        # Convert MaskedArray to regular array and ensure float64
                        if hasattr(pct_clay_data, 'data'):  # MaskedArray
                            pct_clay_data = pct_clay_data.data
                        data_dict[var_name].append(pct_clay_data.astype(np.float64).tolist())
                    except (IndexError, ValueError, TypeError) as e:
                        print(f"  Error processing PCT_CLAY: {e}")
                        data_dict[var_name].append([])
                elif var_name == 'PCT_SAND':
                    # Store PCT_SAND as a list (all levels)
                    # PCT_SAND is from ds1, check ds1's grid format and variable dimensions
                    var_obj = ds1.variables['PCT_SAND']
                    var_dims = var_obj.dimensions
                    grid_format_ds1 = grid_info['format_ds1']
                    dim1_ds1 = grid_info['dim1_ds1']
                    dim2_ds1 = grid_info['dim2_ds1']
                    
                    try:
                        if grid_format_ds1 == '2d' and len(var_dims) == 3 and var_dims[1] == dim1_ds1 and var_dims[2] == dim2_ds1:
                            # Variable is (level, lat, lon) in ds1 - use 2D indexing
                            lat_idx = grid_info['lat_idx_map_ds1'].get(restart_idx, None)
                            lon_idx = grid_info['lon_idx_map_ds1'].get(restart_idx, None)
                            if lat_idx is not None and lon_idx is not None:
                                pct_sand_data = var_obj[:, lat_idx, lon_idx]
                            else:
                                # Fallback: use 1D indexing if mapping not available
                                pct_sand_data = var_obj[:, restart_idx] if restart_idx < var_obj.shape[-1] else var_obj[:, 0]
                        else:
                            # Variable is (level, gridcell) or 1D format - use 1D indexing with restart_idx
                            if len(var_dims) >= 2 and restart_idx < var_obj.shape[-1]:
                                pct_sand_data = var_obj[:, restart_idx]
                            else:
                                print(f"  Warning: restart_idx {restart_idx} out of bounds for PCT_SAND (shape: {var_obj.shape})")
                                pct_sand_data = np.zeros(var_obj.shape[0])
                        
                        # Convert to numpy array
                        pct_sand_data = np.asarray(pct_sand_data)
                        # Convert MaskedArray to regular array and ensure float64
                        if hasattr(pct_sand_data, 'data'):  # MaskedArray
                            pct_sand_data = pct_sand_data.data
                        data_dict[var_name].append(pct_sand_data.astype(np.float64).tolist())
                    except (IndexError, ValueError, TypeError) as e:
                        print(f"  Error processing PCT_SAND: {e}")
                        data_dict[var_name].append([])
                elif var_name.startswith('PCT_NAT_PFT_') or var_name.startswith('PCT_CLAY_') or var_name.startswith('PCT_SAND_'):
                    # Handle 2D variables with level indices
                    if '_' in var_name:
                        level_idx = int(var_name.split('_')[-1])
                        base_var = '_'.join(var_name.split('_')[:-1])  # e.g., 'PCT_CLAY'
                        if base_var in ds1.variables:
                            var_obj = ds1.variables[base_var]
                            var_dims = var_obj.dimensions
                            var_shape = var_obj.shape
                            grid_format_ds1 = grid_info['format_ds1']
                            dim1_ds1 = grid_info['dim1_ds1']
                            dim2_ds1 = grid_info['dim2_ds1']
                            
                            try:
                                if grid_format_ds1 == '2d' and len(var_dims) == 3 and var_dims[1] == dim1_ds1 and var_dims[2] == dim2_ds1:
                                    # Variable is (level, lat, lon) in ds1 - use 2D indexing
                                    lat_idx = grid_info['lat_idx_map_ds1'].get(restart_idx, None)
                                    lon_idx = grid_info['lon_idx_map_ds1'].get(restart_idx, None)
                                    if lat_idx is not None and lon_idx is not None and level_idx < var_shape[0]:
                                        pct_val_raw = var_obj[level_idx, lat_idx, lon_idx]
                                        pct_val = np.asarray(pct_val_raw).item()
                                    else:
                                        # Fallback: use 1D indexing if mapping not available
                                        if level_idx < var_shape[0] and restart_idx < var_shape[1]:
                                            pct_val_raw = var_obj[level_idx, restart_idx]
                                            pct_val = np.asarray(pct_val_raw).item()
                                        else:
                                            pct_val = 0.0
                                else:
                                    # Variable is (level, gridcell) or 1D format - use 1D indexing with restart_idx
                                    if level_idx < var_shape[0] and restart_idx < var_shape[1]:
                                        pct_val_raw = var_obj[level_idx, restart_idx]
                                        pct_val = np.asarray(pct_val_raw).item()
                                    else:
                                        print(f"  Warning: Index out of bounds for {var_name} (level_idx={level_idx}, restart_idx={restart_idx}, shape={var_shape})")
                                        pct_val = 0.0
                                
                                # Final check: ensure we have a numeric value
                                if not isinstance(pct_val, (int, float, np.integer, np.floating)):
                                    pct_val = float(pct_val)
                                
                                data_dict[var_name].append(float(pct_val))
                            except (IndexError, ValueError, TypeError, AttributeError) as e:
                                print(f"  Error processing {var_name}: {e}")
                                data_dict[var_name].append(0.0)
                        else:
                            data_dict[var_name].append(0.0)
                    else:
                        data_dict[var_name].append(0.0)
                elif var_name in ds1.variables:
                    # ds1 variables are indexed by restart_idx (gridcell index), not gridcell_idx_history
                    var_obj = ds1.variables[var_name]
                    var_dims = var_obj.dimensions
                    var_shape = var_obj.shape
                    
                    try:
                        if len(var_dims) == 1:
                            # Variable is 1D (gridcell) - typical for surface scalar variables
                            if restart_idx < var_shape[0]:
                                # Read value and immediately convert to scalar using np.asarray().item()
                                # This handles memoryview, MaskedArray, and other netCDF types
                                val = np.asarray(var_obj[restart_idx]).item()
                            else:
                                print(f"  Error: restart_idx {restart_idx} out of bounds for {var_name} (shape: {var_shape})")
                                val = 0.0
                        else:
                            # Multi-dimensional variable - surface variables should typically be 1D
                            # This might be incorrectly classified, but try to handle it
                            print(f"  Warning: {var_name} has {len(var_dims)} dimensions but is in surface_vars")
                            # For multi-dim, assume last dimension is gridcell, take first element of other dims
                            if restart_idx < var_shape[-1]:
                                indices = [0] * (len(var_shape) - 1) + [restart_idx]
                                val = np.asarray(var_obj[tuple(indices)]).item()
                            else:
                                val = 0.0
                    except (IndexError, ValueError, TypeError, AttributeError) as e:
                        print(f"  Error indexing {var_name} (dims: {var_dims}, shape: {var_shape}, restart_idx: {restart_idx}): {e}")
                        val = 0.0
                    
                    # Final check: ensure we have a numeric value
                    if not isinstance(val, (int, float, np.integer, np.floating)):
                        try:
                            val = float(val)
                        except (ValueError, TypeError) as e:
                            print(f"  Error: Could not convert {var_name} to float: {val}, type: {type(val)}, error: {e}")
                            val = 0.0
                    
                    # Handle integer variables
                    if var_name in ['SOIL_COLOR', 'SOIL_ORDER']:
                        data_dict[var_name].append(int(val))
                    else:
                        data_dict[var_name].append(float(val))
                else:
                    data_dict[var_name].append(0.0)  # Add default value if variable not found
            
            # Process scalar variables from history file (X only, no Y_ variables for initial-only mode)
            for var_name in variable_definitions['scalar_vars']:
                if var_name in ds2.variables:
                    # ds2 variables are indexed by gridcell_idx_history
                    val_raw = ds2.variables[var_name][0, gridcell_idx_history]
                    val = np.asarray(val_raw).item()
                    # Handle integer variables
                    if var_name in ['SOIL_COLOR', 'SOIL_ORDER']:
                        data_dict[var_name].append(int(val))
                    else:
                        data_dict[var_name].append(float(val))
                else:
                    data_dict[var_name].append(0.0)  # Add default value if variable not found
            
            # Process PFT variables (X only, no Y_ variables)
            for var_name in variable_definitions['pft_1d_vars']:
                if var_name in ds10.variables:
                    # X values from restart file
                    x_val = ds10.variables[var_name][pft_indices_for_cell]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
            
            # Process 2D soil variables (X only, no Y_ variables)
            for var_name in variable_definitions['variables_2d_soil']:
                if var_name in ds10.variables:
                    # X values
                    x_val = ds10.variables[var_name][column_indices_for_cell, :]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
            
            # Process water variables (X only, no Y_ variables)
            for var_name in variable_definitions['water_vars']:
                if var_name in ds10.variables:
                    # X values
                    x_val = ds10.variables[var_name][column_indices_for_cell, :]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
            
            # Process pool variables (X only, no Y_ variables)
            for var_name in variable_definitions['pool_vars']:
                # Skip if already processed as PFT variable
                if var_name in variable_definitions['pft_1d_vars']:
                    continue
                    
                if var_name in ds10.variables:
                    # X values
                    x_val = ds10.variables[var_name][column_indices_for_cell]
                    # Convert MaskedArray to regular array and ensure float64
                    if hasattr(x_val, 'data'):  # MaskedArray
                        x_val = x_val.data
                    data_dict[var_name].append(x_val.astype(np.float64).tolist())
                else:
                    data_dict[var_name].append([])  # Add empty list if variable not found
        
        # Create DataFrame and save
        print(f"  Creating DataFrame...")
        
        # Debug: Check array lengths
        print(f"  Debug: Checking array lengths...")
        for key, values in data_dict.items():
            print(f"    {key}: {len(values)} items")
        
        # Check for length mismatches
        lengths = {key: len(values) for key, values in data_dict.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            print(f"Length mismatch detected!")
            for length in unique_lengths:
                vars_with_length = [k for k, v in lengths.items() if v == length]
                print(f"    Length {length}: {len(vars_with_length)} variables")
                if length != max(unique_lengths):
                    print(f"      Variables: {vars_with_length[:5]}{'...' if len(vars_with_length) > 5 else ''}")
        
        df_batch = pd.DataFrame(data_dict)
        
        print(f"  Saving to disk...")
        batch_save_path = f"{output_dir}/initial_condition_batch_{batch_number:02d}.pkl"
        df_batch.to_pickle(batch_save_path)
        batch_files.append(batch_save_path)  # Add to list for post-processing
        
        batch_time = time.time() - batch_start_time
        print(f"    Batch {batch_number} completed: {batch_time:.2f}s")
        print(f"    Path: {batch_save_path}")
        print(f"    Shape: {df_batch.shape}")
        print(f"    Columns: {len(df_batch.columns)}")
        
        batch_number += 1
    
    # Cleanup NetCDF files
    print(f"\nCleaning up NetCDF files...")
    ds1.close()
    ds2.close()
    ds10.close()
    
    print("All NetCDF files closed")
    print(f"Base dataset generation completed!")
    print(f"Total batches: {batch_number - 1}")
    print(f"Output directory: {output_dir}")
    
    # Post-processing like original script
    print(f"\n{'='*80}")
    if use_monthly_forcing:
        print("POST-PROCESSING: Expanding variables (skipping monthly average conversion - data already monthly)")
    else:
        print("POST-PROCESSING: Converting to monthly averages and expanding variables")
    print(f"{'='*80}")
    
    # Process all generated files
    for file_path in batch_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Load the file
        df = pd.read_pickle(file_path)
        
        # Process time series columns (convert to monthly averages only if not using pre-computed monthly data)
        if not use_monthly_forcing:
            print(f"  Processing time series columns (converting to monthly averages)...")
            for col in variable_definitions['time_series_vars']:
                if col in df.columns:
                    print(f"    Processing {col}...")
                    df[col] = df[col].apply(calculate_monthly_avg)
        else:
            print(f"  Skipping monthly average conversion (data already monthly)")
        
        # Process list columns (expand PCT variables)
        print(f"  Processing list columns...")
        list_like_columns = ['PCT_CLAY', 'PCT_SAND', 'PCT_NAT_PFT']
        for col in list_like_columns:
            if col in df.columns:
                print(f"    Expanding {col}...")
                expanded_cols = df[col].apply(pd.Series).fillna(0)
                # Convert to float64 to match reference file data types
                expanded_cols = expanded_cols.astype(np.float64)
                expanded_cols = expanded_cols.add_prefix(f"{col}_")
                df = df.drop(col, axis=1).join(expanded_cols)
        
        # Save processed file
        df.to_pickle(file_path)
        print(f"Post-processing completed for {os.path.basename(file_path)}")
    
    return output_dir

def generate_enhanced_dataset(base_output_dir, variable_definitions, initial_only_mode=False):
    """Generate enhanced dataset (37_dataset.py logic)"""
    print(f"\n{'='*80}")
    print("STEP 2: Enhanced Dataset Generation (37_dataset.py)")
    print(f"{'='*80}")
    
    # Enhanced output directory
    if initial_only_mode:
        # For initial-only mode, keep output in initial_condition_dataset directory
        enhanced_output_dir = base_output_dir
    else:
        # For regular enhanced mode, use enhanced_training_dataset directory
        enhanced_output_dir = os.path.join(os.path.dirname(base_output_dir), 'enhanced_training_dataset')
    os.makedirs(enhanced_output_dir, exist_ok=True)
    
    # Get base PKL files - handle different naming patterns
    base_files = sorted(glob.glob(os.path.join(base_output_dir, "training_data_batch_*.pkl")))
    if not base_files:
        # Try initial_condition_batch pattern for initial-only mode
        base_files = sorted(glob.glob(os.path.join(base_output_dir, "initial_condition_batch_*.pkl")))
    print(f"Found {len(base_files)} base PKL files to enhance")
    
    if not base_files:
        print("No base PKL files found")
        return base_output_dir

    if initial_only_mode and (not config.final_spinup_history_files or not config.final_spinup_restart_files):
        print("Initial-only mode detected with no final spinup files; skipping enhanced dataset generation.")
        return base_output_dir
    
    # Load restart files for enhancement
    file_path10 = config.ad_spinup_restart_files[0]
    file_path17 = config.final_spinup_restart_files[0]
    
    # Create multiple restart files for Y variables (using same file for demo)
    ds_r_list = [nc.Dataset(file_path17) for _ in range(5)]  # 5 copies for averaging
    
    try:
        ds_restart = nc.Dataset(file_path10)
        restart_tree, restart_coords = build_restart_kdtree(ds_restart)
        col_index_map = build_column_index_map(ds_restart)
        pft_index_map = build_pft_index_map(ds_restart)
        
        for i, base_file in enumerate(base_files, 1):
            print(f"\nProcessing file {i}/{len(base_files)}: {os.path.basename(base_file)}")
            
            try:
                # Read base dataset
                df = pd.read_pickle(base_file)
                print(f"  Base dataset shape: {df.shape}")
                
                # Add pool variables
                print("  Adding pool variables...")
                for pool_var in variable_definitions['pool_vars']:
                    if pool_var not in df.columns:
                        # Create dummy pool variable (should be loaded from restart files)
                        df[pool_var] = [0.0] * len(df)
                        if not initial_only_mode:
                            df[f'Y_{pool_var}'] = [0.0] * len(df)
                            print(f"    Added {pool_var} and Y_{pool_var}")
                        else:
                            print(f"    Added {pool_var} (skipped Y_{pool_var} for initial-only mode)")
                
                # Add Y_ variables for 1D PFT variables (skip in initial-only mode)
                if not initial_only_mode:
                    print("  Adding Y_ variables for 1D PFT...")
                    for var in variable_definitions['pft_1d_vars']:
                        y_var = f"Y_{var}"
                        if y_var not in df.columns and var in df.columns:
                            # Create dummy Y_ variable
                            df[y_var] = [0.0] * len(df)
                            print(f"    Added {y_var}")
                else:
                    print("  Skipping Y_ variables for 1D PFT (initial-only mode)")
                
                # Add Y_ variables for 2D soil variables (skip in initial-only mode)
                if not initial_only_mode:
                    print("  Adding Y_ variables for 2D soil...")
                    for var in variable_definitions['variables_2d_soil']:
                        y_var = f"Y_{var}"
                        if y_var not in df.columns and var in df.columns:
                            # Create dummy Y_ variable
                            df[y_var] = [0.0] * len(df)
                            print(f"    Added {y_var}")
                else:
                    print("  Skipping Y_ variables for 2D soil (initial-only mode)")
                
                # Add Y_ variables for water variables (skip in initial-only mode)
                if not initial_only_mode:
                    print("  Adding Y_ variables for water...")
                    for var in variable_definitions['water_vars']:
                        y_var = f"Y_{var}"
                        if y_var not in df.columns and var in df.columns:
                            # Create dummy Y_ variable
                            df[y_var] = [0.0] * len(df)
                            print(f"    Added {y_var}")
                else:
                    print("  Skipping Y_ variables for water (initial-only mode)")
                
                # Filter to keep only variables defined in CNP_IO file
                all_cnp_vars = []
                for key, vars_list in variable_definitions.items():
                    if isinstance(vars_list, list):
                        all_cnp_vars.extend(vars_list)
                        # Also add Y_ counterparts
                        all_cnp_vars.extend([f"Y_{v}" for v in vars_list])
                
                cnp_vars_set = set(all_cnp_vars)
                current_vars = set(df.columns)
                
                # Find variables to keep
                vars_to_keep = current_vars & cnp_vars_set
                vars_to_remove = current_vars - cnp_vars_set
                
                print(f"  Variables to keep: {len(vars_to_keep)}")
                print(f"  Variables to remove: {len(vars_to_remove)}")
                
                if vars_to_remove:
                    print(f"  🗑️  Removing {len(vars_to_remove)} variables not in CNP_IO file")
                    df_enhanced = df[list(vars_to_keep)]
                else:
                    df_enhanced = df.copy()
                
                # Save enhanced dataset
                enhanced_file = os.path.join(enhanced_output_dir, f"enhanced_monthly_training_data_batch_{i:02d}.pkl")
                df_enhanced.to_pickle(enhanced_file)
                print(f"Enhanced dataset saved: {os.path.basename(enhanced_file)}")
                print(f"Enhanced shape: {df_enhanced.shape}")
                
            except Exception as e:
                print(f"Failed to process file: {e}")
                continue
        
        return enhanced_output_dir
        
    finally:
        ds_restart.close()
        for ds in ds_r_list:
            try:
                ds.close()
            except Exception:
                pass

def add_pft_variables(enhanced_output_dir, variable_definitions):
    """Add PFT variables from CLM parameters file (1_add_pft_to_dataset.py logic)"""
    print(f"\n{'='*80}")
    print("STEP 3: Adding PFT Variables (1_add_pft_to_dataset.py)")
    print(f"{'='*80}")
    
    # Load CLM parameters file
    print("Reading CLM parameters NetCDF file...")
    print(f"File path: {config.clm_params_nc_path}")
    
    if not os.path.exists(config.clm_params_nc_path):
        print(f"Error: CLM parameters file not found: {config.clm_params_nc_path}")
        return enhanced_output_dir
    
    ds = nc.Dataset(config.clm_params_nc_path)
    
    # Get PFT variables dynamically from CNP_IO file
    pft_params = variable_definitions.get('pft_parameters', [])
    print(f"CNP_IO file defines {len(pft_params)} PFT parameter variables")
    
    # Remove 'pft_' prefix from variable names for NetCDF lookup
    target_vars = []
    for var in pft_params:
        if var.startswith('pft_'):
            target_vars.append(var[4:])  # Remove 'pft_' prefix
        else:
            target_vars.append(var)
    
    print(f"PFT parameters to add: {len(target_vars)}")
    print(f"PFT variables: {target_vars[:10]}...")
    
    broadcast_feature_dict = {}
    for var in target_vars:
        if var in ds.variables:
            raw_vals = ds.variables[var][:17]
            # Skip variables if any value is NaN or masked (missing)
            if np.any(np.isnan(raw_vals)) or np.ma.is_masked(raw_vals):
                print(f"Skipped {var}: contains NaN or masked values")
                continue
            broadcast_feature_dict[var] = list(map(float, raw_vals))
            print(f"Added: {var} (length {len(raw_vals)})")
        else:
            print(f"Skipped {var}: not found in NetCDF")
    
    print(f"\nSuccessfully loaded {len(broadcast_feature_dict)} PFT variables from NetCDF")
    
    # Get enhanced PKL files - handle different naming patterns
    input_files = sorted(glob.glob(os.path.join(enhanced_output_dir, "enhanced_monthly_training_data_batch_*.pkl")))
    if not input_files:
        # Try initial_condition_batch pattern for initial-only mode
        input_files = sorted(glob.glob(os.path.join(enhanced_output_dir, "initial_condition_batch_*.pkl")))
    
    print(f"\n🔍 Found {len(input_files)} enhanced PKL files to process")
    
    if len(input_files) == 0:
        print("❌ No enhanced PKL files found.")
        return enhanced_output_dir
    
    # Process each PKL file
    for i, file_path in enumerate(input_files, 1):
        print(f"\nProcessing file {i}/{len(input_files)}: {os.path.basename(file_path)}")
        
        try:
            # Read PKL file
            df = pd.read_pickle(file_path)
            original_shape = df.shape
            print(f"  Original shape: {original_shape}")
            
            # Check if PFT variables already exist
            existing_pft_cols = [col for col in df.columns if col.startswith("pft_")]
            if existing_pft_cols:
                print(f"File already contains {len(existing_pft_cols)} PFT variables, skipping addition")
                continue
            
            # Add each variable as a vector column with pft_ prefix
            print("  Adding PFT variables...")
            for var, val_list in broadcast_feature_dict.items():
                df["pft_" + var] = [val_list] * len(df)  # Add the same list to each row
            
            new_shape = df.shape
            print(f"Successfully added {len(broadcast_feature_dict)} PFT variables")
            print(f"New data shape: {original_shape} → {new_shape}")
            
            # Save in-place (overwrite original file)
            df.to_pickle(file_path)
            print(f"File saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Failed to process file: {e}")
            continue
    
    ds.close()
    
    print(f"\nPFT variables addition completed!")
    print(f"   - Total files: {len(input_files)}")
    print(f"   - PFT variables added: {len(broadcast_feature_dict)}")
    
    return enhanced_output_dir

def final_variable_cleanup(enhanced_output_dir, variable_definitions):
    """Final variable cleanup to ensure only CNP_IO variables remain (2_rm_variables.py logic)"""
    print(f"\n{'='*80}")
    print("STEP 4: Final Variable Cleanup (2_rm_variables.py)")
    print("Keeping only variables defined in CNP_IO file")
    print(f"{'='*80}")
    
    # Get all expected variables from CNP_IO file
    all_expected_vars = set()
    
    # Add all variables from CNP_IO file
    for key, vars_list in variable_definitions.items():
        if isinstance(vars_list, list):
            all_expected_vars.update(vars_list)
            # Also add Y_ counterparts
            all_expected_vars.update([f"Y_{v}" for v in vars_list])
    
    # Add PFT variables with pft_ prefix
    pft_vars = variable_definitions.get('pft_1d_vars', [])
    for var in pft_vars:
        if not var.startswith('pft_'):
            all_expected_vars.add(f"pft_{var}")
        else:
            all_expected_vars.add(var)
    
    # Add PFT parameter variables from CNP_IO file
    pft_params = variable_definitions.get('pft_parameters', [])
    for var in pft_params:
        all_expected_vars.add(var)  # These already have pft_ prefix
    
    print(f"CNP_IO file defines {len(all_expected_vars)} variables")
    
    # Get enhanced PKL files - handle different naming patterns
    input_files = sorted(glob.glob(os.path.join(enhanced_output_dir, "enhanced_monthly_training_data_batch_*.pkl")))
    if not input_files:
        # Try initial_condition_batch pattern for initial-only mode
        input_files = sorted(glob.glob(os.path.join(enhanced_output_dir, "initial_condition_batch_*.pkl")))
    
    print(f"\n🔍 Found {len(input_files)} enhanced PKL files to process")
    
    if len(input_files) == 0:
        print("No enhanced PKL files found.")
        return enhanced_output_dir
    
    # Process each PKL file
    for i, file_path in enumerate(input_files, 1):
        print(f"\nProcessing file {i}/{len(input_files)}: {os.path.basename(file_path)}")
        
        try:
            # Read PKL file
            df = pd.read_pickle(file_path)
            original_shape = df.shape
            print(f"  Original shape: {original_shape}")
            
            # Expand list-like columns (PCT_CLAY, PCT_SAND)
            list_like_columns = ['PCT_CLAY', 'PCT_SAND']
            for col in list_like_columns:
                if col in df.columns:
                    print(f"    Expanding {col}...")
                    expanded_cols = df[col].apply(pd.Series).fillna(0)
                    expanded_cols = expanded_cols.add_prefix(f"{col}_")
                    df = df.drop(col, axis=1).join(expanded_cols)
                    print(f"      Expanded {col} into {len(expanded_cols.columns)} columns")
            
            current_vars = set(df.columns)
            vars_to_keep = current_vars & all_expected_vars
            vars_to_remove = current_vars - all_expected_vars
            
            print(f"  Variables in file: {len(current_vars)}")
            print(f"  Variables to keep: {len(vars_to_keep)}")
            print(f"  Variables to remove: {len(vars_to_remove)}")
            
            if vars_to_remove:
                print(f"  🗑️  Removing {len(vars_to_remove)} variables not in CNP_IO file")
                if len(vars_to_remove) <= 10:
                    print(f"     Variables to remove: {sorted(list(vars_to_remove))}")
                else:
                    print(f"     Sample variables to remove: {sorted(list(vars_to_remove))[:10]}...")
                
                # Keep only expected variables
                df_cleaned = df[list(vars_to_keep)]
                
                new_shape = df_cleaned.shape
                print(f"Successfully removed {len(vars_to_remove)} variables")
                print(f"New data shape: {original_shape} → {new_shape}")
                
                # Save in-place (overwrite original file)
                df_cleaned.to_pickle(file_path)
                print(f"File saved: {os.path.basename(file_path)}")
            else:
                print("No extra variables found to remove. File already matches CNP_IO file.")
                if len(current_vars) != len(all_expected_vars):
                    print(f"  Warning: Column count mismatch. Current: {len(current_vars)}, Expected: {len(all_expected_vars)}")
                    print(f"  Missing from current: {list(all_expected_vars - current_vars)[:5]}...")
                    print(f"  Extra in current: {list(current_vars - all_expected_vars)[:5]}...")
            
        except Exception as e:
            print(f"Failed to process file: {e}")
            continue
    
    print(f"\nVariable cleanup completed!")
    print(f"   - Total files: {len(input_files)}")
    print(f"   - Kept only variables defined in CNP_IO file")
    print(f"   - Target variable count: {len(all_expected_vars)}")
    
    return enhanced_output_dir

def generate_forcing_only_dataset():
    """Generate forcing-only dataset (raw time series, no monthly averaging)"""
    print(f"\n{'='*80}")
    print("FORCING-ONLY DATASET GENERATION")
    print(f"{'='*80}")
    
    # File paths from config
    surface_data_files = config.surface_data_files
    ad_spinup_history_files = config.ad_spinup_history_files
    
    # Forcing data files - dynamically find files containing variable names
    forcing_data_files = {}
    forcing_variables = ['FLDS', 'FSDS', 'PSRF', 'QBOT', 'PRECTmms', 'TBOT']
    
    for var_name in forcing_variables:
        # Look for files containing the variable name in forcing_netcdf directory
        pattern = os.path.join(config.forcing_netcdf_output_dir, f'*{var_name}*1980-1999.nc')
        matching_files = glob.glob(pattern)
        if matching_files:
            forcing_data_files[var_name] = matching_files[0]  # Use first match
            print(f"Found forcing file: {os.path.basename(matching_files[0])}")
        else:
            print(f"Forcing file not found for {var_name}: {pattern}")
    
    print(f"Found {len(forcing_data_files)} forcing files")
    
    # Output directory
    output_dir = os.path.join(config.output_dir, 'forcing_only_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load NetCDF files
    print("Loading NetCDF files...")
    ds1 = nc.Dataset(surface_data_files[0])  # Surface data
    ds2 = nc.Dataset(ad_spinup_history_files[0])  # History file
    
    # Load forcing files
    ds_forcing = {}
    for var_name, file_path in forcing_data_files.items():
        ds_forcing[var_name] = nc.Dataset(file_path)
        print(f"  Loaded {var_name}: {file_path}")
    
    # Get coordinates and land mask
    lats = ds2.variables['lat'][:]
    lons = ds2.variables['lon'][:]
    landmask = ds2.variables['landfrac'][:]
    
    # Filter land gridcells
    valid_mask = (landmask > 0)
    valid_gridcells = np.where(valid_mask)[0]
    
    print(f"Total land gridcells: {len(valid_gridcells)}")
    print(f"Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")
    
    # Build KDTree for forcing data mapping
    print("Building KDTree for forcing data mapping...")
    query_coords = np.array([(lats[i], lons[i]) for i in valid_gridcells])
    
    # Use first forcing file for coordinate mapping
    first_forcing_var = list(forcing_data_files.keys())[0]
    first_forcing_ds = ds_forcing[first_forcing_var]
    forcing_lats = first_forcing_ds.variables['LATIXY'][:].flatten()
    forcing_lons = first_forcing_ds.variables['LONGXY'][:].flatten()
    forcing_coords = np.vstack((forcing_lats, forcing_lons)).T
    
    forcing_tree = cKDTree(forcing_coords)
    _, all_forcing_indices = forcing_tree.query(query_coords, k=1)
    
    print("KDTree indices built")
    
    # Pre-load forcing data into memory for optimization
    print("Pre-loading forcing data into memory...")
    forcing_data = {}
    for var_name, ds in ds_forcing.items():
        forcing_data[var_name] = ds.variables[var_name][:, 0, :]  # (time, 1, grid_cells)
        print(f"  Loaded {var_name}: {forcing_data[var_name].shape}")
    
    # Close forcing NetCDF files (data is now in memory)
    for ds in ds_forcing.values():
        ds.close()
    
    # Process data in batches
    batch_size = 1000
    batch_number = 1
    batch_files = []  # Track all generated files
    
    print(f"\nProcessing {len(valid_gridcells)} gridcells in batches of {batch_size}...")
    
    for start_idx in range(0, len(valid_gridcells), batch_size):
        end_idx = min(start_idx + batch_size, len(valid_gridcells))
        batch_gridcells = valid_gridcells[start_idx:end_idx]
        batch_forcing_indices = all_forcing_indices[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_number}: gridcells {start_idx+1}-{end_idx}")
        print(f"  Batch size: {len(batch_gridcells)} gridcells")
        batch_start_time = time.time()
        
        # Initialize data dictionary - only forcing data and basic geographic info
        data_dict = {
            'landfrac': [],
            'Latitude': [],
            'Longitude': [],
            'FLDS': [],
            'PSRF': [],
            'FSDS': [],
            'QBOT': [],
            'PRECTmms': [],
            'TBOT': [],
        }
        
        # Process each gridcell
        for k, gridcell_idx in enumerate(batch_gridcells):
            if k % 100 == 0:
                print(f"  Processing gridcell {k}/{len(batch_gridcells)} (idx={gridcell_idx})")
            
            # Get forcing index
            forcing_idx = batch_forcing_indices[k]
            
            # Basic geographic info
            landfrac_val = ds2.variables['landfrac'][gridcell_idx]
            if hasattr(landfrac_val, 'data'):  # MaskedArray
                landfrac_val = landfrac_val.data
            data_dict['landfrac'].append(float(landfrac_val))
            data_dict['Latitude'].append(float(lats[gridcell_idx]))
            data_dict['Longitude'].append(float(lons[gridcell_idx]))
            
            # Forcing data (raw time series, no monthly averaging) - convert to list format
            data_dict['FLDS'].append(forcing_data['FLDS'][:, forcing_idx].tolist())
            data_dict['PSRF'].append(forcing_data['PSRF'][:, forcing_idx].tolist())
            data_dict['FSDS'].append(forcing_data['FSDS'][:, forcing_idx].tolist())
            data_dict['QBOT'].append(forcing_data['QBOT'][:, forcing_idx].tolist())
            data_dict['PRECTmms'].append(forcing_data['PRECTmms'][:, forcing_idx].tolist())
            data_dict['TBOT'].append(forcing_data['TBOT'][:, forcing_idx].tolist())
        
        # Create DataFrame and save
        print(f"  Creating DataFrame...")
        df_batch = pd.DataFrame(data_dict)
        
        print(f"  Saving to disk...")
        batch_save_path = f"{output_dir}/forcing_data_batch_{batch_number:02d}.pkl"
        df_batch.to_pickle(batch_save_path)
        batch_files.append(batch_save_path)  # Add to list for tracking
        
        batch_time = time.time() - batch_start_time
        print(f"    Batch {batch_number} completed: {batch_time:.2f}s")
        print(f"    Path: {batch_save_path}")
        print(f"    Shape: {df_batch.shape}")
        print(f"    Forcing data length: {len(df_batch['FLDS'].iloc[0])}")
        
        batch_number += 1
    
    # Cleanup NetCDF files
    print(f"\nCleaning up NetCDF files...")
    ds1.close()
    ds2.close()
    
    print("All NetCDF files closed")
    print(f"Forcing-only dataset generation completed!")
    print(f"Total batches: {batch_number - 1}")
    print(f"Output directory: {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Training Dataset Generation - Three modes: forcing-only, enhanced dataset, or initial-only"
    )
    parser.add_argument(
        "--forcing_only",
        action="store_true",
        help="Generate only forcing data (raw time series, no monthly averaging)"
    )
    parser.add_argument(
        "--enhanced_dataset",
        action="store_true",
        help="Generate complete enhanced dataset (includes PFT variables and all processing steps)"
    )
    parser.add_argument(
        "--initial_only",
        action="store_true",
        help="Generate initial condition dataset (excludes Y_ variables from final_spinup files)"
    )
    parser.add_argument(
        "--use_monthly_forcing",
        action="store_true",
        help="Use pre-computed monthly average forcing data (skip calculate_monthly_avg processing)"
    )
    parser.add_argument(
        "--forcing_year_range",
        type=str,
        default="1980-1999",
        help="Year range for forcing files (e.g., '1980-1999' or '2004-2023'). Default: 1980-1999"
    )
    args = parser.parse_args()
    
    # Validate arguments
    options = [args.forcing_only, args.enhanced_dataset, args.initial_only]
    if sum(options) > 1:
        print("❌ Error: Cannot specify multiple options. Choose only one of: --forcing_only, --enhanced_dataset, or --initial_only")
        sys.exit(1)
    if sum(options) == 0:
        print("❌ Error: Must specify one of: --forcing_only, --enhanced_dataset, or --initial_only")
        sys.exit(1)
    
    try:
        start_time = time.time()
        
        if args.forcing_only:
            # Generate forcing-only dataset
            print("🚀 Starting FORCING-ONLY dataset generation...")
            final_output_dir = generate_forcing_only_dataset()
            
            total_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("🎉 FORCING-ONLY DATASET GENERATION COMPLETED!")
            print(f"{'='*80}")
            print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Final output directory: {final_output_dir}")
            
            # List final files
            final_files = sorted(glob.glob(os.path.join(final_output_dir, "*.pkl")))
            print(f"Generated {len(final_files)} forcing-only dataset files:")
            for file in final_files:
                file_size = os.path.getsize(file) / (1024**3)  # GB
                print(f"  {os.path.basename(file)}: {file_size:.2f} GB")
            
            print(f"\n✅ Forcing-only dataset ready!")
            print(f"✅ Raw time series data (no monthly averaging)")
            print(f"✅ Only forcing variables and basic geographic info")
            
        elif args.enhanced_dataset:
            # Generate complete enhanced dataset
            print("🚀 Starting COMPLETE ENHANCED DATASET generation...")
            
            # Parse CNP_IO variables
            variable_definitions = parse_cnp_io_variables()
            
            # Step 1: Generate base dataset
            base_output_dir = generate_base_dataset(
                variable_definitions, 
                use_monthly_forcing=args.use_monthly_forcing,
                forcing_year_range=args.forcing_year_range
            )
            
            # Step 2: Generate enhanced dataset
            enhanced_output_dir = generate_enhanced_dataset(base_output_dir, variable_definitions)
            
            # Step 3: Add PFT variables
            pft_output_dir = add_pft_variables(enhanced_output_dir, variable_definitions)
            
            # Step 4: Final variable cleanup
            final_output_dir = final_variable_cleanup(pft_output_dir, variable_definitions)
            
            total_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("🎉 COMPLETE ENHANCED TRAINING DATASET GENERATION COMPLETED!")
            print(f"{'='*80}")
            print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Final output directory: {final_output_dir}")
            
            # List final files
            final_files = sorted(glob.glob(os.path.join(final_output_dir, "*.pkl")))
            print(f"Generated {len(final_files)} enhanced dataset files:")
            for file in final_files:
                file_size = os.path.getsize(file) / (1024**3)  # GB
                print(f"  {os.path.basename(file)}: {file_size:.2f} GB")
            
            print(f"\n✅ Complete enhanced training dataset ready for machine learning!")
            print(f"✅ All variables dynamically extracted from CNP_IO file!")
            print(f"✅ Combines all 4 original scripts into one integrated workflow!")
            
        elif args.initial_only:
            # Generate initial condition dataset (without Y_ variables)
            print("🚀 Starting INITIAL-ONLY dataset generation...")
            
            # Parse CNP_IO variables
            variable_definitions = parse_cnp_io_variables()
            
            # Step 1: Generate base dataset (without Y_ variables)
            base_output_dir = generate_base_dataset_initial_only(
                variable_definitions,
                use_monthly_forcing=args.use_monthly_forcing,
                forcing_year_range=args.forcing_year_range
            )
            
            # Step 2: Generate enhanced dataset (same as regular enhanced dataset)
            enhanced_output_dir = generate_enhanced_dataset(base_output_dir, variable_definitions, initial_only_mode=True)
            
            # Step 3: Add PFT variables (same as regular enhanced dataset)
            pft_output_dir = add_pft_variables(enhanced_output_dir, variable_definitions)
            
            # Step 4: Final variable cleanup (same as regular enhanced dataset)
            final_output_dir = final_variable_cleanup(pft_output_dir, variable_definitions)
            
            total_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("🎉 INITIAL-ONLY TRAINING DATASET GENERATION COMPLETED!")
            print(f"{'='*80}")
            print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Final output directory: {final_output_dir}")
            
            # List final files
            final_files = sorted(glob.glob(os.path.join(final_output_dir, "*.pkl")))
            print(f"Generated {len(final_files)} initial-only dataset files:")
            for file in final_files:
                file_size = os.path.getsize(file) / (1024**3)  # GB
                print(f"  {os.path.basename(file)}: {file_size:.2f} GB")
            
            print(f"\n✅ Initial-only training dataset ready!")
            print(f"✅ Excludes all Y_ variables from final_spinup files!")
            print(f"✅ Contains only initial condition variables!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
