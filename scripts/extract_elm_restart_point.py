#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELM Restart File Single Point Data Extraction Tool 
Function: Extract all data for specified coordinates from global ELM restart files (correctly handles multi-level structure)
"""

import xarray as xr
import numpy as np
import os
import argparse
from datetime import datetime

# ==================== Configuration Parameters (Default Values) ====================

SOURCE_NC = '/global/cfs/cdirs/m4814/daweigao/14_Code/all_dataset_1_degree/20250117_trendytest_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc'
OUTPUT_NC = 'single_point_20_year_restart_extracted_43_56.nc'  # Output filename
# Target coordinates
TARGET_LAT = 35.833332
TARGET_LON = -84.208336

# ==================== Command Line Argument Parser ====================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Extract single point data from ELM restart file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_elm_restart_point.py --restart-file /path/to/restart.nc --lat 35.83 --lon -84.21
  python extract_elm_restart_point.py --restart-file /path/to/restart.nc --lat 35.83 --lon -84.21 --output-file output.nc
        """
    )
    
    parser.add_argument(
        '--restart-file',
        type=str,
        default=None,
        help='Path to input ELM restart NetCDF file (default: uses hardcoded SOURCE_NC)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to output NetCDF file (default: uses hardcoded OUTPUT_NC)'
    )
    
    parser.add_argument(
        '--lat',
        type=float,
        default=None,
        help='Target latitude coordinate (default: uses hardcoded TARGET_LAT)'
    )
    
    parser.add_argument(
        '--lon',
        type=float,
        default=None,
        help='Target longitude coordinate (default: uses hardcoded TARGET_LON)'
    )
    
    return parser.parse_args()


# ==================== Main Function ====================
def extract_single_point_elm(source_nc=None, output_nc=None, target_lat=None, target_lon=None):
    """
    Extract single point data from global ELM restart file
    Correctly handles ELM's multi-level structure (gridcell -> topounit -> landunit -> column -> pft)
    
    Args:
        source_nc: Path to source NetCDF file (if None, uses SOURCE_NC)
        output_nc: Path to output NetCDF file (if None, uses OUTPUT_NC)
        target_lat: Target latitude (if None, uses TARGET_LAT)
        target_lon: Target longitude (if None, uses TARGET_LON)
    """
    # Use provided parameters or fall back to defaults
    source_file = source_nc if source_nc is not None else SOURCE_NC
    output_file = output_nc if output_nc is not None else OUTPUT_NC
    lat = target_lat if target_lat is not None else TARGET_LAT
    lon = target_lon if target_lon is not None else TARGET_LON
    
    print("="*70)
    print("ELM Restart File Single Point Data Extraction (Multi-level Version)")
    print("="*70)
    print(f"Source file: {source_file}")
    print(f"Target coordinates: Lat={lat}, Lon={lon}")
    print("="*70)
    
    # Check file existence
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Error: Source file not found {source_file}")
    
    # ============ Step 1: Open dataset ============
    print("\n[1/5] Loading dataset...")
    ds = xr.open_dataset(source_file, decode_times=False, mask_and_scale=False)
    
    print(f"   ✓ Dataset loaded successfully")
    print(f"   - File size: {os.path.getsize(source_file) / (1024**3):.2f} GB")
    print(f"   - Dimensions: gridcell={ds.dims['gridcell']}, topounit={ds.dims['topounit']}, " +
          f"landunit={ds.dims['landunit']}, column={ds.dims['column']}, pft={ds.dims['pft']}")
    print(f"   - Number of variables: {len(ds.data_vars)}")
    
    # ============ Step 2: Find nearest neighbor gridcell ============
    print("\n[2/5] Locating nearest neighbor gridcell coordinates...")
    
    # Read gridcell-level latitude and longitude
    lat_vals = ds['grid1d_lat'].values
    lon_vals = ds['grid1d_lon'].values
    
    # Handle longitude format conversion (0-360° vs -180 to 180°)
    lon_adjusted = lon_vals.copy()
    if lon_vals.max() > 200:
        lon_adjusted = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
        print(f"   Detected 0-360° longitude format, converted to -180 to 180° format")
    
    # Calculate Euclidean distance and find nearest neighbor
    dist = np.sqrt((lat_vals - lat)**2 + (lon_adjusted - lon)**2)
    gridcell_idx_py = int(np.argmin(dist))  # Python 0-based index
    gridcell_idx_elm = gridcell_idx_py + 1  # ELM 1-based index
    
    found_lat = float(lat_vals[gridcell_idx_py])
    found_lon = float(lon_vals[gridcell_idx_py])
    found_lon_adjusted = float(lon_adjusted[gridcell_idx_py])
    
    print(f"   ✓ Found nearest neighbor gridcell")
    print(f"   - Python index (0-based): {gridcell_idx_py}")
    print(f"   - ELM index (1-based): {gridcell_idx_elm}")
    print(f"   - Target coordinates: ({lat:.6f}, {lon:.6f})")
    print(f"   - Matched coordinates: ({found_lat:.6f}, {found_lon_adjusted:.6f})")
    print(f"   - Euclidean distance: {dist[gridcell_idx_py]:.6f}°")
    
    # ============ Step 3: Find indices for all levels ============
    print("\n[3/5] Finding all level data for this gridcell...")
    
    # Find all level indices belonging to this gridcell
    indices = {}
    
    # Gridcell level (use found index directly)
    indices['gridcell'] = [gridcell_idx_py]
    
    # Topounit level
    if 'topo1d_gridcell_index' in ds.variables:
        topo_gc_idx = ds['topo1d_gridcell_index'].values
        topo_match = np.where(topo_gc_idx == gridcell_idx_elm)[0]
        indices['topounit'] = topo_match.tolist()
        print(f"   - Found {len(topo_match)} topounits")
    
    # Landunit level
    if 'land1d_gridcell_index' in ds.variables:
        land_gc_idx = ds['land1d_gridcell_index'].values
        land_match = np.where(land_gc_idx == gridcell_idx_elm)[0]
        indices['landunit'] = land_match.tolist()
        print(f"   - Found {len(land_match)} landunits")
    
    # Column level
    if 'cols1d_gridcell_index' in ds.variables:
        cols_gc_idx = ds['cols1d_gridcell_index'].values
        cols_match = np.where(cols_gc_idx == gridcell_idx_elm)[0]
        indices['column'] = cols_match.tolist()
        print(f"   - Found {len(cols_match)} columns")
    
    # PFT level
    if 'pfts1d_gridcell_index' in ds.variables:
        pfts_gc_idx = ds['pfts1d_gridcell_index'].values
        pfts_match = np.where(pfts_gc_idx == gridcell_idx_elm)[0]
        indices['pft'] = pfts_match.tolist()
        print(f"   - Found {len(pfts_match)} pfts")
    
    print(f"   ✓ All level index search completed")
    
    # ============ Step 4: Extract data ============
    print("\n[4/5] Extracting single point data...")
    
    # Filter data by level
    subset = ds.copy()
    
    # Filter each dimension
    for dim_name, dim_indices in indices.items():
        if dim_name in subset.dims and len(dim_indices) > 0:
            subset = subset.isel({dim_name: dim_indices})
            print(f"   ✓ Filtered {dim_name}: {len(dim_indices)} elements")
    
    # Calculate data compression ratio
    original_size_estimate = sum([
        ds.dims['gridcell'], 
        ds.dims['topounit'], 
        ds.dims['landunit'], 
        ds.dims['column'], 
        ds.dims['pft']
    ])
    subset_size_estimate = sum([
        len(indices.get('gridcell', [])),
        len(indices.get('topounit', [])),
        len(indices.get('landunit', [])),
        len(indices.get('column', [])),
        len(indices.get('pft', []))
    ])
    compression_ratio = original_size_estimate / subset_size_estimate if subset_size_estimate > 0 else 0
    
    print(f"   ✓ Data extraction completed")
    print(f"   - Compression ratio: {compression_ratio:.1f}x (original {original_size_estimate} -> extracted {subset_size_estimate} elements)")
    
    # ============ Step 5: Preserve metadata and save ============
    print("\n[5/5] Saving file...")
    
    # Preserve original encoding information for each variable
    encoding = {}
    for var_name in subset.variables:
        encoding[var_name] = {}
        
        if var_name in ds.variables:
            original_var = ds[var_name]
            
            if hasattr(original_var, 'encoding'):
                for key in ['dtype', 'scale_factor', 'add_offset', '_FillValue', 
                           'missing_value', 'zlib', 'complevel', 'shuffle', 
                           'chunksizes', 'fletcher32', 'contiguous']:
                    if key in original_var.encoding:
                        encoding[var_name][key] = original_var.encoding[key]
    
    # Add extraction information to global attributes
    subset.attrs['extraction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    subset.attrs['extraction_target_lat'] = lat
    subset.attrs['extraction_target_lon'] = lon
    subset.attrs['extraction_actual_lat'] = found_lat
    subset.attrs['extraction_actual_lon'] = found_lon_adjusted
    subset.attrs['extraction_source_file'] = os.path.basename(source_file)
    subset.attrs['extraction_gridcell_index_python'] = gridcell_idx_py
    subset.attrs['extraction_gridcell_index_elm'] = gridcell_idx_elm
    
    print(f"   Writing to: {output_file}")
    
    # Save file
    subset.to_netcdf(
        output_file, 
        format='NETCDF4',
        encoding=encoding,
        unlimited_dims=None
    )
    
    # Verify output
    output_size = os.path.getsize(output_file) / (1024**2)  # MB
    print(f"   ✓ File saved successfully")
    print(f"   - Output size: {output_size:.2f} MB")
    print(f"   - Number of variables: {len(subset.data_vars)}")
    print(f"   - Global attributes: {len(subset.attrs)}")
    
    # Close dataset
    ds.close()
    
    print("\n" + "="*70)
    print("✓ Processing completed!")
    print("="*70)
    print("\nDimension statistics:")
    for dim in subset.dims:
        print(f"  {dim}: {subset.dims[dim]}")
    
    print("\nTip: You can use the following commands to view the output file:")
    print(f"  ncdump -h {output_file}")
    print(f"Or in Python:")
    print(f"  import xarray as xr")
    print(f"  ds = xr.open_dataset('{output_file}')")
    print(f"  print(ds)")
    
    return subset


# ==================== Helper Function: Quick File Structure Inspection ====================
def inspect_structure(source_nc=None):
    """Quickly view the hierarchical structure information of NetCDF file"""
    source_file = source_nc if source_nc is not None else SOURCE_NC
    print("Checking file structure...")
    ds = xr.open_dataset(source_file, decode_times=False)
    
    print("\nDimensions:")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    print("\nIndex mapping variables:")
    for var in ds.variables:
        if 'index' in var.lower():
            print(f"  {var}: {ds[var].dims} - shape={ds[var].shape}")
            if ds[var].size < 50:
                print(f"    Example values: {ds[var].values[:10]}")
    
    ds.close()


# ==================== Main Program Entry ====================
if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Uncomment the line below to view file structure first
        # inspect_structure(args.restart_file)
        
        # Execute extraction with command line arguments (or defaults)
        result = extract_single_point_elm(
            source_nc=args.restart_file,
            output_nc=args.output_file,
            target_lat=args.lat,
            target_lon=args.lon
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

