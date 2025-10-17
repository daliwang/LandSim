#!/usr/bin/env python3
"""
Forcing PKL Validation Script
Validates generated PKL files against processed NetCDF files
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
import os
import sys
import glob
import time

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_netcdf_data(netcdf_dir, variable):
    """
    Load processed NetCDF data for a specific variable
    """
    print(f"  Loading {variable} NetCDF data...")
    
    file_path = os.path.join(netcdf_dir, f"TVA_{variable}_1980-1999.nc")
    
    if not os.path.exists(file_path):
        print(f"    ❌ NetCDF file not found: {file_path}")
        return None
    
    with nc.Dataset(file_path, 'r') as ds:
        if variable in ds.variables:
            # NetCDF data is (time, nj, ni), we need (time, ni)
            data = ds.variables[variable][:, 0, :]  # Remove nj dimension
            print(f"    Loaded {variable}: shape {data.shape}")
            return data
        else:
            print(f"    ❌ Variable {variable} not found in NetCDF file")
            return None

def get_netcdf_coordinates(netcdf_dir):
    """
    Get coordinate information from processed NetCDF files
    """
    # Use any NetCDF file to get coordinates
    sample_file = os.path.join(netcdf_dir, "TVA_FLDS_1980-1999.nc")
    if os.path.exists(sample_file):
        with nc.Dataset(sample_file, 'r') as ds:
            # NetCDF files use LATIXY and LONGXY
            lat = ds.variables['LATIXY'][0, :].data  # (11357,)
            lon = ds.variables['LONGXY'][0, :].data  # (11357,)
            return lat, lon
    else:
        print("❌ ERROR: Cannot find sample NetCDF file for coordinates")
        return None, None

def create_coordinate_mapping(pkl_coords, netcdf_coords, tolerance=1e-6):
    """
    Create mapping between PKL coordinates and NetCDF coordinates using coordinate matching
    """
    print("  Creating coordinate-based mapping...")
    
    pkl_lat, pkl_lon = pkl_coords
    netcdf_lat, netcdf_lon = netcdf_coords
    
    # Create coordinate pairs
    pkl_coord_pairs = np.column_stack([pkl_lat, pkl_lon])
    netcdf_coord_pairs = np.column_stack([netcdf_lat, netcdf_lon])
    
    mapping = []
    unmatched_count = 0
    
    for i, pkl_coord in enumerate(pkl_coord_pairs):
        # Find matching NetCDF coordinate
        distances = np.sqrt(np.sum((netcdf_coord_pairs - pkl_coord)**2, axis=1))
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance < tolerance:
            mapping.append(min_idx)
        else:
            mapping.append(-1)  # No match found
            unmatched_count += 1
    
    print(f"    Mapped {len(mapping) - unmatched_count}/{len(mapping)} coordinates")
    print(f"    Unmatched coordinates: {unmatched_count}")
    
    return np.array(mapping)

def validate_pkl_batch_against_netcdf(pkl_file, netcdf_data, variable, batch_index, batch_size=1000):
    """
    Validate a PKL batch against corresponding NetCDF data by gridcell range
    """
    print(f"  Validating {variable} (batch {batch_index + 1})...")
    
    # Load PKL data
    pkl_data = pd.read_pickle(pkl_file)
    
    if variable not in pkl_data.columns:
        print(f"    ❌ Variable {variable} not found in PKL file")
        return False
    
    # Calculate the corresponding NetCDF gridcell range for this batch
    start_idx = batch_index * batch_size
    end_idx = min(start_idx + len(pkl_data), netcdf_data.shape[1])
    
    print(f"    PKL batch size: {len(pkl_data)} gridcells")
    print(f"    NetCDF range: gridcells {start_idx}-{end_idx-1}")
    
    # Sample validation parameters
    sample_gridcells = min(20, len(pkl_data))  # Sample fewer gridcells for accuracy
    sample_timesteps = min(50, len(pkl_data[variable].iloc[0]))  # Sample fewer timesteps
    
    # Random sampling
    np.random.seed(42)  # For reproducible results
    pkl_gridcell_indices = np.random.choice(len(pkl_data), sample_gridcells, replace=False)
    timestep_indices = np.random.choice(len(pkl_data[variable].iloc[0]), sample_timesteps, replace=False)
    
    print(f"    Sampling {sample_gridcells} gridcells and {sample_timesteps} timesteps")
    
    matches = 0
    total_checks = 0
    
    for pkl_gc_idx in pkl_gridcell_indices:
        # Get PKL data for this gridcell
        pkl_gridcell_data = pkl_data[variable].iloc[pkl_gc_idx]
        
        # Map PKL gridcell index to NetCDF gridcell index
        netcdf_gc_idx = start_idx + pkl_gc_idx
        
        if netcdf_gc_idx >= netcdf_data.shape[1]:
            print(f"    ⚠️ NetCDF index {netcdf_gc_idx} out of range")
            continue
        
        for ts_idx in timestep_indices:
            if ts_idx < len(pkl_gridcell_data) and ts_idx < netcdf_data.shape[0]:
                pkl_value = pkl_gridcell_data[ts_idx]
                netcdf_value = netcdf_data[ts_idx, netcdf_gc_idx]
                
                # Check for NaN values
                if np.isnan(pkl_value) and np.isnan(netcdf_value):
                    matches += 1
                elif not np.isnan(pkl_value) and not np.isnan(netcdf_value):
                    # Check if values are close (allowing for small numerical differences)
                    if np.allclose(pkl_value, netcdf_value, rtol=1e-8, atol=1e-8):
                        matches += 1
                
                total_checks += 1
    
    if total_checks > 0:
        match_rate = matches / total_checks
        print(f"    Match rate: {match_rate:.4f} ({matches}/{total_checks})")
        return match_rate > 0.95  # 95% match rate threshold
    else:
        print(f"    ❌ No valid samples checked")
        return False

def main():
    print("="*80)
    print("Forcing PKL Validation Against NetCDF Files")
    print("="*80)
    
    # Configuration
    netcdf_dir = config.forcing_netcdf_output_dir
    pkl_output_dir = config.forcing_pkl_output_dir
    
    # Forcing variables to validate
    forcing_variables = ['FLDS', 'FSDS', 'PSRF', 'QBOT', 'PRECTmms', 'TBOT']
    
    print(f"Configuration:")
    print(f"  NetCDF directory: {netcdf_dir}")
    print(f"  PKL output directory: {pkl_output_dir}")
    print(f"  Variables to validate: {forcing_variables}")
    
    # Check if directories exist
    if not os.path.exists(netcdf_dir):
        print(f"❌ ERROR: NetCDF directory not found: {netcdf_dir}")
        return False
    
    if not os.path.exists(pkl_output_dir):
        print(f"❌ ERROR: PKL output directory not found: {pkl_output_dir}")
        return False
    
    print(f"\n{'='*60}")
    print("Step 1: Getting coordinate information")
    print(f"{'='*60}")
    
    # Get NetCDF coordinate information
    netcdf_lat, netcdf_lon = get_netcdf_coordinates(netcdf_dir)
    if netcdf_lat is None or netcdf_lon is None:
        return False
    
    print(f"  NetCDF grid: {len(netcdf_lat)} points")
    
    # Get PKL coordinate information
    pkl_files = sorted(glob.glob(os.path.join(pkl_output_dir, "TVA_forcing_batch_*.pkl")))
    if not pkl_files:
        print(f"❌ ERROR: No PKL files found in {pkl_output_dir}")
        return False
    
    print(f"  Loading PKL coordinates from first batch...")
    first_pkl = pd.read_pickle(pkl_files[0])
    pkl_lat = first_pkl['Latitude'].values
    pkl_lon = first_pkl['Longitude'].values
    print(f"  PKL grid: {len(pkl_lat)} points")
    
    print(f"\n{'='*60}")
    print("Step 2: Creating coordinate mapping")
    print(f"{'='*60}")
    
    # Create coordinate mapping
    coordinate_mapping = create_coordinate_mapping(
        (pkl_lat, pkl_lon), 
        (netcdf_lat, netcdf_lon)
    )
    
    print(f"\n{'='*60}")
    print("Step 3: Loading NetCDF data")
    print(f"{'='*60}")
    
    # Load NetCDF data for each variable
    netcdf_data = {}
    for var in forcing_variables:
        print(f"\nLoading {var}...")
        data = load_netcdf_data(netcdf_dir, var)
        if data is not None:
            netcdf_data[var] = data
        else:
            print(f"❌ ERROR: Failed to load {var} NetCDF data")
            return False
    
    print(f"\n{'='*60}")
    print("Step 4: Validating PKL files")
    print(f"{'='*60}")
    
    print(f"Found {len(pkl_files)} PKL files to validate")
    
    # Validate first 5 PKL files (as requested)
    num_files_to_validate = min(5, len(pkl_files))
    print(f"Validating first {num_files_to_validate} PKL files")
    
    # Validate each PKL file using sequential gridcell mapping
    all_validations_passed = True
    validation_results = {}
    
    for i in range(num_files_to_validate):
        pkl_file = pkl_files[i]
        print(f"\n{'='*50}")
        print(f"Validating batch {i+1}/{num_files_to_validate}: {os.path.basename(pkl_file)}")
        print(f"{'='*50}")
        
        batch_passed = True
        batch_results = {}
        
        for var in forcing_variables:
            if var in netcdf_data:
                passed = validate_pkl_batch_against_netcdf(
                    pkl_file, 
                    netcdf_data[var], 
                    var, 
                    i,  # batch index
                    1000  # batch size
                )
                batch_results[var] = passed
                if not passed:
                    batch_passed = False
        
        validation_results[os.path.basename(pkl_file)] = {
            'passed': batch_passed,
            'details': batch_results
        }
        
        if not batch_passed:
            all_validations_passed = False
            print(f"❌ Batch {i+1} validation FAILED")
        else:
            print(f"✅ Batch {i+1} validation PASSED")
    
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    passed_count = 0
    for pkl_file, result in validation_results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{pkl_file:<30} - {status}")
        if result['passed']:
            passed_count += 1
        
        # Show detailed results
        for var, var_result in result['details'].items():
            var_status = "✅" if var_result else "❌"
            print(f"  {var:<12} - {var_status}")
    
    print(f"\nResults: {passed_count}/{num_files_to_validate} PKL files passed validation")
    
    if all_validations_passed:
        print("🎉 ALL PKL VALIDATIONS PASSED!")
        print("✅ Generated PKL files are consistent with NetCDF files")
        return True
    else:
        print("⚠️  Some PKL validations failed")
        print("❌ Please check the PKL generation process")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
