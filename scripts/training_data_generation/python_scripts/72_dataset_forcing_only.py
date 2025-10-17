#!/usr/bin/env python3
"""
TVA Forcing Data Only - PKL Generation Script (Optimized with List Format)
Extract only forcing data and basic geographical information
Automatically converts forcing variables to list format for training compatibility
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os
import sys
import time

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("="*80)
print("TVA Forcing Data Only - PKL Generation with List Format Conversion")
print("="*80)

# File paths using config
history_file = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_adspinref.elm.h0.0021-01-01-00000.nc'

# Forcing files - use newly generated ones
forcing_files = {
    'FLDS': os.path.join(config.forcing_netcdf_output_dir, 'TVA_FLDS_1980-1999.nc'),
    'FSDS': os.path.join(config.forcing_netcdf_output_dir, 'TVA_FSDS_1980-1999.nc'),
    'PSRF': os.path.join(config.forcing_netcdf_output_dir, 'TVA_PSRF_1980-1999.nc'),
    'QBOT': os.path.join(config.forcing_netcdf_output_dir, 'TVA_QBOT_1980-1999.nc'),
    'PRECTmms': os.path.join(config.forcing_netcdf_output_dir, 'TVA_PRECTmms_1980-1999.nc'),
    'TBOT': os.path.join(config.forcing_netcdf_output_dir, 'TVA_TBOT_1980-1999.nc'),
}

# Output directory
output_dir = config.forcing_pkl_output_dir
os.makedirs(output_dir, exist_ok=True)

# Configuration
batch_size = 1000
batch_number = 1

print(f"Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Output directory: {output_dir}")
print(f"  History file: {history_file}")
for var, path in forcing_files.items():
    print(f"  {var} file: {path}")

# Load NetCDF files
print("\nLoading NetCDF files...")
start_time = time.time()

# Load history file for coordinates
ds_history = nc.Dataset(history_file)

# Load forcing files
forcing_datasets = {}
for var, file_path in forcing_files.items():
    print(f"  Loading {var}...")
    forcing_datasets[var] = nc.Dataset(file_path)

print(f"✅ All files loaded: {time.time() - start_time:.2f}s")

# Get coordinate information
print("\nSetting up spatial filtering...")
lats = ds_history.variables['lat'][:]
lons = ds_history.variables['lon'][:]
landmask = ds_history.variables['landfrac'][:]

# Filter for land gridcells
valid_mask = (landmask > 0)
valid_gridcells = np.where(valid_mask)[0]

print(f"Total land gridcells: {len(valid_gridcells)}")
print(f"Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
print(f"Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")

# Build KDTree index
print("\nBuilding KDTree index...")
query_coords = np.array([(lats[i], lons[i]) for i in valid_gridcells])

# Forcing file coordinates
forcing_lats = forcing_datasets['FLDS'].variables['LATIXY'][:].flatten()
forcing_lons = forcing_datasets['FLDS'].variables['LONGXY'][:].flatten()
forcing_coords = np.vstack((forcing_lats, forcing_lons)).T

forcing_tree = cKDTree(forcing_coords)
_, all_forcing_indices = forcing_tree.query(query_coords, k=1)

print("✅ Forcing mapping completed")

# 🚀 KEY OPTIMIZATION: Pre-load all forcing data into memory
print("\n🚀 Pre-loading all forcing data into memory...")
start_time = time.time()

forcing_data = {}
for var in forcing_files.keys():
    print(f"  Loading {var} data...")
    # Load all data: time × nj × ni (58400 × 1 × 11357)
    forcing_data[var] = forcing_datasets[var].variables[var][:, 0, :]
    print(f"    {var} shape: {forcing_data[var].shape}, memory: {forcing_data[var].nbytes / 1024**3:.2f} GB")

total_forcing_memory = sum(data.nbytes for data in forcing_data.values()) / 1024**3

print(f"✅ All forcing data pre-loaded: {time.time() - start_time:.2f}s")
print(f"   Total memory usage: {total_forcing_memory:.2f} GB")

# Close forcing NetCDF files (data is now in memory)
for ds in forcing_datasets.values():
    ds.close()

print("✅ Forcing NetCDF files closed, data in memory")

# Process data
print(f"\nStarting to process {len(valid_gridcells)} gridcells...")

for start_idx in range(0, len(valid_gridcells), batch_size):
    end_idx = min(start_idx + batch_size, len(valid_gridcells))
    batch_gridcells = valid_gridcells[start_idx:end_idx]
    batch_forcing_indices = all_forcing_indices[start_idx:end_idx]
    
    print(f"\nProcessing batch {batch_number}: gridcells {start_idx+1}-{end_idx}")
    batch_start_time = time.time()
    
    # Initialize data dictionary - only forcing data and basic geographical information
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
        
        # Basic geographical information
        data_dict['landfrac'].append(float(landmask[gridcell_idx]))
        data_dict['Latitude'].append(float(lats[gridcell_idx]))
        data_dict['Longitude'].append(float(lons[gridcell_idx]))
        
        # Forcing data (directly from pre-loaded memory arrays)
        data_dict['FLDS'].append(forcing_data['FLDS'][:, forcing_idx])
        data_dict['PSRF'].append(forcing_data['PSRF'][:, forcing_idx])
        data_dict['FSDS'].append(forcing_data['FSDS'][:, forcing_idx])
        data_dict['QBOT'].append(forcing_data['QBOT'][:, forcing_idx])
        data_dict['PRECTmms'].append(forcing_data['PRECTmms'][:, forcing_idx])
        data_dict['TBOT'].append(forcing_data['TBOT'][:, forcing_idx])
    
    # Create DataFrame and save
    print(f"  Creating DataFrame...")
    df_batch = pd.DataFrame(data_dict)
    
    # Convert forcing variables to list format for training compatibility
    print(f"  Converting forcing variables to list format...")
    forcing_vars = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
    for var in forcing_vars:
        df_batch[var] = df_batch[var].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    print(f"  Saving to disk...")
    batch_save_path = f"{output_dir}/TVA_forcing_batch_{batch_number:02d}.pkl"
    df_batch.to_pickle(batch_save_path)
    
    batch_time = time.time() - batch_start_time
    print(f"✅ Batch {batch_number} completed in {batch_time:.2f}s:")
    print(f"    Path: {batch_save_path}")
    print(f"    Shape: {df_batch.shape}")
    print(f"    Columns: {len(df_batch.columns)}")
    print(f"    Forcing data length: {len(df_batch['FLDS'].iloc[0])}")
    print(f"    Data format: All forcing variables converted to list format")
    
    batch_number += 1

# Cleanup
print(f"\n{'='*80}")
print("Cleaning up...")
ds_history.close()

print("✅ All NetCDF files closed")

print(f"\n🎉 Forcing data PKL generation completed!")
print(f"Total batches: {batch_number - 1}")
print(f"Output directory: {output_dir}")
print(f"Each PKL file contains: 9 variables (landfrac, Latitude, Longitude, 6 forcing variables)")
print(f"Forcing data: 58400 time steps (3-hour resolution, 20 years)")
print(f"Data format: All forcing variables automatically converted to list format")
print(f"Training compatibility: Ready for machine learning training pipelines")
print(f"Memory optimization: All forcing data pre-loaded for faster processing")