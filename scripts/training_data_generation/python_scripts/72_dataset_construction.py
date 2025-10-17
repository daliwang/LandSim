#!/usr/bin/env python3
"""
TVA Complete Training Dataset Generation Script
Generates complete PKL files with forcing data, ecosystem variables, and monthly averaging
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
print("TVA Complete Training Dataset Generation")
print("="*80)

# File paths using config and hardcoded paths
file_path1 = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/domain_surfdata/TVA_surfdata.TES_SE.4km.1d.NLCD.c241219.nc'
file_path2 = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_adspinref.elm.h0.0021-01-01-00000.nc'

# Use generated forcing NetCDF files
file_path4 = os.path.join(config.forcing_netcdf_output_dir, 'TVA_FLDS_1980-1999.nc')
file_path5 = os.path.join(config.forcing_netcdf_output_dir, 'TVA_FSDS_1980-1999.nc')
file_path6 = os.path.join(config.forcing_netcdf_output_dir, 'TVA_PRECTmms_1980-1999.nc')
file_path7 = os.path.join(config.forcing_netcdf_output_dir, 'TVA_PSRF_1980-1999.nc')
file_path8 = os.path.join(config.forcing_netcdf_output_dir, 'TVA_QBOT_1980-1999.nc')
file_path9 = os.path.join(config.forcing_netcdf_output_dir, 'TVA_TBOT_1980-1999.nc')

file_path10 = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_adspinref.elm.r.0021-01-01-00000.nc'

file_path12 = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_runs/uELM_TVA_finalspinref/run/uELM_TVA_finalspinref.elm.h0.0781-01.nc'
file_path17 = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc'

# Output directory using config
output_dir = os.path.join(config.output_dir, 'training_dataset_pkl')
os.makedirs(output_dir, exist_ok=True)

# TVA region coordinates (1D domain)
# TVA: lat [32.33, 37.58], lon [-90.33, -81.71]

print("Loading NetCDF files...")
start_time = time.time()

ds1 = nc.Dataset(file_path1)   # Surface data (TVA_surfdata.TES_SE.4km.1d.NLCD.c241219.nc)
ds2 = nc.Dataset(file_path2)   # History file (uELM_TVA_adspinref.elm.h0.0021-01-01-00000.nc)
ds4 = nc.Dataset(file_path4)   # FLDS forcing (TVA_FLDS_1980-1999.nc)
ds5 = nc.Dataset(file_path5)   # FSDS forcing (TVA_FSDS_1980-1999.nc)
ds6 = nc.Dataset(file_path6)   # PRECTmms forcing (TVA_PRECTmms_1980-1999.nc)
ds7 = nc.Dataset(file_path7)   # PSRF forcing (TVA_PSRF_1980-1999.nc)
ds8 = nc.Dataset(file_path8)   # QBOT forcing (TVA_QBOT_1980-1999.nc)
ds9 = nc.Dataset(file_path9)   # TBOT forcing (TVA_TBOT_1980-1999.nc)
ds10 = nc.Dataset(file_path10) # Restart file (uELM_TVA_adspinref.elm.r.0021-01-01-00000.nc)

# For Y (future) values - using single file for demo
ds_h0_list = [nc.Dataset(file_path12)]  # Future history file (uELM_TVA_finalspinref.elm.h0.0781-01.nc)
ds_r_list = [nc.Dataset(file_path17)]   # Future restart file (uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc)

print(f"✅ All files loaded: {time.time() - start_time:.2f}s")

# TVA data is 1D (lndgrid), not 2D
lats = ds2.variables['lat'][:]      # 1D array
lons = ds2.variables['lon'][:]      # 1D array
landmask = ds2.variables['landfrac'][:]  # Use landfrac instead of landmask

# Filter for land gridcells (1D domain)
valid_mask = (landmask > 0)
valid_gridcells = np.where(valid_mask)[0]

print(f"✅ Spatial filtering completed: {time.time() - start_time:.2f}s")
print(f"Total land gridcells: {len(valid_gridcells)}")
print(f"Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
print(f"Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")

batch_size = 1000
batch_number = 1

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Output directory: {output_dir}")

print("\nBuilding KDTree index...")
start_time = time.time()

# Build query coordinates for valid gridcells (1D domain)
query_coords = np.array([(lats[i], lons[i]) for i in valid_gridcells])

# Restart file coordinates (1D)
gridcell_lat = ds10.variables['grid1d_lat'][:]
gridcell_lon = ds10.variables['grid1d_lon'][:]
restart_grid_coords = np.vstack((gridcell_lat, gridcell_lon)).T

restart_tree = cKDTree(restart_grid_coords)
_, all_restart_indices = restart_tree.query(query_coords, k=1)

# Forcing file coordinates (1D)
forcing_lats = ds4.variables['LATIXY'][:].flatten()
forcing_lons = ds4.variables['LONGXY'][:].flatten()
forcing_grid_coords = np.vstack((forcing_lats, forcing_lons)).T

forcing_tree = cKDTree(forcing_grid_coords)
_, all_forcing_indices = forcing_tree.query(query_coords, k=1)

print(f"✅ KDTree index construction completed: {time.time() - start_time:.2f}s")

# 🚀 KEY OPTIMIZATION: Pre-load all forcing data into memory
print("\n🚀 Pre-loading all forcing data into memory...")
start_time = time.time()

print("  Loading FLDS data...")
flds_data = ds4.variables['FLDS'][:, 0, :]  # 58400 × 11357
print(f"    FLDS shape: {flds_data.shape}, memory: {flds_data.nbytes / 1024**3:.2f} GB")

print("  Loading PSRF data...")
psrf_data = ds7.variables['PSRF'][:, 0, :]
print(f"    PSRF shape: {psrf_data.shape}, memory: {psrf_data.nbytes / 1024**3:.2f} GB")

print("  Loading FSDS data...")
fsds_data = ds5.variables['FSDS'][:, 0, :]
print(f"    FSDS shape: {fsds_data.shape}, memory: {fsds_data.nbytes / 1024**3:.2f} GB")

print("  Loading QBOT data...")
qbot_data = ds8.variables['QBOT'][:, 0, :]
print(f"    QBOT shape: {qbot_data.shape}, memory: {qbot_data.nbytes / 1024**3:.2f} GB")

print("  Loading PRECTmms data...")
prect_data = ds6.variables['PRECTmms'][:, 0, :]
print(f"    PRECTmms shape: {prect_data.shape}, memory: {prect_data.nbytes / 1024**3:.2f} GB")

print("  Loading TBOT data...")
tbot_data = ds9.variables['TBOT'][:, 0, :]
print(f"    TBOT shape: {tbot_data.shape}, memory: {tbot_data.nbytes / 1024**3:.2f} GB")

total_forcing_memory = (flds_data.nbytes + psrf_data.nbytes + fsds_data.nbytes + 
                        qbot_data.nbytes + prect_data.nbytes + tbot_data.nbytes) / 1024**3

print(f"✅ All forcing data pre-loaded: {time.time() - start_time:.2f}s")
print(f"   Total memory usage: {total_forcing_memory:.2f} GB")

# Close forcing NetCDF files (data is now in memory)
ds4.close()
ds5.close()
ds6.close()
ds7.close()
ds8.close()
ds9.close()

print("✅ Forcing NetCDF files closed, data in memory")

# Define variable lists
pft_based_vars = [
    'totvegc', 'deadstemn', 'deadcrootn', 'deadstemp', 'deadcrootp',
    'leafc', 'leafc_storage', 'frootc', 'frootc_storage',
    'deadcrootc', 'deadstemc', 'tlai',
    'leafn', 'leafn_storage', 'frootn','frootn_storage',
    'leafp', 'leafp_storage', 'frootp','frootp_storage',
    'livestemc', 'livestemc_storage', 
    'livestemn', 'livestemn_storage', 
    'livestemp', 'livestemp_storage',
    'deadcrootc_storage', 'deadstemc_storage', 
    'livecrootc', 'livecrootc_storage', 
    'deadcrootn_storage', 'deadstemn_storage', 
    'livecrootn', 'livecrootn_storage', 
    'deadcrootp_storage', 'deadstemp_storage', 
    'livecrootp', 'livecrootp_storage'
]

col_based_1d_vars = ['cwdp', 'totcolp', 'totlitc']

col_based_2d_vars = [
    'cwdn_vr', 'secondp_vr', 'cwdp_vr', 'soil3c_vr', 'soil4c_vr', 'cwdc_vr',
    'soil1c_vr', 'soil1n_vr', 'soil1p_vr',
    'soil2c_vr', 'soil2n_vr', 'soil2p_vr',
    'soil3n_vr', 'soil3p_vr',
    'soil4n_vr', 'soil4p_vr',
    'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
    'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
    'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
    'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
    'labilep_vr', 'occlp_vr', 'primp_vr'
]

all_x_vars = pft_based_vars + col_based_1d_vars + col_based_2d_vars 

# Pre-load X variable data
x_values = {}
for var_name in all_x_vars:
    print(f"  Loading X variable: {var_name}")
    x_values[var_name] = ds10.variables[var_name][:]

# Pre-load Y variable data
stacked_y_values = {}
for var_name in all_x_vars:
    print(f"  Loading Y variable: {var_name}")
    list_of_arrays = [ds_r.variables[var_name][:] for ds_r in ds_r_list]
    stacked_y_values[var_name] = np.stack(list_of_arrays, axis=0)

print(f"✅ X and Y variables pre-loaded: {time.time() - start_time:.2f}s")

# Build index mapping
print("\nBuilding index mapping...")
start_time = time.time()

pft_gridcell_index = ds10.variables['pfts1d_gridcell_index'][:]
column_gridcell_index = ds10.variables['cols1d_gridcell_index'][:]

pft_map = {}
column_map = {}

unique_gridcell_ids = np.unique(pft_gridcell_index)
for grid_id in unique_gridcell_ids:
    pft_map[grid_id] = np.where(pft_gridcell_index == grid_id)[0]
    column_map[grid_id] = np.where(column_gridcell_index == grid_id)[0]

print(f"✅ Index mapping construction completed: {time.time() - start_time:.2f}s")

# Process data
print(f"\nStarting to process {len(valid_gridcells)} gridcells...")

for start_idx in range(0, len(valid_gridcells), batch_size):
    end_idx = min(start_idx + batch_size, len(valid_gridcells))
    batch_gridcells = valid_gridcells[start_idx:end_idx]
    batch_restart_indices = all_restart_indices[start_idx:end_idx]
    batch_forcing_indices = all_forcing_indices[start_idx:end_idx]
    
    print(f"\nProcessing batch {batch_number}: gridcells {start_idx+1}-{end_idx}")
    batch_start_time = time.time()

    data_dict = {
        'landfrac':[],
        'Latitude': [],
        'Longitude': [],
        'FLDS': [], 'PSRF': [], 'FSDS': [], 'QBOT': [], 'PRECTmms': [], 'TBOT': [],
        'LANDFRAC_PFT': [], 'PCT_NATVEG': [], 'AREA': [], 'peatf': [], 'abm': [],
        'SOIL_COLOR': [], 'SOIL_ORDER': [], 'PCT_NAT_PFT': [], 'PCT_SAND': [],
        'soil3c_vr': [], 'soil4c_vr': [], 'cwdc_vr': [], 'deadcrootc': [], 'deadstemc': [],'tlai': [],
        'GPP': [],
        'Y_soil3c_vr': [], 'Y_soil4c_vr': [], 'Y_cwdc_vr': [],  'Y_deadcrootc': [],  'Y_deadstemc': [], 'Y_tlai': [],
        'Y_GPP': [],
        'SCALARAVG_vr': [],
        'PCT_CLAY': [],
        'SNOWDP': [],
        'H2OSOI_10CM': [],
        'HR': [],  'AR': [],  'NPP': [], 'COL_FIRE_CLOSS': [],
        'Y_HR': [],  'Y_AR': [],  'Y_NPP': [], 'Y_COL_FIRE_CLOSS': [],
        'OCCLUDED_P': [],
        'SECONDARY_P': [],
        'LABILE_P': [],
        'APATITE_P': [],

        'cwdn_vr': [], 'secondp_vr': [], 'cwdp_vr': [],'cwdp': [], 'totcolp': [], 'totvegc': [], 'deadstemn': [], 'deadcrootn': [],
        'deadstemp': [], 'deadcrootp': [], 'leafc': [], 'leafc_storage': [], 'frootc': [], 'frootc_storage': [],
        'Y_cwdn_vr': [], 'Y_secondp_vr': [], 'Y_cwdp_vr': [], 'Y_cwdp': [], 'Y_totcolp': [], 'Y_totvegc': [], 'Y_deadstemn': [], 'Y_deadcrootn': [],
        'Y_deadstemp': [], 'Y_deadcrootp': [], 'Y_leafc': [], 'Y_leafc_storage': [], 'Y_frootc': [], 'Y_frootc_storage': [],
        'totlitc': [],

    'leafn': [], 'leafn_storage': [], 'frootn': [],'frootn_storage': [],
    'leafp': [], 'leafp_storage': [], 'frootp': [],'frootp_storage': [],
    'livestemc': [], 'livestemc_storage': [], 
    'livestemn': [], 'livestemn_storage': [], 
    'livestemp': [], 'livestemp_storage': [],
    'labilep_vr': [], 'occlp_vr': [], 'primp_vr': [],

    'deadcrootc_storage': [], 'deadstemc_storage': [], 
    'livecrootc': [], 'livecrootc_storage': [], 
    'deadcrootn_storage': [], 'deadstemn_storage': [], 
    'livecrootn': [], 'livecrootn_storage': [], 
    'deadcrootp_storage': [], 'deadstemp_storage': [],
    'livecrootp': [], 'livecrootp_storage': [], 

    'Y_leafn': [], 'Y_leafn_storage': [], 'Y_frootn': [],'Y_frootn_storage': [],
    'Y_leafp': [], 'Y_leafp_storage': [], 'Y_frootp': [],'Y_frootp_storage': [],
    'Y_livestemc': [], 'Y_livestemc_storage': [], 
    'Y_livestemn': [], 'Y_livestemn_storage': [], 
    'Y_livestemp': [], 'Y_livestemp_storage': [],
    'Y_labilep_vr': [], 'Y_occlp_vr': [], 'Y_primp_vr': [],

    'Y_deadcrootc_storage': [], 'Y_deadstemc_storage': [], 
    'Y_livecrootc': [], 'Y_livecrootc_storage': [], 
    'Y_deadcrootn_storage': [], 'Y_deadstemn_storage': [], 
    'Y_livecrootn': [], 'Y_livecrootn_storage': [], 
    'Y_deadcrootp_storage': [], 'Y_deadstemp_storage': [],
    'Y_livecrootp': [], 'Y_livecrootp_storage': [], 
        'Y_totlitc': [],
        # 'H2OCAN': [], 'T_VEG': [], 'T10_VALUE': [],
        # 'Y_H2OCAN': [], 'Y_T_VEG': [], 'Y_T10_VALUE': [],
        # 'H2OSFC': [], 'H2OSNO': [], 'TH2OSFC': [], 'T_GRND': [], 'T_GRND_R': [], 'T_GRND_U': [],
        # 'Y_H2OSFC': [], 'Y_H2OSNO': [], 'Y_TH2OSFC': [], 'Y_T_GRND': [], 'Y_T_GRND_R': [], 'Y_T_GRND_U': [],
        # 'H2OSOI_LIQ': [], 'H2OSOI_ICE': [], 'T_SOISNO': [], 'LAKE_SOILC': [], 'T_LAKE': [],
        # 'Y_H2OSOI_LIQ': [], 'Y_H2OSOI_ICE': [], 'Y_T_SOISNO': [], 'Y_LAKE_SOILC': [], 'Y_T_LAKE': [],
        # 'taf': [],
        # 'Y_taf': [],
        # 'TS_TOPO': [],
        # 'Y_TS_TOPO': [],
        'soil1c_vr': [], 'soil1n_vr': [], 'soil1p_vr': [],
        'soil2c_vr': [], 'soil2n_vr': [], 'soil2p_vr': [],
        'soil3n_vr': [], 'soil3p_vr': [],
        'soil4n_vr': [], 'soil4p_vr': [],
        'litr1c_vr': [], 'litr2c_vr': [], 'litr3c_vr': [],
        'litr1n_vr': [], 'litr2n_vr': [], 'litr3n_vr': [],
        'litr1p_vr': [], 'litr2p_vr': [], 'litr3p_vr': [],
        'sminn_vr': [], 'smin_no3_vr': [], 'smin_nh4_vr': [],
        'Y_soil1c_vr': [], 'Y_soil1n_vr': [], 'Y_soil1p_vr': [],
        'Y_soil2c_vr': [], 'Y_soil2n_vr': [], 'Y_soil2p_vr': [],
        'Y_soil3n_vr': [], 'Y_soil3p_vr': [],
        'Y_soil4n_vr': [], 'Y_soil4p_vr': [],
        'Y_litr1c_vr': [], 'Y_litr2c_vr': [], 'Y_litr3c_vr': [],
        'Y_litr1n_vr': [], 'Y_litr2n_vr': [], 'Y_litr3n_vr': [],
        'Y_litr1p_vr': [], 'Y_litr2p_vr': [], 'Y_litr3p_vr': [],
        'Y_sminn_vr': [], 'Y_smin_no3_vr': [], 'Y_smin_nh4_vr': []
    }



    for k, gridcell_idx in enumerate(batch_gridcells):
        if k % 100 == 0:
            print(f"  Processing gridcell {k}/{len(batch_gridcells)} (idx={gridcell_idx})")
        
        # Get indices
        restart_idx = batch_restart_indices[k]
        gridcell_id = restart_idx + 1

        pft_indices_for_cell = pft_map.get(gridcell_id, [])
        column_indices_for_cell = column_map.get(gridcell_id, [])

        for var_name in pft_based_vars:
            x_val = x_values[var_name][pft_indices_for_cell]
            data_dict[var_name].append(x_val.tolist())

            y_slice = stacked_y_values[var_name][:, pft_indices_for_cell]
            avg_y_val = np.mean(y_slice, axis=0)
            data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        for var_name in col_based_1d_vars:
            x_val = x_values[var_name][column_indices_for_cell]
            data_dict[var_name].append(x_val.tolist())

            y_slice = stacked_y_values[var_name][:, column_indices_for_cell]
            avg_y_val = np.mean(y_slice, axis=0)
            data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        for var_name in col_based_2d_vars:
            x_val = x_values[var_name][column_indices_for_cell, :]
            data_dict[var_name].append(x_val.tolist())

            y_slice = stacked_y_values[var_name][:, column_indices_for_cell, :]
            avg_y_val = np.mean(y_slice, axis=0)
            data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        # landunit_indices_for_cell = landunit_map.get(gridcell_id, [])
        # for var_name in landunit_based_vars:
        #     x_val = x_values[var_name][landunit_indices_for_cell]
        #     data_dict[var_name].append(x_val.tolist())

        #     y_slice = stacked_y_values[var_name][:, landunit_indices_for_cell]
        #     avg_y_val = np.mean(y_slice, axis=0)
        #     data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        # topounit_indices_for_cell = topounit_map.get(gridcell_id, [])
        # for var_name in topounit_based_vars:
        #     x_val = x_values[var_name][topounit_indices_for_cell]
        #     data_dict[var_name].append(x_val.tolist())

        #     y_slice = stacked_y_values[var_name][:, topounit_indices_for_cell]
        #     avg_y_val = np.mean(y_slice, axis=0)
        #     data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        data_dict['landfrac'].append(ds2.variables['landfrac'][gridcell_idx])
        data_dict['Latitude'].append(lats[gridcell_idx])
        data_dict['Longitude'].append(lons[gridcell_idx])
        data_dict['LANDFRAC_PFT'].append(ds1.variables['LANDFRAC_PFT'][gridcell_idx])
        data_dict['PCT_NATVEG'].append(ds1.variables['PCT_NATVEG'][gridcell_idx])
        data_dict['AREA'].append(ds1.variables['AREA'][gridcell_idx])
        data_dict['peatf'].append(ds1.variables['peatf'][gridcell_idx])
        data_dict['abm'].append(ds1.variables['abm'][gridcell_idx])
        data_dict['SOIL_COLOR'].append(ds1.variables['SOIL_COLOR'][gridcell_idx])
        data_dict['SOIL_ORDER'].append(ds1.variables['SOIL_ORDER'][gridcell_idx])
        data_dict['PCT_SAND'].append(ds1.variables['PCT_SAND'][:, gridcell_idx])
        data_dict['PCT_NAT_PFT'].append(ds1.variables['PCT_NAT_PFT'][:, gridcell_idx])

        data_dict['OCCLUDED_P'].append(ds1.variables['OCCLUDED_P'][gridcell_idx])
        data_dict['SECONDARY_P'].append(ds1.variables['SECONDARY_P'][gridcell_idx])
        data_dict['LABILE_P'].append(ds1.variables['LABILE_P'][gridcell_idx])
        data_dict['APATITE_P'].append(ds1.variables['APATITE_P'][gridcell_idx])


        data_dict['GPP'].append(ds2.variables['GPP'][0, gridcell_idx])
        data_dict['SCALARAVG_vr'].append(ds2.variables['SCALARAVG_vr'][0, :, gridcell_idx])
        data_dict['HR'].append(ds2.variables['HR'][0, gridcell_idx])
        data_dict['AR'].append(ds2.variables['AR'][0, gridcell_idx])
        data_dict['NPP'].append(ds2.variables['NPP'][0, gridcell_idx])
        # COL_FIRE_CLOSS may not exist in TVA history, use FIRE instead if available
        if 'COL_FIRE_CLOSS' in ds2.variables:
            data_dict['COL_FIRE_CLOSS'].append(ds2.variables['COL_FIRE_CLOSS'][0, gridcell_idx])
        elif 'FIRE' in ds2.variables:
            data_dict['COL_FIRE_CLOSS'].append(ds2.variables['FIRE'][0, gridcell_idx])
        else:
            data_dict['COL_FIRE_CLOSS'].append(0.0)

        data_dict['SNOWDP'].append(ds2.variables['SNOWDP'][0, gridcell_idx])
        data_dict['H2OSOI_10CM'].append(ds2.variables['H2OSOI'][0,3, gridcell_idx])
        data_dict['PCT_CLAY'].append(ds1.variables['PCT_CLAY'][:, gridcell_idx])

        h0_gpp_vals = []
        for ds_h0 in ds_h0_list:
            h0_gpp_vals.append(ds_h0.variables['GPP'][0, gridcell_idx])
        avg_h0_gpp = np.mean(h0_gpp_vals)
        data_dict['Y_GPP'].append(avg_h0_gpp)

        h0_HR_vals = []
        for ds_h0 in ds_h0_list:
            h0_HR_vals.append(ds_h0.variables['HR'][0, gridcell_idx])
        avg_h0_HR = np.mean(h0_HR_vals)
        data_dict['Y_HR'].append(avg_h0_HR)

        h0_AR_vals = []
        for ds_h0 in ds_h0_list:
            h0_AR_vals.append(ds_h0.variables['AR'][0, gridcell_idx])
        avg_h0_AR = np.mean(h0_AR_vals)
        data_dict['Y_AR'].append(avg_h0_AR)

        h0_NPP_vals = []
        for ds_h0 in ds_h0_list:
            h0_NPP_vals.append(ds_h0.variables['NPP'][0, gridcell_idx])
        avg_h0_NPP = np.mean(h0_NPP_vals)
        data_dict['Y_NPP'].append(avg_h0_NPP)

        h0_COL_FIRE_CLOSS_vals = []
        for ds_h0 in ds_h0_list:
            # COL_FIRE_CLOSS may not exist in TVA history
            if 'COL_FIRE_CLOSS' in ds_h0.variables:
                h0_COL_FIRE_CLOSS_vals.append(ds_h0.variables['COL_FIRE_CLOSS'][0, gridcell_idx])
            elif 'FIRE' in ds_h0.variables:
                h0_COL_FIRE_CLOSS_vals.append(ds_h0.variables['FIRE'][0, gridcell_idx])
            else:
                h0_COL_FIRE_CLOSS_vals.append(0.0)
        avg_h0_COL_FIRE_CLOSS = np.mean(h0_COL_FIRE_CLOSS_vals)
        data_dict['Y_COL_FIRE_CLOSS'].append(avg_h0_COL_FIRE_CLOSS)

        # Get forcing index
        forcing_idx = batch_forcing_indices[k]
        
        # 🚀 Fast access to forcing data from memory
        data_dict['FLDS'].append(flds_data[:, forcing_idx])
        data_dict['PSRF'].append(psrf_data[:, forcing_idx])
        data_dict['FSDS'].append(fsds_data[:, forcing_idx])
        data_dict['QBOT'].append(qbot_data[:, forcing_idx])
        data_dict['PRECTmms'].append(prect_data[:, forcing_idx])
        data_dict['TBOT'].append(tbot_data[:, forcing_idx])
    # Create DataFrame and save
    print(f"  Creating DataFrame...")
    df_batch = pd.DataFrame(data_dict)
    
    print(f"  Saving to disk...")
    batch_save_path = f"{output_dir}/training_data_batch_{batch_number:02d}.pkl"
    df_batch.to_pickle(batch_save_path)
    
    batch_time = time.time() - batch_start_time
    print(f"✅ Batch {batch_number} completed: {batch_time:.2f}s")
    print(f"    Path: {batch_save_path}")
    print(f"    Shape: {df_batch.shape}")
    print(f"    Columns: {len(df_batch.columns)}")
    print(f"    Forcing data length: {len(df_batch['FLDS'].iloc[0])}")
    
    batch_number += 1

# Cleanup NetCDF files
print(f"\n{'='*80}")
print("Cleaning up NetCDF files...")
ds1.close()
ds2.close()
ds10.close()
for ds_h0 in ds_h0_list:
    ds_h0.close()
for ds_r in ds_r_list:
    ds_r.close()

print("✅ All NetCDF files closed")

print(f"\n🎉 Complete dataset PKL generation completed!")
print(f"Total batches: {batch_number - 1}")
print(f"Output directory: {output_dir}")
print(f"Each PKL file contains: {len(data_dict)} variables")
print(f"Forcing data: 58400 time steps")
print(f"Optimization strategy: Pre-load all forcing data to memory, avoid repeated NetCDF access")

# =============================================================================
# POST-PROCESSING: MONTHLY AVERAGING (Simplified)
# =============================================================================

print(f"\n{'='*80}")
print("POST-PROCESSING: Monthly Averaging")
print(f"{'='*80}")

# Import glob for file processing
import glob

# Get all generated PKL files
input_files = sorted(glob.glob(f'{output_dir}/training_data_batch_*.pkl'))
print(f"Found {len(input_files)} PKL files for post-processing")

# TVA data parameters (20 years, 3-hour interval)
time_series_length = 58400  # 20 years × 365 days × 8 steps/day
steps_per_day = 8           # 3-hour interval = 8 steps/day
days_per_year = 365
years_in_data = 20          # 1980-1999
months_per_year = 12
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

print(f"TVA data parameters:")
print(f"  Time series length: {time_series_length}")
print(f"  Steps per day: {steps_per_day}")
print(f"  Years: {years_in_data}")
print(f"  Expected monthly values: {years_in_data * months_per_year}")

def calculate_monthly_avg(time_series):
    """
    Calculate monthly averages from high-resolution time series
    """
    if not isinstance(time_series, (list, np.ndarray)):
        return []
    
    if len(time_series) != time_series_length:
        return []

    monthly_averages = []
    start_idx = 0
    
    for year in range(years_in_data):
        for month_idx, month_days in enumerate(days_per_month):
            # Handle leap year February
            if year % 4 == 0 and month_idx == 1:  # Leap year February
                month_days = 29
            
            end_idx = start_idx + month_days * steps_per_day
            monthly_avg = np.mean(time_series[start_idx:end_idx])
            monthly_averages.append(monthly_avg)
            start_idx = end_idx
    
    return monthly_averages

# Define columns to process
time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
single_value_columns = [
    'landfrac', 'LANDFRAC_PFT', 'PCT_NATVEG', 'AREA', 'peatf', 'abm', 
    'SOIL_COLOR', 'SOIL_ORDER', 'GPP', 'SNOWDP', 'H2OSOI_10CM',
    'Y_GPP', 'HR', 'AR', 'NPP', 'COL_FIRE_CLOSS',
    'Y_HR', 'Y_AR', 'Y_NPP', 'Y_COL_FIRE_CLOSS',
    'OCCLUDED_P', 'SECONDARY_P', 'LABILE_P', 'APATITE_P'
]
list_like_columns = ['PCT_NAT_PFT', 'PCT_SAND', 'SCALARAVG_vr', 'PCT_CLAY']

print(f"\nStarting post-processing...")

# Process all files
print(f"Processing all {len(input_files)} files...")

for file_idx, file_path in enumerate(input_files, 1):
    print(f"\nProcessing file {file_idx}/{len(input_files)}: {os.path.basename(file_path)}")
    
    try:
        # Read PKL file
        df = pd.read_pickle(file_path)
        print(f"  Original shape: {df.shape}")
        print(f"  Original columns: {len(df.columns)}")
    
        # Process time series columns (convert to monthly averages)
        print(f"  Processing time series columns...")
        for col in time_series_columns:
            if col in df.columns:
                print(f"    Processing {col}...")
                df[col] = df[col].apply(calculate_monthly_avg)
                
                # Verify processing results
                sample_data = df[col].apply(lambda x: x if isinstance(x, list) else [])
                lengths = sample_data.apply(len).unique()
                print(f"      {col} monthly values length: {lengths}")
    
        # Process single value columns
        print(f"  Processing single value columns...")
        for col in single_value_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process list columns
        print(f"  Processing list columns...")
        for col in list_like_columns:
            if col in df.columns:
                print(f"    Expanding {col}...")
                expanded_cols = df[col].apply(pd.Series).fillna(0)
                expanded_cols = expanded_cols.add_prefix(f"{col}_")
                df = df.drop(col, axis=1).join(expanded_cols)
        
        # Reorder columns
        y_columns = [col for col in df.columns if col.startswith('Y_')]
        other_columns = [col for col in df.columns if not col.startswith('Y_')]
        df = df[other_columns + y_columns]
        
        # Save processed file
        output_file = f"{output_dir}/monthly_{os.path.basename(file_path)}"
        df.to_pickle(output_file)
        
        print(f"  ✅ Post-processing completed")
        print(f"    Processed shape: {df.shape}")
        print(f"    Processed columns: {len(df.columns)}")
        print(f"    Saved to: {output_file}")
        
        # Display sample data
        print(f"    Sample data:")
        print(f"      FLDS monthly values length: {len(df['FLDS'].iloc[0]) if 'FLDS' in df.columns else 'N/A'}")
        print(f"      First 3 coordinates: {df[['Latitude', 'Longitude']].head(3).values.tolist()}")
        
    except Exception as e:
        print(f"  ❌ Error processing {os.path.basename(file_path)}: {e}")
        continue

print(f"\n{'='*80}")
print("POST-PROCESSING COMPLETED!")
print(f"Input directory: {output_dir}")
print(f"Output directory: {output_dir}")
print(f"Processed files: {len(input_files)}")

# Check output files
output_files = glob.glob(f'{output_dir}/monthly_*.pkl')
print(f"Generated files: {len(output_files)}")

if output_files:
    print(f"\nOutput file list:")
    for file in output_files:
        file_size = os.path.getsize(file) / (1024**3)  # GB
        print(f"  {os.path.basename(file)}: {file_size:.2f} GB")

print("="*80)

