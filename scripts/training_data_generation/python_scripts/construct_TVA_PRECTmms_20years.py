#!/usr/bin/env python3
"""
Generate TVA PRECTmms forcing data (1980-1999, 20 years, 3-hour resolution)
Corresponding to crujra.v2.5.5d_PRECTmms_1901-2023_z01.nc

Author: Zhuowei Gu
Date: 2024
"""

import xarray as xr
import os
import cftime
import numpy as np
import sys
import traceback
import datetime
from pathlib import Path

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- Configuration ---
data_dir = config.forcing_raw_data_path
out_dir = config.forcing_netcdf_output_dir
os.makedirs(out_dir, exist_ok=True)

start_year = 1980
end_year = 1999
final_output_file = os.path.join(out_dir, f"TVA_PRECTmms_{start_year}-{end_year}.nc")

print("="*80)
print("Generate TVA PRECTmms forcing data")
print("1. Merge monthly files (1980-1999, 240 months)")
print("2. Correct time axis discontinuities")
print("3. Maintain 3-hour resolution (no downsampling)")
print("="*80)

print(f"Source data directory: {data_dir}")
print(f"Target output file: {final_output_file}")

# --- Find and build all monthly file list ---
print("Searching for monthly files...")
all_monthly_files = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        file_name = f"clmforc.Daymet.km.1d.Prec.{year}-{month:02d}.nc"
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            all_monthly_files.append(file_path)
        else:
            print(f"   Warning: File {file_name} does not exist, skipping.")

if not all_monthly_files:
    print(f"Error: No valid monthly files found in directory {data_dir} for years {start_year}-{end_year}.")
    sys.exit(1)

print(f"Found {len(all_monthly_files)} valid monthly files.")

# --- Define Dask chunks ---
dask_chunks = {'time': 366*8}

try:
    with xr.open_mfdataset(all_monthly_files, combine='nested', concat_dim='time',
                          decode_times=False, chunks=dask_chunks, parallel=False) as ds:

        # === Separate static variables ===
        print("Separating static coordinate/ID variables...")
        static_data = {}
        for var_name in ['gridID', 'LONGXY', 'LATIXY']:
            if var_name in ds:
                if 'time' in ds[var_name].dims:
                    static_data[var_name] = ds[var_name].isel(time=0, drop=True).load()
                else:
                    static_data[var_name] = ds[var_name].load()

        # === Process PRECTmms variable ===
        if 'PRECTmms' not in ds.data_vars:
            print("Error: PRECTmms variable not found.")
            sys.exit(1)
        
        print("Processing PRECTmms variable...")
        ds_temporal = ds[['time', 'PRECTmms']]
        
        # --- Time axis correction ---
        print("Loading raw time coordinates...")
        time_values_raw = ds_temporal['time'].load().values
        units = ds_temporal['time'].attrs['units']
        calendar = ds_temporal['time'].attrs.get('calendar', 'standard')
        if calendar.lower() == 'no_leap': 
            calendar = 'noleap'
        
        print(f"  Time points: {len(time_values_raw)}")
        print("Starting time coordinate correction...")

        time_values_corrected = np.copy(time_values_raw).astype(float)
        cumulative_offset_days = 0.0
        expected_step_days = 3.0 / 24.0
        jump_count = 0
        
        for i in range(len(time_values_raw) - 1):
            corrected_i = time_values_raw[i] + cumulative_offset_days
            time_values_corrected[i] = corrected_i
            raw_diff = time_values_raw[i+1] - time_values_raw[i]
            if raw_diff < expected_step_days * 0.5:
                jump_count += 1
                expected_next = corrected_i + expected_step_days
                cumulative_offset_days = expected_next - time_values_raw[i+1]
            time_values_corrected[i+1] = time_values_raw[i+1] + cumulative_offset_days
        
        print(f"Time correction completed. Corrected {jump_count} jumps.")

        print("Decoding corrected time values...")
        try:
            dates = cftime.num2date(time_values_corrected, units, calendar=calendar, only_use_cftime_datetimes=True)
        except ValueError:
            dates = cftime.num2date(time_values_corrected, units, calendar=calendar)

        # --- Create dataset with corrected time ---
        ds_corrected = ds_temporal.copy(deep=False)
        ds_corrected['time'] = xr.DataArray(dates, dims='time', coords={'time': dates})
        ds_corrected['time'].encoding['units'] = units
        ds_corrected['time'].encoding['calendar'] = calendar

        # === Merge static variables ===
        for var_name, data_array in static_data.items():
            ds_corrected[var_name] = data_array

        # --- Write to file ---
        print(f"Writing to file: {final_output_file}")
        output_encoding = {
            'PRECTmms': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'time': {'units': units, 'calendar': calendar, 'dtype': 'float64'}
        }
        if 'gridID' in ds_corrected:
            output_encoding['gridID'] = {'dtype': ds_corrected['gridID'].dtype}
        if 'LONGXY' in ds_corrected:
            output_encoding['LONGXY'] = {'dtype': ds_corrected['LONGXY'].dtype, '_FillValue': np.nan}
        if 'LATIXY' in ds_corrected:
            output_encoding['LATIXY'] = {'dtype': ds_corrected['LATIXY'].dtype, '_FillValue': np.nan}

        ds_corrected.to_netcdf(final_output_file, encoding=output_encoding, unlimited_dims=['time'])
        print(f"✓ Successfully generated: {final_output_file}")
        print(f"  File size: {os.path.getsize(final_output_file) / (1024**3):.2f} GB")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nScript execution completed.")

