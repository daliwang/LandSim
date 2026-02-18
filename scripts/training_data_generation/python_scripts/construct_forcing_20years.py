#!/usr/bin/env python3
"""
Generate all forcing data (1980-1999, 20 years, 3-hour resolution)
Integrates all 6 forcing variables: FLDS, FSDS, PRECTmms, PSRF, QBOT, TBOT
Based on optimized individual scripts

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
import argparse
from pathlib import Path
import re

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- Configuration ---
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate forcing NetCDF files for 1980-1999')
    parser.add_argument('--input-dir', type=str, 
                       default=config.forcing_raw_data_path,
                       help='Input directory containing raw forcing data')
    parser.add_argument('--output-dir', type=str,
                       default=config.forcing_netcdf_output_dir,
                       help='Output directory for generated NetCDF files')
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year (default: 1980)')
    parser.add_argument('--end-year', type=int, default=1999,
                       help='End year (default: 1999)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
data_dir = args.input_dir
out_dir = args.output_dir
os.makedirs(out_dir, exist_ok=True)

start_year = args.start_year
end_year = args.end_year

# Define all forcing variables and their file patterns
forcing_variables = {
    'FLDS': {
        'token': 'TPQWL',
        'description': 'Downward longwave radiation'
    },
    'FSDS': {
        'token': 'Solr',
        'description': 'Downward shortwave radiation'
    },
    'PRECTmms': {
        'token': 'Prec',
        'description': 'Precipitation rate'
    },
    'PSRF': {
        'token': 'TPQWL',
        'description': 'Surface pressure'
    },
    'QBOT': {
        'token': 'TPQWL',
        'description': 'Specific humidity'
    },
    'TBOT': {
        'token': 'TPQWL',
        'description': 'Air temperature'
    }
}

print("="*80)
print("Generate All Forcing Data (1980-1999, 20 years)")
print("Processing 6 forcing variables:")
for var, info in forcing_variables.items():
    print(f"  - {var}: {info['description']}")
print("="*80)

print(f"Source data directory: {data_dir}")
print(f"Output directory: {out_dir}")

def process_forcing_variable(var_name, var_info):
    """Process a single forcing variable using optimized method"""
    print(f"\n{'='*60}")
    print(f"Processing {var_name}: {var_info['description']}")
    print(f"{'='*60}")
    
    final_output_file = os.path.join(out_dir, f"{var_name}_{start_year}-{end_year}.nc")
    
    # Check if output file already exists
    if os.path.exists(final_output_file):
        print(f"✅ Output file already exists: {final_output_file}")
        return True
    
    print(f"Target output file: {final_output_file}")
    
    # --- Find and build all monthly file list ---
    print(f"[{datetime.datetime.now()}] Searching for monthly files...")
    all_monthly_files = []
    
    # Pre-scan directory for efficiency (especially important for TES_NORTH with 4000+ files)
    print(f"[{datetime.datetime.now()}] Pre-scanning directory for available files (recursive)...")
    files_by_token = {}  # {(token, year, month): [paths]}
    total_files = 0
    pattern = re.compile(r'(?:.*_)?clmforc\..*\.(Prec|Solr|TPQWL)\.(\d{4})-(\d{2})\.nc$')
    for root, _, files in os.walk(data_dir):
        for fname in files:
            total_files += 1
            match = pattern.search(fname)
            if match:
                token, year_s, month_s = match.groups()
                key = (token, int(year_s), int(month_s))
                files_by_token.setdefault(key, []).append(os.path.join(root, fname))
    print(f"[{datetime.datetime.now()}] Found {total_files} files across {len(files_by_token)} token-year-month combinations")
    
    token = var_info['token']
    for year in range(start_year, end_year + 1):
        year_found = 0
        for month in range(1, 13):
            key = (token, year, month)
            if key in files_by_token:
                file_paths = files_by_token[key]
                if len(file_paths) > 1:
                    print(f"   Warning: Multiple matches for {token} {year}-{month:02d}; using {file_paths[0]}")
                file_path = file_paths[0]
                all_monthly_files.append(file_path)
                year_found += 1
            else:
                print(f"   Warning: File with token {token} for {year}-{month:02d} does not exist, skipping.")
        print(f"[{datetime.datetime.now()}] {var_name}: year {year} -> found {year_found}/12 monthly files")
    
    if not all_monthly_files:
        print(f"❌ Error: No valid monthly files found for {var_name} in directory {data_dir}")
        return False
    
    print(f"[{datetime.datetime.now()}] {var_name}: total valid monthly files: {len(all_monthly_files)}")
    
    # --- Define Dask chunks ---
    dask_chunks = {'time': 366*8}
    
    try:
        print(f"[{datetime.datetime.now()}] Opening {len(all_monthly_files)} monthly files with xarray.open_mfdataset ...")
        with xr.open_mfdataset(
            all_monthly_files,
            combine='nested',
            concat_dim='time',
            decode_times=False,
            chunks=dask_chunks,
            parallel=False,
        ) as ds:
            print(f"[{datetime.datetime.now()}] Dataset opened. Dims: {dict(ds.dims)} | Vars: {list(ds.data_vars)}")

            # === Separate static variables ===
            print(f"[{datetime.datetime.now()}] Separating static coordinate/ID variables...")
            static_var_names = ['gridID', 'LONGXY', 'LATIXY']
            static_data = {}
            
            for var_name_static in static_var_names:
                if var_name_static in ds:
                    if 'time' in ds[var_name_static].dims:
                        static_data[var_name_static] = ds[var_name_static].isel(time=0, drop=True).load()
                    else:
                        static_data[var_name_static] = ds[var_name_static].load()

            # === Process target variable ===
            time_varying_vars = [var_name]
            if var_name not in ds.data_vars:
                print(f"Error: {var_name} variable not found.")
                return False
            
            print(f"[{datetime.datetime.now()}] Processing variables: {time_varying_vars}")
            ds_temporal = ds[['time'] + time_varying_vars]

            # --- Time axis correction ---
            print(f"[{datetime.datetime.now()}] Loading raw time coordinates...")
            time_values_raw = ds_temporal['time'].load().values
            units = ds_temporal['time'].attrs['units']
            calendar = ds_temporal['time'].attrs.get('calendar', 'standard')
            if calendar.lower() == 'no_leap': 
                calendar = 'noleap'
            
            print(f"  Time points: {len(time_values_raw)}")

            print(f"[{datetime.datetime.now()}] Starting time coordinate correction...")
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
                    expected_next_corrected_value = corrected_i + expected_step_days
                    offset_needed = expected_next_corrected_value - time_values_raw[i+1]
                    cumulative_offset_days = offset_needed
                
                time_values_corrected[i+1] = time_values_raw[i+1] + cumulative_offset_days
            
            print(f"[{datetime.datetime.now()}] Time correction completed. Corrected {jump_count} jumps.")

            print(f"[{datetime.datetime.now()}] Decoding corrected time values...")
            try:
                dates = cftime.num2date(time_values_corrected, units, calendar=calendar, only_use_cftime_datetimes=True)
            except ValueError:
                dates = cftime.num2date(time_values_corrected, units, calendar=calendar)

            # --- Check time monotonicity ---
            print(f"[{datetime.datetime.now()}] Checking time monotonicity...")
            if len(dates) >= 2:
                diffs_corrected = np.diff(dates)
                zero_timedelta = datetime.timedelta(0)
                problem_indices = np.where(diffs_corrected <= zero_timedelta)[0]
                
                if len(problem_indices) > 0:
                    print(f"Error! Corrected time is still not monotonic!")
                    return False
                else:
                    print(f"[{datetime.datetime.now()}] ✓ Time coordinate check passed.")

            # --- Create dataset with corrected time ---
            ds_corrected_time = ds_temporal.copy(deep=False)
            ds_corrected_time['time'] = xr.DataArray(dates, dims='time', coords={'time': dates})
            ds_corrected_time['time'].encoding['units'] = units
            ds_corrected_time['time'].encoding['calendar'] = calendar

            # === Merge static variables ===
            for var_name_static, data_array in static_data.items():
                ds_corrected_time[var_name_static] = data_array

            final_dataset = ds_corrected_time
            print(f"[{datetime.datetime.now()}] Final dataset ready. Summary dims: {dict(final_dataset.dims)}")

            # --- Write to file ---
            print(f"[{datetime.datetime.now()}] Writing to file: {final_output_file}")
            output_encoding = {
                var_name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
                'time': {'units': units, 'calendar': calendar, 'dtype': 'float64'}
            }
            
            if 'gridID' in final_dataset:
                output_encoding['gridID'] = {'dtype': final_dataset['gridID'].dtype}
            if 'LONGXY' in final_dataset:
                output_encoding['LONGXY'] = {'dtype': final_dataset['LONGXY'].dtype, '_FillValue': np.nan}
            if 'LATIXY' in final_dataset:
                output_encoding['LATIXY'] = {'dtype': final_dataset['LATIXY'].dtype, '_FillValue': np.nan}

            final_dataset.to_netcdf(final_output_file, encoding=output_encoding, unlimited_dims=['time'])
            print(f"[{datetime.datetime.now()}] ✓ Successfully generated: {final_output_file}")
            print(f"[{datetime.datetime.now()}]   File size: {os.path.getsize(final_output_file) / (1024**3):.2f} GB")

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        return False

    return True

def main():
    """Main function to process all forcing variables"""
    print("Starting processing of all forcing variables...")
    
    success_count = 0
    total_count = len(forcing_variables)
    
    for var_name, var_info in forcing_variables.items():
        try:
            success = process_forcing_variable(var_name, var_info)
            if success:
                success_count += 1
                print(f"✅ {var_name} processing completed successfully")
            else:
                print(f"❌ {var_name} processing failed")
        except Exception as e:
            print(f"❌ {var_name} processing failed with exception: {e}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total variables: {total_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count == total_count:
        print("🎉 All forcing variables processed successfully!")
        print(f"Output files saved in: {out_dir}")
        
        # List generated files
        print("\nGenerated files:")
        for var_name in forcing_variables.keys():
            output_file = os.path.join(out_dir, f"{var_name}_{start_year}-{end_year}.nc")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024**3)
                print(f"  ✅ {var_name}_{start_year}-{end_year}.nc ({file_size:.2f} GB)")
            else:
                print(f"  ❌ {var_name}_{start_year}-{end_year}.nc (not found)")
        
        return True
    else:
        print(f"⚠️  Some variables failed to process. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)