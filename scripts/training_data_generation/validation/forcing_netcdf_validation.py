#!/usr/bin/env python3
"""
Simple validation script for forcing data extraction
Compares generated NetCDF files with reference files
"""

import xarray as xr
import numpy as np
import os
import sys

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def validate_simple(var_name):
    """Simple validation by comparing with reference file"""
    print(f"\n{'='*50}")
    print(f"Validating {var_name}")
    print(f"{'='*50}")
    
    # File paths
    generated_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 config.forcing_netcdf_output_dir, f"TVA_{var_name}_1980-1999.nc")
    reference_file = f"/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/TVA_forcing_dataset/data/TVA_{var_name}_1980-1999.nc"
    
    print(f"Generated: {generated_file}")
    print(f"Reference: {reference_file}")
    
    # Check if files exist
    if not os.path.exists(generated_file):
        print(f"❌ Generated file not found")
        return False
    
    if not os.path.exists(reference_file):
        print(f"❌ Reference file not found")
        return False
    
    try:
        # Load both files
        print("Loading files...")
        ds_gen = xr.open_dataset(generated_file)
        ds_ref = xr.open_dataset(reference_file)
        
        # Compare data values
        print("Comparing data values...")
        data_match = np.allclose(ds_gen[var_name].values, ds_ref[var_name].values, equal_nan=True)
        
        # Compare basic properties
        shape_match = ds_gen[var_name].shape == ds_ref[var_name].shape
        dtype_match = ds_gen[var_name].dtype == ds_ref[var_name].dtype
        
        print(f"Data values match: {data_match}")
        print(f"Shape match: {shape_match}")
        print(f"Dtype match: {dtype_match}")
        
        if data_match and shape_match and dtype_match:
            print(f"✅ {var_name} validation PASSED")
            return True
        else:
            print(f"❌ {var_name} validation FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        try:
            ds_gen.close()
            ds_ref.close()
        except:
            pass

def main():
    """Main validation function"""
    print("="*60)
    print("Simple TVA Forcing Data Validation")
    print("="*60)
    
    variables = ['FLDS', 'FSDS', 'PSRF', 'QBOT', 'PRECTmms', 'TBOT']
    results = {}
    
    for var_name in variables:
        results[var_name] = validate_simple(var_name)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = 0
    for var_name in variables:
        status = "✅ PASS" if results[var_name] else "❌ FAIL"
        print(f"{var_name:12} - {status}")
        if results[var_name]:
            passed += 1
    
    print(f"\nResults: {passed}/{len(variables)} variables passed")
    
    if passed == len(variables):
        print("🎉 ALL VALIDATIONS PASSED! Script is working correctly.")
        return 0
    else:
        print(f"⚠️  {len(variables) - passed} validation(s) failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
