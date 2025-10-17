import netCDF4 as nc
import numpy as np
import pandas as pd
import glob
import os
import sys

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clm_params_nc_path, training_dataset_pkl_output_dir

# === 1. Read PFT vectors for all variables from NetCDF ===
print("Reading CLM parameters NetCDF file...")
print(f"File path: {clm_params_nc_path}")

if not os.path.exists(clm_params_nc_path):
    print(f"❌ Error: CLM parameters file not found: {clm_params_nc_path}")
    sys.exit(1)

ds = nc.Dataset(clm_params_nc_path)

target_vars = [
    "aleaff", "allconsl", "allconss", "arootf", "arooti", "astemf", "baset", "bfact", "c3psn", "cc_dstem",
    "cc_leaf", "cc_lstem", "cc_other", "croot_stem", "crop", "deadwdcn", "declfact", "displar", "dleaf", "dsladlai",
    "evergreen", "fcur", "fcurdv", "fd_pft", "fertnitro", "ffrootcn", "fleafcn", "fleafi", "flivewd", "flnr",
    "fm_droot", "fm_dstem", "fm_leaf", "fm_lroot", "fm_lstem", "fm_other", "fm_root", "fnitr", "fr_fcel", "fr_flab",
    "fr_flig", "froot_leaf", "frootcn", "fsr_pft", "fstemcn", "gddmin", "graincn", "grnfill", "grperc", "grpnow",
    "hybgdd", "irrigated", "laimx", "leaf_long", "leafcn", "lf_fcel", "lf_flab", "lf_flig", "lfemerg", "lflitcn",
    "livewdcn", "mxtmp", "pconv", "pftpar20", "pftpar28", "pftpar29", "pftpar30", "pftpar31", "planting_temp",
    "pprod10", "pprod100", "pprodharv10", "rholnir", "rholvis", "rhosnir", "rhosvis", "roota_par", "rootb_par",
    "rootprof_beta", "season_decid", "slatop", "smpsc", "smpso", "stem_leaf", "stress_decid", "taulnir", "taulvis",
    "tausnir", "tausvis", "woody", "xl", "z0mr", "ztopmx"
]

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

print(f"\n✅ Successfully loaded {len(broadcast_feature_dict)} PFT variables from NetCDF")

# === 2. Process all PKL files in enhanced_training_dataset directory ===
enhanced_output_dir = os.path.join(os.path.dirname(training_dataset_pkl_output_dir), "enhanced_training_dataset")
input_files = sorted(glob.glob(os.path.join(enhanced_output_dir, "enhanced_monthly_training_data_batch_*.pkl")))

print(f"\n🔍 Found {len(input_files)} PKL files to process")
print("📋 File list:")
for i, file in enumerate(input_files, 1):
    print(f"   {i:2d}. {os.path.basename(file)}")

if len(input_files) == 0:
    print("❌ No enhanced PKL files found. Please run enhanced dataset generation first.")
    sys.exit(1)

# === 3. Process each PKL file individually ===
for i, file_path in enumerate(input_files, 1):
    print(f"\n{'='*80}")
    print(f"Processing file {i}/{len(input_files)}: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    try:
        # Read PKL file
        print("📖 Reading PKL file...")
        df = pd.read_pickle(file_path)
        original_shape = df.shape
        print(f"✅ File loaded successfully, original shape: {original_shape}")
        
        # Check if PFT variables already exist
        existing_pft_cols = [col for col in df.columns if col.startswith("pft_")]
        if existing_pft_cols:
            print(f"⚠️  File already contains {len(existing_pft_cols)} PFT variables, skipping addition")
            print(f"   Existing PFT variables: {existing_pft_cols[:5]}...")
            continue
        
        # Add each variable as a vector column with pft_ prefix
        print("🔧 Adding PFT variables...")
        for var, val_list in broadcast_feature_dict.items():
            df["pft_" + var] = [val_list] * len(df)  # Add the same list to each row
        
        new_shape = df.shape
        print(f"✅ Successfully added {len(broadcast_feature_dict)} PFT variables")
        print(f"📐 New data shape: {original_shape} → {new_shape}")
        
        # Save in-place (overwrite original file)
        print("💾 Saving modified file...")
        df.to_pickle(file_path)
        print(f"✅ File saved: {os.path.basename(file_path)}")
        
        # Display sample PFT variables
        pft_cols = [col for col in df.columns if col.startswith("pft_")]
        if pft_cols:
            print("🧾 Sample PFT variables:")
            for col in pft_cols[:3]:
                print(f"   {col}: {df[col].iloc[0]}")
        
    except Exception as e:
        print(f"❌ Failed to process file: {e}")
        continue

print(f"\n{'='*80}")
print("🎉 All files processed successfully!")
print("📊 Summary:")
print(f"   - Total files: {len(input_files)}")
print(f"   - PFT variables added: {len(broadcast_feature_dict)}")
print("="*80)