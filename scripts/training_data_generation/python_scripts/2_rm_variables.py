import pandas as pd
import glob
import os
import sys

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import training_dataset_pkl_output_dir

# === 1. Define variables to delete ===
# Delete columns starting with SCALARAVG_vr
scalaravg_cols = [col for col in [] if col.startswith("SCALARAVG_vr")]  # This list will be dynamically generated at runtime

# Delete COL_FIRE_CLOSS and Y_COL_FIRE_CLOSS
fire_cols = ["COL_FIRE_CLOSS", "Y_COL_FIRE_CLOSS"]

# Delete unwanted PFT variables
unwanted_pft_cols = [
    "pft_aleaff", "pft_baset", "pft_cc_dstem", "pft_cc_leaf", "pft_cc_lstem", "pft_cc_other", "pft_displar", 
    "pft_fcurdv", "pft_fd_pft", "pft_fertnitro", "pft_ffrootcn", "pft_fleafcn", "pft_fm_droot", "pft_fm_dstem",
    "pft_fm_leaf", "pft_fm_lroot", "pft_fm_lstem", "pft_fm_other", "pft_fm_root", "pft_fnitr", "pft_fsr_pft",
    "pft_fstemcn", "pft_irrigated", "pft_pconv", "pft_pftpar20", "pft_pftpar28", "pft_pftpar29", "pft_pftpar30",
    "pft_pftpar31", "pft_pprod10", "pft_pprod100", "pft_pprodharv10"
]

# Combine all columns to delete
all_drop_cols = fire_cols + unwanted_pft_cols

print(f"🗑️  Variables to delete:")
print(f"   - Fire-related variables: {len(fire_cols)} variables")
print(f"   - Unwanted PFT variables: {len(unwanted_pft_cols)} variables")
print(f"   - Total: {len(all_drop_cols)} variables")

# === 2. Process all PKL files in enhanced_training_dataset directory ===
enhanced_output_dir = os.path.join(os.path.dirname(training_dataset_pkl_output_dir), "enhanced_training_dataset")
input_files = sorted(glob.glob(os.path.join(enhanced_output_dir, "*.pkl")))

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
        
        # Dynamically find columns starting with SCALARAVG_vr
        scalaravg_cols = [col for col in df.columns if col.startswith("SCALARAVG_vr")]
        if scalaravg_cols:
            print(f"🔍 Found {len(scalaravg_cols)} SCALARAVG_vr variables: {scalaravg_cols[:5]}...")
        
        # Combine all columns to delete
        drop_cols = scalaravg_cols + all_drop_cols
        
        # Check which variables actually exist
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        missing_cols = [col for col in drop_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Following variables do not exist in file: {missing_cols[:5]}...")
        
        if existing_drop_cols:
            print(f"🗑️  Preparing to delete {len(existing_drop_cols)} variables")
            
            # Delete variables
            df.drop(columns=existing_drop_cols, inplace=True, errors='ignore')
            
            new_shape = df.shape
            print(f"✅ Successfully deleted {len(existing_drop_cols)} variables")
            print(f"📐 New data shape: {original_shape} → {new_shape}")
            
            # Save in-place (overwrite original file)
            print("💾 Saving modified file...")
            df.to_pickle(file_path)
            print(f"✅ File saved: {os.path.basename(file_path)}")
            
            # Display deleted variables statistics
            print("📊 Deleted variables statistics:")
            if scalaravg_cols:
                print(f"   - SCALARAVG_vr variables: {len([col for col in scalaravg_cols if col in existing_drop_cols])} variables")
            print(f"   - Fire-related variables: {len([col for col in fire_cols if col in existing_drop_cols])} variables")
            print(f"   - Unwanted PFT variables: {len([col for col in unwanted_pft_cols if col in existing_drop_cols])} variables")
            
        else:
            print("ℹ️  No variables found to delete, skipping processing")
        
    except Exception as e:
        print(f"❌ Failed to process file: {e}")
        continue

print(f"\n{'='*80}")
print("🎉 All files processed successfully!")
print("📊 Summary:")
print(f"   - Total files: {len(input_files)}")
print(f"   - Deleted variable types:")
print(f"     * SCALARAVG_vr variables (dynamically detected)")
print(f"     * Fire-related variables: {len(fire_cols)} variables")
print(f"     * Unwanted PFT variables: {len(unwanted_pft_cols)} variables")
print("="*80)