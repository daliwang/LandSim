#!/bin/bash
# Adding PFT variables script runner

echo "=========================================="
echo "TVA Adding PFT Variables"
echo "=========================================="

# Get the project directory
PROJECT_DIR="/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation"

# Activate Python virtual environment
source ${PROJECT_DIR}/venv_py311/bin/activate

# Check if enhanced dataset exists
if [ ! -d "${PROJECT_DIR}/output/enhanced_training_dataset" ]; then
    echo "❌ Error: Enhanced dataset directory not found!"
    echo "   Please run enhanced dataset generation first:"
    echo "   ./bash_script/run_enhanced_dataset_generation.sh"
    exit 1
fi

# Check if CLM parameters file exists
if [ ! -f "${PROJECT_DIR}/clm_params.c130821.nc" ]; then
    echo "❌ Error: CLM parameters file not found!"
    echo "   Expected file: ${PROJECT_DIR}/clm_params.c130821.nc"
    echo "   Please ensure the CLM parameters file is in the correct location."
    exit 1
fi

echo "✅ Enhanced dataset directory found"
echo "✅ CLM parameters file found"

# Step 1: Add PFT variables
echo ""
echo "=========================================="
echo "Step 1: Adding PFT Variables"
echo "=========================================="

python ${PROJECT_DIR}/python_scripts/1_add_pft_to_dataset.py

echo ""
echo "✅ PFT variables addition completed!"

# Step 2: Remove unwanted variables
echo ""
echo "=========================================="
echo "Step 2: Removing Unwanted Variables"
echo "=========================================="

python ${PROJECT_DIR}/python_scripts/2_rm_variables.py

echo ""
echo "🎉 Adding PFT variables completed!"
echo "   - PFT variables added to enhanced dataset"
echo "   - Dataset ready for further processing"

echo ""
echo "Final enhanced dataset files:"
ls -la ${PROJECT_DIR}/output/enhanced_training_dataset/
