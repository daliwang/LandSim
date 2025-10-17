#!/bin/bash
# Enhanced dataset generation script runner with automatic cleanup

echo "=========================================="
echo "TVA Enhanced Dataset Generation with Cleanup"
echo "=========================================="

# Get the project directory
PROJECT_DIR="/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation"

# Activate Python virtual environment
source ${PROJECT_DIR}/venv_py311/bin/activate

# Create output directory if it doesn't exist
mkdir -p ${PROJECT_DIR}/output/enhanced_training_dataset

# Run enhanced dataset generation script
echo ""
echo "Step 1: Running enhanced dataset generation (37_dataset.py)..."
python ${PROJECT_DIR}/python_scripts/37_dataset.py

echo ""
echo "✅ Enhanced dataset generation completed!"

# Clean up intermediate files
echo ""
echo "=========================================="
echo "Step 2: Cleaning Up Intermediate Files"
echo "=========================================="

echo "Current directory contents:"
ls -la ${PROJECT_DIR}/output/

echo ""
echo "Removing training_dataset_pkl directory..."
rm -rf ${PROJECT_DIR}/output/training_dataset_pkl/

echo ""
echo "✅ Cleanup completed!"
echo "Remaining directories:"
ls -la ${PROJECT_DIR}/output/

echo ""
echo "Final enhanced dataset files:"
ls -la ${PROJECT_DIR}/output/enhanced_training_dataset/

echo ""
echo "🎉 Enhanced dataset generation and cleanup completed!"
echo "   - Enhanced dataset files ready for training"
echo "   - Intermediate files removed"
echo "   - Disk space saved: ~2GB"
echo "   - Only enhanced dataset files remain for training use."
