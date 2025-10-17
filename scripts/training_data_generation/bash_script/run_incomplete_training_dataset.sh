#!/bin/bash
# Incomplete training dataset generation script runner

echo "=========================================="
echo "TVA Incomplete Training Dataset Generation"
echo "=========================================="

# Get the project directory
PROJECT_DIR="/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation"

# Activate Python virtual environment
source ${PROJECT_DIR}/venv_py311/bin/activate

# Run incomplete training dataset generation script
python ${PROJECT_DIR}/python_scripts/72_dataset_construction.py

echo ""
echo "Incomplete training dataset generation completed!"

# Clean up original PKL files (keep only monthly averaged files)
echo ""
echo "Cleaning up original PKL files..."
echo "Keeping only monthly averaged files..."

# Remove original training_data_batch_*.pkl files
rm -f ${PROJECT_DIR}/output/training_dataset_pkl/training_data_batch_*.pkl

echo "✅ Original PKL files removed"
echo "✅ Only monthly averaged files retained"
echo ""
echo "🎉 Incomplete training dataset generation completed!"
echo "   - Monthly averaged PKL files created in output/training_dataset_pkl/"
echo "   - Ready for enhanced dataset generation (37_dataset.py)"
echo "   - Use run_enhanced_dataset_generation.sh to complete the workflow"
