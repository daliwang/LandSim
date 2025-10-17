#!/bin/bash
# Forcing data PKL generation script runner

echo "=========================================="
echo "TVA Forcing Data PKL Generation"
echo "=========================================="

# Get the project directory
PROJECT_DIR="/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation"

# Activate Python virtual environment
source ${PROJECT_DIR}/venv_py311/bin/activate

# Run forcing PKL generation script
python ${PROJECT_DIR}/python_scripts/72_dataset_forcing_only.py

echo ""
echo "Forcing PKL generation completed!"
