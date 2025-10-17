#!/bin/bash
# Forcing PKL validation script runner

echo "=========================================="
echo "TVA Forcing PKL Validation"
echo "=========================================="

# Get the project directory
PROJECT_DIR="/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation"

# Activate Python virtual environment
source ${PROJECT_DIR}/venv_py311/bin/activate

# Run forcing PKL validation script
python ${PROJECT_DIR}/validation/forcing_pkl_validation.py

echo ""
echo "Forcing PKL validation completed!"
