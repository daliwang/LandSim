#!/bin/bash
# Forcing data validation script runner

echo "=========================================="
echo "TVA Forcing Data Validation"
echo "=========================================="

# Get the project directory
PROJECT_DIR="/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation"

# Activate Python virtual environment
source ${PROJECT_DIR}/venv_py311/bin/activate

# Run validation script
python ${PROJECT_DIR}/validation/forcing_netcdf_validation.py

echo ""
echo "Validation completed!"
