#!/bin/bash
# Run all TVA forcing data extraction scripts
# This script processes 6 forcing variables: FLDS, FSDS, PSRF, QBOT, PRECTmms, TBOT

echo "=========================================="
echo "TVA Forcing Data Extraction Pipeline"
echo "=========================================="
echo "Processing 6 forcing variables (1980-1999, 20 years)"
echo "Output directory: /gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation/output/forcing_netcdf"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Activate Python virtual environment
echo "Activating Python virtual environment..."
source venv_py311/bin/activate

# Create output directory
echo "Creating output directory..."
mkdir -p output/forcing_netcdf

# Create log directory
mkdir -p logs

echo ""
echo "Starting forcing data extraction..."
echo ""

# Initialize counters
success_count=0
total_count=6

# Function to run a script and check result
run_script() {
    local script_name="$1"
    local variable="$2"
    local log_file="logs/${variable}_extraction.log"
    
    echo "=========================================="
    echo "Processing $variable forcing data"
    echo "=========================================="
    echo "Script: $script_name"
    echo "Log file: $log_file"
    echo ""
    
    # Run the script and capture output
    python "python_scripts/$script_name" > "$log_file" 2>&1
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ $variable forcing data extraction completed successfully"
        ((success_count++))
    else
        echo "✗ $variable forcing data extraction failed"
        echo "   Check log file: $log_file"
    fi
    
    echo ""
}

# Run each forcing variable extraction script
run_script "construct_TVA_FLDS_20years.py" "FLDS"
run_script "construct_TVA_FSDS_20years.py" "FSDS" 
run_script "construct_TVA_PSRF_20years.py" "PSRF"
run_script "construct_TVA_QBOT_20years.py" "QBOT"
run_script "construct_TVA_PRECTmms_20years.py" "PRECTmms"
run_script "construct_TVA_TBOT_20years.py" "TBOT"

echo "=========================================="
echo "All Tasks Completed!"
echo "=========================================="
echo "Successfully processed: $success_count/$total_count variables"

# List generated files
echo ""
echo "Generated NetCDF files:"
if [ -d "output/forcing_netcdf" ]; then
    ls -lh output/forcing_netcdf/*.nc 2>/dev/null || echo "No NetCDF files found"
else
    echo "Output directory not found"
fi

echo ""
echo "Log files:"
if [ -d "logs" ]; then
    ls -lh logs/*.log 2>/dev/null || echo "No log files found"
else
    echo "Log directory not found"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Variables processed:"
echo "  FLDS    - Longwave radiation"
echo "  FSDS    - Shortwave radiation" 
echo "  PSRF    - Surface pressure"
echo "  QBOT    - Specific humidity"
echo "  PRECTmms - Precipitation"
echo "  TBOT    - Air temperature"
echo ""
echo "Output location: $PROJECT_DIR/output/forcing_netcdf/"
echo "Log location: $PROJECT_DIR/logs/"
echo "=========================================="

# Exit with error code if any script failed
if [ $success_count -ne $total_count ]; then
    echo "WARNING: Some extractions failed. Check log files for details."
    exit 1
else
    echo "✓ All forcing data extractions completed successfully!"
    exit 0
fi


