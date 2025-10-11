#!/bin/bash
# PyTorch Profiler execution script
# Based on setup_and_train_amd.sh environment configuration

echo "=== PyTorch Profiler Performance Analysis Script ==="

# Load necessary modules
echo "Loading modules..."
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

# Activate conda environment
echo "Activating conda environment..."
source activate ../amd_env

# Set ROCm/HIP cache environment variables (avoid cache errors)
echo "Setting ROCm environment variables..."
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=../miopen_cache
mkdir -p "$MIOPEN_USER_DB_PATH"
chmod 700 "$MIOPEN_USER_DB_PATH"

export HIP_COMPILE_CACHE_DIR=../hip_cache
mkdir -p "$HIP_COMPILE_CACHE_DIR"
chmod 700 "$HIP_COMPILE_CACHE_DIR"

export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Switch to project directory
cd ../LandSim

echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA/ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run PyTorch Profiler
echo ""
echo "Starting PyTorch Profiler analysis..."
echo "=================================="

python profile_train_with_pytorch_profiler.py

echo ""
echo "=================================="
echo "PyTorch Profiler analysis completed!"
echo "Check results directory: pytorch_profiler_logs/"
echo "Main files:"
echo "  - train_model_trace.json (Chrome timeline)"
echo "  - train_model_stats.txt (Performance statistics)"
echo "  - train_model_memory_stats.txt (Memory statistics, if GPU available)"
