#!/bin/bash

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

conda create -p ../amd_env python=3.12 -c conda-forge
source activate ../amd_env

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

module unload rocm
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
module load rocm/6.4.1

pip install -r requirements_amd.txt

# If you get MIOpen errors (miopenStatusInternalError, readonly database), uncomment the line below:
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=../miopen_cache
mkdir -p "$MIOPEN_USER_DB_PATH"


# HIP cache path
# export HIP_COMPILE_CACHE_DIR=/lustre/orion/csc665/world-shared/zhuowei/LandSim/hip_cache
# mkdir -p "$HIP_COMPILE_CACHE_DIR"
# chmod -R 700 "$HIP_COMPILE_CACHE_DIR"

python ../train_model.py
