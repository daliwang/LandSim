#!/usr/bin/env bash
# Shared environment for E3SMv3_h0 phase-1/2/3 workflow.
# Source from repo root:
#   source config/e3smv3_h0_env.sh
#
# Before Phase 1 restart generation, set RESTART_TEMPLATE to your ELM restart file.

: "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# --- Training data (E3SMv3_h0) ---
export E3SM_DATA_DIR="/mnt/proj-shared/AI4BGC_7xw/TrainingData/E3SMv3_h0"
export VARIABLE_LIST="${VARIABLE_LIST:-CNP_IO_e3smv3_h0.txt}"
export DATA_PATHS="${DATA_PATHS:-$E3SM_DATA_DIR}"
export FILE_PATTERN="${FILE_PATTERN:-training_data_batch_*.pkl}"

# --- Training configs (same hyperparameters as Trendy reference runs) ---
# Phase 1 reference: cnp_results/run_20260315_113900_phase1_global
export CONFIG_GLOBAL="${CONFIG_GLOBAL:-config/training_config_experiment_3_global_natveg_improved.json}"
# Phase 2 reference: cnp_results/run_20260315_175250_phase2_tropical
export CONFIG_TROPICAL="${CONFIG_TROPICAL:-config/training_config_phase2_tropical_soilp_only.json}"

# --- ELM restart template (Phase 1 base restart) ---
export RESTART_TEMPLATE="${RESTART_TEMPLATE:-/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/E3SMV3_025/20240214.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.adsp.elm.r.0021-01-01-00000.nc}"

# --- Run directory suffixes (identify E3SMv3_h0 runs in cnp_results/) ---
export PHASE1_SUFFIX="${PHASE1_SUFFIX:-e3smv3_h0_phase1_global}"
export PHASE2_SUFFIX="${PHASE2_SUFFIX:-e3smv3_h0_phase2_tropical}"
export PHASE3_SUFFIX="${PHASE3_SUFFIX:-e3smv3_h0_phase3_tworegions}"

# --- Optional: set these after each phase completes ---
# export NATVEG_RUN_DIR=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase1_global
# export PHASE1_RUN_DIR=$NATVEG_RUN_DIR
# export BASE_RESTART=$NATVEG_RUN_DIR/updated_restart_base.nc
# export PHASE2_RUN_DIR=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase2_tropical
# export RUN3_DIR=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase3_tworegions

# --- Training options (H200-friendly; global default in train_cnp_model.py remains 128) ---
export TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-1024}"
# Parallel pickle reads for ~396 E3SM batch files (DataLoaderIndividual)
export LOAD_WORKERS="${LOAD_WORKERS:-8}"

# Preprocessed tensor cache (opt-in): skips ~20 min preprocess+normalize on reruns
export USE_PREPROCESSED_CACHE="${USE_PREPROCESSED_CACHE:-0}"
export PREPROCESSED_CACHE_DIR="${PREPROCESSED_CACHE_DIR:-$E3SM_DATA_DIR/.preprocessed_cache}"
# export REBUILD_PREPROCESSED_CACHE=1   # force cache rebuild on next training run

# --- Inference options ---
# Phase 1 global: derive N/P from C; batched inference (~396k E3SM gridcells)
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4096}"
export PHASE1_INFERENCE_EXTRA_FLAGS="${PHASE1_INFERENCE_EXTRA_FLAGS:---derive-np-from-c --inference-batch-size ${INFERENCE_BATCH_SIZE}}"
# Phase 2/3 (see docs/RERUN_PHASE2_PHASE3_NO_CNP_FIX.md); batched to avoid GPU OOM on E3SM grid
export INFERENCE_EXTRA_FLAGS="${INFERENCE_EXTRA_FLAGS:---no-derive-np-from-c --inference-batch-size ${INFERENCE_BATCH_SIZE}}"

_check_restart_template() {
  if [[ ! -f "$RESTART_TEMPLATE" ]]; then
    echo "WARNING: RESTART_TEMPLATE is not set to an existing file:" >&2
    echo "  $RESTART_TEMPLATE" >&2
    echo "  Update RESTART_TEMPLATE before running Phase 1 restart generation." >&2
    return 1
  fi
  return 0
}
