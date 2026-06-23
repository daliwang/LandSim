#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# run_phase1_global.sh
#
# Phase 1 for new users:
# - Train a global natveg_improved-like model (if needed)
# - Create a base restart from that model using an ELM restart template
#
# Requirements:
# - Training data configured via training_config_experiment_3_global_natveg_improved.json
# - ELM restart template (RESTART_TEMPLATE)
#
# Environment (can be overridden):
# - CONFIG_GLOBAL   training config JSON for global model
# - VARIABLE_LIST   CNP_IO variable list file
# - RESTART_TEMPLATE  path to original ELM restart .nc (required)

# e.g. /mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/
# 20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc
#
# Outputs:
# - NATVEG_RUN_DIR  (exported): global model run directory
# - BASE_RESTART    (exported): base restart .nc created from NATVEG_RUN_DIR
#
# Usage (from repo root):
#   bash scripts/run_phase1_global.sh
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_GLOBAL="${CONFIG_GLOBAL:-config/training_config_experiment_3_global_natveg_improved.json}"
VARIABLE_LIST="${VARIABLE_LIST:-CNP_IO_updated9_dev_dw.txt}"

if [[ -z "${RESTART_TEMPLATE:-}" ]] || [[ ! -f "$RESTART_TEMPLATE" ]]; then
  echo "Error: RESTART_TEMPLATE must point to an existing ELM restart .nc file." >&2
  echo "Example:" >&2
  echo "  export RESTART_TEMPLATE=/path/to/your/elm_restart_template.nc" >&2
  exit 1
fi

_has_model() {
  local d="$1"
  [[ -f "$d/cnp_model.pt" ]] || [[ -f "$d/cnp_predictions/model.pth" ]]
}

_latest_run_with_model() {
  local pattern="$1"
  for d in $(ls -td $pattern 2>/dev/null); do
    [[ -d "$d" ]] && _has_model "$d" && { echo "$d"; return 0; }
  done
  return 1
}

echo "=== Phase 1: Global model training + base restart creation ==="

if [[ -n "${NATVEG_RUN_DIR:-}" ]]; then
  if ! _has_model "$NATVEG_RUN_DIR"; then
    echo "Error: NATVEG_RUN_DIR is set but no model found at $NATVEG_RUN_DIR" >&2
    exit 1
  fi
  echo "Using existing global model at $NATVEG_RUN_DIR."
elif _latest_run_with_model "cnp_results/run_*_phase1_global" >/dev/null; then
  NATVEG_RUN_DIR="$(_latest_run_with_model "cnp_results/run_*_phase1_global")"
  echo "Reusing existing global model at $NATVEG_RUN_DIR."
else
  echo "Training global (natveg_improved-like) model (may take ~30 min)..."
  t0=$(date +%s)
  python train_cnp_model.py \
    --training-config-json "$CONFIG_GLOBAL" \
    --output-dir cnp_results \
    --output-dir-suffix phase1_global \
    --variable-list "$VARIABLE_LIST"
  t1=$(date +%s)
  echo "Global training finished in $(( (t1 - t0) / 60 )) minutes."
  NATVEG_RUN_DIR="$(ls -td cnp_results/run_*_phase1_global 2>/dev/null | head -1)"
  if [[ -z "$NATVEG_RUN_DIR" ]] || [[ ! -d "$NATVEG_RUN_DIR" ]]; then
    echo "Error: Could not find global model run directory (run_*_phase1_global)." >&2
    exit 1
  fi
  echo "Global model run: $NATVEG_RUN_DIR"
fi

INF_GLOBAL="$NATVEG_RUN_DIR/cnp_inference_entire_dataset"
mkdir -p "$INF_GLOBAL"

echo "Running full-grid inference for global model..."
python scripts/run_inference_all.py \
  --model "$NATVEG_RUN_DIR/cnp_model.pt" \
  --output-dir "$INF_GLOBAL" \
  --variable-list "$VARIABLE_LIST" \
  --inference-full-grid

echo "Converting global predictions to NetCDF..."
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$INF_GLOBAL/cnp_predictions" \
  --variable-list "$VARIABLE_LIST" \
  --output "$NATVEG_RUN_DIR/ai_predictions_global.nc"

echo "Creating base restart from template + global predictions..."
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$NATVEG_RUN_DIR/ai_predictions_global.nc" \
  --restart-file "$RESTART_TEMPLATE" \
  --output "$NATVEG_RUN_DIR/updated_restart_base.nc" \
  --variable-list "$VARIABLE_LIST" \
  "--tropical-lat-range=-90,90"

BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_base.nc"
echo "Base restart created at: $BASE_RESTART"

echo
echo "Phase 1 complete."
echo "  NATVEG_RUN_DIR = $NATVEG_RUN_DIR"
echo "  BASE_RESTART   = $BASE_RESTART"

