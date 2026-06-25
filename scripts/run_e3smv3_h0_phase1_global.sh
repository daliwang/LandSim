#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# E3SMv3_h0 Phase 1: global natveg training + full-grid inference + base restart
#
# Reference Trendy run: cnp_results/run_20260315_113900_phase1_global
#
# Usage (from repo root):
#   source config/e3smv3_h0_env.sh
#   source config/e3smv3_h0_env.sh   # sets RESTART_TEMPLATE to E3SMV3_025 ELM restart
#   bash scripts/run_e3smv3_h0_phase1_global.sh
#
# Skip training if model already exists:
#   export NATVEG_RUN_DIR=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase1_global
#   bash scripts/run_e3smv3_h0_phase1_global.sh
#
# Run only training (no inference/restart):
#   TRAIN_ONLY=1 bash scripts/run_e3smv3_h0_phase1_global.sh
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=config/e3smv3_h0_env.sh
source config/e3smv3_h0_env.sh

_has_model() {
  local d="$1"
  [[ -f "$d/cnp_model.pt" ]] || [[ -f "$d/cnp_predictions/model.pth" ]]
}

echo "=== E3SMv3_h0 Phase 1: global training + base restart ==="
echo "  DATA_PATHS=$DATA_PATHS"
echo "  VARIABLE_LIST=$VARIABLE_LIST"
echo "  CONFIG_GLOBAL=$CONFIG_GLOBAL"
echo "  TRAINING_BATCH_SIZE=$TRAINING_BATCH_SIZE"
echo "  USE_PREPROCESSED_CACHE=$USE_PREPROCESSED_CACHE"
echo "  PREPROCESSED_CACHE_DIR=$PREPROCESSED_CACHE_DIR"
echo "  PHASE1_INFERENCE_EXTRA_FLAGS=$PHASE1_INFERENCE_EXTRA_FLAGS"

if [[ -n "${NATVEG_RUN_DIR:-}" ]]; then
  if ! _has_model "$NATVEG_RUN_DIR"; then
    echo "Error: NATVEG_RUN_DIR is set but no model found at $NATVEG_RUN_DIR" >&2
    exit 1
  fi
  echo "Using existing global model at $NATVEG_RUN_DIR."
else
  echo "Training global model on E3SMv3_h0 (396 batches; expect long runtime)..."
  t0=$(date +%s)
  python train_cnp_model.py \
    --training-config-json "$CONFIG_GLOBAL" \
    --variable-list "$VARIABLE_LIST" \
    --data-paths "$DATA_PATHS" \
    --file-pattern "$FILE_PATTERN" \
    --batch-size "${TRAINING_BATCH_SIZE}" \
    --output-dir cnp_results \
    --output-dir-suffix "$PHASE1_SUFFIX"
  t1=$(date +%s)
  echo "Global training finished in $(( (t1 - t0) / 60 )) minutes."
  NATVEG_RUN_DIR="$(ls -td cnp_results/run_*_"${PHASE1_SUFFIX}" 2>/dev/null | head -1)"
  if [[ -z "$NATVEG_RUN_DIR" ]] || [[ ! -d "$NATVEG_RUN_DIR" ]]; then
    echo "Error: Could not find run directory (run_*_${PHASE1_SUFFIX})." >&2
    exit 1
  fi
  echo "Global model run: $NATVEG_RUN_DIR"
fi

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  echo "TRAIN_ONLY=1: skipping inference and restart generation."
  echo "  NATVEG_RUN_DIR = $NATVEG_RUN_DIR"
  exit 0
fi

if ! _check_restart_template; then
  echo "Set RESTART_TEMPLATE and re-run, or run inference/restart steps manually (see docs/WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md)." >&2
  exit 1
fi

INF_GLOBAL="$NATVEG_RUN_DIR/cnp_inference_entire_dataset"
mkdir -p "$INF_GLOBAL"

echo "Running full-grid inference..."
# shellcheck disable=SC2086
python scripts/run_inference_all.py \
  --model "$NATVEG_RUN_DIR/cnp_model.pt" \
  --output-dir "$INF_GLOBAL" \
  --variable-list "$VARIABLE_LIST" \
  --inference-full-grid \
  $PHASE1_INFERENCE_EXTRA_FLAGS

echo "Converting global predictions to NetCDF..."
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$INF_GLOBAL/cnp_predictions" \
  --variable-list "$VARIABLE_LIST" \
  --output "$NATVEG_RUN_DIR/ai_predictions_global.nc"

echo "Creating base restart from ELM template..."
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$NATVEG_RUN_DIR/ai_predictions_global.nc" \
  --restart-file "$RESTART_TEMPLATE" \
  --output "$NATVEG_RUN_DIR/updated_restart_base.nc" \
  --variable-list "$VARIABLE_LIST" \
  "--tropical-lat-range=-90,90"

BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_base.nc"
echo "phase1_global E3SMv3_h0 on $(date)" > "$NATVEG_RUN_DIR/README_phase1_global.txt"

echo
echo "Phase 1 complete."
echo "  export NATVEG_RUN_DIR=$NATVEG_RUN_DIR"
echo "  export BASE_RESTART=$BASE_RESTART"
