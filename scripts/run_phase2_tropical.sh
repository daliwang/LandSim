#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# run_phase2_tropical.sh
#
# Phase 2 for new users:
# - Train a tropical Phase2 P-focused model (if needed)
# - Create a tropical restart with raw Phase2 5P overwriting base restart
#
# Requirements:
# - NATVEG_RUN_DIR and BASE_RESTART from phase 1
#
# Env (can be overridden):
# - CONFIG_TROPICAL   training config JSON for tropical model
# - VARIABLE_LIST     CNP_IO variable list file
#
# Usage:
#   export NATVEG_RUN_DIR=...   # from run_phase1_global.sh
#   export BASE_RESTART=...     # from run_phase1_global.sh
#   bash scripts/run_phase2_tropical.sh
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# If not set, try to use latest phase1 run and its base restart
if [[ -z "${NATVEG_RUN_DIR:-}" ]] || [[ -z "${BASE_RESTART:-}" ]]; then
  _latest_phase1() {
    local d
    for d in $(ls -td cnp_results/run_*_phase1_global 2>/dev/null); do
      [[ -d "$d" ]] && [[ -f "$d/updated_restart_base.nc" ]] && { echo "$d"; return 0; }
    done
    return 1
  }
  if _latest_phase1 >/dev/null; then
    NATVEG_RUN_DIR="$(_latest_phase1)"
    BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_base.nc"
    echo "Using phase 1 outputs: NATVEG_RUN_DIR=$NATVEG_RUN_DIR, BASE_RESTART=$BASE_RESTART"
  else
    echo "Error: NATVEG_RUN_DIR and BASE_RESTART must be set (from phase 1), or run phase 1 first." >&2
    echo "  export NATVEG_RUN_DIR=cnp_results/run_XXXX_phase1_global" >&2
    echo "  export BASE_RESTART=\$NATVEG_RUN_DIR/updated_restart_base.nc" >&2
    exit 1
  fi
fi

CONFIG_TROPICAL="${CONFIG_TROPICAL:-config/training_config_phase2_tropical_soilp_only.json}"
VARIABLE_LIST="${VARIABLE_LIST:-CNP_IO_updated9_dev_dw.txt}"

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

echo "=== Phase 2: Tropical model training + tropical restart (raw Phase2 5P) ==="

if [[ -n "${PHASE2_RUN_DIR:-}" ]]; then
  if ! _has_model "$PHASE2_RUN_DIR"; then
    echo "Error: PHASE2_RUN_DIR is set but no model found at $PHASE2_RUN_DIR" >&2
    exit 1
  fi
  echo "Using existing tropical model at $PHASE2_RUN_DIR."
elif _latest_run_with_model "cnp_results/run_*_phase2_tropical" >/dev/null; then
  PHASE2_RUN_DIR="$(_latest_run_with_model "cnp_results/run_*_phase2_tropical")"
  echo "Reusing existing tropical model at $PHASE2_RUN_DIR."
else
  echo "Training tropical (phase2 P-focused) model (may take ~30 min)..."
  t0=$(date +%s)
  python train_cnp_model.py \
    --training-config-json "$CONFIG_TROPICAL" \
    --output-dir cnp_results \
    --output-dir-suffix phase2_tropical \
    --variable-list "$VARIABLE_LIST"
  t1=$(date +%s)
  echo "Tropical training finished in $(( (t1 - t0) / 60 )) minutes."
  PHASE2_RUN_DIR="$(ls -td cnp_results/run_*_phase2_tropical 2>/dev/null | head -1)"
  if [[ -z "$PHASE2_RUN_DIR" ]] || [[ ! -d "$PHASE2_RUN_DIR" ]]; then
    echo "Error: Could not find tropical model run directory (run_*_phase2_tropical)." >&2
    exit 1
  fi
  echo "Tropical model run: $PHASE2_RUN_DIR"
fi

TROP_INF_DIR="$PHASE2_RUN_DIR/cnp_inference_tropical_only"
if [[ ! -d "$TROP_INF_DIR/cnp_predictions" ]]; then
  echo "Running tropical-only inference..."
  python scripts/run_inference_all.py \
    --model "$PHASE2_RUN_DIR/cnp_model.pt" \
    --output-dir "$TROP_INF_DIR" \
    --variable-list "$VARIABLE_LIST"
fi

TROP_NETCDF="$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc"
mkdir -p "$PHASE2_RUN_DIR/comparison_results"
if [[ ! -f "$TROP_NETCDF" ]]; then
  echo "Converting tropical predictions to NetCDF..."
  python scripts/ai_predictions_to_netcdf.py \
    --ai-predictions "$TROP_INF_DIR/cnp_predictions" \
    --variable-list "$VARIABLE_LIST" \
    --output "$TROP_NETCDF"
fi

echo "Creating phase2_tropical restart (raw Phase2 5P in tropics)..."
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$TROP_NETCDF" \
  --restart-file "$BASE_RESTART" \
  --output "$PHASE2_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc" \
  --variable-list "$VARIABLE_LIST" \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"

echo "phase2_tropical: natveg base + raw Phase2 5P in tropics on $(date)" > "$PHASE2_RUN_DIR/README_phase2_tropical.txt"

echo

echo "Phase 2 complete."
echo "  PHASE2_RUN_DIR = $PHASE2_RUN_DIR"
echo "  Tropical restart: $PHASE2_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc"

