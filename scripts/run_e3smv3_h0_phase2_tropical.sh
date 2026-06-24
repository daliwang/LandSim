#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# E3SMv3_h0 Phase 2: tropical P-focused training + tropical restart (raw 5P)
#
# Reference Trendy run: cnp_results/run_20260315_175250_phase2_tropical
#
# Usage:
#   source config/e3smv3_h0_env.sh
#   export BASE_RESTART=cnp_results/run_.../updated_restart_base.nc   # from Phase 1
#   bash scripts/run_e3smv3_h0_phase2_tropical.sh
#
# Skip training:
#   export PHASE2_RUN_DIR=cnp_results/run_..._e3smv3_h0_phase2_tropical
#   bash scripts/run_e3smv3_h0_phase2_tropical.sh
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=config/e3smv3_h0_env.sh
source config/e3smv3_h0_env.sh

if [[ -z "${BASE_RESTART:-}" ]]; then
  if [[ -n "${NATVEG_RUN_DIR:-}" ]] && [[ -f "$NATVEG_RUN_DIR/updated_restart_base.nc" ]]; then
    BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_base.nc"
  else
    _latest_phase1() {
      local d
      for d in $(ls -td cnp_results/run_*_e3smv3_h0_phase1_global 2>/dev/null); do
        [[ -d "$d" ]] && [[ -f "$d/updated_restart_base.nc" ]] && { echo "$d/updated_restart_base.nc"; return 0; }
      done
      return 1
    }
    if _latest_phase1 >/dev/null; then
      BASE_RESTART="$(_latest_phase1)"
      NATVEG_RUN_DIR="$(dirname "$BASE_RESTART")"
      echo "Using latest E3SMv3_h0 Phase 1 base restart: $BASE_RESTART"
    else
      echo "Error: BASE_RESTART must be set (from Phase 1)." >&2
      exit 1
    fi
  fi
fi

_has_model() {
  local d="$1"
  [[ -f "$d/cnp_model.pt" ]] || [[ -f "$d/cnp_predictions/model.pth" ]]
}

echo "=== E3SMv3_h0 Phase 2: tropical training + raw 5P restart ==="
echo "  BASE_RESTART=$BASE_RESTART"
echo "  CONFIG_TROPICAL=$CONFIG_TROPICAL"
echo "  INFERENCE_EXTRA_FLAGS=$INFERENCE_EXTRA_FLAGS"

if [[ -n "${PHASE2_RUN_DIR:-}" ]]; then
  if ! _has_model "$PHASE2_RUN_DIR"; then
    echo "Error: PHASE2_RUN_DIR is set but no model found at $PHASE2_RUN_DIR" >&2
    exit 1
  fi
  echo "Using existing tropical model at $PHASE2_RUN_DIR."
else
  echo "Training tropical model on E3SMv3_h0 (expect long runtime)..."
  t0=$(date +%s)
  python train_cnp_model.py \
    --training-config-json "$CONFIG_TROPICAL" \
    --variable-list "$VARIABLE_LIST" \
    --data-paths "$DATA_PATHS" \
    --file-pattern "$FILE_PATTERN" \
    --output-dir cnp_results \
    --output-dir-suffix "$PHASE2_SUFFIX"
  t1=$(date +%s)
  echo "Tropical training finished in $(( (t1 - t0) / 60 )) minutes."
  PHASE2_RUN_DIR="$(ls -td cnp_results/run_*_"${PHASE2_SUFFIX}" 2>/dev/null | head -1)"
  if [[ -z "$PHASE2_RUN_DIR" ]] || [[ ! -d "$PHASE2_RUN_DIR" ]]; then
    echo "Error: Could not find run directory (run_*_${PHASE2_SUFFIX})." >&2
    exit 1
  fi
  echo "Tropical model run: $PHASE2_RUN_DIR"
fi

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  echo "TRAIN_ONLY=1: skipping inference and restart generation."
  echo "  export PHASE2_RUN_DIR=$PHASE2_RUN_DIR"
  exit 0
fi

TROP_INF_DIR="$PHASE2_RUN_DIR/cnp_inference_tropical_only"
mkdir -p "$TROP_INF_DIR"

echo "Running tropical-only inference..."
# shellcheck disable=SC2086
python scripts/run_inference_all.py \
  --model "$PHASE2_RUN_DIR/cnp_model.pt" \
  --output-dir "$TROP_INF_DIR" \
  --variable-list "$VARIABLE_LIST" \
  $INFERENCE_EXTRA_FLAGS

TROP_NETCDF="$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc"
mkdir -p "$PHASE2_RUN_DIR/comparison_results"

echo "Converting tropical predictions to NetCDF..."
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$TROP_INF_DIR/cnp_predictions" \
  --variable-list "$VARIABLE_LIST" \
  --output "$TROP_NETCDF"

echo "Creating phase2_tropical restart (raw Phase2 5P in tropics)..."
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$TROP_NETCDF" \
  --restart-file "$BASE_RESTART" \
  --output "$PHASE2_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc" \
  --variable-list "$VARIABLE_LIST" \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"

echo "phase2_tropical E3SMv3_h0 on $(date)" > "$PHASE2_RUN_DIR/README_phase2_tropical.txt"

echo
echo "Phase 2 complete."
echo "  export PHASE2_RUN_DIR=$PHASE2_RUN_DIR"
echo "  Phase2 restart: $PHASE2_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc"
