#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# run_phase3_tworegions.sh
#
# Phase 3 for new users:
# - Full-grid inference with Phase2 model
# - Apply 5P bias/scale correction: Amazon-only, then Africa-only, then merge
#   (same workflow as run_20260313_224805_phase2_tropical_soilp_amazon_africa)
# - Create a tropical restart with bias-corrected Phase2 5P
#
# Requirements:
# - PHASE2_RUN_DIR and BASE_RESTART from previous phases
#
# Env (can be overridden):
# - VARIABLE_LIST    CNP_IO variable list file
# - TS               timestamp suffix (optional; only affects output dir name)
#
# Usage:
#   export PHASE2_RUN_DIR=...   # optional; default prefers run_*_phase2_tropical_soilp* then latest phase2_tropical
#   export BASE_RESTART=...
#   bash scripts/run_phase3_tworegions.sh
#
# For best Africa/Amazon 5P, use: PHASE2_RUN_DIR=cnp_results/run_20260313_224805_phase2_tropical_soilp_amazon_africa
# See docs/PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md for why phase3 results can differ from that run.
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# If not set, try to use latest phase2 run and phase1 base restart
if [[ -z "${PHASE2_RUN_DIR:-}" ]] || [[ -z "${BASE_RESTART:-}" ]]; then
  _has_model() {
    local d="$1"
    [[ -f "$d/cnp_model.pt" ]] || [[ -f "$d/cnp_predictions/model.pth" ]]
  }
  # Prefer phase2_tropical_soilp* (e.g. phase2_tropical_soilp_amazon_africa) for best Africa/Amazon 5P.
  _latest_phase2() {
    local d
    for d in $(ls -td cnp_results/run_*_phase2_tropical_soilp* 2>/dev/null); do
      [[ -d "$d" ]] && _has_model "$d" && { echo "$d"; return 0; }
    done
    for d in $(ls -td cnp_results/run_*_phase2_tropical* 2>/dev/null); do
      [[ -d "$d" ]] && _has_model "$d" && { echo "$d"; return 0; }
    done
    return 1
  }
  _latest_phase1_restart() {
    local d
    for d in $(ls -td cnp_results/run_*_phase1_global 2>/dev/null); do
      [[ -d "$d" ]] && [[ -f "$d/updated_restart_base.nc" ]] && { echo "$d/updated_restart_base.nc"; return 0; }
    done
    return 1
  }
  if [[ -z "${PHASE2_RUN_DIR:-}" ]] && _latest_phase2 >/dev/null; then
    PHASE2_RUN_DIR="$(_latest_phase2)"
    echo "Using latest phase 2 run: PHASE2_RUN_DIR=$PHASE2_RUN_DIR"
  fi
  if [[ -z "${BASE_RESTART:-}" ]] && _latest_phase1_restart >/dev/null; then
    BASE_RESTART="$(_latest_phase1_restart)"
    echo "Using latest phase 1 base restart: BASE_RESTART=$BASE_RESTART"
  fi
  if [[ -z "${PHASE2_RUN_DIR:-}" ]] || [[ -z "${BASE_RESTART:-}" ]]; then
    echo "Error: PHASE2_RUN_DIR and BASE_RESTART must be set (from earlier phases), or run phase 1 and phase 2 first." >&2
    echo "  export PHASE2_RUN_DIR=cnp_results/run_XXXX_phase2_tropical" >&2
    echo "  export BASE_RESTART=cnp_results/run_XXXX_phase1_global/updated_restart_base.nc" >&2
    exit 1
  fi
fi

VARIABLE_LIST="${VARIABLE_LIST:-CNP_IO_updated9_dev_dw.txt}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN3_DIR="cnp_results/run_${TS}_phase3_tworegions"
mkdir -p "$RUN3_DIR"

INF_FULL_DIR="$RUN3_DIR/cnp_inference_entire_dataset"
if [[ ! -d "$INF_FULL_DIR/cnp_predictions" ]]; then
  echo "Running full-grid Phase2 inference into $INF_FULL_DIR ..."
  python scripts/run_inference_all.py \
    --model "$PHASE2_RUN_DIR/cnp_model.pt" \
    --output-dir "$INF_FULL_DIR" \
    --variable-list "$VARIABLE_LIST" \
    --inference-full-grid
fi

# Match good-run workflow: Amazon-only correction, Africa-only correction, then merge.
echo "Applying 5P bias/scale correction (Amazon only)..."
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_amazon_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_amazon

echo "Applying 5P bias/scale correction (Africa only)..."
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_africa_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_africa

echo "Merging Amazon and Africa bias-corrected 5P..."
python scripts/merge_5p_bias_corrected_amazon_africa.py --run-dir "$RUN3_DIR"

mkdir -p "$RUN3_DIR/comparison_results"
BC_NETCDF="$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa.nc"

echo "Converting bias-corrected predictions to NetCDF..."
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$INF_FULL_DIR/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_amazon_africa \
  --variable-list "$VARIABLE_LIST" \
  --output "$BC_NETCDF"

echo "Creating phase3_tworegions restart (bias-corrected 5P in tropics)..."
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$BC_NETCDF" \
  --restart-file "$BASE_RESTART" \
  --output "$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_tropical.nc" \
  --variable-list "$VARIABLE_LIST" \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"

echo "phase3_tworegions: natveg base + Phase2 5P with Amazon-only + Africa-only bias correction merged (same workflow as good run) in tropics on $(date)" > "$RUN3_DIR/README_phase3_tworegions.txt"

echo

echo "Phase 3 complete."
echo "  RUN3_DIR       = $RUN3_DIR"
echo "  Phase3 restart = $RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_tropical.nc"

