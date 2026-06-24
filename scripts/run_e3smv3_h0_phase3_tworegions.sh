#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# E3SMv3_h0 Phase 3: two-region bias correction (Amazon + Africa, regional_fit_v3)
#
# Reference Trendy run: cnp_results/run_20260512_083650_phase3_tworegions_(africa_improvement)
# Workflow: full-grid inference → Amazon v3 → Africa v3 → merge → global restart
#
# Usage:
#   source config/e3smv3_h0_env.sh
#   export BASE_RESTART=.../updated_restart_base.nc
#   export PHASE2_RUN_DIR=.../run_*_e3smv3_h0_phase2_tropical
#   bash scripts/run_e3smv3_h0_phase3_tworegions.sh
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=config/e3smv3_h0_env.sh
source config/e3smv3_h0_env.sh

if [[ -z "${PHASE2_RUN_DIR:-}" ]] || [[ -z "${BASE_RESTART:-}" ]]; then
  _has_model() {
    local d="$1"
    [[ -f "$d/cnp_model.pt" ]] || [[ -f "$d/cnp_predictions/model.pth" ]]
  }
  if [[ -z "${PHASE2_RUN_DIR:-}" ]]; then
    for d in $(ls -td cnp_results/run_*_e3smv3_h0_phase2_tropical 2>/dev/null); do
      if [[ -d "$d" ]] && _has_model "$d"; then
        PHASE2_RUN_DIR="$d"
        echo "Using latest E3SMv3_h0 Phase 2 run: $PHASE2_RUN_DIR"
        break
      fi
    done
  fi
  if [[ -z "${BASE_RESTART:-}" ]]; then
    for d in $(ls -td cnp_results/run_*_e3smv3_h0_phase1_global 2>/dev/null); do
      if [[ -d "$d" ]] && [[ -f "$d/updated_restart_base.nc" ]]; then
        BASE_RESTART="$d/updated_restart_base.nc"
        echo "Using latest E3SMv3_h0 Phase 1 base restart: $BASE_RESTART"
        break
      fi
    done
  fi
  if [[ -z "${PHASE2_RUN_DIR:-}" ]] || [[ -z "${BASE_RESTART:-}" ]]; then
    echo "Error: PHASE2_RUN_DIR and BASE_RESTART must be set." >&2
    exit 1
  fi
fi

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN3_DIR="${RUN3_DIR:-cnp_results/run_${TS}_${PHASE3_SUFFIX}}"
mkdir -p "$RUN3_DIR/analysis" "$RUN3_DIR/comparison_results"

AMAZON_SUBDIR="soil_2d_predictions_5P_bias_corrected_amazon_v3"
AFRICA_SUBDIR="soil_2d_predictions_5P_bias_corrected_africa_v3"
MERGED_SUBDIR="soil_2d_predictions_5P_bias_corrected_amazon_africa_v3"
MERGED_SUFFIX="_bias_corrected_amazon_africa_v3"

echo "=== E3SMv3_h0 Phase 3: two-region v3 bias correction + restart ==="
echo "  PHASE2_RUN_DIR=$PHASE2_RUN_DIR"
echo "  BASE_RESTART=$BASE_RESTART"
echo "  RUN3_DIR=$RUN3_DIR"
echo "  INFERENCE_EXTRA_FLAGS=$INFERENCE_EXTRA_FLAGS"

INF_FULL_DIR="$RUN3_DIR/cnp_inference_entire_dataset"
if [[ ! -d "$INF_FULL_DIR/cnp_predictions" ]]; then
  echo "Running full-grid Phase2 inference (long step)..."
  # shellcheck disable=SC2086
  python scripts/run_inference_all.py \
    --model "$PHASE2_RUN_DIR/cnp_model.pt" \
    --output-dir "$INF_FULL_DIR" \
    --variable-list "$VARIABLE_LIST" \
    --inference-full-grid \
    $INFERENCE_EXTRA_FLAGS
fi

echo "Applying 5P bias/scale correction (Amazon, regional_fit_v3)..."
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_amazon_5p_box.json \
  --output-subdir "$AMAZON_SUBDIR" \
  --regional-fit-v3

echo "Applying 5P bias/scale correction (Africa, regional_fit_v3)..."
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_africa_5p_box.json \
  --output-subdir "$AFRICA_SUBDIR" \
  --regional-fit-v3

echo "Merging Amazon and Africa bias-corrected 5P..."
python scripts/merge_5p_bias_corrected_amazon_africa.py \
  --run-dir "$RUN3_DIR" \
  --amazon-subdir "$AMAZON_SUBDIR" \
  --africa-subdir "$AFRICA_SUBDIR" \
  --output-subdir "$MERGED_SUBDIR" \
  --output-filename-suffix "$MERGED_SUFFIX"

BC_NETCDF="$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa_v3.nc"
RESTART_OUT="$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc"

echo "Converting bias-corrected predictions to NetCDF..."
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$INF_FULL_DIR/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir "$MERGED_SUBDIR" \
  --variable-list "$VARIABLE_LIST" \
  --output "$BC_NETCDF"

echo "Creating phase3_tworegions restart..."
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$BC_NETCDF" \
  --restart-file "$BASE_RESTART" \
  --output "$RESTART_OUT" \
  --variable-list "$VARIABLE_LIST" \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"

echo "phase3_tworegions E3SMv3_h0: regional_fit_v3 Amazon+Africa on $(date)" \
  > "$RUN3_DIR/README_phase3_tworegions.txt"

echo
echo "Phase 3 complete."
echo "  export RUN3_DIR=$RUN3_DIR"
echo "  Phase3 restart: $RESTART_OUT"
