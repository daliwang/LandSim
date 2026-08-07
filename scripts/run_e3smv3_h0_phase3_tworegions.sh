#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# E3SMv3_h0 Phase 3: two-region bias correction (Amazon + Africa, regional_fit_v3)
#
# Seeds inference from Phase 1 global grid (correct E3SM lon/lat), overlays Phase 2
# tropical 5P predictions by coordinate, then Amazon v3 + Africa v3 + merge → restart.
# Does NOT re-run full-grid inference (avoids lat/lon mislabelling on tropical model).
#
# Usage:
#   source config/e3smv3_h0_env.sh
#   export PHASE1_RUN_DIR=.../run_*_e3smv3_h0_phase1_global
#   export PHASE2_RUN_DIR=.../run_*_e3smv3_h0_phase2_tropical
#   export BASE_RESTART=.../updated_restart_base.nc
#   bash scripts/run_e3smv3_h0_phase3_tworegions.sh
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=config/e3smv3_h0_env.sh
source config/e3smv3_h0_env.sh

_has_model() {
  local d="$1"
  [[ -f "$d/cnp_model.pt" ]] || [[ -f "$d/cnp_predictions/model.pth" ]]
}

_has_phase1_inference() {
  local d="$1"
  [[ -d "$d/cnp_inference_entire_dataset/cnp_predictions/soil_2d_ground_truth" ]]
}

if [[ -z "${PHASE1_RUN_DIR:-}" ]]; then
  for d in $(ls -td cnp_results/run_*_e3smv3_h0_phase1_global 2>/dev/null); do
    if [[ -d "$d" ]] && _has_phase1_inference "$d"; then
      PHASE1_RUN_DIR="$d"
      echo "Using latest E3SMv3_h0 Phase 1 run: $PHASE1_RUN_DIR"
      break
    fi
  done
fi

if [[ -z "${PHASE2_RUN_DIR:-}" ]] || [[ -z "${BASE_RESTART:-}" ]]; then
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
fi

if [[ -z "${PHASE1_RUN_DIR:-}" ]] || [[ -z "${PHASE2_RUN_DIR:-}" ]] || [[ -z "${BASE_RESTART:-}" ]]; then
  echo "Error: PHASE1_RUN_DIR, PHASE2_RUN_DIR, and BASE_RESTART must be set." >&2
  exit 1
fi

if ! _has_phase1_inference "$PHASE1_RUN_DIR"; then
  echo "Error: Phase 1 inference not found under $PHASE1_RUN_DIR/cnp_inference_entire_dataset" >&2
  exit 1
fi

PHASE2_INF="$PHASE2_RUN_DIR/cnp_inference_tropical_only/cnp_predictions"
if [[ ! -d "$PHASE2_INF/soil_2d_predictions" ]]; then
  echo "Error: Phase 2 tropical inference not found: $PHASE2_INF" >&2
  exit 1
fi

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN3_DIR="${RUN3_DIR:-cnp_results/run_${TS}_${PHASE3_SUFFIX}}"
mkdir -p "$RUN3_DIR/analysis" "$RUN3_DIR/comparison_results"

AMAZON_SUBDIR="soil_2d_predictions_5P_bias_corrected_amazon_v3"
AFRICA_SUBDIR="soil_2d_predictions_5P_bias_corrected_africa_v3"
MERGED_SUBDIR="soil_2d_predictions_5P_bias_corrected_amazon_africa_v3"
MERGED_SUFFIX="_bias_corrected_amazon_africa_v3"

echo "=== E3SMv3_h0 Phase 3: two-region v3 bias correction + restart ==="
echo "  PHASE1_RUN_DIR=$PHASE1_RUN_DIR  (inference seed)"
echo "  PHASE2_RUN_DIR=$PHASE2_RUN_DIR  (5P overlay source)"
echo "  BASE_RESTART=$BASE_RESTART"
echo "  TROPICAL_LAT_RANGE=$TROPICAL_LAT_RANGE"
echo "  AMAZON_REGION_CONFIG=$AMAZON_REGION_CONFIG"
echo "  AFRICA_REGION_CONFIG=$AFRICA_REGION_CONFIG"
echo "  RUN3_DIR=$RUN3_DIR"

INF_FULL_DIR="$RUN3_DIR/cnp_inference_entire_dataset"
PRED_ROOT="$INF_FULL_DIR/cnp_predictions"
OVERLAY_STAMP="$PRED_ROOT/.phase2_5p_overlay_done"

if [[ ! -d "$PRED_ROOT/soil_2d_ground_truth" ]]; then
  echo "Seeding inference from Phase 1 global grid (symlink)..."
  ln -sfn "$(readlink -f "$PHASE1_RUN_DIR/cnp_inference_entire_dataset")" "$INF_FULL_DIR"
fi

if [[ ! -f "$OVERLAY_STAMP" ]]; then
  echo "Overlaying Phase 2 tropical 5P predictions onto global grid by (lon, lat)..."
  python scripts/overlay_5p_by_coords.py \
    --base-run-dir "$RUN3_DIR" \
    --source-predictions "$PHASE2_INF"
  date -Iseconds > "$OVERLAY_STAMP"
else
  echo "Phase 2 5P overlay already done ($OVERLAY_STAMP); skipping."
fi

echo "Applying 5P bias/scale correction (Amazon, regional_fit_v3)..."
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json "$AMAZON_REGION_CONFIG" \
  --output-subdir "$AMAZON_SUBDIR" \
  --regional-fit-v3

echo "Applying 5P bias/scale correction (Africa, regional_fit_v3)..."
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json "$AFRICA_REGION_CONFIG" \
  --output-subdir "$AFRICA_SUBDIR" \
  --regional-fit-v3

echo "Merging Amazon and Africa bias-corrected 5P..."
python scripts/merge_5p_bias_corrected_amazon_africa.py \
  --run-dir "$RUN3_DIR" \
  --amazon-subdir "$AMAZON_SUBDIR" \
  --africa-subdir "$AFRICA_SUBDIR" \
  --output-subdir "$MERGED_SUBDIR" \
  --output-filename-suffix "$MERGED_SUFFIX" \
  --amazon-box "$AMAZON_BOX" \
  --africa-box "$AFRICA_BOX"

BC_NETCDF="$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa_v3.nc"
RESTART_OUT="$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc"

echo "Converting bias-corrected predictions to NetCDF..."
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$PRED_ROOT/" \
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
  "--tropical-lat-range=${TROPICAL_LAT_RANGE}"

cat > "$RUN3_DIR/README_phase3_tworegions.txt" <<EOF
E3SMv3_h0 Phase 3 ($(date -Iseconds))

Inference seed: $PHASE1_RUN_DIR/cnp_inference_entire_dataset (symlink)
5P overlay:     $PHASE2_RUN_DIR/cnp_inference_tropical_only (by lon/lat)
Bias correction: regional_fit_v3 Amazon + Africa
Restart base:   $BASE_RESTART
EOF

echo
echo "Phase 3 complete."
echo "  export RUN3_DIR=$RUN3_DIR"
echo "  Phase3 restart: $RESTART_OUT"
