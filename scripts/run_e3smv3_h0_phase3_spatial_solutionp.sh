#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# E3SMv3_h0 Phase 3: spatial learned-k correction for solutionp_vr + restart
#
# Seeds inference from Phase 1 global grid, overlays Phase 2 tropical 5P,
# applies fitted spatial learned-k to solutionp in Amazon/Africa boxes, then
# writes tropical 5P restart on top of Phase 1 base restart.
#
# Usage:
#   source config/e3smv3_h0_env.sh
#   export PHASE1_RUN_DIR=.../run_*_e3smv3_h0_phase1_global
#   export PHASE2_RUN_DIR=.../run_*_e3smv3_h0_phase2_tropical
#   export BASE_RESTART=.../updated_restart_base.nc
#   bash scripts/run_e3smv3_h0_phase3_spatial_solutionp.sh
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

SPATIAL_PARAMS="${SPATIAL_PARAMS:-${PHASE2_RUN_DIR}/analysis/solutionp_spatial_phase3_learned_params.json}"
if [[ ! -f "$SPATIAL_PARAMS" ]]; then
  echo "Spatial params not found at $SPATIAL_PARAMS — fitting on Phase 2 tropical inference..."
  python scripts/calibrate_solutionp_spatial_phase3.py \
    --inference-dir "$PHASE2_RUN_DIR/cnp_inference_tropical_only" \
    --method learned \
    --output-subdir soil_2d_predictions_solutionp_spatial_learned \
    --params-json "$SPATIAL_PARAMS" \
    --eval-json "${PHASE2_RUN_DIR}/analysis/solutionp_spatial_phase3_learned_eval.json"
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

if [[ ! -f "$SPATIAL_PARAMS" ]]; then
  echo "Error: spatial params JSON missing: $SPATIAL_PARAMS" >&2
  exit 1
fi

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN3_DIR="${RUN3_DIR:-cnp_results/run_${TS}_${PHASE3_SPATIAL_SUFFIX}}"
mkdir -p "$RUN3_DIR/analysis" "$RUN3_DIR/comparison_results"

SPATIAL_SUBDIR="${SPATIAL_SUBDIR:-soil_2d_predictions_solutionp_spatial_learned}"
RESTART_OUT="$RUN3_DIR/updated_restart_phase3_spatial_solutionp_tropical_5P.nc"
NETCDF_OUT="$RUN3_DIR/comparison_results/ai_predictions_phase3_spatial_solutionp.nc"

if [[ -f "$RESTART_OUT" ]] && [[ -f "$NETCDF_OUT" ]]; then
  echo "Phase 3 spatial restart already exists: $RESTART_OUT"
  echo "  export RUN3_DIR=$RUN3_DIR"
  exit 0
fi

echo "=== E3SMv3_h0 Phase 3: spatial learned-k (solutionp) + restart ==="
echo "  PHASE1_RUN_DIR=$PHASE1_RUN_DIR"
echo "  PHASE2_RUN_DIR=$PHASE2_RUN_DIR"
echo "  BASE_RESTART=$BASE_RESTART"
echo "  SPATIAL_PARAMS=$SPATIAL_PARAMS"
echo "  TROPICAL_LAT_RANGE=$TROPICAL_LAT_RANGE"
echo "  RUN3_DIR=$RUN3_DIR"

INF_FULL_DIR="$RUN3_DIR/cnp_inference_entire_dataset"
PRED_ROOT="$INF_FULL_DIR/cnp_predictions"
OVERLAY_STAMP="$RUN3_DIR/.phase2_5p_overlay_done"
SPATIAL_STAMP="$RUN3_DIR/.spatial_solutionp_applied"

if [[ ! -d "$PRED_ROOT/soil_2d_ground_truth" ]]; then
  echo "Seeding inference from Phase 1 global grid (symlink)..."
  ln -sfn "$(readlink -f "$PHASE1_RUN_DIR/cnp_inference_entire_dataset")" "$INF_FULL_DIR"
fi

if [[ ! -f "$OVERLAY_STAMP" ]]; then
  echo "Overlaying Phase 2 tropical 5P onto global grid..."
  python scripts/overlay_5p_by_coords.py \
    --base-run-dir "$RUN3_DIR" \
    --source-predictions "$PHASE2_INF"
  date -Iseconds > "$OVERLAY_STAMP"
else
  echo "Phase 2 5P overlay already done; skipping."
fi

if [[ ! -f "$SPATIAL_STAMP" ]]; then
  echo "Applying spatial learned-k to solutionp_vr (Amazon + Africa boxes)..."
  python scripts/apply_solutionp_spatial_phase3.py \
    --inference-dir "$INF_FULL_DIR" \
    --params-json "$SPATIAL_PARAMS" \
    --output-subdir "$SPATIAL_SUBDIR"
  cp -f "$PRED_ROOT/$SPATIAL_SUBDIR/predictions_Y_solutionp_vr.csv" \
    "$PRED_ROOT/soil_2d_predictions/predictions_Y_solutionp_vr.csv"
  date -Iseconds > "$SPATIAL_STAMP"
else
  echo "Spatial solutionp correction already applied; skipping."
fi

echo "Converting predictions to NetCDF..."
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$PRED_ROOT/" \
  --variable-list "$VARIABLE_LIST" \
  --output "$NETCDF_OUT"

echo "Creating Phase 3 spatial restart (~53 GB, may take several minutes)..."
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$NETCDF_OUT" \
  --restart-file "$BASE_RESTART" \
  --output "$RESTART_OUT" \
  --variable-list "$VARIABLE_LIST" \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=${TROPICAL_LAT_RANGE}"

ln -sfn "$(readlink -f "$SPATIAL_PARAMS")" "$RUN3_DIR/analysis/solutionp_spatial_phase3_learned_params.json"

cat > "$RUN3_DIR/README_phase3_spatial_solutionp.txt" <<EOF
E3SMv3_h0 Phase 3 — spatial learned-k for solutionp_vr ($(date -Iseconds))

Inference seed:  $PHASE1_RUN_DIR/cnp_inference_entire_dataset (symlink)
5P overlay:      $PHASE2_RUN_DIR/cnp_inference_tropical_only (by lon/lat)
solutionp Phase3: spatial learned-k Ridge (Amazon + Africa boxes)
Spatial params:  $SPATIAL_PARAMS
Restart base:    $BASE_RESTART
Tropical band:   $TROPICAL_LAT_RANGE

Outputs:
  NetCDF:  $NETCDF_OUT
  Restart: $RESTART_OUT

Compare with:
  Phase 2 raw:     $PHASE2_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc
  Phase 3 v3 affine: cnp_results/run_*_e3smv3_h0_phase3_tworegions/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc
EOF

echo
echo "Phase 3 spatial complete."
echo "  export RUN3_DIR=$RUN3_DIR"
echo "  Restart: $RESTART_OUT"
if [[ -f "$RESTART_OUT" ]]; then
  ls -lh "$RESTART_OUT"
fi
