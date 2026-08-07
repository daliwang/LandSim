#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# E3SMv3_h0 validation: training metrics, 5P pred-eval, site restart comparison
#
# Usage:
#   source config/e3smv3_h0_env.sh
#   export NATVEG_RUN_DIR=... PHASE2_RUN_DIR=... RUN3_DIR=...
#   bash scripts/run_e3smv3_h0_validation.sh
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=config/e3smv3_h0_env.sh
source config/e3smv3_h0_env.sh

for var in NATVEG_RUN_DIR PHASE2_RUN_DIR RUN3_DIR; do
  if [[ -z "${!var:-}" ]]; then
    echo "Error: $var must be set." >&2
    exit 1
  fi
done

echo "=== E3SMv3_h0 validation ==="
echo "  Phase1: $NATVEG_RUN_DIR"
echo "  Phase2: $PHASE2_RUN_DIR"
echo "  Phase3: $RUN3_DIR"

find_validation_stats() {
  local run_dir="$1"
  for candidate in \
    "$run_dir/validation_stats.csv" \
    "$run_dir/analysis/validation_stats.csv" \
    "$run_dir/cnp_inference_entire_dataset/validation_stats.csv"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# --- Training quality reports (Phase 1) ---
VALIDATION_STATS="$(find_validation_stats "$NATVEG_RUN_DIR" || true)"
if [[ -z "${VALIDATION_STATS:-}" && -d "$NATVEG_RUN_DIR/cnp_inference_entire_dataset/cnp_predictions" ]]; then
  echo "validation_stats.csv not found; generating with cnp_result_validationplot.py --stats-only ..."
  python scripts/cnp_result_validationplot.py \
    "$NATVEG_RUN_DIR/cnp_inference_entire_dataset" \
    --stats-only --no-scatter --no-plot-loss || true
  VALIDATION_STATS="$(find_validation_stats "$NATVEG_RUN_DIR" || true)"
fi

if [[ -n "${VALIDATION_STATS:-}" ]]; then
  echo "Generating Phase 1 prediction quality report from $VALIDATION_STATS ..."
  python scripts/generate_prediction_quality_report.py \
    --input "$VALIDATION_STATS" \
    --output-dir "$NATVEG_RUN_DIR/analysis" || true
else
  echo "Skipping prediction quality report (validation_stats.csv not found under $NATVEG_RUN_DIR)."
fi

# --- NPOOL/PPOOL per-PFT validation ---
echo "NPOOL/PPOOL validation (Phase 1)..."
python scripts/validation_npool_ppool_per_pft.py "$NATVEG_RUN_DIR" || true

# --- 5P pred-eval: Amazon and Africa boxes ---
RUNS_JSON="$RUN3_DIR/analysis/5p_pred_eval_runs.json"
mkdir -p "$RUN3_DIR/analysis"
cat > "$RUNS_JSON" <<EOF
[
  {
    "label": "e3smv3_h0_phase1_global",
    "inference_dir": "$NATVEG_RUN_DIR/cnp_inference_entire_dataset",
    "predictions_subdir": "soil_2d_predictions",
    "predictions_filename_suffix": ""
  },
  {
    "label": "e3smv3_h0_phase2_tropical",
    "inference_dir": "$PHASE2_RUN_DIR/cnp_inference_tropical_only",
    "predictions_subdir": "soil_2d_predictions",
    "predictions_filename_suffix": ""
  },
  {
    "label": "e3smv3_h0_phase3_tworegions_v3",
    "inference_dir": "$RUN3_DIR/cnp_inference_entire_dataset",
    "predictions_subdir": "soil_2d_predictions_5P_bias_corrected_amazon_africa_v3",
    "predictions_filename_suffix": "_bias_corrected_amazon_africa_v3"
  }
]
EOF

echo "5P pred-eval (Amazon + Africa)..."
python scripts/compare_5p_gt_two_regions_inference.py pred-eval \
  --runs-json "$RUNS_JSON" \
  --output-summary "$RUN3_DIR/analysis/5p_pred_eval_e3smv3_h0_summary.csv"

# --- Site restart comparison (Africa + Amazon) ---
export PHASE1_RUN_DIR="$NATVEG_RUN_DIR"
export PHASE2_RUN_DIR="$PHASE2_RUN_DIR"
export PHASE3_RUN_DIR="$RUN3_DIR"

echo "Site 5P restart comparison (Africa lon=28, lat=0)..."
python scripts/generate_site_5p_restart_comparison.py \
  --lon 28.0 \
  --lat 0.0 \
  --site-name africa_site_lon28p00_lat0p00 \
  --phase3-run-dir "$RUN3_DIR" \
  --phase3-restart "$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc"

echo "Site 5P restart comparison (Amazon lon=303.75, lat=-17.43)..."
python scripts/generate_site_5p_restart_comparison.py \
  --lon 303.75 \
  --lat -17.434553 \
  --site-name amazon_site_lon303p75_latm17p43 \
  --phase3-run-dir "$RUN3_DIR" \
  --phase3-restart "$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc"

echo
echo "Validation complete. Key outputs:"
echo "  $RUN3_DIR/analysis/5p_pred_eval_e3smv3_h0_summary.csv"
echo "  $RUN3_DIR/analysis/africa_site_lon28p00_lat0p00_5p_restart_comparison/"
echo "  $RUN3_DIR/analysis/amazon_site_lon303p75_latm17p43_5p_restart_comparison/"
