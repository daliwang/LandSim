#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Build preprocessed tensor cache for E3SMv3_h0 training (no model training).
#
# Usage (from repo root):
#   bash scripts/build_e3smv3_preprocessed_cache.sh           # Phase 1 global
#   bash scripts/build_e3smv3_preprocessed_cache.sh tropical # Phase 2 tropical
#   bash scripts/build_e3smv3_preprocessed_cache.sh both
#
# Cache directory: PREPROCESSED_CACHE_DIR (default: E3SM_DATA_DIR/.preprocessed_cache)
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=config/e3smv3_h0_env.sh
source config/e3smv3_h0_env.sh

PHASE="${1:-global}"
export USE_PREPROCESSED_CACHE=1
export REBUILD_PREPROCESSED_CACHE="${REBUILD_PREPROCESSED_CACHE:-1}"

_build_one() {
  local phase="$1"
  local config extra_suffix

  if [[ "$phase" == "tropical" ]]; then
    config="$CONFIG_TROPICAL"
    extra_suffix="cache_build_phase2_tropical"
    echo "=== Building preprocessed cache: Phase 2 tropical ==="
    echo "  TROPICAL_LAT_RANGE=$TROPICAL_LAT_RANGE"
  else
    config="$CONFIG_GLOBAL"
    extra_suffix="cache_build_phase1_global"
    echo "=== Building preprocessed cache: Phase 1 global ==="
  fi

  echo "  CONFIG=$config"
  echo "  PREPROCESSED_CACHE_DIR=$PREPROCESSED_CACHE_DIR"
  echo "  LOAD_WORKERS=$LOAD_WORKERS"

  if [[ "$phase" == "tropical" ]]; then
    python train_cnp_model.py \
      --training-config-json "$config" \
      --variable-list "$VARIABLE_LIST" \
      --data-paths "$DATA_PATHS" \
      --file-pattern "$FILE_PATTERN" \
      --use-preprocessed-cache \
      --preprocessed-cache-dir "$PREPROCESSED_CACHE_DIR" \
      --rebuild-preprocessed-cache \
      --preprocessed-cache-only \
      --output-dir cnp_results \
      --output-dir-suffix "$extra_suffix" \
      --tropical-only \
      "--tropical-lat-range=${TROPICAL_LAT_RANGE}"
  else
    python train_cnp_model.py \
      --training-config-json "$config" \
      --variable-list "$VARIABLE_LIST" \
      --data-paths "$DATA_PATHS" \
      --file-pattern "$FILE_PATTERN" \
      --use-preprocessed-cache \
      --preprocessed-cache-dir "$PREPROCESSED_CACHE_DIR" \
      --rebuild-preprocessed-cache \
      --preprocessed-cache-only \
      --output-dir cnp_results \
      --output-dir-suffix "$extra_suffix"
  fi
}

case "$PHASE" in
  global)
    _build_one global
    ;;
  tropical)
    _build_one tropical
    ;;
  both)
    _build_one global
    _build_one tropical
    ;;
  *)
    echo "Usage: $0 [global|tropical|both]" >&2
    exit 1
    ;;
esac

echo
echo "Cache build complete."
echo "  PREPROCESSED_CACHE_DIR=$PREPROCESSED_CACHE_DIR"
ls -lh "$PREPROCESSED_CACHE_DIR" 2>/dev/null || true
