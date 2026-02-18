#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/prepare_data.sh /path/to/target_dir [gdrive_folder_url]

# Inputs
DATA_DIR="${1:-/tmp}"
GDRIVE_URL="${2:-https://drive.google.com/drive/folders/1djTADtAqxlYml8GwdwxUEAi-1xR_Yqn5}"

if [[ "${1:-}" == "" ]]; then
  echo "[prepare_data] No target directory provided; defaulting to /tmp"
fi

echo "[prepare_data] Target dir: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

# If directory already has at least one pkl file matching expected pattern, skip
shopt -s nullglob
existing=("${DATA_DIR}"/enhanced_1_training_data_batch_*.pkl)
if (( ${#existing[@]} > 0 )); then
  echo "[prepare_data] Found ${#existing[@]} data files, skipping download."
  exit 0
fi

echo "[prepare_data] No data found. Ensuring gdown is available..."

# If a virtualenv is active, use it; otherwise try local .venv; else system python
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  : # already in a venv
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
fi

python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade || true
python -m pip install -q --upgrade pip || true
python -m pip install -q gdown || true

echo "[prepare_data] Downloading dataset from Google Drive to ${DATA_DIR}..."
gdown --folder --continue --remaining-ok --fuzzy "${GDRIVE_URL}" -O "${DATA_DIR}"

echo "[prepare_data] Done. Listing downloaded files:"
ls -lh "${DATA_DIR}" | sed 's/^/[prepare_data] /'


