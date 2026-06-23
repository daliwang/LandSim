# Instructions: TESNORTHERA5 restart update from a trained CNP

This document walks through:
1) running inference with your trained model,
2) exporting predictions to a NetCDF (`.nc`),
3) updating the target ELM restart NetCDF using those predictions.

All commands below are based on the repo scripts:
- `scripts/run_inference_all.py`
- `scripts/ai_predictions_to_netcdf.py`
- `scripts/ai_predictions_to_restart.py`

---

## Inputs (your paths)

### Trained model run (source of checkpoint)
- `TRAIN_RUN_DIR=/mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20260319_134216_tesnorth10pc`
- Model checkpoint:
  - `$TRAIN_RUN_DIR/cnp_predictions/model.pth`

### CNP IO variable list (drives variables + dataset defaults)
- `VAR_LIST=/mnt/proj-shared/AI4BGC_7xw/AI4BGC/CNP_IO_tesnorth10pct.txt`

### Target restart NetCDF to update
- `RESTART_FILE=/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/TESNORTHERA5_20adspin_restartfile/uELM_NORTHERA5_ERA5REF_I1850uELMCNPRDCTCBC.elm.r.0021-01-01-00000.nc`

---

## Step 0: Set up run output paths

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

export TRAIN_RUN_DIR="/mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20260319_134216_tesnorth10pc"
export VAR_LIST="/mnt/proj-shared/AI4BGC_7xw/AI4BGC/CNP_IO_tesnorth10pct.txt"
export RESTART_FILE="/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/TESNORTHERA5_20adspin_restartfile/uELM_NORTHERA5_ERA5REF_I1850uELMCNPRDCTCBC.elm.r.0021-01-01-00000.nc"

export OUT_ROOT="$TRAIN_RUN_DIR/inference_restart_update"
mkdir -p "$OUT_ROOT"

# Where inference predictions (CSV) will land
export INFER_DIR="$OUT_ROOT/cnp_inference_entire_dataset"

# Where the converter will write the NetCDF needed by the restart updater
export AI_PRED_NC="$OUT_ROOT/ai_predictions_for_restart.nc"

# Where the restart updater will write the updated restart
export OUT_RESTART="$OUT_ROOT/updated_restart_tesnorth10pc.nc"
```

---

## Step 1: Run inference with your trained checkpoint

### Recommended mode (use dataset defaults referenced by `CNP_IO_tesnorth10pct.txt` / training config)

```bash
python scripts/run_inference_all.py \
  --model "$TRAIN_RUN_DIR/cnp_predictions/model.pth" \
  --variable-list "$VAR_LIST" \
  --output-dir "$INFER_DIR"
```

This runs inference on the entire dataset available to the script (it will use `data_paths` / `file_pattern` from the training run config if available; otherwise it falls back to defaults from the provided IO file).

### If your "new dataset" is stored in different PKL directories

If the inference data (`training_data_batch_*.pkl`) live somewhere else, pass:
- `--data-paths` (comma-separated directories)
- `--file-pattern` (e.g. `training_data_batch_*.pkl`)

Example:
```bash
python scripts/run_inference_all.py \
  --model "$TRAIN_RUN_DIR/cnp_predictions/model.pth" \
  --variable-list "$VAR_LIST" \
  --data-paths "/path/to/new/pkl_root_1,/path/to/new/pkl_root_2" \
  --file-pattern "training_data_batch_*.pkl" \
  --output-dir "$INFER_DIR"
```

---

## Step 2: Convert predictions to NetCDF

The restart updater expects the NetCDF layout produced by `ai_predictions_to_netcdf.py`.

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$INFER_DIR/cnp_predictions" \
  --variable-list "$VAR_LIST" \
  --output "$AI_PRED_NC"
```

### Longitude wrapping (only if mapping fails)

`ai_predictions_to_restart.py` maps AI gridcells to model gridcells by nearest-neighbor in `(lon,lat)`.
So the longitude convention must match.

If your AI predictions NetCDF uses `0..360` longitudes but your restart uses `-180..180`, re-run Step 2 with:

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$INFER_DIR/cnp_predictions" \
  --variable-list "$VAR_LIST" \
  --wrap-longitude \
  --output "$AI_PRED_NC"
```

For your specific restart file (`NORTHERA5`), it already uses negative longitudes, and your existing TESNORTH10PC predictions appear to use negative longitudes too—so `--wrap-longitude` is usually not needed.

---

## Step 3: Update the ELM restart NetCDF

### (A) Preview first (no file is modified)

Update only the 5 soil-P variables that correspond to the common “5P” update:
`labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr`

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$AI_PRED_NC" \
  --restart-file "$RESTART_FILE" \
  --output "$OUT_RESTART" \
  --variable-list "$VAR_LIST" \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  --preview-only
```

Check the console output for:
- the variables detected as “to update”
- the coordinate mapping details

### (B) Write the updated restart (recommended: keep a backup)

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$AI_PRED_NC" \
  --restart-file "$RESTART_FILE" \
  --output "$OUT_RESTART" \
  --variable-list "$VAR_LIST" \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  --backup
```

This produces:
- `$OUT_RESTART`
- plus a backup next to `$RESTART_FILE` if `--backup` is set.

---

## Step 4: Sanity checks (recommended)

1) Confirm the updated restart exists:
```bash
ls -lah "$OUT_RESTART"
```

2) (Optional) Visual comparison / debugging:
- `scripts/restart_variable_plot.py` (often requires editing its `FILE_NEW` path)
- other comparison scripts under `scripts/` (e.g. scatter / profile comparisons)

---

## Notes / gotchas

- `ai_predictions_to_restart.py` updates:
  - PFT1D variables in the model’s first “column mapping” slot (skips PFT0)
  - soil2D variables in the first column and first 10 layers
- If your “new dataset” changes the variable set or ordering, you must keep using the correct `--variable-list` for TESNORTHERA5 (`CNP_IO_tesnorth10pct.txt`).

