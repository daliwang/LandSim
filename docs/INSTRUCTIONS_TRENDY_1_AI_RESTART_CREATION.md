# Instructions: Repeat workflow and create new restart in branch `trendy_1_ai_restart_creation`

This document gives step-by-step instructions to repeat the process in **docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md** and produce a new restart file in a dedicated run branch **trendy_1_ai_restart_creation**.

---

## Prerequisites (for new users and existing runs)

This workflow assumes you have **two trained CNP models**:

1. A **global natveg_improved-like model** with a global restart file.
2. A **tropical Phase2 P-focused model** (`phase2_pvariable_focus`-like) with a
   trained checkpoint and inference outputs.

If you are a **new user** and do not yet have these models, first follow
Section 0 in `docs/WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md` to train them
with `train_cnp_model.py`. In short:

- Global model (natveg_improved-like):

  ```bash
  cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

  python train_cnp_model.py \
    --config config/training_config_experiment_3_global_natveg_improved.json \
    --run-dir cnp_results/run_YYYYMMDD_HHMMSS_natveg_improved_custom \
    --variable-list CNP_IO_updated9_dev_dw.txt
  ```

  Then produce a global restart NetCDF from this run and set:

  ```bash
  export NATVEG_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_natveg_improved_custom"
  export BASE_RESTART_FILE="$NATVEG_RUN_DIR/updated_restart_...your_file.nc"
  ```

- Tropical P-focused model (phase2_tropical-like):

  ```bash
  cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

  python train_cnp_model.py \
    --config config/training_config_phase2_tropical_soilp_only.json \
    --run-dir cnp_results/run_YYYYMMDD_HHMMSS_phase2_pvariable_focus_custom \
    --variable-list CNP_IO_updated9_dev_dw.txt
  ```

  Then set:

  ```bash
  export PHASE2_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_phase2_pvariable_focus_custom"
  ```

If you already have trained runs, you can reuse them. The paths used below
assume the **existing** experiments (adjust if your run IDs differ):

- **Natveg run:** `cnp_results/run_20260228_214757_natveg_improved`
- **Base restart file:**  
  `cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc`
- **Phase2 run:** `cnp_results/run_20260305_153217_phase2_pvariable_focus`

---

## Step 0: Create git branch and run directory

From the repo root:

```bash
git checkout -b trendy_1_ai_restart_creation
```

Create a run directory for this workflow’s outputs (restart, logs, copies of params):

```bash
export RUN_TS=$(date +%Y%m%d_%H%M%S)
export TRENDY_RUN_DIR="cnp_results/run_${RUN_TS}_trendy_1_ai_restart_creation"
mkdir -p "$TRENDY_RUN_DIR"
```

Use `TRENDY_RUN_DIR` in the steps below so the new restart and any logs live under this branch’s run.

---

## Path A: Tropical-only inference → 4P or 5P restart (no bias/scale, or with bias/scale)

Use this if you only have (or only want) **tropical** Phase2 predictions.

### A1. Phase2 tropical-only inference (if not already done)

```bash
cd cnp_results/run_20260305_153217_phase2_pvariable_focus
python ../../scripts/run_inference_all.py \
  --model cnp_predictions/model.pth \
  --output-dir cnp_inference_tropical_only
cd ../..
```

### A2. NetCDF from tropical predictions

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_inference_tropical_only/cnp_predictions \
  --output cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_tropical_only.nc
```

### A3. Update natveg restart (4 P variables, raw Phase2)

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_tropical_only.nc \
  --restart-file cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc \
  --output "$TRENDY_RUN_DIR/updated_restart_phase2_tropical_4p.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update occlp_vr,labilep_vr,solutionp_vr,secondp_vr \
  "--tropical-lat-range=-30,30"
```

**New restart file:** `$TRENDY_RUN_DIR/updated_restart_phase2_tropical_4p.nc`

To use **5P with bias/scale** on tropical-only: first run the bias/scale step (Section B2 below) on the **tropical** CSVs (script expects `cnp_inference_entire_dataset` layout; if your tropical run writes elsewhere, either symlink or run bias/scale from a copy that matches that layout), then build a NetCDF from the bias-corrected CSVs and run `ai_predictions_to_restart.py` with `--variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr` and `--output "$TRENDY_RUN_DIR/updated_restart_phase2_5P_bias_corrected_tropical.nc"`.

---

## Path B: Full-grid inference → 5P bias/scale → tropical restart (recommended for 5P bias/scale)

Use this to get **bias-corrected 5P in Amazon + Africa** and a single global NetCDF.

### B1. Phase2 full-grid inference

From repo root. You can set the output directory with `--output-dir` (default is `cnp_inference_entire_dataset` under the current directory). Example: write outputs into the Phase2 run directory, or into your trendy run directory.

** Output into your trendy run directory** (then bias/scale script must be pointed at this run’s paths or you copy/symlink the expected structure):

```bash
# TRENDY_RUN_DIR set in Step 0
python scripts/run_inference_all.py \
  --model cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_predictions/model.pth \
  --output-dir "$TRENDY_RUN_DIR/cnp_inference_entire_dataset" \
  --inference-full-grid
```

This creates `.../cnp_predictions/` and `.../soil_2d_ground_truth/` under the given `--output-dir`. The bias/scale script (B2) expects `--run-dir` to be the run that contains `cnp_inference_entire_dataset`; if you use Option 2, set `--run-dir "$TRENDY_RUN_DIR"` in B2 and ensure the run has that subdir.

### B2. Apply 5P bias/scale correction (Amazon + Africa)

Use as `--run-dir` the directory that **contains** `cnp_inference_entire_dataset/` (where B1 wrote the CSVs). If you used **Option 1** in B1, that is the Phase2 run dir; if you used **Option 2**, that is `$TRENDY_RUN_DIR`.

**If you used Option 1 (output under Phase2 run):**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir cnp_results/run_20260305_153217_phase2_pvariable_focus \
  --region-config-json config/training_config_two_region_five_p.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_phase2
```

**If you used Option 2 (output under TRENDY_RUN_DIR):**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$TRENDY_RUN_DIR" \
  --region-config-json config/training_config_two_region_five_p.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_phase2
```

Corrected CSVs end up under `<run-dir>/cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_5P_bias_corrected_phase2/`. Bias/scale params are written to `<run-dir>/analysis/bias_scale_params_5P_two_regions.json`.

Optional: copy bias/scale params into the trendy run for provenance (if Phase2 run was used for B2, copy from there):

```bash
cp cnp_results/run_20260305_153217_phase2_pvariable_focus/analysis/bias_scale_params_5P_two_regions.json \
   "$TRENDY_RUN_DIR/bias_scale_params_5P_two_regions_phase2.json"
```

### B3. NetCDF from bias-corrected 5P predictions

Use the same run directory you used in B2 (Phase2 run dir or `$TRENDY_RUN_DIR`).

```bash
# If you used Option 1 in B1/B2 (Phase2 run dir):
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_inference_entire_dataset/cnp_predictions/ \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_phase2 \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --output cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc

# If you used Option 2, write NetCDF into your run dir:
mkdir -p "$TRENDY_RUN_DIR/comparison_results"

python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$TRENDY_RUN_DIR/cnp_inference_entire_dataset/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_phase2 \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --output "$TRENDY_RUN_DIR/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc"
```

### B4. Generate new restart (5P in tropics, from natveg base)

Use the path to the NetCDF you produced in B3 (Phase2 run dir or `$TRENDY_RUN_DIR/comparison_results/...`).

```bash
# If Option 1 was used (NetCDF in Phase2 run dir):
python scripts/ai_predictions_to_restart.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc \
  --restart-file cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc \
  --output "$TRENDY_RUN_DIR/updated_restart_phase2_5P_bias_corrected_tropical.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"

# If Option 2 was used (NetCDF in trendy run dir):
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$TRENDY_RUN_DIR/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc" \
  --restart-file cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc \
  --output "$TRENDY_RUN_DIR/updated_restart_phase2_5P_bias_corrected_tropical.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

**New restart file:** `$TRENDY_RUN_DIR/updated_restart_phase2_5P_bias_corrected_tropical.nc`

---

## Step 7: Record and validate

1. **Record** in `$TRENDY_RUN_DIR`:
   - Which path (A or B) was used.
   - Base restart path and Phase2 run path.
   - Command lines for `ai_predictions_to_netcdf.py` and `ai_predictions_to_restart.py`.

   Example README:

   ```bash
   echo "Restart created: $(date). Path B (full-grid + 5P bias/scale). Base: natveg_improved. Phase2: run_20260305_153217_phase2_pvariable_focus." > "$TRENDY_RUN_DIR/README_restart.txt"
   ```

2. **Run the land model** using the new restart and validate (e.g. Amazon/Africa 5P profiles, NPP, fluxes) as in **docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md**.

---

## Summary checklist (Path B – 5P bias/scale)

- [ ] `git checkout -b trendy_1_ai_restart_creation`
- [ ] Create `TRENDY_RUN_DIR=cnp_results/run_<timestamp>_trendy_1_ai_restart_creation`
- [ ] Run Phase2 **full-grid** inference (`--inference-full-grid`)
- [ ] Run **apply_5p_bias_scale_correction.py** (Phase2 run-dir, two-region config)
- [ ] Run **ai_predictions_to_netcdf.py** on bias-corrected 5P CSVs
- [ ] Run **ai_predictions_to_restart.py** (natveg base restart → `$TRENDY_RUN_DIR/updated_restart_phase2_5P_bias_corrected_tropical.nc`, 5P, `--tropical-lat-range=-30,30`)
- [ ] Document and validate

Reference: **docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md**.
