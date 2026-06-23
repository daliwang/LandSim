# Instructions: Repeat workflow and create new restart in branch `trendy_1_ai_restart_creation`

This document gives step-by-step instructions to repeat the process in **docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md** and produce a new restart file in a dedicated run branch **trendy_1_ai_restart_creation**.

---

## Prerequisites (for new users and existing runs)

This workflow assumes you have **two trained CNP models**:

1. A **global natveg_improved-like model** with a global restart file.
2. A **tropical-only Phase2 model** trained with the updated
   `config/training_config_phase2_tropical_soilp_only.json` (emphasis on the
   five soil P variables), with a trained checkpoint and inference outputs
   (e.g. a run like `run_20260313_224805_phase2_tropical_soilp_amazon_africa`).

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

- Tropical-only Phase2 model (updated config, emphasis on 5 P variables):

  ```bash
  cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

  python train_cnp_model.py \
    --training-config-json config/training_config_phase2_tropical_soilp_only.json \
    --output-dir cnp_results \
    --output-dir-suffix phase2_tropical_soilp_amazon_africa \
    --variable-list CNP_IO_updated9_dev_dw.txt
  ```

  Then set:

  ```bash
  export PHASE2_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_phase2_tropical_soilp_amazon_africa"
  ```

If you already have trained runs, you can reuse them. The paths used below
assume the **existing** experiments (adjust if your run IDs differ):

- **Natveg run:** `cnp_results/run_20260228_214757_natveg_improved`
- **Base restart file:**  
  `cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc`
- **Phase2 run:** Tropical-only run with the updated Phase2 config (5 P emphasis), e.g. `cnp_results/run_20260313_224805_phase2_tropical_soilp_amazon_africa`. This is the basis run for Phase3.

For the full **three-phase workflow** (phase1_global, phase2_tropical, phase3_tworegions): Phase2 is tropical-only with `config/training_config_phase2_tropical_soilp_only.json`; Phase3 is **two-region (Amazon + Africa) bias correction and merge** of the 5 P variables. See **docs/WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md**.

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

## Path A: Tropical-only inference → Phase2 restart (raw 5P in tropics)

Use this if you only have (or only want) **tropical** Phase2 predictions from
the updated Phase2 run (tropical-only, 5 P emphasis). Set `PHASE2_RUN_DIR` to
your run (e.g. `cnp_results/run_20260313_224805_phase2_tropical_soilp_amazon_africa`).

### A1. Phase2 tropical-only inference (if not already done)

```bash
cd "$PHASE2_RUN_DIR"
python ../../scripts/run_inference_all.py \
  --model cnp_model.pt \
  --output-dir cnp_inference_tropical_only
cd ../..
```

### A2. NetCDF from tropical predictions

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$PHASE2_RUN_DIR/cnp_inference_tropical_only/cnp_predictions" \
  --output "$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt
```

### A3. Update natveg restart (5 P variables, raw Phase2)

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc" \
  --restart-file "$BASE_RESTART_FILE" \
  --output "$TRENDY_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

**New restart file:** `$TRENDY_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc`

For **5P with two-region bias/scale** (Phase3): use Path B below — full-grid inference, then **Amazon and Africa bias correction separately**, then **merge** with `scripts/merge_5p_bias_corrected_amazon_africa.py`, then build NetCDF and restart. See **docs/WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md** §3 Path 3B.

---

## Path B: Full-grid inference → two-region 5P bias correction and merge → global restart

Use this to get **bias-corrected 5P in Amazon + Africa**: apply bias/scale for
Amazon only and Africa only, **merge** with `scripts/merge_5p_bias_corrected_amazon_africa.py`,
then build a global restart from the Phase1 base. This is Phase3 of the
three-phase workflow.

### B1. Phase2 full-grid inference

From repo root. Run full-grid inference into your **Phase2 run** (the tropical-only run with updated config, e.g. `run_20260313_224805_phase2_tropical_soilp_amazon_africa`). Set `PHASE2_RUN_DIR` to that path.

```bash
python scripts/run_inference_all.py \
  --model "$PHASE2_RUN_DIR/cnp_model.pt" \
  --output-dir "$PHASE2_RUN_DIR/cnp_inference_entire_dataset" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --inference-full-grid
```

This creates `cnp_predictions/` and `soil_2d_ground_truth/` under `$PHASE2_RUN_DIR/cnp_inference_entire_dataset`. The next steps use `--run-dir "$PHASE2_RUN_DIR"`.

### B2. Apply 5P bias/scale for Amazon only, then Africa only

Apply bias correction **separately** for the two regions, then merge (B3). Use `--run-dir "$PHASE2_RUN_DIR"`.

**Amazon only:**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$PHASE2_RUN_DIR" \
  --region-config-json config/training_config_amazon_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_amazon
```

**Africa only:**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$PHASE2_RUN_DIR" \
  --region-config-json config/training_config_africa_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_africa
```

### B3. Merge Amazon and Africa corrected 5P

```bash
python scripts/merge_5p_bias_corrected_amazon_africa.py \
  --run-dir "$PHASE2_RUN_DIR"
```

This creates `$PHASE2_RUN_DIR/cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_5P_bias_corrected_amazon_africa/`.

### B4. NetCDF from merged bias-corrected 5P (amazon_africa)

```bash
mkdir -p "$PHASE2_RUN_DIR/comparison_results"

python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$PHASE2_RUN_DIR/cnp_inference_entire_dataset/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_amazon_africa \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --output "$PHASE2_RUN_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa.nc"
```

### B5. Generate global restart (Phase1 base + 5P in tropics)

Use the Phase1 **base restart** (`$BASE_RESTART_FILE`) so the result is a full global restart with 5P updated only in the tropics (Amazon and Africa bias-corrected; other tropics raw Phase2).

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$PHASE2_RUN_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa.nc" \
  --restart-file "$BASE_RESTART_FILE" \
  --output "$TRENDY_RUN_DIR/updated_restart_global_5P_bias_corrected_amazon_africa.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

**New restart file:** `$TRENDY_RUN_DIR/updated_restart_global_5P_bias_corrected_amazon_africa.nc`

---

## Step 7: Record and validate

1. **Record** in `$TRENDY_RUN_DIR`:
   - Which path (A or B) was used.
   - Base restart path and Phase2 run path.
   - Command lines for `ai_predictions_to_netcdf.py` and `ai_predictions_to_restart.py`.

   Example README:

   ```bash
   echo "Restart created: $(date). Path B (two-region 5P bias correction + merge). Base: natveg_improved. Phase2: run_*_phase2_tropical_soilp_amazon_africa." > "$TRENDY_RUN_DIR/README_restart.txt"
   ```

2. **Run the land model** using the new restart and validate (e.g. Amazon/Africa 5P profiles, NPP, fluxes) as in **docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md**.

---

## Summary checklist (Path B – two-region 5P bias correction and merge)

- [ ] `git checkout -b trendy_1_ai_restart_creation`
- [ ] Create `TRENDY_RUN_DIR=cnp_results/run_<timestamp>_trendy_1_ai_restart_creation`
- [ ] Set `PHASE2_RUN_DIR` to your tropical-only Phase2 run (e.g. `run_*_phase2_tropical_soilp_amazon_africa`)
- [ ] Run Phase2 **full-grid** inference into `$PHASE2_RUN_DIR/cnp_inference_entire_dataset`
- [ ] Run **apply_5p_bias_scale_correction.py** for Amazon only (`training_config_amazon_5p_box.json`)
- [ ] Run **apply_5p_bias_scale_correction.py** for Africa only (`training_config_africa_5p_box.json`)
- [ ] Run **merge_5p_bias_corrected_amazon_africa.py** (`--run-dir "$PHASE2_RUN_DIR"`)
- [ ] Run **ai_predictions_to_netcdf.py** on merged `soil_2d_predictions_5P_bias_corrected_amazon_africa`
- [ ] Run **ai_predictions_to_restart.py** (Phase1 base → `updated_restart_global_5P_bias_corrected_amazon_africa.nc`, 5P, `--tropical-lat-range=-30,30`)
- [ ] Document and validate

Reference: **docs/WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md**.
