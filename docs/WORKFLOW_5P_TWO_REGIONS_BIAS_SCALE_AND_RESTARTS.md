## Workflow: natveg_improved baseline + Phase2 tropical 5P model → 5P bias/scale correction → tropical/two-region restart runs

This document records the end-to-end procedure to:

1. Train or reuse a **global baseline CNP model** (`natveg_improved`) and obtain a full global restart.
2. Train or reuse the **Phase2 tropical, P-focused model** (`phase2_pvariable_focus`, using `training_config_phase2_tropical_soilp_only.json`).
3. Run inference with Phase2 (tropical-only or full grid) to generate prediction and ground truth CSVs.
4. Fit and apply **5P bias/scale corrections** for **Amazon + Africa** on top of the Phase2 predictions.
5. Convert the bias-corrected Phase2 5P predictions to NetCDF.
6. Use `ai_predictions_to_restart.py` to overwrite 5P in the **tropical band** of the `natveg_improved` restart (with bias-corrected values in Amazon + Africa), creating updated restart files.
7. Run new simulations from these restarts and validate the results.

The goal is to have a reproducible protocol you can follow again for future runs.

**See also**

- [REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md](./REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md) — methodology review, reference-site weighting vs full-box fitting, and recommendations for `solutionp_vr` / `occlp_vr`.
- [PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md](./PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md) — why Phase3 uses Amazon-only → Africa-only → `merge_5p_bias_corrected_amazon_africa.py` like the “good” Phase2 run.
- [WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md](./WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md) — three-phase restart orchestration (`run_phase3_tworegions.sh`, merged 5P CSV layout).
- Regional **prediction vs ground truth** tables: `scripts/compare_5p_gt_two_regions_inference.py` (`pred-eval`); broader pipeline: [CNP_pipeline_runbook.md](./CNP_pipeline_runbook.md).

---

### 1. Train (or select) the global model: `natveg_improved`

**Objective:** Obtain a trained global model that will serve as the baseline for 5P corrections in the two focus regions.

- **Inputs/config:**
  - Training config JSON used for the `natveg_improved` experiment (e.g. `config/training_config_experiment_3_global_natveg_improved_occlp.json` or the exact file you used).
  - Global dataset with all necessary inputs (climate, soil, etc.) and outputs (including soil 2D P variables).
- **Action:**
  - Train the model using your standard training pipeline (e.g. `train_cnp_repeat.py` or the corresponding training script).
  - Ensure that the run is saved under something like:
    - `cnp_results/run_20260228_214757_natveg_improved/`
- **Result:**
  - A trained checkpoint and associated metadata under the `natveg_improved` run directory.

If the model is already trained, you can **reuse** the existing `natveg_improved` run and skip retraining.

---

### 2. Run entire-domain inference for natveg_improved

**Objective:** Produce per-variable CSVs with **predictions** and **ground truth** for the full domain, including Amazon and Africa.

- **Script:** your standard full-domain inference script (e.g. `run_inference_all.py` or its updated equivalent).
- **Key arguments:**
  - `--run-dir` pointing to the `natveg_improved` run directory, e.g.  
    `cnp_results/run_20260228_214757_natveg_improved`
  - Any additional options needed for full-domain inference (dataset split, etc.).
- **Expected outputs (directory structure):**
  - Under the run directory:
    - `cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions/`
      - Files like `predictions_Y_labilep_vr.csv`, `predictions_Y_occlp_vr.csv`, etc.
    - `cnp_inference_entire_dataset/cnp_predictions/soil_2d_ground_truth/`
      - Files like `ground_truth_Y_labilep_vr.csv`, `ground_truth_Y_occlp_vr.csv`, etc.

Each CSV should contain:

- `Longitude`, `Latitude` columns.
- One column per layer, named like:
  - `Y_<var>_col1_layer1`, `Y_<var>_col1_layer2`, …, `Y_<var>_col1_layer10`.

These paired GT and prediction CSVs are the basis for fitting the bias/scale corrections.

---

### 2. How the Phase2 tropical 5P model is trained

The Phase2 run at:

- `cnp_results/run_20260305_153217_phase2_pvariable_focus`

uses the unified config:

- `config/training_config_phase2_tropical_soilp_only.json`

Key points from that config:

- **Training domain (data_filtering_config):**
  - `"tropical_only": true`
  - `"tropical_lat_range": [-30.0, 30.0]`
  - `"natveg_only": true`
  - → Phase2 is trained **only on natveg grid cells in the latitude band [-30°, 30°]**.
- **Loss focus (variable_weights and tail_aware_weights):**
  - Soil 2D P variables get **very large weights**:
    - `primp_vr`: 26, `occlp_vr`: 45, `labilep_vr`: 18, `secondp_vr`: 14, `solutionp_vr`: 18.
  - Other soil 2D variables (C/N) have more moderate weights.
  - Tail-aware weights mirror this emphasis, so **extremes of 5P** are also prioritized.

**Interpretation:** `phase2_pvariable_focus` is a **tropical-only, P-specialist model**:

- It sees only **tropical natveg** data during training.
- Within that band, it allocates a large fraction of capacity to **matching soil P pools** (5P) and their vertical structure.

This explains why, in diagnostics (e.g. `prediction_quality_by_variable.png` and the Amazon
site profiles), Phase2 often outperforms `natveg_improved` on the 5P variables in the
tropics, especially after the additional bias/scale correction step.

---

### 3. Run inference for Phase2 (tropical-only vs full grid)

You have two useful inference paths. Both ultimately feed into `ai_predictions_to_netcdf.py`
and `ai_predictions_to_restart.py`, but they differ in spatial coverage.

#### 3.1 Tropical-only inference (replicating the existing 4P tropical restart)

As documented in `README_updated_restart_phase2_tropical_4p.md`, you can run inference on
only the tropical subset (same filter as training):

```bash
cd cnp_results/run_20260305_153217_phase2_pvariable_focus
python ../../scripts/run_inference_all.py \
  --model cnp_predictions/model.pth \
  --output-dir cnp_inference_tropical_only
```

- This uses the same `tropical_only` filter as training:
  - Only cells in [-30°, 30°] appear in the prediction CSVs / NetCDF.

You can then convert these predictions to NetCDF and update the natveg restart in the
**tropical band only**:

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_inference_tropical_only/cnp_predictions \
  --output cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_tropical_only.nc

python scripts/ai_predictions_to_restart.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_tropical_only.nc \
  --restart-file cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc \
  --output cnp_results/run_20260305_153217_phase2_pvariable_focus/updated_restart_phase2_tropical_4p.nc \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update occlp_vr,labilep_vr,solutionp_vr,secondp_vr \
  "--tropical-lat-range=-30,30"
```

- **Effect:**
  - For tropical cells (\([-30, 30]\)) that exist in the AI NetCDF:
    - The 4 listed P variables are overwritten with **raw Phase2** predictions.
  - Outside \([-30, 30]\), and for all non-P variables:
    - Values remain from the **natveg_improved** base restart.

If you later rerun inference after bias/scale correction (so that the tropical CSVs already
contain bias-corrected 5P values for Amazon + Africa), you can:

- Point `ai_predictions_to_netcdf.py` at the **bias-corrected tropical CSVs** instead.
- Use `--variables-to-update` including all 5P variables:
  - `labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr`.

#### 3.2 Full-grid inference (for a single global NetCDF)

For a cleaner global bias/scale workflow (global CSVs, corrections only in Amazon + Africa,
but predictions everywhere), you can instead run:

```bash
python scripts/run_inference_all.py \
  --run-dir cnp_results/run_20260305_153217_phase2_pvariable_focus \
  --inference-full-grid
```

- With `--inference-full-grid`, `run_inference_all.py` explicitly sets:
  - `tropical_only=False`
  - Clears `region_boxes`
  - → Predictions are written for **all natveg grid cells globally**.

This is the recommended path if you want a **single global NetCDF** where:

- Inside Amazon + Africa, 5P values are **bias/scale–corrected Phase2**.
- Elsewhere (tropics and extratropics), 5P values are **raw Phase2**.

---

### 4. Apply 5P bias/scale correction to Phase2 (Amazon + Africa only)

**Objective:** Fit and apply regional 5P corrections, but using Phase2
predictions as the baseline. The fit and corrections are restricted to the two-region
boxes (Amazon + Africa); elsewhere the CSVs remain raw Phase2.

For **Phase3-style** workflows, the recommended pattern is **two** passes of `apply_5p_bias_scale_correction.py` (Amazon-only config, then Africa-only config) followed by **`scripts/merge_5p_bias_corrected_amazon_africa.py`**. Rationale and tuning notes: [REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md](./REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md), [PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md](./PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md).

- **Script:** `scripts/apply_5p_bias_scale_correction.py`
- **Merge (split regional fits into one CSV set):** `scripts/merge_5p_bias_corrected_amazon_africa.py`
- **Key command (example):**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir cnp_results/run_20260305_153217_phase2_pvariable_focus \
  --region-config-json config/training_config_two_region_five_p.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_phase2
```

- **Outputs:**
  - Corrected predictions under:
    - `cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_5P_bias_corrected_phase2/`
      - `predictions_Y_<var>_bias_corrected.csv` for each of the 5 P variables.
    - Cells inside Amazon + Africa use bias-corrected values; other cells remain raw Phase2.
  - Bias/scale parameters under:
    - `cnp_results/run_20260305_153217_phase2_pvariable_focus/analysis/bias_scale_params_5P_two_regions.json`
  - (Optional) copy this JSON into `docs/` with a descriptive name, e.g.:
    - `docs/bias_scale_params_5P_two_regions_phase2_pvariable_focus.json`

These corrected Phase2 CSVs correspond to the `phase2_pvariable_focus_bias_corrected`
curve you see in the Amazon site profile plots.

---

### 5. Convert Phase2 bias-corrected CSVs to NetCDF (global, with two-region corrections)

Use `scripts/ai_predictions_to_netcdf.py` to build a global NetCDF of Phase2 predictions
with bias/scale corrections applied only in Amazon + Africa. **You must pass
`--soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_phase2`** so that
the script reads the 5P CSVs produced by `apply_5p_bias_scale_correction.py`; otherwise
it only loads raw predictions from `soil_2d_predictions` and the restart will not use
bias-corrected 5P.

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_inference_entire_dataset/cnp_predictions/ \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_phase2 \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --output cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc
```

The resulting NetCDF has:

- Global domain (all natveg grid cells).
- Inside Amazon + Africa: 5P values are bias/scale–corrected Phase2.
- Outside Amazon + Africa: 5P values match raw Phase2 predictions.

---

### 6. Generate Phase2-based bias-corrected tropical restart files

**Objective:** Use `phase2_pvariable_focus_bias_corrected` predictions as the 5P inputs
for new restart files in the **tropical band**, while keeping extratropical cells as in
the natveg_improved base restart.

1. **Select the base restart and Phase2 NetCDF:**
   - Base restart:
     - `cnp_results/run_20260228_214757_natveg_improved/updated_restart_...elm.r.0021-01-01-00000.nc`
   - Phase2 bias-corrected NetCDF:
     - `cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc`

2. **Map corrected 5P values into the restart template (tropics only):**

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc \
  --restart-file cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc  \
  --output cnp_results/run_20260305_153217_phase2_pvariable_focus/updated_restart_phase2_5P_bias_corrected_tropical.nc \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

- `--tropical-lat-range=-30,30`:
  - Only tropical cells (lat in [-30, 30]) have their 5P overwritten.
  - Outside the tropics, all variables, including 5P, remain as in the natveg_improved base restart.
- Within the tropical band:
  - Amazon + Africa cells use **bias-corrected Phase2** 5P.
  - Other tropical cells use **raw Phase2** 5P (since those cells were not bias-corrected).

3. **Write and label Phase2-based restart files:**
   - Save in a separate directory, e.g.:
     - `cnp_results/run_20260305_153217_phase2_pvariable_focus/updated_restart_phase2_5P_bias_corrected_tropical.nc`
   - Use filenames that clearly distinguish:
     - natveg-based restarts vs. Phase2-based 5P bias/scale tropical restarts.

4. **Run and compare:**
   - Run the land model from:
     - **Run A:** natveg-based 5P bias/scale two-region restarts.
     - **Run B:** Phase2-based 5P bias/scale tropical restarts.
   - Use your existing analysis scripts (profiles, regional stats, etc.) to compare:
     - 5P behavior at the Amazon site and across Amazon/Africa.
     - Any downstream variables of interest (e.g., NPP, fluxes).

This gives you a clean, repeatable way to reproduce the `phase2_pvariable_focus_bias_corrected`
behavior you liked in the Amazon solutionp profile, and to propagate that improvement into
restart-based simulations in the tropics.


