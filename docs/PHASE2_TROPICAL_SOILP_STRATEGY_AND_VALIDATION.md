# Phase 2 Tropical Soil-P Strategy: Configuration and Validation Matrix

**Purpose:** Define the two-phase global + tropical P strategy, the Phase 2 training configuration, and the validation/verification matrix for model efficiency.

---

## 1. Strategy Overview

| Phase | Scope | Role |
|-------|--------|------|
| **Phase 1** | Global domain | Train global natveg model (existing Phase 1 config). Run full pipeline to produce a **global** AI-updated restart file. |
| **Phase 2** | Tropical-only | Train a **separate** AI model on **tropical-only** data with **extreme attention to soil P variables**. Use Phase 2 only to improve P predictions in the tropics. |

**Restart merge:** Start from the Phase 1 global restart. Run Phase 2 inference (on the same grid or tropical subset). Then update the restart file so that **only in tropical grid cells** the CNP_IO variables are replaced by **Phase 2 predictions** (using `ai_predictions_to_restart.py --tropical-lat-range`). Extratropical cells keep Phase 1 values.

- **Config (Phase 1):** Use existing global natveg config (e.g. `training_config_experiment_3_global_natveg_improved_occlp.json` or your current Phase 1).
- **Config (Phase 2):** `config/training_config_phase2_tropical_soilp_only.json` (tropical-only, extreme soil-P weights).

---

## 2. Phase 2 Configuration

- **File:** `config/training_config_phase2_tropical_soilp_only.json`
- **Description:** Tropical-only natveg training with extreme focus on soil P. Same variable set as global CNP_IO; loss and variable weights emphasize P.

**Main settings:**

| Section | Key settings |
|---------|----------------|
| **data_filtering_config** | `tropical_only: true`, `tropical_lat_range: [-30, 30]`, `natveg_only: true` |
| **training_hyperparameters** | `matrix_loss_weight: 2.5`, low scalar/vector (0.06 / 0.4), `litter_p_loss_weight: 4` |
| **soil2d_weights** | `occlp_vr: 45`, `primp_vr: 26`, `labilep_vr: 18`, `solutionp_vr: 18`, `secondp_vr: 14`, soil*p_vr and litr*p_vr elevated |
| **tail_aware_weights** | Same P variables emphasized for tail-aware loss |

**Training command (from repo root):**

```bash
python train_cnp_model.py \
  --training-config-json config/training_config_phase2_tropical_soilp_only.json \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --use-tva4km \
  --tropical-only \
  --tropical-lat-range -30,30 \
  --output-dir-suffix phase2_tropical_soilp_only
```

The config’s `data_filtering_config.tropical_only` and `tropical_lat_range` are read by the trainer when `--tropical-only` and `--tropical-lat-range` are passed (see `train_cnp_model.py`).

---

## 3. Validation Matrix (Model Efficiency)

Use this matrix to validate and verify both phases and the merged restart.

### 3.1 Phase 1 (Global) – Validation

| Check | Metric / method | Target / note |
|-------|------------------|----------------|
| Global scalar | GPP, NPP, AR, HR: RMSE, R² from `cnp_metrics.json` | R² > 0.97 typical |
| Global PFT1D | pft_1d_rmse, per-PFT R² | No severe degradation |
| Global soil 2D | Per-variable RMSE/R² (e.g. occlp_vr, labilep_vr, solutionp_vr) | Baseline for comparison |
| Restart file | Run `ai_predictions_to_restart.py`, then ELM restart run | No runtime errors; restart loads |

### 3.2 Phase 2 (Tropical-only) – Validation

| Check | Metric / method | Target / note |
|-------|------------------|----------------|
| Tropical holdout scalar | GPP, NPP, AR, HR R² on **tropical test set** | Comparable or better than Phase 1 on tropics |
| Tropical holdout P (primary) | **occlp_vr**, **labilep_vr**, **solutionp_vr**: R², RMSE, NRMSE (all layers or layer-agg) | R² > 0.99 for occlp_vr; clear improvement over Phase 1 tropical |
| Tropical holdout P (other) | primp_vr, secondp_vr, soil1p_vr–soil4p_vr, litr2p_vr, litr3p_vr | Improved vs Phase 1 in tropics |
| Phase 2 cnp_metrics.json | Y_occlp_vr_layer*_r2, Y_labilep_vr_*, Y_solutionp_vr_* | Record for reporting |
| Site: Amazon | occlp_vr, solutionp_vr, labilep_vr at (303.75, −17.43) vs ground truth | Phase 2 closer to GT than Phase 1 (see §4) |

### 3.2b Entire-dataset inference for Phase 2 merge

Phase 2 is trained **tropical-only**. The run’s `cnp_predictions` folder contains only the **validation subset** (~1k gridcells) written at training time. Inference **without** `--inference-full-grid` uses the same data scope as training: tropical filter is read from the run’s `cnp_config.json` (since we now save `tropical_only` and `tropical_lat_range`), so you get the **full tropical train+validation** set (~6k samples), not just 1k. To merge P variables into the **global** restart, you need predictions on the full model grid (~21k gridcells).

1. **Run inference on full grid:** From the Phase 2 run directory, run `run_inference_all.py` with **`--inference-full-grid`** so the dataloader uses all gridcells (tropical_only=False). Output goes to `cnp_inference_entire_dataset/cnp_predictions`.
2. **Build NetCDF** from `cnp_inference_entire_dataset/cnp_predictions` (not from `cnp_predictions`).
3. **Run `ai_predictions_to_restart.py`** with that NetCDF and `--variables-to-update @config/phase2_soilp_variables_to_update.txt`.

- **Without** `--inference-full-grid`: inference uses the training data scope (tropical-only run → ~6k samples; global run → ~21k). Good for evaluation on the same domain as training.
- **With** `--inference-full-grid`: inference always uses the full global grid (~21k). Required for merging Phase 2 P predictions into a global restart.

If you use the small NetCDF built from `cnp_predictions` (~1k) or from inference without the flag (~6k tropical), the restart updater will map every model gridcell to the nearest of those points; for a proper global P merge use `--inference-full-grid` and the resulting NetCDF.

### 3.3 Merged Restart (Phase 1 + Phase 2 tropical) – Verification

| Check | Method | Target / note |
|-------|--------|----------------|
| Tropical cells | Use Phase 2 predictions for lat in `tropical_lat_range` | Only tropical cells updated with Phase 2 |
| Extratropical cells | Unchanged from Phase 1 | No overwrite outside tropics |
| Restart consistency | Run ELM with merged restart | No NaNs; no crashes; NEE/fluxes plausible in tropics |
| Site: Amazon | Extract Amazon point from merged restart; compare occlp_vr (and labilep_vr, solutionp_vr) to Phase 2 prediction and to GT | Values match Phase 2 at Amazon; closer to GT than Phase 1 |

---

## 4. Verification: Amazon and Optional Tropical Sites

### 4.1 Reference site (Amazon)

- **Lon:** 303.75°E  
- **Lat:** −17.4246° (≈ −17.434553° in grid)

**Verification steps:**

1. **Phase 1 only:** From Phase 1 run dir, get predictions at Amazon (e.g. from `cnp_predictions/` or inference CSVs). Record occlp_vr, solutionp_vr, labilep_vr (layers 1–10) vs ground truth.
2. **Phase 2 only:** From Phase 2 run dir, get predictions at Amazon. Compare same variables to GT and to Phase 1 (Phase 2 should be closer; see `docs/AMAZON_SITE_SOILP_COMPARISON_REPORT.md` for comparison template).
3. **Merged restart:** Extract Amazon point from the final restart (e.g. `scripts/extract_elm_restart_point.py`). Compare restart values for occlp_vr (and labilep_vr, solutionp_vr) to Phase 2 predictions and to GT.

**Commands (from run directory or with absolute paths):**

```bash
# Site validation stats (Phase 1 or Phase 2 run dir)
python scripts/cnp_result_validationplot_site.py <RUN_DIR> --lon 303.75 --lat -17.4246 --tolerance 0.1 --stats-only

# Extract Amazon from merged restart
python scripts/extract_elm_restart_point.py \
  --restart-file <PATH_TO_MERGED_RESTART.nc> \
  --lat -17.4246 --lon 303.75 \
  --output-file <PATH_TO_AMAZON_SINGLE_POINT.nc>
```

### 4.2 Optional: Additional tropical sites

Define other tropical (lon, lat) pairs and repeat the same checks (Phase 1 vs Phase 2 vs merged restart) for occlp_vr, solutionp_vr, labilep_vr. Document in the same format as the Amazon report.

---

## 5. Workflow Summary (Commands)

1. **Phase 1 – Global restart**
   - Train: existing Phase 1 global natveg config.
   - Inference: full grid.
   - Restart: `ai_predictions_to_restart.py` **without** `--tropical-lat-range` → global restart.

2. **Phase 2 – Tropical P model**
   - Train: `config/training_config_phase2_tropical_soilp_only.json` with `--tropical-only --tropical-lat-range -30,30`.
   - Inference: run on full grid (or tropical-only subset); only tropical cells will be used when merging.

3. **Merge Phase 2 into Phase 1 restart**
   - Input: Phase 1 global restart.
   - Predictions: Phase 2 inference output.
   - Update only tropical cells and only P (or selected) variables:  
     `ai_predictions_to_restart.py ... --tropical-lat-range -30,30 --variables-to-update occlp_vr,labilep_vr,solutionp_vr,...`  
   - `--variables-to-update`: comma-separated list of variable names (e.g. `occlp_vr,labilep_vr,solutionp_vr,...`) or a file path with `@` prefix (e.g. `@config/phase2_soilp_variables_to_update.txt`). Omit to update all CNP_IO variables in tropical cells.
   - Output: merged restart (extratropical = Phase 1; tropical = Phase 1 except selected variables replaced by Phase 2).

4. **Validation**
   - Phase 1: `cnp_metrics.json`, global validation_stats, quality report.
   - Phase 2: `cnp_metrics.json`, tropical validation_stats, Amazon (and optional sites) P comparison.
   - Merged: Amazon (and optional) extraction and comparison; ELM run with merged restart.

---

## 6. Checklist (Validation and Verification)

- [ ] Phase 1 trained and global restart produced; metrics and quality report reviewed.
- [ ] Phase 2 trained with `training_config_phase2_tropical_soilp_only.json`; tropical holdout metrics recorded.
- [ ] Phase 2 P variables (occlp_vr, labilep_vr, solutionp_vr) on tropical holdout: R²/RMSE meet targets and improve vs Phase 1.
- [ ] Amazon site: Phase 2 predictions vs GT and vs Phase 1 documented (e.g. table as in `AMAZON_SITE_SOILP_COMPARISON_REPORT.md`).
- [ ] Merged restart created with `--tropical-lat-range -30,30`.
- [ ] Amazon (and optional sites) extracted from merged restart; P variables match Phase 2 and are closer to GT than Phase 1.
- [ ] ELM run with merged restart: no errors; tropical NEE/fluxes plausible.

This completes the configuration and validation matrix for the Phase 2 tropical soil-P strategy.
