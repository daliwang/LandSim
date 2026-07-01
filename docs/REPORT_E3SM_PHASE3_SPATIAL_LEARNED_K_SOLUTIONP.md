# E3SMv3_h0 Phase 3: spatial learned-k for solutionp_vr — report

**Date:** 2026-07-01  
**Status:** Completed — restart ready for ELM comparison runs  
**Branch:** `e3smv3case`  
**Related:**

- [REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md](REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md) — scale vs R², why affine Phase 3 fails on E3SM  
- [PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md](PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md) — P3-focus / P3-moderate weight experiments  
- [WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md](WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md) — full three-phase workflow  

---

## 1. Executive summary

Phase 2 baseline tropical (`run_20260624_092639`) remains the **5P backbone**. High pooled R² on `solutionp_vr` masks **order-of-magnitude under-prediction** at Amazon/Africa reference sites. Uniform or regional-quantile multiplicative correction cannot fix both extreme anchor sites and the bulk region simultaneously.

**Recommended Phase 3 for E3SM:** **spatial learned-k** — per-cell multiplicative scale for `solutionp_vr` only, fitted in Amazon and Africa boxes using pred-only features (Ridge on log-k per soil layer). Other 4P pools stay at Phase 2 overlay values.

**Deliverable restart (Jul 2026):**

`cnp_results/run_20260630_phase3_spatial_solutionp_e3smv3_h0/updated_restart_phase3_spatial_solutionp_tropical_5P.nc` (~54 GB)

---

## 2. Problem and motivation

| Metric (Phase 2 baseline) | Amazon region | Africa region | Amazon site | Africa site |
|---------------------------|---------------|---------------|-------------|-------------|
| Pooled R² (solutionp) | ~0.997 | ~0.997 | — | — |
| Site median \|log(pred/GT)\| | — | — | **4.48** (~88× low) | **2.33** (~10× low) |
| Region p90 rel error | 0.97 | 0.98 | — | — |

Prior attempts:

- **P3-focus / P3-moderate** reweighting: did not fix scale; P3-focus worse than baseline.  
- **Affine Phase 3 v3** (`run_20260624_153307`): wrong functional form for multiplicative bias; Trendy hybrid-protect logic inapplicable to E3SM.  
- **Uniform / regional-quantile k**: fixes some cells, over- or under-corrects others; Amazon anchor remains extreme outlier vs regional distribution.

---

## 3. Diagnostics added (log-ratio + p90, not R² alone)

**Script:** `scripts/report_sensitive_p_logratio_diagnostics.py`

Metrics for `solutionp_vr`, `occlp_vr`, `labilep_vr`:

- Site **median / max \|log(pred/GT)\|** at Amazon (303.75°E, −17.43°N) and Africa (28°E, 0°N)  
- Region **p90 relative error** over Amazon / Africa boxes  

Example output (Phase 2 baseline):

`cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical/analysis/sensitive_p_logratio_diagnostics_all_runs.csv`

---

## 4. Calibration prototypes (Phase 2 tropical holdout)

### 4.1 Multiplicative calibrator (`scripts/calibrate_solutionp_multiplicative_sites.py`)

| Method | Description | Outcome |
|--------|-------------|---------|
| `site` | k = GT/PRED at anchor | Fixes anchors; **overcorrects** region (Amazon p90 rel → 91) |
| `regional_quantile` | k = clip(p50, p10, p90) of regional ratios | Africa site improves; region p90 worsens |
| `regional_optimal` | k per layer minimizes regional p90 | Best **uniform** k (~1.5–1.9×); anchors still bad |
| `site_capped` | min(site_k, regional p90) | Same as high quantile; region blow-up |

**Conclusion:** single uniform k is insufficient; spatial variation is required.

### 4.2 Spatial Phase 3 calibrator (`scripts/calibrate_solutionp_spatial_phase3.py`)

Fits on **80/20 train/test** within each region box; deployable without GT at apply time.

| Method | Deployable | Mean test p90 rel (Amazon+Africa) | Notes |
|--------|------------|-----------------------------------|-------|
| **learned** (Ridge log-k) | Yes | **1.39** | Selected for restart |
| binned (8 bins, occlp feature) | Yes | ~1.69 | Alternative |
| oracle (per-cell k) | No (needs GT) | **0.10** | Upper bound — proves spatial k works |

**Learned model features (pred-only):** log1p(solutionp colsum), log1p(occlp colsum), log1p(total P colsum), log1p(solutionp layer-1), lat, lon360_norm.

**Params:** `cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical/analysis/solutionp_spatial_phase3_learned_params.json`

**Full method comparison:** `.../analysis/solutionp_spatial_phase3_compare_eval.json`  
**CSV summary:** `.../analysis/solutionp_calibration_methods_comparison.csv`

### 4.3 Spatial learned-k — holdout test metrics (learned)

| Region | p90 rel before → after | median \|log\| before → after | frac rel < 0.5 before → after |
|--------|------------------------|-------------------------------|-------------------------------|
| Amazon test | 0.97 → 1.16 | 1.21 → **0.34** | 0.47 → **0.65** |
| Africa test | 0.98 → 1.63 | 1.61 → **0.47** | 0.40 → **0.53** |

| Site | median \|log\| before → after |
|------|-------------------------------|
| Amazon | 4.48 → **1.28** |
| Africa | 2.33 → **0.28** |

Regional p90 on held-out test can still exceed 1.0 (tail cells); bulk median log-ratio and fraction of “good” cells improve strongly. Oracle ceiling (p90 ~0.08) indicates room to improve features or move to Phase 2.5 log-relative retrain.

---

## 5. Site profile plots

Under `cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical/analysis/site_plots/`:

| Plot | Content |
|------|---------|
| `amazon_africa_5p_summary_baseline.png` | GT vs Phase 2 baseline |
| `amazon_africa_5p_summary_spatial_learned_k.png` | GT + baseline + spatial k |
| `amazon_africa_5p_summary_baseline_and_spatial.png` | 4-row side-by-side |
| `amazon_5p_summary_baseline.png`, `africa_5p_summary_baseline.png` | Per-site baseline |
| `amazon_5p_summary_spatial_learned_k.png`, `africa_5p_summary_spatial_learned_k.png` | Per-site with spatial k |

**Visual summary:** Africa `solutionp` tracks GT much better after spatial k; Amazon `solutionp` improves but remains below GT; other 4P unchanged (solutionp-only correction).

---

## 6. Phase 3 restart pipeline (spatial learned-k)

**Script:** `scripts/run_e3smv3_h0_phase3_spatial_solutionp.sh`

```bash
source config/e3smv3_h0_env.sh
export PHASE1_RUN_DIR=cnp_results/run_20260623_172201_e3smv3_h0_phase1_global
export PHASE2_RUN_DIR=cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical
export BASE_RESTART=$PHASE1_RUN_DIR/updated_restart_base.nc
bash scripts/run_e3smv3_h0_phase3_spatial_solutionp.sh
```

**Steps:**

1. Symlink Phase 1 `cnp_inference_entire_dataset` (global E3SM grid).  
2. Overlay Phase 2 tropical 5P onto global `soil_2d_predictions` by `(lon, lat)` (`overlay_5p_by_coords.py`).  
3. Apply spatial learned-k to `solutionp_vr` in Amazon + Africa boxes (`apply_solutionp_spatial_phase3.py`).  
4. Replace `soil_2d_predictions/predictions_Y_solutionp_vr.csv` with corrected values.  
5. NetCDF → `ai_predictions_to_restart.py` (tropical band ±23.5°, all 5P updated).

**Run directory:** `cnp_results/run_20260630_phase3_spatial_solutionp_e3smv3_h0`

| Output | Path |
|--------|------|
| **Restart** | `updated_restart_phase3_spatial_solutionp_tropical_5P.nc` |
| NetCDF | `comparison_results/ai_predictions_phase3_spatial_solutionp.nc` |
| README | `README_phase3_spatial_solutionp.txt` |

---

## 7. Restart comparison matrix (for ELM runs)

| Case | Restart path |
|------|----------------|
| Phase 1 base | `run_20260623_172201.../updated_restart_base.nc` |
| Phase 2 raw 5P | `run_20260624_092639.../updated_restart_phase2_tropical_5P_raw.nc` |
| Phase 3 affine v3 | `run_20260624_153307.../updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc` |
| **Phase 3 spatial k (new)** | `run_20260630_phase3_spatial_solutionp_e3smv3_h0/updated_restart_phase3_spatial_solutionp_tropical_5P.nc` |

**Difference vs Phase 2 restart:** only `solutionp_vr` is spatially bias-corrected in Amazon/Africa; labilep, occlp, secondp, primp are Phase 2 overlay (same as Phase 2 restart for those pools).

---

## 8. Scripts reference

| Script | Role |
|--------|------|
| `report_sensitive_p_logratio_diagnostics.py` | Site log-ratio + region p90 diagnostics |
| `calibrate_solutionp_multiplicative_sites.py` | Uniform-k prototypes (site, quantile, optimal) |
| `calibrate_solutionp_spatial_phase3.py` | Fit binned / learned / oracle spatial k |
| `apply_solutionp_spatial_phase3.py` | Apply saved params at inference (no GT) |
| `run_e3smv3_h0_phase3_spatial_solutionp.sh` | End-to-end Phase 3 spatial + restart |
| `overlay_5p_by_coords.py` | Overlay Phase 2 5P onto global grid |
| `region_box_utils.py` | Amazon/Africa box masking (lon 0–360) |

**Env:** `PHASE3_SPATIAL_SUFFIX=e3smv3_h0_phase3_spatial_solutionp` in `config/e3smv3_h0_env.sh`

---

## 9. Recommended next steps

1. **ELM restart simulation** with `updated_restart_phase3_spatial_solutionp_tropical_5P.nc`; compare against Phase 2 raw and Phase 3 v3 affine.  
2. **Richer features** for learned k (climate/soil inputs from model X) to close gap toward oracle ceiling.  
3. **Phase 2.5 fine-tune** with log-relative loss on denormalized `solutionp_vr`, checkpoint selected by regional log-ratio metrics.  
4. Do **not** use validation R² alone to accept solutionp corrections.

---

## 10. Key run IDs

| Run | Role |
|-----|------|
| `run_20260623_172201_e3smv3_h0_phase1_global` | Phase 1 global + base restart |
| `run_20260624_092639_e3smv3_h0_phase2_tropical` | **Phase 2 backbone** + spatial k fit |
| `run_20260624_153307_e3smv3_h0_phase3_tworegions` | Legacy affine Phase 3 |
| `run_20260630_phase3_spatial_solutionp_e3smv3_h0` | **New spatial-k Phase 3 restart** |
