# Trendy vs E3SMv3_h0: three-phase CNP workflow comparison

**Document type:** Methods & results comparison (manuscript draft material)  
**Date:** 2026-07-01  
**Repository branch:** `e3smv3case`  
**Audience:** Paper publication, internal reproducibility  

**Related internal reports**

- [REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md](REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md) — training-data distribution analysis  
- [REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md](REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md) — R² vs magnitude for ELM-sensitive P  
- [REPORT_E3SM_PHASE3_SPATIAL_LEARNED_K_SOLUTIONP.md](REPORT_E3SM_PHASE3_SPATIAL_LEARNED_K_SOLUTIONP.md) — E3SM spatial-k Phase 3  
- [PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md](PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md) — Trendy hybrid Phase 3  
- [WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md](WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md) — E3SM runbook  
- [WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md](WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md) — Trendy runbook  

---

## Abstract

We compare two applications of the same three-phase machine-learning workflow for initializing ELM soil phosphorus (5P) pools: a **Trendy** reference case and an **E3SMv3_h0** case. Both cases use matched Phase 1 (global natveg) and Phase 2 (tropical P-focused) training hyperparameters and loss weighting, but differ in training trajectories, variable I/O schema, tropical latitude definition, and Phase 3 bias-correction methodology. Phase 1 and Phase 2 are **configuration-equivalent in design** (same JSON templates aside from tropical lat bounds); models are **not interchangeable** because ground-truth distributions and grid extent differ. Phase 3 follows a **common regional strategy** (Amazon + Africa boxes, Phase 1 grid seed, Phase 2 5P overlay, tropical restart update) but **diverges in correction method**: Trendy production uses **regional affine v3 bias correction with hybrid protection** of `solutionp_vr` and `occlp_vr`, whereas E3SM requires **multiplicative, spatially varying correction of `solutionp_vr`** because high pooled R² masks order-of-magnitude scale error at reference sites. We document reference runs, evaluation metrics suitable for publication (log-ratio error, regional relative error quantiles), and recommended restart artifacts for paired ELM experiments.

---

## 1. Introduction

Land-surface models such as ELM depend sensitively on soil solution and occluded phosphorus (`solutionp_vr`, `occlp_vr`, `labilep_vr`). A three-phase AI workflow has been developed to produce restart files that inject learned 5P profiles into ELM:

1. **Phase 1** — global natveg model and base restart.  
2. **Phase 2** — tropical-only fine-tuning with emphasis on soil P pools.  
3. **Phase 3** — regional bias correction in Amazon and Africa, merged into a tropical-band restart.

The Trendy case established this pipeline on TRENDY-style training batches. The E3SMv3_h0 case **reuses the same pipeline architecture and baseline training configs** on E3SMv3 trajectory data (~396k gridcells). A central scientific question for publication is whether **the same Phase 3 strategy** (affine regional correction) transfers from Trendy to E3SM. Empirical evidence shows it does **not** for `solutionp_vr`: error structure differs (correlation vs scale), requiring a case-specific Phase 3 method while preserving the shared workflow skeleton.

---

## 2. Materials and methods

### 2.1 Study cases

| Item | Trendy | E3SMv3_h0 |
|------|--------|-----------|
| **Training data root** | `TrainingData/Trendy_1_data_CNP` | `TrainingData/E3SMv3_h0` |
| **Variable list** | `CNP_IO_updated9_dev_dw.txt` | `CNP_IO_e3smv3_h0.txt` |
| **Approx. global gridcells (inference)** | Trendy full grid | ~395,784 (Phase 1); ~105k tropical (Phase 2 only) |
| **ELM restart template** | TRENDY2024 / ICB1850 default IC | `E3SMV3_025` land restart (20240214) |
| **Git branch (E3SM adaptations)** | main workflow | `e3smv3case` (zero-scalar I/O, batched inference) |

### 2.2 Reference model runs

**Table 1 — Canonical reference runs**

| Phase | Trendy | E3SMv3_h0 |
|-------|--------|-----------|
| Phase 1 global | `run_20260315_113900_phase1_global` | `run_20260623_172201_e3smv3_h0_phase1_global` |
| Phase 2 tropical (baseline) | `run_20260315_175250_phase2_tropical` | `run_20260624_092639_e3smv3_h0_phase2_tropical` |
| Phase 3 (affine v3, all 5P) | `run_20260512_083650_phase3_tworegions_(africa_improvement)` | `run_20260624_153307_e3smv3_h0_phase3_tworegions` |
| Phase 3 (recommended production) | `run_20260617_181123_phase3_hybrid_protect_occlp_solutionp` | `run_20260630_phase3_spatial_solutionp_e3smv3_h0` |

Restart filenames (for paired ELM experiments):

| Case | Restart path |
|------|----------------|
| Trendy Phase 2 raw | `run_20260315_175250_phase2_tropical/updated_restart_phase2_tropical_5P_raw.nc` |
| Trendy Phase 3 hybrid | `run_20260617_181123_phase3_hybrid_protect_occlp_solutionp/updated_restart_phase3_hybrid_labilep_secondp_primp_protect_occlp_solutionp_tropical.nc` |
| E3SM Phase 2 raw | `run_20260624_092639_e3smv3_h0_phase2_tropical/updated_restart_phase2_tropical_5P_raw.nc` |
| E3SM Phase 3 affine v3 | `run_20260624_153307_e3smv3_h0_phase3_tworegions/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc` |
| E3SM Phase 3 spatial k | `run_20260630_phase3_spatial_solutionp_e3smv3_h0/updated_restart_phase3_spatial_solutionp_tropical_5P.nc` |

### 2.3 Shared three-phase architecture

Both cases implement the same logical pipeline:

```text
Phase 1: train global model → full-grid inference → base restart
Phase 2: train tropical model → tropical inference → overlay 5P on base (tropical band)
Phase 3: seed Phase 1 global grid → overlay Phase 2 5P by (lon, lat)
         → regional correction (Amazon box, Africa box)
         → merge → NetCDF → update tropical 5P in restart
```

Shared software (representative):

- Training: `train_cnp_model.py`  
- Inference: `scripts/run_inference_all.py`  
- Phase 3 overlay: `scripts/overlay_5p_by_coords.py`  
- Restart generation: `scripts/ai_predictions_to_netcdf.py`, `scripts/ai_predictions_to_restart.py`  

Five soil P variables throughout: `labilep_vr`, `occlp_vr`, `solutionp_vr`, `secondp_vr`, `primp_vr`.

### 2.4 Phase 1 and Phase 2 — training configuration parity

**Table 2 — Phase 1 training config (identical JSON)**

| Setting | Value |
|---------|--------|
| Config file | `config/training_config_experiment_3_global_natveg_improved.json` |
| Epochs | 120 |
| Optimizer | AdamW, lr = 8×10⁻⁵, cosine schedule |
| Normalization | Individual (per-variable MinMax) |
| Data filter | `natveg_only: true`, drop lon 0° and 358.75° |

**Table 3 — Phase 2 baseline training config (equivalent design)**

| Setting | Trendy (`training_config_phase2_tropical_soilp_only.json`) | E3SM (`training_config_e3smv3_h0_phase2_tropical.json`) |
|---------|-----------------------------------------------------------|--------------------------------------------------------|
| Epochs | 150 | 150 |
| Learning rate | 5×10⁻⁵ | 5×10⁻⁵ |
| `matrix_loss_weight` | 2.5 | 2.5 |
| `solutionp_vr` weight | 50 | 50 |
| `occlp_vr` weight | 42 | 42 |
| `labilep_vr` weight | 28 | 28 |
| Tropical training filter | `tropical_only: true` | `tropical_only: true` |
| **Tropical lat range** | **±30°** | **±23.5°** |

Verified: the two Phase 2 JSON files differ **only** in `_description` and `tropical_lat_range` ([−30, 30] vs [−23.5, 23.5]). The Trendy reference run `run_20260315_175250` used `training_config_phase2_tropical_soilp_only.json` (confirmed in `cnp_config.json`).

**Table 4 — Structural differences outside JSON configs**

| Aspect | Trendy | E3SMv3_h0 |
|--------|--------|-----------|
| Scalar outputs (GPP, NPP, AR, HR) | 4 | **0** (absent from batches) |
| `landfrac` vs `LANDFRAC_PFT` | `landfrac` | `LANDFRAC_PFT` only |
| Phase 1 inference | Standard | Batched (`--inference-batch-size 4096`), `--derive-np-from-c` |
| Phase 2/3 inference | `--no-derive-np-from-c` | Same (explicit in E3SM scripts) |
| Tropical restart latitude band | ±30° (typical) | ±23.5° |

### 2.5 Ground-truth distribution differences (Phase 2 tropical, ±23.5°)

From [REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md](REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md) (50-batch sample, natveg tropical filter):

**Table 5 — Raw tropical target statistics (all layers pooled)**

| Variable | Case | mean | p50 | p95 | max |
|----------|------|------|-----|-----|-----|
| `Y_solutionp_vr` | Trendy | 0.092 | 0.0086 | 0.32 | 4.91 |
| `Y_solutionp_vr` | E3SM | 0.018 | 0.0096 | 0.069 | 0.42 |
| `Y_labilep_vr` | Trendy | 67.1 | 54.5 | 218 | 503 |
| `Y_labilep_vr` | E3SM | 24.6 | 20.0 | 69.4 | 199 |
| `Y_occlp_vr` | Trendy | 375 | 373 | 623 | 904 |
| `Y_occlp_vr` | E3SM | 159 | 140 | 424 | 1102 |

E3SM tropical phosphorus targets are **lower and less heavy-tailed** than Trendy despite shared loss weights. Missing scalar fluxes in E3SM are confirmed but **weakly correlated** with `solutionp_vr` (|r| < 0.14 on Trendy sample); they are unlikely to be the primary cause of the Phase 2 gap.

### 2.6 Phase 3 — shared strategy, different methods

**Table 6 — Phase 3 workflow steps (common)**

| Step | Description |
|------|-------------|
| 1 | Symlink or reuse Phase 1 **full-grid** inference (preserves E3SM/Trendy lon–lat) |
| 2 | Overlay Phase 2 **raw** 5P predictions by rounded `(lon, lat)` |
| 3 | Apply **region-specific** correction inside Amazon and Africa boxes |
| 4 | Merge corrected fields; build NetCDF; patch tropical band of Phase 1 restart |

**Table 7 — Region boxes**

| Region | Trendy (typical) | E3SMv3_h0 |
|--------|------------------|-----------|
| Amazon lat | −30° … 10° | −23.5° … 10° |
| Amazon lon | 270° … 330° (0–360) | 270° … 330° |
| Africa lat | −15° … 15° (eval) / ±30° (some configs) | −23.5° … 23.5° |
| Africa lon | 0° … 30° | 0° … 30° |
| Region config (E3SM) | — | `training_config_e3smv3_h0_amazon_5p_box.json`, `..._africa_5p_box.json` |
| Region config (Trendy) | `training_config_amazon_5p_box.json`, `..._africa_5p_box.json` | — |

**Table 8 — Phase 3 correction methods (divergence)**

| Element | Trendy (recommended) | E3SMv3_h0 (recommended) |
|---------|---------------------|---------------------------|
| **Functional form** | Per-layer **affine**: GT ≈ a·pred + b | Per-cell **multiplicative** (solutionp only): pred_corr = k·pred |
| **Fitter** | `apply_5p_bias_scale_correction.py --regional-fit-v3` | `calibrate_solutionp_spatial_phase3.py` (Ridge on log-k) + `apply_solutionp_spatial_phase3.py` |
| **Variables corrected** | **Hybrid:** v3 on `labilep_vr`, `secondp_vr`, `primp_vr` only; **raw Phase 2** for `solutionp_vr`, `occlp_vr` | **Spatial k** on `solutionp_vr` only; other 4P = Phase 2 overlay |
| **Rationale** | Raw solutionp/occlp already high R²; v3 **degrades** them | High R² but **scale error** at sites; affine v3 **fails** to boost magnitude |
| **Merge script** | `merge_5p_hybrid_selective_v3.py` | In-place solutionp replace (no 5P affine merge) |
| **Orchestration** | `run_phase3_tworegions.sh` / hybrid workflow | `run_e3smv3_h0_phase3_spatial_solutionp.sh` |

Reference sites used for site-level diagnostics (both cases):

- **Amazon:** 303.75°E, −17.434553°N (nearest grid ≈ −56.375°E, −17.375°N on −180…180 grid)  
- **Africa:** 28°E, 0°N (nearest grid ≈ 27.875°E, 0.125°N)  

### 2.7 Evaluation metrics (publication-oriented)

Standard pooled R² is **insufficient** for ELM-sensitive P when most gridcells are near zero. Recommended metrics:

1. **Site median absolute log-ratio:** median over layers of |log(pred/GT)| at reference sites.  
2. **Regional p90 relative error:** 90th percentile of |pred−GT|/|GT| over all cell×layer pairs in Amazon/Africa boxes.  
3. **Fraction with relative error < 0.5:** bulk “good cell” fraction.  
4. **GT-decile stratified error:** avoid domination by near-zero cells.  

Implementation: `scripts/report_sensitive_p_logratio_diagnostics.py`, `scripts/compare_5p_gt_two_regions_inference.py pred-eval`.

For manuscript figures: vertical 5P profiles at reference sites (`analysis/site_plots/*_5p_summary_*.png`).

---

## 3. Results

### 3.1 Phase 2 — correlation vs scale (E3SM baseline)

E3SM Phase 2 baseline (`run_20260624_092639`) on tropical holdout:

**Table 9 — E3SM Phase 2: pooled R² vs site scale error (`solutionp_vr`)**

| Metric | Amazon region | Africa region | Amazon site | Africa site |
|--------|---------------|---------------|-------------|-------------|
| Pooled R² | ~0.997 | ~0.997 | — | — |
| Site median \|log(pred/GT)\| | — | — | **4.48** | **2.33** |
| Site column-sum rel. error | — | — | ~99% | ~86% |
| Region p90 rel. error | 0.97 | 0.98 | — | — |

The model learns **rank order and profile shape** (high R²) but **under-predicts magnitude** at ELM-sensitive sites.

### 3.2 Phase 3 — Trendy hybrid protect

On Trendy Phase 2 inference (`run_20260315_175250`), full v3 correction of all 5P **degraded** `occlp_vr` and `solutionp_vr` (e.g. Amazon occlp R² 0.91 → 0.19). The adopted **hybrid protect** strategy leaves `solutionp_vr` and `occlp_vr` at raw Phase 2 values and applies v3 only to `labilep_vr`, `secondp_vr`, `primp_vr` ([PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md](PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md)).

### 3.3 Phase 3 — E3SM affine v3 (Trendy method transplanted)

E3SM run `run_20260624_153307` applied the **same affine v3 + merge-all-5P** approach as early Trendy Phase 3. This did **not** resolve E3SM `solutionp_vr` scale error because:

- Affine fits are dominated by near-zero cells (a ≈ 1, b ≈ 0).  
- Multiplicative bias at peaks requires scale factors, not intercept-dominated maps.  
- Trendy “protect solutionp” logic is **inverted** for E3SM (solutionp needs correction, not protection).

### 3.4 Phase 3 — E3SM spatial learned-k (case-specific method)

Spatial Ridge log-k calibration (80/20 train/test per region, pred-only features):

**Table 10 — E3SM spatial learned-k holdout test (Phase 2 tropical inference)**

| Region | p90 rel before → after | median \|log\| before → after | frac rel < 0.5 before → after |
|--------|------------------------|-------------------------------|-------------------------------|
| Amazon | 0.97 → 1.16 | 1.21 → **0.34** | 0.47 → **0.65** |
| Africa | 0.98 → 1.63 | 1.61 → **0.47** | 0.40 → **0.53** |

| Site | median \|log\| before → after |
|------|-------------------------------|
| Amazon | 4.48 → **1.28** |
| Africa | 2.33 → **0.28** |

**Oracle per-cell k** (upper bound, not deployable) achieves p90 rel ≈ 0.08–0.12, showing remaining gap is **predicting k from features**, not the multiplicative form.

Site 5P plots: Africa `solutionp_vr` profile aligns much more closely with ground truth after spatial k; Amazon improves but remains below ground truth at the anchor.

### 3.5 Failed E3SM Phase 2 variants (weight-only)

| Experiment | Outcome vs baseline |
|------------|---------------------|
| P3-focus (aggressive 3P weights) | Worse on all 5P regionally and at sites |
| P3-moderate | Better than P3-focus; still below baseline regionally |

Confirms that **loss reweighting without log-ratio / scale-aware objectives** does not fix E3SM `solutionp_vr` magnitude ([PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md](PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md)).

---

## 4. Discussion

### 4.1 What is shared between Trendy and E3SMv3

- Three-phase **architecture** and restart patching logic.  
- Phase 1 JSON and Phase 2 **hyperparameters / P loss weights** (Trendy-equivalent baseline).  
- Amazon + Africa **regional framing** for Phase 3.  
- Five pool coupling through shared training and overlay.

### 4.2 What must differ

- **Training data** and **I/O schema** (E3SM lacks scalar flux targets).  
- **Tropical latitude band** (±23.5° vs ±30°) for E3SM consistency.  
- **Phase 3 correction method** driven by **error geometry**:  
  - Trendy: errors are primarily **bias in weak pools**; strong pools must be **protected** from affine correction.  
  - E3SM: errors for `solutionp_vr` are **multiplicative scale** at high-R² sites; affine regional fits and hybrid protect are **mis-specified**.

### 4.3 Implications for paired ELM experiments

For publication-quality comparison, we recommend **four restart arms**:

1. Phase 1 base (optional control)  
2. Phase 2 raw tropical 5P (both cases)  
3. Case-native Phase 3 production (Trendy hybrid vs E3SM spatial k)  
4. Cross-method control (E3SM affine v3 — expected poor for solutionp — optional)

Report ELM outcomes with both **bulk regional statistics** and **reference-site log-ratio metrics**, not R² alone.

### 4.4 Limitations

- Spatial learned-k is fitted on **holdout tropical inference** with ground truth; deployment uses **pred-only features** — generalization to unseen climates within boxes is not fully validated.  
- Amazon anchor site remains an **extreme outlier** relative to regional k distribution; uniform or low-dimensional k cannot fully resolve anchor and bulk simultaneously.  
- Restart files (~54 GB) and training batches are **not** versioned in git; paths in Table 1 are the reproducibility anchor.  
- Trendy and E3SM ELM templates differ; strict cross-case ELM comparison requires careful experimental design (same model version, comparable spinup protocol).

### 4.5 Future work (manuscript-ready bullet list)

1. Phase 2.5 fine-tune with **log-relative loss** on denormalized `solutionp_vr`, checkpoint selection by log-ratio metrics.  
2. Richer spatial-k features (climate / soil inputs from model X).  
3. GT-decile and tail-stratified Phase 3 objectives.  
4. ELM sensitivity experiments linking restart `solutionp_vr` to plant P limitation metrics.

---

## 5. Conclusions

1. Trendy and E3SMv3_h0 share **the same Phase 1 and Phase 2 baseline training configuration in design**, but train on **different ELM trajectories** with different I/O and tropical extent.  
2. Both use the **same three-phase regional strategy** for Phase 3 (Amazon + Africa, overlay, restart).  
3. Phase 3 **methods diverge by necessity**: Trendy **affine v3 + hybrid protect** preserves strong `solutionp_vr`/`occlp_vr`; E3SM requires **spatial multiplicative correction of `solutionp_vr`**.  
4. **Pooled R² is misleading** for E3SM `solutionp_vr`; log-ratio and regional quantile errors should be reported in publications.  
5. Recommended E3SM production restart: **`run_20260630_phase3_spatial_solutionp_e3smv3_h0`**. Recommended Trendy production restart: **`run_20260617_181123_phase3_hybrid_protect_occlp_solutionp`**.

---

## 6. Suggested figures for manuscript

| Figure | Content | Source artifact |
|--------|---------|-----------------|
| Fig. 1 | Three-phase workflow schematic (shared) | New diagram from §2.3 |
| Fig. 2 | Raw tropical GT distributions Trendy vs E3SM (solutionp, labilep, occlp) | Table 5 / data comparison report |
| Fig. 3 | Site 5P profiles: GT vs Phase 2 vs Phase 3 (both cases, 2×5 panels) | `analysis/site_plots/amazon_africa_5p_summary_baseline_and_spatial.png` (E3SM); Trendy equivalent from hybrid run |
| Fig. 4 | R² vs median \|log ratio\| schematic or scatter across methods | Table 9–10 |
| Fig. 5 | Phase 3 method comparison (affine vs spatial k vs oracle p90) | `solutionp_spatial_phase3_compare_eval.json` |

---

## 7. Software and data availability

| Resource | Location |
|----------|----------|
| E3SM workflow env | `config/e3smv3_h0_env.sh` |
| Phase 3 spatial (E3SM) | `scripts/run_e3smv3_h0_phase3_spatial_solutionp.sh` |
| Phase 3 affine (both) | `scripts/apply_5p_bias_scale_correction.py` |
| Diagnostics | `scripts/report_sensitive_p_logratio_diagnostics.py` |
| Spatial k fit/apply | `scripts/calibrate_solutionp_spatial_phase3.py`, `scripts/apply_solutionp_spatial_phase3.py` |
| Training data | `/mnt/proj-shared/AI4BGC_7xw/TrainingData/` (not in repository) |
| Model outputs | `cnp_results/run_*` (not in repository) |

---

## 8. Appendix — Quick parity checklist

| Question | Trendy | E3SMv3 |
|----------|--------|--------|
| Same Phase 1 config JSON? | Yes | Yes |
| Same Phase 2 weights/epochs? | Yes (±30° lat) | Yes (±23.5° lat) |
| Same Phase 3 workflow steps? | Yes | Yes |
| Same Phase 3 correction equation? | **No** (affine + hybrid) | **No** (spatial multiplicative k) |
| Same trained model weights? | **No** | **No** |
| Same restart file? | **No** | **No** |

---

*Document prepared for internal use and adaptation into a methods/results section. Update Table 1 run IDs when new production runs supersede Jul 2026 artifacts.*
