# TES-SFA North 10% Run — Prediction Quality Analysis Report

**Run directory:** `cnp_results/run_20260319_134216_tesnorth10pc`  
**Dataset:** TES North ERA5 10% (`TESNorthERA510PCT`)  
**Branch context:** `tessfa_north`  
**Analysis date:** 2026-08-05 / 2026-08-06  

This report documents the post-training quality analysis and paper-oriented scatter plots generated for the TES-SFA North 10% CNP model run.

---

## 1. Purpose

1. Quantify test-split prediction quality (good / ok / bad) across scalar, PFT-1D, and soil-2D outputs.
2. Produce summary plots and “top bad” GT vs Pred scatters for diagnosis.
3. Produce **good-example** GT vs Pred scatters suitable for manuscript figures.

---

## 2. Run overview

| Item | Value |
|------|--------|
| Run ID | `run_20260319_134216_tesnorth10pc` |
| Config | `cnp_config.json` (individual normalization) |
| Training log | `cnp_training_20260319_134216.log` |
| Model | `cnp_predictions/model.pth` |
| Test predictions | `cnp_predictions/` (scalar, `pft_1d_*`, `soil_2d_*`) |
| Full-domain inference | `cnp_inference_entire_dataset/` |
| AI restart product | `updated_restart_AI_uELM_TESNorthERA510PCT_I1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc` |

Quality metrics below are for the **held-out test split** under `cnp_predictions/` (not the full-domain inference directory), with the default natveg filter applied (exclude `PCT_NATVEG=0` or `PCT_NAT_PFT_0=100`).

---

## 3. Methods and commands

All commands were run from the repository root with the project Python environment
(`/mnt/proj-shared/AI4BGC_7xw/ai4bgc_env`).

### 3.1 Validation statistics

```bash
cd cnp_results/run_20260319_134216_tesnorth10pc
python ../../scripts/cnp_result_validationplot.py . --stats-only
```

**Output:** `validation_stats.csv` (920 PFT/layer/scalar entries with RMSE, MAE, R², GT/Pred ranges).

### 3.2 Quality report and worst-case scatters

```bash
python ../../scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --output-dir analysis
```

This categorizes each entry and writes summary CSVs/HTML/PNG under `analysis/`, then calls
`cnp_result_validationplot.py --worst-only` to populate `analysis/top_bad_plots/`.

**Classification thresholds (defaults):**

| Class | Criteria |
|-------|----------|
| **Good** | R² ≥ 0.9 and relative RMSE ≤ 0.1 and relative MAE ≤ 0.1 |
| **OK** | R² ≥ 0.7 and relative RMSE ≤ 0.25 and relative MAE ≤ 0.25 |
| **Bad** | Below OK thresholds |

Relative errors use GT range: `RMSE / (gt_max − gt_min)` (same for MAE).

### 3.3 Paper-oriented good-example scatters

```bash
python scripts/plot_good_prediction_examples.py \
  cnp_results/run_20260319_134216_tesnorth10pc
```

**Script:** `scripts/plot_good_prediction_examples.py`  
**Output:** `analysis/good_examples/` (300 DPI PNGs with annotated R², RMSE, MAE, N, and 1:1 line).

Default variables:

- **Scalar:** `Y_GPP`, `Y_NPP`, `Y_AR`, `Y_HR`
- **1D (All-PFTs + 2 strongest PFTs each):** `leafc`, `deadstemc`, `totvegc`, `livecrootc`, `cpool`
- **2D (All-layers + 2 strongest layers each):** `soil4c_vr`, `litr3c_vr`, `secondp_vr`, `occlp_vr`

---

## 4. Overall quality results

From `analysis/quality_summary_report.txt`:

| Metric | Value |
|--------|--------|
| Variables analyzed | 71 |
| Total prediction entries | 920 |
| **Good** | **859 (93.4%)** |
| OK | 39 (4.2%) |
| Bad | 22 (2.4%) |

**Interpretation:** Overall test performance is strong. Nearly all “bad” flags are concentrated in plant N/P pools (`npool`, `ppool`) and sparse PFT 13 vegetation states—not in bulk C pools or scalar fluxes.

---

## 5. Strongest and weakest variables

### 5.1 Best (100% good examples)

Scalar fluxes and many structural pools are excellent, including:

- `Y_GPP`, `Y_NPP`, `Y_AR`, `Y_HR`
- `totvegc`, `deadstemc`, `deadcrootc`, `cpool`, and related storage terms
- Deep soil / mineral P pools such as `soil4p_vr`, `litr3*_vr`, `secondp_vr`, `occlp_vr` (variable-level summaries)

### 5.2 Worst / diagnostic focus

| Variable | Quality pattern | Notes |
|----------|-----------------|-------|
| `ppool` | 37.5% good, 50% bad | Bad PFTs: 1, 2, 7, 8, 10, 13, 14, 15 |
| `npool` | 43.8% good, 43.8% bad | Bad PFTs: 1, 2, 7, 8, 10, 14, 15 |
| `primp_vr` | 0% good, 100% ok | R² typically ~0.79–0.82 (OK band) |
| `solutionp_vr` | 0% good, 100% ok | R² typically ~0.77–0.81 (OK band) |
| `soil1c/n/p_vr` | ~60% good, ~40% ok | Surface soil C/N/P weaker than deeper pools |
| leaf/froot/`tlai` | ~93.8% good | Single bad PFT (13) driven by sparsity |

All 22 “bad” entries are **1D**; no soil-layer entry was classified bad under default thresholds.

---

## 6. Output inventory

### 6.1 Quality analysis (`analysis/`)

| File / directory | Description |
|------------------|-------------|
| `quality_summary_report.txt` | Text summary (overall, best/worst, bad-count) |
| `prediction_quality_report.html` | Interactive HTML report |
| `detailed_quality_assessment.csv` | Per-entry quality labels |
| `variable_quality_summary.csv` | Per-variable good/ok/bad counts |
| `overall_prediction_quality.png` | Pie chart of quality classes |
| `prediction_quality_by_variable.png` | Stacked bars by variable |
| `r2_vs_rmse.png` | R² vs relative RMSE scatter |
| `top_bad_plots/` | GT vs Pred for worst variables (~211 PNGs) |
| `../validation_stats.csv` | Full validation metrics table |

`top_bad_plots/` layout:

- `aggregate_all/` — all PFTs or all layers combined  
- `aggregate_bad/` — bad PFTs/layers only  
- `by_pft_layer/` — individual PFT or layer panels  
- `train_val_loss.png` — training/validation loss curve  

### 6.2 Good examples for papers (`analysis/good_examples/`)

| Subdirectory | Contents |
|--------------|----------|
| `scalar/` | GPP, NPP, AR, HR |
| `aggregate_1d/` | All-PFT panels for selected pools |
| `by_pft/` | Two strong PFTs per selected 1D variable |
| `aggregate_2d/` | All-layer panels for selected soil variables |
| `by_layer/` | Two strong layers per selected 2D variable |

**Recommended manuscript figures (high R², clear dynamic range):**

| Plot | Approx. R² |
|------|------------|
| `scalar/Y_GPP_gt_vs_pred.png` | 0.988 |
| `scalar/Y_NPP_gt_vs_pred.png` | 0.986 |
| `aggregate_1d/1D_totvegc_AllPFTs_gt_vs_pred.png` | 0.997 |
| `aggregate_1d/1D_leafc_AllPFTs_gt_vs_pred.png` | 0.996 |
| `aggregate_1d/1D_deadstemc_AllPFTs_gt_vs_pred.png` | 0.996 |
| `aggregate_2d/2D_soil4c_vr_AllLayers_gt_vs_pred.png` | 0.993 |
| `aggregate_2d/2D_secondp_vr_AllLayers_gt_vs_pred.png` | 0.993 |
| `aggregate_2d/2D_litr3c_vr_AllLayers_gt_vs_pred.png` | 0.991 |

Pair these with a small “challenging cases” panel from `top_bad_plots/` (e.g. `npool`/`ppool` All-PFTs or `solutionp_vr`) for a balanced figure set.

---

## 7. How to regenerate

```bash
# From repo root
RUN=cnp_results/run_20260319_134216_tesnorth10pc
PY=/mnt/proj-shared/AI4BGC_7xw/ai4bgc_env/bin/python   # or your env

cd "$RUN"
$PY ../../scripts/cnp_result_validationplot.py . --stats-only
$PY ../../scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv --output-dir analysis

cd ../..
$PY scripts/plot_good_prediction_examples.py "$RUN" \
  --vars-1d leafc,deadstemc,totvegc \
  --vars-2d soil4c_vr,secondp_vr,litr3c_vr
```

Optional: customize thresholds via `generate_prediction_quality_report.py`
(`--r2-good`, `--rmse-good`, etc.). See `docs/README_prediction_quality.md`
and `docs/CNP_pipeline_runbook.md` (sections 4 / 4.0.1).

---

## 8. Related artifacts (not committed)

Large run products remain under `cnp_results/run_20260319_134216_tesnorth10pc/` and are
**not** versioned in git (model weights, CSVs, PNGs, NetCDF restart). This report and
`scripts/plot_good_prediction_examples.py` document how to reproduce the analysis on the
stored run directory.

---

## 9. Summary

The TES North 10% model achieves **93.4% good** test predictions under standard
thresholds. Scalar C fluxes and major vegetation/soil C pools are manuscript-ready;
remaining weaknesses are primarily `npool`/`ppool` across several PFTs and OK-tier
(not bad) performance for `primp_vr` / `solutionp_vr`. Quality summary plots,
top-bad scatters, and annotated good-example scatters are available under
`analysis/` for paper preparation and further diagnosis.
