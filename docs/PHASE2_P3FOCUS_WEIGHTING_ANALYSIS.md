# Phase 2 tropical P3-focus vs Trendy-style weighting — analysis and next steps

**Date:** 2026-06-30  
**Context:** E3SMv3_h0 Phase 2 tropical retrain targeting `solutionp_vr`, `occlp_vr`, `labilep_vr`. See also [REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md](REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md) for the data comparison that motivated stronger P emphasis.

**Decision (2026-06-30):** Run **P3-focus first**, evaluate results, then decide whether a Trendy-equivalent or moderate-boost config is needed.

---

## 1. Config comparison

| Pool / group | Trendy / baseline E3SM (`training_config_e3smv3_h0_phase2_tropical.json`) | P3-focus (`training_config_e3smv3_h0_phase2_tropical_p3focus.json`) |
|--------------|---------------------------------------------------------------------------|---------------------------------------------------------------------|
| `solutionp_vr` | 50 | **90** |
| `occlp_vr` | 42 | **75** |
| `labilep_vr` | 28 | **70** |
| `secondp_vr`, `primp_vr` | 24, 28 | 30, 30 (modest) |
| `soil1p_vr`, `litr*p_vr` | 8–16 | **0.1** |
| Litter / soil C/N (`litr*c/n`, `soil*c/n`) | 2–4 | **0.1** |
| PFT (e.g. `leafp`, `ppool`) | 2–6 | 1–3 (reduced) |
| `matrix_loss_weight` | 2.5 | 3.5 |
| `litter_p_loss_weight` | 4 | 2 |

**Trendy strategy:** boost all **five** soil P pools strongly; keep **moderate** weights on coupled litter/soil C/N/P and PFT variables.

**P3-focus strategy:** boost **three** headline P pools heavily; set most non-P soil2d weights to **0.1** (near-zero gradient); keep `secondp_vr` / `primp_vr` at moderate levels only.

---

## 2. What loss reweighting does (and does not do)

The CNP model still:

- **Predicts all variables** (scalar, PFT1d, soil2d) through shared trunk + multi-head outputs.
- **Uses all pools as inputs** (litter P, `soil1p_vr`, C/N pools, PFT state, forcing).

Reweighting only changes **how much each prediction error contributes to gradients** during tropical training. It does **not** remove ELM couplings from the architecture or from the input features.

Physical links among the five P pools (uptake, mineralization, occlusion, etc.) are learned indirectly through:

1. Shared transformer representations.
2. Multi-output loss on **related** targets (all 5P, `soil1p_vr`, `litr*p_vr`, plant P, …).

---

## 3. Theoretical risks of P3-focus

### 3.1 Weak constraint on coupled pools

If `solutionp_vr` / `occlp_vr` / `labilep_vr` improve but `soil1p_vr`, `litr2p_vr`, or `secondp_vr` drift, the loss barely penalizes that. Predictions can become **internally inconsistent across the P cycle** even when the three headline variables look better at validation sites.

### 3.2 Shared trunk can drift

Gradients are dominated by the three P variables. Representations that encoded P–litter–mineral coherence in Phase 1 (or in a balanced tropical baseline) can erode over 150 epochs—especially when Phase 2 trains **from scratch on tropical data only** (current E3SM path via `train_cnp_model.py`), rather than a short fine-tune from Phase 1 weights.

### 3.3 Precedent from Trendy experiments

The Phase 3 report ([PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md](PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md)) notes that aggressive P reweighting (`labilep_weight42`, `run_20260617_161538`) **hurt all 5P** vs baseline Phase 2 tropical. Extreme emphasis is not guaranteed to help.

### 3.4 What is lower risk

- Fine-tune **from Phase 1 checkpoint** (`run_finetuning_json.py`) with Trendy-like weights—preserves global couplings while adapting tropics.
- **Moderate** boost on 3P only, keeping `soil1p` / `litr*p` / C/N at baseline levels (not 0.1).
- Emphasize all **five** P pools together, as in the Trendy Phase 2 config.

### 3.5 Bottom line

P3-focus does **not** remove physical relationships from the model, but it **largely stops the loss from enforcing them** during tropical training. That can fix weak targets while breaking consistency elsewhere—a high-variance experiment, not the conservative analogue of the Trendy strategy.

---

## 4. P3-focus run in progress (2026-06-30)

| Item | Value |
|------|--------|
| Run directory | `cnp_results/run_20260630_130344_e3smv3_h0_phase2_tropical_p3focus` |
| Config | `config/training_config_e3smv3_h0_phase2_tropical_p3focus.json` |
| Log | `logs/e3smv3_phase2_p3focus_20260630_130340.log` |
| Cache | `TrainingData/E3SMv3_h0/.preprocessed_cache/` (fingerprint `c138bda0…`) |
| Filter | lat ±23.5°, natveg, drop lon 0 / 358.75 |
| Status | Training ~epoch 63/150 at last check (2026-06-30 ~13:21); ~16 s/epoch → ~25 min remaining |

Monitor:

```bash
tail -f logs/e3smv3_phase2_p3focus_20260630_130340.log
grep "Epoch \[" logs/e3smv3_phase2_p3focus_20260630_130340.log | tail -5
```

---

## 5. Evaluation checklist (after P3-focus completes)

Compare P3-focus against **baseline E3SM Phase 2** (`run_20260624_092639_e3smv3_h0_phase2_tropical`) and, where useful, Phase 1 global.

### 5.1 Training metrics

- `cnp_metrics.json`: R² / RMSE for `Y_solutionp_vr`, `Y_occlp_vr`, `Y_labilep_vr` (all layers + layer-0).
- Same for `secondp_vr`, `primp_vr`, `soil1p_vr`, `litr2p_vr` — **watch for degradation** (coupling check).
- Train vs val loss gap (overfitting under extreme weights).

### 5.2 Site-level ground truth

- Africa: 28°E, 0°N (and other sites used in prior reports).
- Amazon: reference lon/lat from `compare_5p_gt_two_regions_inference.py` / workflow docs.
- Primary: did **solutionp_vr** move toward GT without **occlp** / **labilep** / **secondp** / **primp** getting worse?

### 5.3 Inference → restart

1. Tropical (or full-grid) inference from P3-focus run.
2. Merge into Phase 1 global restart (`ai_predictions_to_restart.py --tropical-lat-range`).
3. Re-run site 5P comparison scripts used for baseline Phase 2.

### 5.4 Decision matrix

| Outcome | Next step |
|---------|-----------|
| 3P clearly better at sites; 5P + coupled pools stable | Proceed to Phase 3 merge / hybrid protect as planned; document P3-focus as chosen Phase 2 |
| 3P better but secondp/primp/soil1p/litr*p worse | Try **moderate boost** config (§6) or Phase 1 → tropical **finetune** with Trendy weights |
| 3P not better; or val metrics unstable | Revert to **baseline** `training_config_e3smv3_h0_phase2_tropical.json` (Trendy-equivalent weights) |
| Mixed (e.g. solutionp up, occlp down) | Region-specific finetune or Phase 3 hybrid protect; avoid further global weight slashing |

---

## 6. Fallback configs (if P3-focus is insufficient)

Not scheduled until P3-focus results are reviewed.

### 6.1 Baseline (Trendy-equivalent) — conservative default

- Config: `config/training_config_e3smv3_h0_phase2_tropical.json`
- Same 5P emphasis (24–50) with moderate litter/soil/PFT weights.
- Addresses E3SM **distribution shift** ([REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md](REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md)) without slashing coupled losses.

### 6.2 Moderate 3P boost — middle ground

If P3-focus helps solutionp but hurts couplings, consider a new config that only scales the three targets ~1.3–1.5× baseline, e.g.:

| Variable | Baseline | Moderate boost (example) |
|----------|----------|---------------------------|
| `solutionp_vr` | 50 | 65–75 |
| `occlp_vr` | 42 | 55–60 |
| `labilep_vr` | 28 | 40–45 |
| `secondp_vr`, `primp_vr` | 24, 28 | unchanged or +10% |
| All other soil2d | unchanged | **unchanged** (not 0.1) |

### 6.3 Phase 1 fine-tune path

- Script: `run_finetuning_json.py` from Phase 1 checkpoint.
- Weights: baseline or moderate boost JSON.
- Preserves global representation; shorter tropical adaptation.

---

## 7. Related files

| File | Role |
|------|------|
| `config/training_config_e3smv3_h0_phase2_tropical.json` | Baseline / Trendy-equivalent Phase 2 |
| `config/training_config_e3smv3_h0_phase2_tropical_p3focus.json` | Aggressive 3P experiment |
| `commands.txt` (2026-06-30 section) | Launch command with fixed `--tropical-lat-range=-23.5,23.5` |
| `docs/WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md` | Full three-phase pipeline |
---

## 8. Results after P3-focus training (2026-06-30)

**Conclusion: P3-focus is worse than Phase 1 global on all five tropical P pools vs ground truth.** It also underperforms baseline E3SM Phase 2 tropical (`run_20260624_092639`).

### 8.1 Tropical band (|lat| ≤ 23.5°), pooled over cells × layers

| Variable | Phase 1 pooled R² | P3-focus pooled R² | Phase 2 baseline pooled R² |
|----------|-------------------|----------------------|----------------------------|
| `labilep_vr` | 0.9897 | 0.9250 | 0.9901 |
| `occlp_vr` | 0.9890 | 0.9105 | 0.9891 |
| `solutionp_vr` | **0.9923** | 0.9432 | **0.9930** |
| `secondp_vr` | 0.9901 | 0.9203 | 0.9905 |
| `primp_vr` | 0.7592 | 0.3150 | 0.7602 |

P3-focus RMSE is ~2–3× higher than Phase 1 on `labilep`, `occlp`, and `secondp` (denormalized units).

### 8.2 Same gridcells (Phase 1 inference ∩ P3-focus holdout, n = 15,827)

Fair head-to-head on identical tropical test cells — P3-focus still loses on every variable (e.g. `solutionp_vr` R² 0.9924 vs 0.9432).

### 8.3 Africa box site (nearest cell ~0.125°N, 27.875°E)

10-layer column sums vs GT:

| Variable | Phase 1 rel error | P3-focus rel error |
|----------|-------------------|---------------------|
| `labilep_vr` | 2.9% | **40.5%** |
| `occlp_vr` | 2.2% | 3.1% |
| `solutionp_vr` | 85.9% | 78.9% |
| `secondp_vr` | 8.2% | **36.3%** |
| `primp_vr` | 23.8% | 28.8% |

P3-focus matches Phase 1 on `solutionp_vr` only marginally; it clearly degrades `labilep_vr` and `secondp_vr`.

### 8.4 Artifacts

- CSV: `cnp_results/run_20260630_130344_e3smv3_h0_phase2_tropical_p3focus/analysis/5p_vs_gt_phase1_p3focus_comparison.csv`
- **Site profile plots** (GT vs phase1_global / p3focus / phase2_baseline):
  - Combined: `analysis/site_plots/amazon_africa_5p_summary_combined.png`
  - Per site: `analysis/site_plots/amazon_5p_summary_all_runs.png`, `analysis/site_plots/africa_5p_summary_all_runs.png`
  - Per variable: `analysis/site_plots/amazon_site_lon303p75_latm17p43/`, `analysis/site_plots/africa_site_lon28p00_lat0p00/`
- P3-focus used training holdout predictions (`cnp_predictions`, ~16k tropical test cells). Phase 1 used full-grid inference filtered to tropics (~80k cells). Baseline Phase 2 used `cnp_inference_tropical_only` (~80k cells).

### 8.5 Recommended next step

Do **not** adopt P3-focus or P3-moderate for Phase 3 merge. Use **baseline Phase 2 tropical** (`run_20260624_092639`).

**Strategic analysis (2026-06-30):** High R² with order-of-magnitude `solutionp_vr` error is a scale/metric mismatch, not a weight-tuning problem. See [REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md](REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md) for ELM sensitivity, Phase 3 limits, and mitigation plan (log-relative loss, multiplicative calibration).
