# E3SM vs Trendy tropical P-variable training data comparison

**Date:** 2026-06-30  
**Context:** Three-phase E3SMv3_h0 workflow mimics the Trendy reference runs (Phase 1 global → Phase 2 tropical → Phase 3 regional bias correction). Phase 2 tropical training on `solutionp_vr`, `occlp_vr`, and `labilep_vr` underperforms relative to the Trendy case. This report verifies two hypotheses raised during investigation:

1. E3SM lacks scalar flux targets (GPP, NPP, AR, HR) that Trendy uses.
2. `solutionp_vr` per-layer MinMax normalization compresses the E3SM tropical signal differently than Trendy.

**Related docs:** [WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md](./WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md), [PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md](./PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md)

**Reference runs**

| Case | Phase 2 tropical run |
|------|----------------------|
| Trendy | `cnp_results/run_20260315_175250_phase2_tropical` |
| E3SMv3_h0 | `cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical` |

---

## 1. Executive summary

| Hypothesis | Verdict | Role in Phase 2 P gap |
|------------|---------|------------------------|
| Missing scalar fluxes (GPP/NPP/AR/HR) | **Confirmed** as structural difference; **downgraded** as root cause | Minor — low loss weight, weak correlation with solution P |
| MinMax compression differs for `solutionp_vr` | **Partially refuted** as mechanism; same method, similar normalized IQR on full tropical cache | Minor as a normalization artifact |
| **Raw P distribution shift** (E3SM lower magnitudes, tighter tails) | **Confirmed** | **Major** — direct ground-truth mismatch vs Trendy-scale training weights |

The E3SM workflow **mimics Trendy at the hyperparameter and pipeline level**, not at the variable-schema level. Missing scalars are documented and code-adapted (`scalar_output_size=0`, scalar loss skipped). The stronger explanation for worse tropical P Phase 2 is that **E3SMv3_h0 tropical targets live on a different physical scale** than Trendy, while reusing Trendy-tuned loss weights.

---

## 2. Method

### 2.1 Data sources

| Dataset | Path | Tropical cache |
|---------|------|----------------|
| Trendy | `/mnt/proj-shared/AI4BGC_7xw/TrainingData/Trendy_1_data_CNP` | None |
| E3SMv3_h0 | `/mnt/proj-shared/AI4BGC_7xw/TrainingData/E3SMv3_h0` | `.preprocessed_cache/c138bda0688c0be0c765e280a94237ee799c3cbe3e2794fa3eb02946bc6a2031` |

### 2.2 Filters (Phase 2 tropical training match)

- Latitude: **±23.5°**
- `PCT_NATVEG > 0` (natveg only)
- Drop longitudes **0°** and **358.75°**
- Soil variables: first column, 10 layers (`col0`, layers 0–9)

### 2.3 Sample scope

Primary statistics use the first **50** pickle batches per dataset (Trendy: 21 files matched `training_data_batch_*.pkl` in that slice; E3SM: 50 files). E3SM tropical cache scalers use the **full** tropical preprocessed set (**59,814 train + 16,030 test** samples).

---

## 3. Hypothesis 1 — Missing scalar fluxes

### 3.1 Confirmed: schema difference

| Variable | Trendy tropical sample | E3SM tropical sample |
|----------|------------------------|----------------------|
| GPP | Present (n=3,990 cells) | **Absent** |
| NPP | Present | **Absent** |
| AR | Present | **Absent** |
| HR | Present | **Absent** |

Documented in `CNP_IO_e3smv3_h0.txt` and `WORKFLOW_E3SMV3_H0_PHASE1_PHASE2_PHASE3.md`. Trendy Phase 2 config uses `CNP_IO_updated9_dev_dw.txt` with four scalar inputs/outputs; E3SM uses `CNP_IO_e3smv3_h0.txt` with zero scalars.

### 3.2 Model and training behavior

From `cnp_config.json` of completed runs:

| Setting | Trendy Phase 2 | E3SM Phase 2 |
|---------|----------------|--------------|
| `scalar_output_size` | 4 | **0** |
| `scalar_variables_in` / `out` | 4 | **0** |
| `scalar_loss_weight` (config) | 0.06 | 0.06 (unused) |

Trainer skips scalar loss when `x_list_scalar_columns` is empty (`training/trainer.py`: scalar loss set to zero). Phase 1 E3SM global was also trained with `scalar_output_size: 0`; Phase 2 continues the same architecture.

### 3.3 Scalar–solution P coupling (Trendy tropical)

Pearson correlation of **Y_solutionp_vr layer 0** with scalars (mean over batches, 50-file sample):

| Scalar | corr(Y_solutionp_vr L0, scalar) |
|--------|----------------------------------|
| GPP | 0.109 |
| NPP | 0.072 |
| AR | 0.126 |
| HR | 0.137 |

Correlations are **weak**. Scalar loss weight (0.06) is small relative to `matrix_loss_weight` (2.5) for soil 2D.

### 3.4 Conclusion

Missing scalars are a **real, intentional** difference from Trendy, not an setup oversight. They are **unlikely the primary cause** of worse E3SM Phase 2 on `solutionp_vr`, `occlp_vr`, and `labilep_vr`.

---

## 4. Hypothesis 2 — solutionp_vr MinMax normalization

### 4.1 Same normalization method

Both Trendy and E3SM Phase 2 configs use `"normalization": "individual"` — per-variable, per-layer MinMax scaling via `IndividualScalerManager.fit_transform_soil_2d`.

### 4.2 Raw tropical distributions (all layers pooled, 50-file sample)

| Variable | Dataset | n | mean | std | p50 | p95 | p99 | max | % zero |
|----------|---------|---|------|-----|-----|-----|-----|-----|--------|
| Y_solutionp_vr | Trendy | 39,900 | 0.0917 | 0.316 | 0.00858 | 0.323 | 1.88 | 4.91 | 0.00 |
| Y_solutionp_vr | E3SM | 62,280 | 0.0176 | 0.0269 | 0.00963 | 0.0692 | 0.134 | 0.419 | 0.00 |
| Y_labilep_vr | Trendy | 39,900 | 67.1 | 72.5 | 54.5 | 218 | 352 | 503 | 0.00 |
| Y_labilep_vr | E3SM | 62,280 | 24.6 | 23.4 | 20.0 | 69.4 | 104 | 199 | 0.00 |
| Y_occlp_vr | Trendy | 39,900 | 375 | 190 | 373 | 623 | 820 | 904 | 0.00 |
| Y_occlp_vr | E3SM | 62,280 | 159 | 138 | 140 | 424 | 657 | 1102 | 0.00 |

E3SM tropical **Y_solutionp_vr** has similar median to Trendy but **much lower tail** (max ~0.42 vs ~4.9). **Y_labilep_vr** and **Y_occlp_vr** are systematically lower (~2–2.5× on mean/p50).

### 4.3 Layer-0 surface and normalization compression

| Metric | Trendy Y_solutionp_vr L0 | E3SM Y_solutionp_vr L0 | E3SM full-cache scaler on L0 sample |
|--------|----------------------------|-------------------------|-------------------------------------|
| Raw range (min, max) | [3.2e-6, 4.91] | [2.0e-7, 0.419] | [2.0e-7, 0.985] (75.8k tropical train) |
| p50 raw | 0.0249 | 0.0166 | — |
| Norm IQR (after MinMax) | 0.0377 | 0.0957 (sample-fit) | **0.0407** (cache scaler) |
| Fraction in norm [0.4, 0.6] | 3.2% | 1.5% | ~0% |

On the **full E3SM tropical cache**, normalized spread (IQR ≈ 0.04) is **comparable to Trendy**. MinMax is not systematically over-compressing E3SM relative to Trendy when fitted on the full tropical training set.

### 4.4 E3SM tropical cache scaler ranges (Y_solutionp_vr)

From `.preprocessed_cache/...c138bda0....scalers.pkl` (full tropical fingerprint):

| Layer | min | max | range |
|-------|-----|-----|-------|
| 0 | 1.98e-7 | 0.985 | 0.985 |
| 1 | 1.02e-7 | 0.674 | 0.674 |
| 2 | 4.40e-8 | 0.389 | 0.389 |
| … | … | … | … |
| 9 | ~0 | 0.141 | 0.141 |

Deep layers have narrow ranges driven by very small minima; layer 0 is dominated by tropical outliers up to ~0.985.

### 4.5 Input→target delta (solutionp_vr, all layers, 50-file sample)

| Dataset | n | mean(Δ) | std(Δ) | p50(Δ) | % Y≈X | % \|Δ\|>0.01 |
|---------|---|---------|--------|--------|-------|-------------|
| Trendy | 39,900 | 0.0910 | 0.315 | 0.00855 | 0.07% | 47.6% |
| E3SM | 62,280 | 0.0173 | 0.0267 | 0.00961 | 2.22% | 48.2% |

Similar fraction of cells with meaningful change (|Δ|>0.01), but Trendy deltas have **much larger variance** (heavy tail). E3SM tropical fine-tuning sees a **tighter, lower-magnitude** target process.

### 4.6 Conclusion

The issue is **not** primarily “MinMax compresses E3SM differently.” Both use the same scheme; full-tropical E3SM normalized IQR matches Trendy. The actionable finding is **raw distribution shift** between E3SMv3_h0 and Trendy ELM trajectories, while Phase 2 reuses **Trendy-tuned P loss weights**.

---

## 5. Implications for Phase 2 tropical retrain

1. **P3-focus Phase 2** (`training_config_e3smv3_h0_phase2_tropical_p3focus.json`) — higher weights on `solutionp_vr`, `occlp_vr`, `labilep_vr` — targets the verified problem (distribution/weight mismatch), not missing scalars. **Run in progress (2026-06-30):** `run_20260630_130344_e3smv3_h0_phase2_tropical_p3focus`.
2. **Weighting trade-offs:** P3-focus is more aggressive than Trendy (non-P soil2d at 0.1 vs moderate baseline weights). Theoretical risks and evaluation plan: [PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md](PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md).
3. **Use tropical preprocessed cache** (`c138bda0...`) for faster iteration on the same 75.8k filtered samples.
4. **Tail-aware / log1p loss** for `solutionp_vr` remains appropriate given heavy-tail sensitivity in Trendy and outlier-driven scalers in E3SM.
5. **Region upsampling** (Amazon 270–330°E, Africa 0–30°E) may help if site-level GT errors concentrate there.
6. **Do not expect scalar restoration** to close the gap unless E3SM batches are regenerated with GPP/NPP/AR/HR; impact would likely be secondary.
7. **If P3-focus underperforms or hurts coupled pools:** fall back to baseline `training_config_e3smv3_h0_phase2_tropical.json` as-is for Phase 3.
8. **Scale vs R² / ELM sensitivity:** see [REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md](REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md) — magnitude errors on `solutionp_vr` are not resolved by reweighting or standard Phase 3 v3 alone.

---

## 6. Reproduction

Analysis was generated with an ad-hoc Python script over:

- `TrainingData/Trendy_1_data_CNP/training_data_batch_*.pkl`
- `TrainingData/E3SMv3_h0/training_data_batch_*.pkl`
- `TrainingData/E3SMv3_h0/.preprocessed_cache/c138bda0688c0be0c765e280a94237ee799c3cbe3e2794fa3eb02946bc6a2031.{pt,scalers.pkl,meta.json}`

Filters: lat ±23.5°, natveg, drop lon 0 and 358.75. Variable list alignment: `CNP_IO_e3smv3_h0.txt` vs `CNP_IO_updated9_dev_dw.txt`.
