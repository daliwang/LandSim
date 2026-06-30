# E3SMv3_h0: ELM-sensitive P pools — scale vs R², Phase 3 limits, and mitigation strategy

**Date:** 2026-06-30  
**Status:** Strategic analysis after P3-focus and P3-moderate Phase 2 experiments  
**Related:**

- [REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md](REPORT_E3SM_TRENDY_TROPICAL_P_VARIABLES_COMPARISON.md) — E3SM vs Trendy target distributions  
- [PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md](PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md) — weight experiments and results  
- [REPORT_E3SM_MODEL_CAPACITY_VS_GRID_SIZE.md](REPORT_E3SM_MODEL_CAPACITY_VS_GRID_SIZE.md) — gridcell count vs model size  
- [PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md](PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md) — Trendy Phase 3 hybrid logic  
- [REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md](REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md) — bias/scale correction methodology  

---

## 1. Executive summary

The ELM system is **extremely sensitive** to `solutionp_vr`, and also sensitive to `occlp_vr` and `labilep_vr`. For E3SMv3_h0, Phase 2 models can show **excellent pooled R²** on these variables while **under-predicting magnitude by an order of magnitude** at reference sites. Standard **loss reweighting** (P3-focus, P3-moderate) and **Phase 3 regional affine bias correction** do not fix this because they optimize **correlation / normalized MSE**, not **absolute or log-ratio magnitude** in physical units.

**Recommendation:** Stop searching normalized weights alone. Use **baseline Phase 2 tropical** as the coupled-5P backbone. Next work should be **scale-first**: log-relative training loss and/or **multiplicative site-anchored calibration** for `solutionp_vr` (and selectively for other sensitive pools), with model selection based on **site log-ratio error**, not validation R².

---

## 2. Problem statement

| What the AI pipeline optimizes | What ELM needs |
|--------------------------------|----------------|
| Normalized MSE / Huber / log1p in MinMax space | Absolute pool concentrations in physical units |
| High R² across many near-zero gridcells | Correct magnitude at sensitive operating points |
| Regional pooled metrics | Site-level and upper-tail behaviour |

The mismatch is sharpest for **`solutionp_vr`**: values are often \(10^{-3}\)–\(10^{-2}\) in E3SM tropical batches, many cells are near zero, and a few cells carry most of the variance. The model learns **rank order and profile shape** (good R²) but **systematic under-scaling of peaks** (bad for ELM).

---

## 3. Evidence (E3SM Phase 2 baseline)

**Run:** `cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical`  
**Reference sites:** Africa ~(28°E, 0°N) → grid (27.875°E, 0.125°N); Amazon (303.75°E, −17.43°N)

### 3.1 Pooled R² vs site column-sum relative error

| Variable | Africa pooled R² | Africa site rel. error (10-layer sum) |
|----------|------------------|---------------------------------------|
| `labilep_vr` | ~0.996 | ~3% |
| `occlp_vr` | ~0.997 | ~2% |
| `solutionp_vr` | ~0.997 | **~86%** (pred ≪ GT) |
| `secondp_vr` | ~0.996 | ~8% |
| `primp_vr` | ~0.874 | ~24% |

Amazon shows the same pattern for `solutionp_vr`: pooled R² ~0.997 but site rel. error **~99%** on column sums.

### 3.2 E3SM vs Trendy target scale (from tropical data comparison)

| Metric | Trendy tropical | E3SM tropical |
|--------|-----------------|---------------|
| `Y_solutionp_vr` max | ~4.9 | ~0.42 |
| `Y_labilep_vr`, `Y_occlp_vr` | higher means | ~2–2.5× lower |

The model is trained on **E3SM-native targets** but with **Trendy-tuned loss weights** and **MinMax normalization** that compresses small-magnitude dynamics.

---

## 4. Why loss reweighting failed

### 4.1 Experiments summary

| Config | Run | Outcome vs baseline Phase 2 |
|--------|-----|----------------------------|
| Baseline tropical | `run_20260624_092639` | Best overall tropical 5P R²; site `solutionp` still ~10× low |
| P3-focus | `run_20260630_130344` | Worse on all 5P; broke labilep/secondp at sites |
| P3-moderate | `run_20260630_145817` | Better than P3-focus; still worse than baseline regionally; partial site `solutionp` gain only |

See [PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md](PHASE2_P3FOCUS_WEIGHTING_ANALYSIS.md) for full tables, site plots, and pred-eval CSVs.

### 4.2 Mechanism

Reweighting only changes **gradient balance on normalized errors**. It does **not**:

1. Switch the loss to **physical-unit relative or log-ratio** error.  
2. Penalize **order-of-magnitude under-prediction** at sites where GT is above the bulk near-zero cloud.  
3. Preserve **5P coupling** when non-P weights are slashed (P3-focus).

P3-moderate improved Africa `solutionp` site error (86% → 47% rel.) but **worsened** `labilep` and regional pooled R² — pushing magnitude in normalized space without a ratio-aware objective disturbs the rest of the P cycle.

---

## 5. Why Phase 3 bias correction does not help (E3SM)

Phase 3 (`apply_5p_bias_scale_correction.py`, regional v3) fits **per-layer affine** maps inside Amazon/Africa boxes:

\[
\text{GT} \approx a \cdot \text{pred} + b
\]

with relative-error weighting for `solutionp_vr` and `occlp_vr`.

### 5.1 R²-driven fitter thinks raw pred is already good

On Trendy Phase 2 inference, when raw `solutionp`/`occlp` already had R² ≈ 0.97+, **full v3 correction degraded** them (e.g. Amazon `solutionp` 0.98 → 0.89). The adopted Trendy strategy was **hybrid protect**: **do not bias-correct** `solutionp_vr` or `occlp_vr` ([PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md](PHASE3_HYBRID_PROTECT_OCCLP_SOLUTIONP_REPORT.md)).

For E3SM, the problem is the **inverse**: raw `solutionp` **needs** scale correction, but high R² tells Phase 3 **not to touch it**.

### 5.2 Affine correction is wrong for multiplicative bias

If pred ≈ 0.1 × GT at sensitive sites, correction needs **\(k \approx 10\)** (multiplicative). Pooled affine fit over ~10⁴ cells — most with GT ≈ 0 — yields **\(a \approx 1, b \approx 0\)**, which minimizes squared error where most mass is tiny but **does not fix peaks**.

### 5.3 Regional pooling dilutes anchor sites

Reference-site upweighting (500× Amazon, 250–500× Africa) helps when the grid aligns with sites. On E3SM, nearest-grid offsets and the near-zero majority still dominate the fit.

### 5.4 E3SM Phase 3 implication

**Do not copy Trendy hybrid-protect verbatim for E3SM.** Trendy protects strong-R² occlp/solutionp; E3SM needs a **scale-first, solutionp-specific multiplicative** path, not regional affine v3 on all 5P.

---

## 6. Three-layer failure stack

```text
Layer 1 — Training objective
  Normalized MSE / Huber on MinMax targets
  → Good validation R², wrong physical magnitude

Layer 2 — Inference
  Correct profile shape, systematic under-scale on solutionp peaks
  → ELM-sensitive variables wrong in absolute units

Layer 3 — Phase 3 correction
  Regional affine fit optimized for pooled R²
  → Insufficient multiplicative boost; hybrid protect skips solutionp

Layer 4 — ELM integration
  Extreme sensitivity to solutionp / occlp / labilep
  → Small absolute errors → large biogeochemical impact
```

Weight tuning only addressed Layer 1 partially. Phase 3 addresses Layer 3 with the wrong functional for E3SM.

---

## 7. Recommended path forward

### Priority 1 — Change what we measure

Report for sensitive pools:

- Median and p90 **\(|\log(\hat y / y)|\)** at reference sites  
- **Upper-quantile relative error** (e.g. cells with GT > p90)  
- Site profile plots (already under `analysis/site_plots/`)

Do **not** use validation R² alone to accept Phase 2 or Phase 3 for `solutionp_vr`.

### Priority 2 — Multiplicative calibration prototype (no retrain)

On **baseline Phase 2** inference (`run_20260624_092639`):

1. Fit **multiplicative-only** \( \hat y_\text{corr} = k \cdot \hat y \) per layer at Amazon + Africa anchor sites, minimizing log-ratio error.  
2. Optionally extend \(k\) spatially (constant per region, or binned by climate).  
3. Check site profiles and restart before another 150-epoch retrain.

### Priority 3 — Retrain with log-relative loss (one experiment)

For `solutionp_vr`, `occlp_vr`, `labilep_vr`:

- Loss on **denormalized** values: \((\hat y - y)^2 / (|y| + \epsilon)^2\), or direct **`log1p(Y_phys)`** head.  
- Keep baseline weights on coupled pools (P3-moderate-style, not P3-focus).  
- Select checkpoint by **site log-ratio**, not val R².

### Priority 4 — E3SM-specific Phase 3

Invert Trendy hybrid logic:

| Variable | E3SM Phase 3 suggestion |
|----------|------------------------|
| `solutionp_vr` | **Multiplicative** scale correction (site-anchored) |
| `labilep_vr` | Correct only if log-ratio error exceeds threshold |
| `occlp_vr` | Raw Phase 2 if magnitude already good; do not apply destructive v3 |
| `secondp_vr`, `primp_vr` | Optional v3 on labilep/secondp/primp only (Trendy hybrid pattern) |

Enforce **5P partition consistency** (CNP ratio or total-P check) when scaling `solutionp` up.

### Priority 5 — Do not pursue (for now)

- Further aggressive normalized weight sweeps without loss reformulation  
- Full v3 merge of all 5P on E3SM (Trendy failure mode)  
- Larger transformer purely because E3SM has more gridcells ([REPORT_E3SM_MODEL_CAPACITY_VS_GRID_SIZE.md](REPORT_E3SM_MODEL_CAPACITY_VS_GRID_SIZE.md))

---

## 8. Key runs and artifacts

| Run | Role |
|-----|------|
| `run_20260624_092639_e3smv3_h0_phase2_tropical` | **Use for Phase 3 backbone** |
| `run_20260630_130344_e3smv3_h0_phase2_tropical_p3focus` | Failed weight experiment; site plots in `analysis/site_plots/` |
| `run_20260630_145817_e3smv3_h0_phase2_tropical_p3moderate` | Moderate boost; `analysis/5p_pred_eval_summary.csv` |

---

## 9. Conclusion

The E3SM workflow failure mode is **not** “Phase 2 correlation is bad” — it is **“Phase 2 magnitude is wrong where ELM is sensitive, and our metrics and Phase 3 tools do not target that error.”** Fixing it requires **log-ratio-aware training and/or multiplicative calibration**, with **baseline Phase 2** preserving 5P coupling until those tools exist.
