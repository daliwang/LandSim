# Phase3 five-P bias correction: review and recommendations

This report summarizes how the Phase3 soil phosphorus bias / scale correction works in this repository, what we observed when comparing older and newer Phase3 outputs, and concrete directions to improve **`solutionp_vr`** and **`occlp_vr`** across the **entire** Amazon and Africa target boxes—not only at a small set of hand-picked reference coordinates.

**Related workflow documentation**

- [REPORT_TRENDY_VS_E3SMV3_THREE_PHASE_WORKFLOW_COMPARISON.md](./REPORT_TRENDY_VS_E3SMV3_THREE_PHASE_WORKFLOW_COMPARISON.md) — publication-oriented Trendy vs E3SMv3 Phase 1–3 comparison.
- [WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md](./WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md) — end-to-end 5P bias/scale and restart steps.
- [WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md](./WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md) — Phase1 / Phase2 / Phase3 restart orchestration.
- [PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md](./PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md) — Phase3 aligned with Amazon-only → Africa-only → merge workflow.
- [SITE_5P_RESTART_COMPARISON.md](./SITE_5P_RESTART_COMPARISON.md) — site-level 5P profile plots (`generate_site_5p_restart_comparison.py`).
- [CNP_pipeline_runbook.md](./CNP_pipeline_runbook.md) — inference, validation plots, and quality reports.

**Scripts**

- `scripts/apply_5p_bias_scale_correction.py` — fits and applies per-layer affine maps \(\mathrm{GT} \approx a\cdot\mathrm{pred} + b\) inside each region box.
- `scripts/merge_5p_bias_corrected_amazon_africa.py` — merges **separate** Amazon-only and Africa-only corrected CSVs so each geographic box uses its own calibration.
- `scripts/compare_5p_gt_two_regions_inference.py` — `gt-summary` and `pred-eval` for Amazon/Africa boxes (median RMSE across layers per cell, multi-run CSVs).

---

## 1. Executive summary

- The pipeline uses **one affine map per variable, per soil layer, per region** (Amazon box vs Africa box), fitted on **all gridcells** falling in that box. That is already an **overall regional** fit in the sense of pooling every cell in the box.

- For **`solutionp_vr`** and **`occlp_vr`**, the fitter switches to a **relative-error–weighted** least squares objective and can **up-weight** a fixed list of **reference lon/lat pairs**. Those sites were chosen from visualization convenience; they are **not** a substitute for optimizing error over the whole region, and they can **skew** the fit if weights are large or if sites do not lie exactly on the inference grid.

- A **post-fit “snap”** toward reference sites (forcing the corrected value toward GT at selected cells) is applied today for **Amazon only** for the relative-error variables, not symmetrically for Africa. That asymmetry conflicts with the product goal of **near-uniform quality in both regions**.

- Empirical comparison (see Section 3) showed that **merging independent Amazon and Africa corrections** (newer Phase3 style) fixes severe **Africa `occlp_vr`** failure modes seen when Africa inherited a correction dominated elsewhere, at the cost of small regressions on some other variables in Amazon. That tradeoff is structurally plausible for a **single** \((a,b)\) per layer per box.

- **Ideal direction** for “almost perfect” **`solutionp_vr`** / **`occlp_vr`** on the **full** boxes: prioritize **region-wide objectives** (e.g. minimize median or Huber loss over all cells, explicit tail control), **reduce or remove dependence** on a few reference coordinates unless they are replaced by **grid-anchored** targets, and add **stability constraints** (weight caps, log-space, or sub-regions) rather than relying on five visual picks.

---

## 2. What the code does today

### 2.1 Regional affine calibration (`apply_5p_bias_scale_correction.py`)

For each of the five variables (`labilep_vr`, `occlp_vr`, `solutionp_vr`, `secondp_vr`, `primp_vr`), and each soil layer column `Y_<var>_col1_layer{1..10}`:

1. Select all inference rows whose \((\mathrm{lat}, \mathrm{lon})\) lie in the active **region box** (from the training JSON’s `data_filtering_config.region_boxes`, optionally splitting Amazon into south/north by latitude).

2. Fit **one** scalar \(a\) and intercept \(b\) so that \(\mathrm{GT} \approx a\cdot\mathrm{pred} + b\) on those rows.

3. Special cases already in code:

   - **`solutionp_vr`**: optional internal scaling (`SCALE_FACTOR_VARS`) for numerical stability when fitting small concentrations.

   - **`solutionp_vr`** and **`occlp_vr`**: **`RELATIVE_ERROR_WEIGHTED_VARS`** — weights \(w_i \propto 1 / (\mathrm{GT}_i + \varepsilon)^2\) in the weighted normal equations, plus **extra weight** at configured reference sites.

4. After fitting, for relative-error variables in **Amazon-shaped** region names only, an additional loop can **nudge** \((a,b)\) so a reference cell is closer to exact match, subject to a relative error cap.

The asymmetry (Amazon-only nudge) is called out in-code as avoiding “overfitting Africa to noisy GT”; the side effect is that **Africa does not receive the same hard anchoring** for the two most sensitive variables.

### 2.2 Merge of two regional products (`merge_5p_bias_corrected_amazon_africa.py`)

The merged product assigns:

- Amazon box cells → predictions from the **Amazon-only** bias correction run.

- Africa box cells → predictions from the **Africa-only** bias correction run.

- All other cells → **raw** Phase2 neural-network predictions.

This preserves **two independent regional fits** and avoids a single compromised regression over Amazon \(\cup\) Africa.

---

## 3. Empirical context (Phase3 old vs new)

Structured evaluation used `scripts/compare_5p_gt_two_regions_inference.py` `pred-eval` on the Amazon and Africa boxes (same definitions as the merge script: Amazon lat \([-30,10]\), lon \([270,330]\); Africa lat \([-15,15]\), lon \([0,30]\), lon in \(0\)–\(360^\circ\)).

Summary of outcomes (median RMSE across ten layers per gridcell):

- **Older Phase3** (`run_20260315_202304_phase3_tworegions`, `soil_2d_predictions_5P_bias_corrected_phase2`): **Africa `occlp_vr`** showed catastrophic bulk error (median RMSE on the order of \(10^2\)), consistent with restart shock in African simulations.

- **Newer Phase3** (`run_20260315_210715_phase3_tworegions`, merged `soil_2d_predictions_5P_bias_corrected_amazon_africa`): **Africa `occlp_vr`** median RMSE improved by roughly two orders of magnitude, with modest **Amazon** regressions on some variables (e.g. `primp_vr`) and small **solutionp_vr** median changes—aligned with sacrificing a little Amazon quality to rescue Africa.

Detailed tables and CSVs live under the newer run’s `analysis/` directory (for example `5p_phase3_old_vs_new_pred_eval_summary.csv` and `5p_phase3_old_vs_new_report.txt`).

---

## 4. Reference sites vs “entire Amazon and Africa” fitting

### 4.1 What the reference list is doing today

In `apply_5p_bias_scale_correction.py`, **`AMAZON_REFERENCE_SITES`** and **`AFRICA_REFERENCE_SITES`** are used to:

1. **Inflate weights** in the weighted least squares fit for **`solutionp_vr`** and **`occlp_vr`** at rows whose lon/lat match a site within a tight tolerance (`REFERENCE_SITE_ATOL`).

2. Optionally drive the **Amazon-only** post-fit projection toward a match at one of those coordinates.

The coordinates were chosen to align with **site plots** and manual inspection. That is useful for **debugging** and for **forcing** a known anchor, but it is **not** the same as defining the objective “minimize error over every cell in the box.”

### 4.2 Why this can conflict with region-wide goals

- **Coverage**: Reference sites may not coincide with inference grid nodes; tight `isclose` checks can **miss** the intended cell, so extra weight never applies.

- **Optimality**: Heavy weight on a few points **reweights** the global minimum of the pooled loss. If the goal is **uniform** accuracy, the optimum is closer to **unweighted** or **percentile-balanced** losses over **all** regional cells (optionally with **capped** weights so tiny GT does not dominate).

- **Asymmetry**: Hard projection for Amazon but not Africa biases the pipeline toward “Amazon looks good on the map” while Africa still depends only on the weighted LS—misaligned with **equal** standards for **`solutionp_vr`** and **`occlp_vr`**.

### 4.3 What “overall fitting across the entire regions” should mean in practice

Operationally, for each region \(R \in \{\mathrm{Amazon}, \mathrm{Africa}\}\) and each critical variable \(v \in \{\mathrm{solutionp\_vr}, \mathrm{occlp\_vr}\}\), a region-wide objective could be stated as:

- Minimize a **robust** scalar over all cells \(i \in R\) and layers \(\ell\), e.g.  
  \(\sum_{i,\ell} \rho\bigl((a_\ell p_{i,\ell} + b_\ell) - g_{i,\ell}\bigr)\)  
  where \(\rho\) is Huber or a **clipped** relative loss, **without** concentrating mass on five lon/lat pairs unless those pairs are formally part of the specification.

- Report **distribution** of errors (median, p90, worst 1%) per region after fit—not only at reference locations.

Reference locations can remain as **diagnostics** or as **soft** priors with **small** weight, but they should not dominate the objective if the stated goal is **whole-box** performance.

---

## 5. Recommendations (prioritized)

1. **Treat `solutionp_vr` and `occlp_vr` with symmetric, region-wide objectives**  
   Remove or drastically lower reference-site **extra weight** for these variables, **or** replace “exact lon/lat” with **nearest inference gridcell** in the region so anchors always exist. If hard projection is kept, apply **the same rule in Africa as in Amazon** for these two variables only.

2. **Stabilize relative-error fitting for small concentrations**  
   Cap weights \(1/(\mathrm{GT}+\varepsilon)^2\), blend with absolute-error terms below a GT threshold, or fit in **log-space** with back-transform so a thin tail of near-zero GT does not dictate \((a,b)\) for the whole layer.

3. **Keep separate regional fits + merge**  
   Continue using **independent** Amazon and Africa calibration followed by `merge_5p_bias_corrected_amazon_africa.py`. Avoid a single pooled \((a,b)\) over both boxes unless you introduce explicit spatial features (sub-boxes, covariates).

4. **Richer within-region structure (if affine is not enough)**  
   Subdivide Amazon (north/south is already supported via `split_first_region_lat` in `load_region_boxes`) or add a **shallow residual** model of corrections vs static soil / PFT predictors, still fit only on withheld or training-consistent data to avoid double-dipping if that becomes a concern.

5. **Physical post-processing**  
   Enforce non-negativity and plausible bounds after correction; optionally enforce mild **monotonicity across depth** where ELM semantics support it, to reduce restart instability when the affine map wobbles layer-to-layer.

6. **Evaluation loop**  
   Use `compare_5p_gt_two_regions_inference.py pred-eval` routinely on **median, mean, p90, and tail** (e.g. 99th percentile RMSE) for **`solutionp_vr`** and **`occlp_vr`** in each box, not only at reference sites.

---

## 6. Reproducibility pointers

Workflow cross-references: [WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md](./WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md) (Section 4: apply + merge), [WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md](./WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md), [PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md](./PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md).

- Bias application: `scripts/apply_5p_bias_scale_correction.py`  
- Merge: `scripts/merge_5p_bias_corrected_amazon_africa.py`  
- Regional pred vs GT metrics: `scripts/compare_5p_gt_two_regions_inference.py pred-eval`

Example Phase3 comparison artifacts (paths on the shared filesystem used in this project):

- `cnp_results/run_20260315_210715_phase3_tworegions/analysis/5p_phase3_old_vs_new_runs.json`  
- `cnp_results/run_20260315_210715_phase3_tworegions/analysis/5p_phase3_old_vs_new_report.txt`

---

## Document history

- **2026-05-11**: Initial report from code review and Phase3 empirical comparison; emphasis on full-box objectives for `solutionp_vr` / `occlp_vr` and on reducing over-reliance on visualization-driven reference sites.
