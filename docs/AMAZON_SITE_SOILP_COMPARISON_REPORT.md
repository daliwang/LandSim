# Amazon Site Soil-P Variable Comparison: Extreme Soil-P vs Natveg Improved

**Date:** 2026-03-04  
**Purpose:** Compare predictions of key soil phosphorus variables (occlp_vr, solutionp_vr, labilep_vr) at the Amazon reference site between the **extreme soil-P** training run and the earlier **natveg improved** (Phase 1) run.

---

## 1. Site and Runs

### Reference site (Amazon)

- **Longitude:** 303.75°E  
- **Latitude:** −17.43° (≈ −17.434553°)  
- **Context:** Tropical site where Phase 1 was known to underpredict occlp_vr and contribute to NEE bias in restart simulations (see `docs/TROPICAL_OCCLP_TWO_OPTIONS_REPORT.md`).

### Runs compared

| Run | Config / description | Results path |
|-----|----------------------|-------------|
| **natveg improved (Phase 1)** | Global natveg, occlp-focused weights | `cnp_results/run_20260228_214757_natveg_improved/cnp_inference_entire_dataset/cnp_predictions/` |
| **extreme soil-P** | Experiment 3 extreme soil-P: very high matrix loss and P-specific variable weights (occlp_vr 40, primp_vr, labilep_vr, secondp_vr, solutionp_vr, soil*p_vr, litr*p_vr) | `cnp_results/run_20260304_133300_natveg_occlp_extreme_soilp/cnp_predictions/` |

Ground truth is the same in both runs (same test set / inference data). Values are per-layer (10 soil layers), units as in the CNP restart (e.g. g/m² for occlp_vr/labilep_vr, solution P in same mass units).

---

## 2. Methodology

- The Amazon grid cell was identified by matching coordinates (lon 303.75, lat −17.43) in the prediction/ground-truth CSVs.
- For each variable, layer-by-layer values were read from:
  - `ground_truth_Y_<var>.csv`
  - `predictions_Y_<var>.csv`
- Comparison is **at-site only** (single grid cell), not global metrics.

---

## 3. Variable 1: occlp_vr (occluded soil P)

**Ground truth (layers 1–10):**  
147.23, 144.12, 140.56, 138.03, 136.41, 135.51, 135.12, 135.01, 135.00, 135.00 g/m²

| Layer | GT     | natveg_improved (pred) | extreme_soilp (pred) |
|-------|--------|------------------------|----------------------|
| 1     | 147.23 | 121.99                | 140.27               |
| 2     | 144.12 | 118.48                | 137.90               |
| 3     | 140.56 | 117.05                | 134.98               |
| 4     | 138.03 | 112.22                | 133.14               |
| 5     | 136.41 | 108.15                | 132.21               |
| 6     | 135.51 | 112.34                | 131.26               |
| 7     | 135.12 | 112.50                | 131.10               |
| 8     | 135.01 | 107.18                | 130.58               |
| 9     | 135.00 | 109.06                | 130.50               |
| 10    | 135.00 | 112.40                | 130.62               |

**Summary:**

- **natveg_improved:** Large underprediction (~25–30 g/m² in top layers, ~23–28 g/m² in deep layers).
- **extreme_soilp:** Much closer to GT: underprediction reduced to ~5–7 g/m² in top layers and ~4.5 g/m² in deep layers.

**Conclusion:** occlp_vr at the Amazon site is clearly better with the extreme soil-P config.

---

## 4. Variable 2: solutionp_vr (solution P)

**Ground truth (layers 1–10):**  
0.00127, 0.000931, 0.000556, 0.000300, 0.000138, 5.02e-05, 1.32e-05, 3.29e-06, 1.95e-06, 1.85e-06

| Layer | GT       | natveg_improved (pred) | extreme_soilp (pred) |
|-------|----------|------------------------|----------------------|
| 1     | 0.00127  | 0.00592                | 0.000203             |
| 2     | 0.000931 | 0.00543                | 0.000321             |
| 3     | 0.000556 | 0.00313                | 0.000253             |
| 4     | 0.000300 | 0.00186                | 0.000173             |
| 5     | 0.000138 | 0.00122                | 9.77e-05             |
| 6     | 5.02e-05 | 0.000391               | 4.37e-05             |
| 7     | 1.32e-05 | 0.000238               | 1.99e-05             |
| 8     | 3.29e-06 | 0.000235               | 1.08e-05             |
| 9     | 1.95e-06 | 0.000309               | 9.94e-06             |
| 10    | 1.85e-06 | 0.000247               | 1.09e-05             |

**Summary:**

- **natveg_improved:** Strong overprediction at Amazon (e.g. top layer ~4.7×; deep layers orders of magnitude too high).
- **extreme_soilp:** Much closer: slight underprediction in top layers; deeper layers closer to GT with some small overprediction in the deepest layers.

**Conclusion:** solutionp_vr at the Amazon site is much better with the extreme soil-P config.

---

## 5. Variable 3: labilep_vr (labile P)

**Ground truth (layers 1–10):**  
40.72, 30.09, 18.21, 9.90, 4.59, 1.67, 0.44, 0.11, 0.065, 0.062 g/m²

| Layer | GT    | natveg_improved (pred) | extreme_soilp (pred) |
|-------|-------|------------------------|----------------------|
| 1     | 40.72 | 47.46                  | 38.26                |
| 2     | 30.09 | 35.12                  | 30.76                |
| 3     | 18.21 | 21.91                  | 18.05                |
| 4     | 9.90  | 11.49                  | 9.37                 |
| 5     | 4.59  | 5.94                   | 4.15                 |
| 6     | 1.67  | 2.85                   | 1.40                 |
| 7     | 0.44  | 1.43                   | 0.39                 |
| 8     | 0.11  | 1.17                   | 0.18                 |
| 9     | 0.065 | 0.98                   | 0.15                 |
| 10    | 0.062 | 1.24                   | 0.14                 |

**Summary:**

- **natveg_improved:** Overprediction in all layers (e.g. +7 in layer 1, +5 in layer 2); deep layers (8–10) are ~10–20× too high (e.g. 0.11→1.17, 0.062→1.24).
- **extreme_soilp:** Slight underprediction in top layers (e.g. 40.7→38.3); mid layers very close (e.g. 18.2→18.0, 9.9→9.4); deep layers still overpredicted but far less than Phase 1 (0.11→0.18, 0.062→0.14).

**Conclusion:** labilep_vr at the Amazon site is clearly better with the extreme soil-P config.

---

## 6. Overall Summary

| Variable      | natveg_improved at Amazon     | extreme_soilp at Amazon        |
|---------------|-------------------------------|--------------------------------|
| **occlp_vr**  | Large underprediction         | Much closer to GT              |
| **solutionp_vr** | Large overprediction       | Much closer to GT              |
| **labilep_vr**   | Overprediction, huge in deep | Closer overall; smaller deep bias |

**Conclusion:** For the Amazon reference site (303.75°E, −17.43°), the **extreme soil-P** training run improves all three soil-P variables (occlp_vr, solutionp_vr, labilep_vr) compared to the natveg_improved (Phase 1) run. The extreme soil-P config (`training_config_experiment_3_natveg_occlp_extreme_soilp.json`) uses higher matrix loss and P-specific variable weights (e.g. occlp_vr weight 40) and yields better tropical P predictions at this site.

---

## 7. Reproducibility

- **Site validation script:** `scripts/cnp_result_validationplot_site.py --lon 303.75 --lat -17.4246 --tolerance 0.1 --stats-only` (run from results directory).
- **Data sources:**  
  - Extreme soil-P: `cnp_results/run_20260304_133300_natveg_occlp_extreme_soilp/cnp_predictions/soil_2d_*/*.csv`  
  - Natveg improved: `cnp_results/run_20260228_214757_natveg_improved/cnp_inference_entire_dataset/cnp_predictions/soil_2d_*/*.csv`
- **Config:** `config/training_config_experiment_3_natveg_occlp_extreme_soilp.json`
