# Excluded Sample Analysis (Natveg Filter)

This document describes how **excluded gridcells** are defined, how they affect validation metrics and plots, and how to analyze or plot variables at those locations only. It also includes a **training-data exclusion report** for the natveg filter.

---

## Report: Training Data Exclusion by Natveg Filter

**Purpose:** Quantify how many training samples would be excluded if the natveg filter were applied at training time (i.e., training only on gridcells with natural vegetation).

**Filter rule (same as in validation):**
- **Include:** `(PCT_NATVEG > 0) AND (PCT_NAT_PFT_0 < 100)`
- **Exclude:** `(PCT_NATVEG = 0) OR (PCT_NAT_PFT_0 >= 100)`  
  *(PCT_NATVEG is a percentage, so “no natural vegetation” is exactly 0; code may use ≤ 0 for robustness.)*

### Dataset

| Item | Value |
|------|--------|
| Data path | `/mnt/proj-shared/AI4BGC_7xw/TrainingData/Trendy_1_data_CNP` |
| File pattern | `training_data_batch_*.pkl` |
| Number of batch files | 21 |
| Total samples loaded | 20,975 |

### Results

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total samples** | 20,975 | 100% |
| **Included (kept)** | 14,100 | **67.22%** |
| **Excluded** | **6,875** | **32.78%** |

### Exclusion breakdown

| Condition | Count | Note |
|----------|--------|------|
| PCT_NATVEG = 0 | 5,921 | No natural vegetation (0%) |
| PCT_NAT_PFT_0 ≥ 100 | 6,875 | 100% non-vegetated PFT (all excluded rows satisfy this) |
| Both conditions | 5,921 | Overlap: 954 excluded only by PFT_0 ≥ 100 |

### Interpretation

- **~33% of training samples** would be dropped by the natveg filter on this dataset.
- This is a **dramatic** reduction: training with the filter would use **14,100** samples instead of **20,975**.
- If the filter is adopted for training, expect:
  - Fewer samples per epoch and potentially different convergence.
  - A model focused on gridcells with natural vegetation; predictions at excluded gridcells would be extrapolation.

### How to reproduce / re-run the count

Use the script from the repo root:

```bash
# Full dataset (same paths as above)
python scripts/count_natveg_filter_exclusion.py \
  --data-paths /mnt/proj-shared/AI4BGC_7xw/TrainingData/Trendy_1_data_CNP \
  --file-pattern "training_data_batch_*.pkl"

# With config/variable list (uses config data paths)
python scripts/count_natveg_filter_exclusion.py --variable-list CNP_IO_updated9_dev_dw.txt

# Quick check on a subset
python scripts/count_natveg_filter_exclusion.py --data-paths /path/to/data --max-files 5

# After applying tropical filter first
python scripts/count_natveg_filter_exclusion.py --data-paths /path/to/data --after-tropical
```

Script: `scripts/count_natveg_filter_exclusion.py`.

---

## 1. What Are Excluded Gridcells?

For PFT (1D) and soil 2D variables, we optionally exclude gridcells that are not meaningful for vegetation-related evaluation:

- **PCT_NATVEG = 0** — no natural vegetation.
- **PCT_NAT_PFT_0 = 100** — 100% of the gridcell is “PFT 0” (non-vegetated).

So **excluded** = gridcells with no natural vegetation or 100% non-vegetated PFT.  
**Included** = gridcells with `PCT_NATVEG > 0` **and** `PCT_NAT_PFT_0 < 100`.

These flags come from the **static** inputs (e.g. `test_static_inverse.csv`), which must have columns `PCT_NATVEG` and `PCT_NAT_PFT_0`.

## 2. Where the Exclusion Is Used

- **Validation metrics (R², RMSE, MAE)**  
  When the **natveg filter** is enabled (`--natveg-filter`), metrics for PFT and 2D variables are computed **only over included gridcells**. Excluded gridcells are not used in the metric.

- **GT vs Pred scatter plots in `top_bad_plots`**  
  The per-variable, per-layer scatter plots (e.g. `2D_soil1c_vr_Layer1_gt_vs_pred.png`) always plot **all** gridcells (both included and excluded). The filter does **not** remove points from these plots; it only changes how the **single** R²/RMSE for that variable/layer is computed. So:
  - **With** `--natveg-filter`: the plot shows all points, but the R² in `validation_stats.csv` is for included-only.
  - **Without** `--natveg-filter`: the plot is the same (all points), and the R² is over all gridcells.

So **the excluded gridcells are still plotted** in `by_pft_layer` and similar GT-vs-Pred figures; they are only excluded from the **metric** when the filter is on.

## 3. Data Source: Row Alignment

Excluded vs included is determined from:

- **File:** `results_dir/cnp_predictions/test_static_inverse.csv`
- **Row order:** Must match the ground-truth and prediction CSVs (same test set, same row index = same gridcell).

The validation script builds a boolean mask from `PCT_NATVEG` and `PCT_NAT_PFT_0` and applies it by **row index** when computing metrics. It does **not** match by latitude/longitude.

## 4. Excluded Locations List (Lat/Lon)

For a given run, the list of excluded gridcells (lat/lon and PCT values) is written when you run the excluded-locations plotting script:

- **Path:**  
  `results_dir/analysis/excluded_locations/excluded_locations_lat_lon.csv`
- **Columns:** `Latitude`, `Longitude`, `PCT_NATVEG`, `PCT_NAT_PFT_0`
- **Rows:** One per excluded gridcell (e.g. 36 in a typical run).

Example (first rows):

```text
Latitude,Longitude,PCT_NATVEG,PCT_NAT_PFT_0
17.434555,25.0,99.99999,100.0
15.549739,15.000001,99.99999,100.0
22.146597,348.75,99.99999,100.0
...
```

Longitude is in 0–360° (e.g. 348.75° ≈ −11.25°).

## 5. Script: Plot Variables at Excluded Locations Only

To visualize **only** the excluded gridcells (GT vs Pred at those locations):

- **Script:** `scripts/plot_excluded_locations.py`
- **Input:** Results directory (e.g. `cnp_results/run_YYYYMMDD_HHMMSS`).
- **Output directory:**  
  `results_dir/analysis/excluded_locations/`  
  (override with `--output-dir`).

### What it does

1. Reads `test_static_inverse.csv` and computes the same exclude mask (PCT_NATVEG=0 or PCT_NAT_PFT_0=100).
2. For each of a set of variables (cpool, npool, ppool, litr1c_vr, litr1n_vr, litr1p_vr, soil1c_vr, soil1n_vr, soil1p_vr, primp_vr), loads GT and prediction CSVs.
3. Keeps **only rows** whose index is in the excluded set.
4. Plots **GT vs Pred** for those rows (all PFTs or layers flattened for that variable).
5. Writes:
   - One PNG per variable: `excluded_locations_<var>_gt_vs_pred.png`
   - `excluded_locations_lat_lon.csv` (list of excluded lat/lon and PCT_*).

### How to run

```bash
# From repo root; output under results_dir/analysis/excluded_locations/
python scripts/plot_excluded_locations.py cnp_results/run_20260220_145518

# Custom output directory
python scripts/plot_excluded_locations.py cnp_results/run_20260220_145518 -o /path/to/output
```

### Interpreting the plots

- Each point is one (gridcell × PFT) or (gridcell × layer) at **excluded** locations only.
- Points on the 1:1 line = good agreement at those cells; off the line = bias or error at excluded (e.g. 100% PFT0) gridcells.

## 6. Validation: Enabling the Natveg Filter

- **Default:** Filter is **off**. Validation uses all gridcells (same as “nofilter” behaviour).
- **Opt-in:** Pass `--natveg-filter` when generating validation stats so that PFT/2D metrics use only **included** gridcells.

```bash
# Validation stats and top_bad plots *without* filter (default)
python scripts/cnp_result_validationplot.py cnp_results/run_20260220_145518

# Validation stats and top_bad plots *with* filter (excluded cells not used in R²/RMSE)
python scripts/cnp_result_validationplot.py cnp_results/run_20260220_145518 --natveg-filter
```

The **scatter plots** in `top_bad_plots/by_pft_layer/` still show all gridcells in both cases; only the numbers in `validation_stats.csv` (and thus in the quality report) change when the filter is used.

## 7. Typical Counts (Example Run)

For one run (798 test gridcells):

| Metric              | Value   |
|---------------------|---------|
| Total gridcells     | 798     |
| Included (kept)     | 762 (95.5%) |
| Excluded (filtered) | 36 (4.5%)   |

All 36 exclusions in that run had **PCT_NAT_PFT_0 = 100** (and PCT_NATVEG ≈ 100); none had PCT_NATVEG = 0.

## 8. Summary

| Question | Answer |
|----------|--------|
| Are excluded gridcells still in the GT-vs-Pred scatter plots? | **Yes.** `top_bad_plots/by_pft_layer/*.png` always show all gridcells. |
| What changes with `--natveg-filter`? | Only the **metrics** (R², RMSE, MAE) for PFT/2D variables; they are computed over **included** gridcells only. |
| Where is the list of excluded locations? | `analysis/excluded_locations/excluded_locations_lat_lon.csv` after running `plot_excluded_locations.py`. |
| How do I plot only excluded gridcells? | Run `scripts/plot_excluded_locations.py <results_dir>`; see `analysis/excluded_locations/excluded_locations_<var>_gt_vs_pred.png`. |

## 9. Related Files

- **Validation (filter logic):** `scripts/cnp_result_validationplot.py` — `_load_gridcell_metadata()`, `analyze_1d_new_structure(..., gridcell_metadata)`, `analyze_2d_new_structure(..., gridcell_metadata)`.
- **Quality report:** `scripts/generate_prediction_quality_report.py` — optional `--natveg-filter` when invoking validation for top-bad plots.
- **Excluded-only plots:** `scripts/plot_excluded_locations.py`.
- **Prediction quality overview:** `docs/README_prediction_quality.md`.
