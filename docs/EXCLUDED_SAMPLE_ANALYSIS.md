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
  When the **natveg filter** is enabled (default; use `--no-natveg-filter` to turn it off), metrics for PFT and 2D variables are computed **only over included gridcells**. Excluded gridcells are not used in the metric.

- **GT vs Pred scatter plots in `top_bad_plots`**  
  When the natveg filter is **on**, the script first restricts to included gridcells, then plots and computes R²/RMSE on that subset. So both the **plot** and the **metric** in `validation_stats.csv` are for included-only. When the filter is **off**, all test gridcells are used for the plot and the metric.

**Fair comparison (no-filter vs natveg run):** For the **natveg** run we should only evaluate on gridcells that pass the natveg filter (the “relevant” gridcells). The natveg model is trained only on those, so we expect it to perform **somewhat better** on that subset. Therefore:
- Compare **both** runs using **validation_stats** (or the quality report) generated with the natveg filter on (default; do not pass `--no-natveg-filter`). Then both metrics are “included gridcells only,” and the natveg run can be fairly expected to be somewhat better on that subset.
- The **inference-time** metrics (`cnp_predictions/test_metrics.csv`, `cnp_metrics.json`) are computed over the **full** test set. For the natveg run that is unfair (it includes gridcells the model was not trained on), so prefer the validation script with the filter for reporting natveg performance.

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

- **Default:** Natveg filter is **on**. PFT/2D metrics use only **included** gridcells. Pass `--no-natveg-filter` to use all gridcells instead.

```bash
# Validation stats and top_bad plots *with* natveg filter (default; included gridcells only)
python scripts/cnp_result_validationplot.py cnp_results/run_20260220_145518

# Validation stats and top_bad plots *without* filter (all gridcells)
python scripts/cnp_result_validationplot.py cnp_results/run_20260220_145518 --no-natveg-filter
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
| What changes with the natveg filter (on by default)? | Only the **metrics** (R², RMSE, MAE) for PFT/2D variables; they are computed over **included** gridcells only. |
| Where is the list of excluded locations? | `analysis/excluded_locations/excluded_locations_lat_lon.csv` after running `plot_excluded_locations.py`. |
| How do I plot only excluded gridcells? | Run `scripts/plot_excluded_locations.py <results_dir>`; see `analysis/excluded_locations/excluded_locations_<var>_gt_vs_pred.png`. |

## 9. Test Set Alignment: Natveg Run vs No-Filter Run

When comparing a **natveg-only** run to a **no-filter** run, the test sets can differ depending on when the filter is applied.

### Two behaviors

| Option | When filter is applied | Test set |
|--------|-------------------------|----------|
| **Filter before split** (default / legacy) | Filter to natveg **before** shuffle and 80/20 split. | Test = 20% of **natveg-only** data (e.g. ~2802 rows). **Not** a subset of the no-filter test set. |
| **Split then filter** (`--no-natveg-filter-before-split`) | Shuffle and split on **full** data (same as no-filter); then filter **only the training set** to natveg. | Test = same 20% as no-filter run (e.g. ~4166 rows). Natveg validation plots show only the **included** subset of that test set. |

So: **for the natveg run’s validation data to be a subset of the no-filter run’s validation data**, use **split then filter** (do **not** filter before split).

### How to get aligned test sets

- **Training:** use `natveg_only: true` and `natveg_filter_before_split: false` in config, or `--natveg-only --no-natveg-filter-before-split` on the CLI.
- **Inference:** uses the same data config from `cnp_config.json` (including `natveg_filter_before_split`), so test rows and `test_static_inverse.csv` match the no-filter run.
- **Validation:** with the natveg filter on, plots and metrics use only the **included** gridcells from that same test set, so points in e.g. `2D_soil4c_vr_Layer10_gt_vs_pred.png` are a subset of the no-filter run’s plot.

### Verifying overlap

Use the script below to compare two runs’ test gridcells by (Latitude, Longitude):

```bash
python scripts/verify_natveg_test_subset.py \
  cnp_results/run_NOFILTER/cnp_predictions \
  cnp_results/run_NATVEG/cnp_predictions
```

If the natveg run was trained with `natveg_filter_before_split: false`, the natveg test set (rows in `test_static_inverse.csv`) should be **identical** to the no-filter test set; the script reports overlap and subset relationship.

## 10. Related Files

- **Validation (filter logic):** `scripts/cnp_result_validationplot.py` — `_load_gridcell_metadata()`, `analyze_1d_new_structure(..., gridcell_metadata)`, `analyze_2d_new_structure(..., gridcell_metadata)`.
- **Quality report:** `scripts/generate_prediction_quality_report.py` — natveg filter on by default when invoking validation for top-bad plots; use `--no-natveg-filter` to turn it off.
- **Excluded-only plots:** `scripts/plot_excluded_locations.py`.
- **Prediction quality overview:** `docs/README_prediction_quality.md`.
