## CNP Pipeline Runbook

This guide walks through the end-to-end workflow using a user-defined CNP_IO list.

### 1) Create your CNP_IO list
Use the CNP_IO template to create a user-defined list, e.g. `CNP_IO_demo1.txt`).

### 2) Train the AI model
Run training with your variable list (edit the filename as needed):

```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt --epoch 100 2>&1 &
```

Notes:
- This launches training in the background and redirects logs to stdout/stderr.
- The run output directory will be created under `cnp_results/run_YYYYMMDD_HHMMSS`.

### 3) Navigate to the run directory

```bash
cd cnp_results/run_YYYYMMDD_HHMMSS  # e.g., cnp_results/run_20250815_205419
```

### 4) Validate predictions vs ground truth (test split) using the current run directory as default
Generates quick statistics and prediction quality report

```bash
python ../../scripts/cnp_result_validationplot.py --stats-only
python ../../scripts/generate_prediction_quality_report.py
```

(optional)
For a detail scatter plot of each variables and its substructure
```bash
python ../../scripts/cnp_result_validationplot.py > cnp_results_validation.log 2>&1 &
```bash

```bash
python ../../scripts/generate_prediction_quality_report.py > prediction_quality_report.log 2>&1 &
```

**Output**: Creates `analysis/` directory with:
- Quality assessment CSV files
- Interactive HTML report
- Visualization charts (bar charts, pie charts, scatter plots)
- Text summary report

**Quality Classification**:
- **Good**: R² ≥ 0.9, Relative RMSE ≤ 0.1, Relative MAE ≤ 0.1
- **OK**: R² ≥ 0.7, Relative RMSE ≤ 0.25, Relative MAE ≤ 0.25
- **Bad**: Below OK thresholds

**Detailed Documentation**: See `docs/README_prediction_quality.md` for comprehensive usage instructions and advanced features.

### 5) Run inference on the entire dataset
Creates a folder `cnp_inference_entire_dataset` with AI predictions for the entire dataset.

```bash
python ../../scripts/run_inference_all.py > run_inference_all.log 2>&1 &
```

### 6) Export AI predictions to NetCDF
Creates a NetCDF file containing all AI predictions for plotting and comparison:
`comparison_results/ai_predictions_for_plotting.nc`

```bash
python ../../scripts/ai_predictions_to_netcdf.py  > ai_prediction_to_netcdf.log 2>&1 &
```

### 7) Generate comparison plots
Creates map plots for the variables in your CNP_IO list under
`./ai_model_comparison_plots/comparison_CNP_IO_demo1`.
   
```bash
python ../../scripts/ai_model_comparison_plot.py  > ai_model_comparison.log 2>&1 &
```

### 8) Create a new ELM restart file using AI predictions

```bash
python ../../scripts/ai_predictions_to_restart.py > ai_predictions_to_restart.log 2>&1 &
```

Outputs a new restart file derived from
`original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc`.

### 9) Compare restart files
Compares selected layers and PFTs; optionally verify with `restart_variable_plot.py`.

```bash
python ../../scripts/ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt --layers 0,5,9 --pfts 0,1,2,3,4,5
# Optionally use ../../scripts/restart_variable_plot.py for manual verification
```

---

### Old scripts (to be double-checked)

```bash
python ../../scripts/update_restart_with_aipredictions.py \
  --variable-list ../../CNP_IO_demo1.txt \
  --run-dir cnp_inference_entire_dataset \
  --input-nc ../../ELM_data/original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc \
  --output-nc updated_restart_with_CNP_IO_demo1.nc
```

Optional: `inference_map.py` can create statistics and interactive maps (currently not working well).


