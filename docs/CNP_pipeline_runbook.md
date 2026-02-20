## CNP Pipeline Runbook

This guide walks through the end-to-end workflow using a user-defined CNP_IO list.

### 1) Create your CNP_IO list
Use the CNP_IO template to create a user-defined list, e.g. `CNP_IO_demo.txt`).

### 2) Train the AI model
Run training with your variable list (edit the filename as needed):

```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt --epoch 100 2>&1 &
```

Notes:
- This launches training in the background and redirects logs to stdout/stderr.
- The run output directory will be created under `cnp_results/run_YYYYMMDD_HHMMSS`.
- Optional: restrict training/test split to tropical latitudes only:
```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt --epoch 100 \
  --tropical-only
```
- Optional: customize the latitude range or column name:
```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt --epoch 100 \
  --tropical-only --tropical-lat-range -23.5,23.5 --tropical-lat-column Latitude
```

### 2b) Configure Training Parameters (Optional)

You can customize training behavior using a unified configuration JSON file that consolidates all user-defined settings:

**Unified Config File (Recommended):**
Create a single JSON file (`config/training_config_unified.json`) with all training parameters:

```json
{
  "variable_weights": {
    "pft1d_weights": {"cpool": 2.0, "npool": 2.0, "tlai": 3.0},
    "soil2d_weights": {"primp_vr": 3.0, "litr2p_vr": 5.0},
    "scalar_weights": {"GPP": 1.5, "NPP": 1.5}
  },
  "tail_aware_weights": {
    "cpool": 5.0,
    "deadstemc": 5.0,
    "litr2p_vr": 5.0
  },
  "pft_zero_sparsity_weights": {
    "cpool": 1.0,
    "deadstemc": 1.0
  },
  "pft1d_activation_overrides": {
    "cpool": "abs",
    "deadstemc": "abs"
  }
}
```

Then use it during training:
```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt \
  --training-config-json config/training_config_unified.json \
  --epoch 100
```

**Individual Config Files (Legacy, Still Supported):**
You can also use separate JSON files for each configuration type:
```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt \
  --variable-weights-json config/variable_weights_config.json \
  --tail-aware-weights-json list1_tail_weights.json \
  --pft-zero-sparsity-weights-json pft_zero_weights.json \
  --pft1d-activation-overrides-json pft1d_activation_overrides.json \
  --epoch 100
```

**Mixing Unified + Individual Files:**
You can use unified config as a base and override specific sections:
```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt \
  --training-config-json config/training_config_unified.json \
  --tail-aware-weights-json custom_tail_weights.json \
  --epoch 100
```

**Configuration Sections:**
- `variable_weights`: Per-variable loss weights for PFT1D, Soil2D, and Scalar outputs
- `tail_aware_weights`: Multipliers for tail-aware loss on heavy-tailed variables
- `pft_zero_sparsity_weights`: Weights for PFT zero sparsity penalty
- `pft1d_activation_overrides`: Per-variable activation function overrides

**Documentation:**
- See `config/UNIFIED_CONFIG_README.md` for detailed unified config usage
- See `config/VARIABLE_WEIGHTS_README.md` for variable weights details

**Repeat experiment_2 (run_20260212_162802_experiment_2):**  
To reproduce the same setup as the reference experiment_2 run:

```bash
python train_cnp_model.py \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --training-config-json config/training_config_experiment_2.json \
  --pft-zero-sparsity-weight 1.0 \
  --epoch 100 \
  --tropical-only
```

Data paths and file pattern come from your CNP_IO file. The run used `1_training_data_batch_*.pkl` under the paths listed in the variable list. Mask absent PFTs and other options were applied from the unified config.

### 2a) Fine-tune a pretrained model (optional)
If you already have a trained checkpoint and want to continue training on a TVA-style dataset, use the fine-tuning helper. Populate the necessary paths in your CNP_IO file (e.g. `CNP_IO_updated9_dev_gao.txt`):

```
fine_tuning_dataset_path: /path/to/TVA_dataset_root
pretrained_model_path: /path/to/base_run/cnp_predictions/model.pth
output_finetuned_model_dir: ./cnp_results/
finetuned_model_filename: model.pth   
num_epochs: 150                              
```

Then launch fine-tuning:

```bash
python scripts/run_finetuning.py 
```

What happens:
- The script reuses the original training pipeline (same preprocessing, normalization, ModelTrainer).
- `fine_tuning_dataset_path` + `FILE_PATTERN` point to the TVA batches to load.
- `pretrained_model_path` is loaded before training begins so weights pick up where the base run ended.
- Outputs are written to `cnp_results/run_YYYYMMDD_HHMMSS/` (same layout as full training).  
  Check `cnp_predictions/model.pth`, `cnp_config.json`, `cnp_metrics.json`, and the refreshed prediction CSVs.

### 3) Navigate to the run directory

```bash
cd cnp_results/run_YYYYMMDD_HHMMSS  # e.g., cnp_results/run_20250815_205419
```

### 4) Validate predictions vs ground truth (test split)
Generates quick statistics and a prediction quality report. The report now also creates filtered plots for the “top variables by bad-count”.

```bash
python ../../scripts/cnp_result_validationplot.py --stats-only && \
python ../../scripts/generate_prediction_quality_report.py
```

### 4.0.1) Validate predictions in `cnp_inference_entire_dataset` (inference subdirectory)
If inference wrote outputs under a subdirectory (e.g. `cnp_inference_entire_dataset/` with `cnp_predictions/` inside it), run validation by passing that directory. From the **run** directory (e.g. `cnp_results/run_YYYYMMDD_HHMMSS`):

```bash
# One-time: copy run config into the inference dir so validation can read variable lists (if not already there)
cp cnp_config.json cnp_inference_entire_dataset/ 2>/dev/null || true

# Stats + quality report for the inference subdirectory
python ../../scripts/cnp_result_validationplot.py cnp_inference_entire_dataset --stats-only && \
python ../../scripts/generate_prediction_quality_report.py cnp_inference_entire_dataset/validation_stats.csv
```

- First command writes `cnp_inference_entire_dataset/validation_stats.csv`.
- Second reads that CSV and writes `cnp_inference_entire_dataset/analysis/` (quality report, plots, `quality_summary_report.txt`, `top_bad_plots/`, etc.).
- Replace `cnp_inference_entire_dataset` with your actual inference subdirectory name if different.

### 4.1) Highlight a site in all scatter plots (optional)
Highlight a given site (lon/lat) in every GT vs Pred scatter plot. The site will be marked as a red star.

```bash
python ../../scripts/cnp_result_validationplot_site.py . \
  --lon 303.75 --lat -17.4246
```

Options and behavior:
- The script matches samples by coordinate with a tolerance (default: 0.01 degrees).
- Use `--tolerance N` to relax the matching if no point is found.
- Use `--coord-csv /path/to/test_static_inverse.csv` to explicitly provide a coordinate source.
- Supports `--top-bad-only` / `--worst-only` for restricted plotting.

**Output**: scatter plots are saved under `plots_site/`.

Options and behavior:
- The quality report saves outputs under `analysis/`:
  - `detailed_quality_assessment.csv`, `variable_quality_summary.csv`, `overall_prediction_quality.png`, `prediction_quality_by_variable.png`, `r2_vs_rmse.png`, `prediction_quality_report.html`, `quality_summary_report.txt`.
  - `bad_predictions_detailed.csv` (full list of rows classified as bad; can be disabled).
  - `top_bad_plots/` folder with filtered scatter plots for variables listed as “top variables by bad-count”.
- Useful flags:
  - `--no-top-bad-plots`: skip generating `analysis/top_bad_plots/`.
  - `--bad-html-limit N`, `--bad-text-limit N`, `--no-export-bad`.
  - `--force-xlim-01/--no-force-xlim-01`, `--print-scatter-stats/--no-print-scatter-stats`.

**Output**: `analysis/` contains CSV/PNG/HTML reports plus `top_bad_plots/`.

**Quality Classification**:
- **Good**: R² ≥ 0.9, Relative RMSE ≤ 0.1, Relative MAE ≤ 0.1
- **OK**: R² ≥ 0.7, Relative RMSE ≤ 0.25, Relative MAE ≤ 0.25
- **Bad**: Below OK thresholds

**Detailed Documentation**: See `docs/README_prediction_quality.md` for comprehensive usage instructions and advanced features.

### 5) Run inference on the entire dataset
Creates a folder `cnp_inference_entire_dataset` with AI predictions for the entire dataset.

```bash
python ../../scripts/run_inference_all.py --model model.pth --output-dir cnp_inference_entire_dataset > run_inference_all.log 2>&1 &
```

**CNP stoichiometric ratio enforcement (optional):** To enforce C:N and C:P ratios by deriving N/P from C predictions after inference, add `--derive-np-from-c`:

```bash
python ../../scripts/run_inference_all.py --model model.pth --output-dir cnp_inference_entire_dataset --derive-np-from-c > run_inference_all.log 2>&1 &
```

This overwrites N/P prediction files with stoichiometrically consistent values. Training should still use the full CNP variable list. See **CNP ratio and derivation docs** below.

### 6) Export AI predictions to NetCDF
Creates a NetCDF file containing all AI predictions for plotting and comparison:
`comparison_results/ai_predictions_for_plotting.nc`

```bash
python ../../scripts/ai_predictions_to_netcdf.py  > ai_prediction_to_netcdf.log 2>&1 &
```
To verify the CSV predictions match the generated NetCDF:
```bash
python ../../scripts/verify_predictions_in_netcdf.py
```

### 7) Generate comparison plots
Creates map plots for the variables in your CNP_IO list under
`./ai_model_comparison_plots/comparison_CNP_IO_demo`.
   
```bash
python ../../scripts/ai_model_comparison_plot.py  > ai_model_comparison.log 2>&1 &
```
#### 7.1) Stats-only mode (no plotting)
To validate that the variable values from the CSV file are consistent with those in the NetCDF file, you can compare their summary statistics (sum, standard deviation, minimum, and maximum) for all variables and their corresponding layers/PFTs using the --stats-only and --variable-list options:
```bash
python ../../scripts/ai_model_comparison_plot.py \
  --stats-only \
  --variable-list ../../CNP_IO_updated9_dev.txt
```

### 8) Create a new ELM restart file using AI predictions

```bash
python ../../scripts/ai_predictions_to_restart.py > ai_predictions_to_restart.log 2>&1 &
```

Outputs a new restart file derived from
default= `20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc`
old file is :`original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc`.



### 9) Compare restart files
Compares selected layers and PFTs; optionally verify with `restart_variable_plot.py`.

```bash
python ../../scripts/ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt --layers 0,5,9 --pfts 0,1,2,3,4,5
# Optionally use ../../scripts/restart_variable_plot.py for manual verification
```
#### 9.1) Stats-only mode (no plotting)
To validate that the variable values in the predicted NetCDF file are consistent with those in the updated restart file, you can compare their summary statistics (sum, standard deviation, minimum, and maximum) for all variables and their corresponding layers/PFTs using the --stats-only and --variable-list options:
```bash
python ../../scripts/ai_restart_comparison.py \
  --stats-only \
  --variable-list ../../CNP_IO_updated9_dev.txt
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


### Run Inference on the TVA Dataset 
The TVA workflow allows you to perform site-specific inference using a trained model and generate updated restart files for targeted locations. Given one or more geographic coordinates, the script extracts all required variables from the TVA dataset, runs model inference to obtain predicted values, and produces a new restart file reflecting the AI-updated state.

All codes are organized under the ./TVA_1_Sample folder.
Geographic coordinates are defined in locations.csv.
```bash
python ./TVA_1_Sample/run_workflow.py \
  --restart-file /path/to/20year_restart_file.nc \
  --model-path /path/to/trained_model_TVA.pt \
  --dataset-root /path/to/TVA_dataset \
  --variable-list ../CNP_IO_updated9_dev.txt
```

### Extract Single Point Data from ELM Restart File
The `extract_elm_restart_point.py` script extracts all data for specified geographic coordinates from a global ELM restart NetCDF file. It correctly handles ELM's multi-level hierarchical structure (gridcell → topounit → landunit → column → pft) and creates a subset NetCDF file containing only the data for the target location.

**Key Features**:
- Extracts data at all hierarchical levels for a single geographic point
- Automatically finds the nearest neighbor gridcell to target coordinates
- Preserves all metadata and encoding information from the original file
- Supports both 0-360° and -180 to 180° longitude formats
- Can be used via command-line arguments or with hardcoded defaults

**Usage**:
```bash
python scripts/extract_elm_restart_point.py \
  --restart-file /path/to/input_restart_file.nc \
  --lat 35.833332(Target latitude coordinate) \
  --lon -84.208336(Target longitude coordinate) \
  --output-file single_point_extracted.nc
```

---

### CNP stoichiometry and derivation (docs)

Documentation for enforcing C:N and C:P ratios and deriving N/P from C at inference:

| Doc | Description |
|-----|-------------|
| `docs/CNP_RATIO_ENFORCEMENT_USAGE.md` | **How to enable** ratio enforcement: `--derive-np-from-c`, standalone script, validation |
| `docs/CNP_DERIVATION_CLARIFICATION.md` | Why train full CNP (not C-only) and derive N/P at inference |
| `docs/CNP_DERIVATION_APPROACH.md` | Rationale and benefits of the derivation approach |
| `docs/CNP_DERIVATION_IMPLEMENTATION.md` | Implementation details and ratio definitions |
| `docs/DERIVATION_RESULTS_INTERPRETATION.md` | How to interpret derivation and ratio validation results |
| `docs/CNP_STOICHIOMETRIC_RELATIONSHIPS.md` | Underlying CNP ratio relationships and target variables |

Scripts: `scripts/derive_np_from_c.py`, `scripts/validate_cnp_ratios.py`.