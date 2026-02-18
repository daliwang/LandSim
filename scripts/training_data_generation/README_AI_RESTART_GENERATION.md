# LandSim AI Model: Generating New Restart Files

## Step 1: Prepare LandSim Inference Inputs

### 1.1 Locate required input files

You need the following files from your `tes_aoi_release` workflow and uELM run:

- **Domain file**: `knox_domain.lnd.TES_SE.4km.1d.c*.nc` (from `tes_aoi_release/knox/domain_surfdata/`)
- **Surfdata file**: `knox_surfdata.TES_SE.4km.1d.NLCD.c*.nc` (from `tes_aoi_release/knox/domain_surfdata/`)
- **Forcing data**: Three subfolders in `tes_aoi_release/knox/forcing/`
- **History files**: `*.elm.h0.*.nc` (from `${KMELM_ROOT}/e3sm_runs/uELM_knox_*/`)
- **Restart file**: `*.elm.r.*.nc` (from `${KMELM_ROOT}/e3sm_runs/uELM_knox_*/`)

### 1.2 Configure training data generation

1. Navigate to the training data generation directory:

   ```bash
   cd /gpfs/.../LandSim/scripts/training_data_generation
   ```

2. Edit `config.py` to set the paths to your input files:

   ```python
   # Update these paths to match your environment
   domain_file = "/path/to/knox_domain.lnd.TES_SE.4km.1d.c*.nc"
   surfdata_file = "/path/to/knox_surfdata.TES_SE.4km.1d.NLCD.c*.nc"
   forcing_dir = "/path/to/tes_aoi_release/knox/forcing"
   history_files = "/path/to/kmELM/e3sm_runs/uELM_knox_*/history/*.elm.h0.*.nc"
   restart_file = "/path/to/kmELM/e3sm_runs/uELM_knox_*/restart/*.elm.r.*.nc"
   ```

3. Review the README in `scripts/training_data_generation/` for detailed configuration options and command examples.

### 1.3 Generate inference dataset

Run the training data generation script to create the inference dataset:

```bash
cd /gpfs/.../LandSim/scripts/training_data_generation
python enhanced_training_dataset.py --initial_only
# or for enhanced dataset with final spinup data:
# python enhanced_training_dataset.py --enhanced_dataset
```

This creates PKL files in the output directory (typically `output/initial_condition_dataset/` or `output/enhanced_dataset/`).

## Step 2: Run LandSim Inference

### 2.1 Activate LandSim environment

```bash
cd /gpfs/.../LandSim
source .venv/bin/activate
```

### 2.2 Run inference

From the LandSim repository root, run:

```bash
python scripts/run_inference_all.py \
  --model /path/to/trained/model.pth \
  --data-paths /gpfs/.../LandSim/scripts/training_data_generation/output/initial_condition_dataset \
  --file-pattern "enhanced_monthly_training_data_batch_*.pkl" \
  --output-dir /gpfs/.../LandSim/cnp_inference_knox_initial \
  --use-training-config \
  --strict-loading
```

**Parameters:**
- `--model`: Path to your trained model file (`.pth`)
- `--data-paths`: Directory containing the PKL files generated in Step 1
- `--file-pattern`: Pattern matching your PKL files (adjust based on dataset type)
- `--output-dir`: Directory where inference results will be saved
- `--use-training-config`: Use the training configuration for normalization
- `--strict-loading`: Enforce strict loading requirements

**Note:** Adjust the `--file-pattern` based on your dataset:
- For initial condition dataset: `"initial_condition_batch_*.pkl"` or `"enhanced_monthly_training_data_batch_*.pkl"`
- For enhanced dataset with final spinup: `"enhanced_1_training_data_batch_*.pkl"`

### 2.3 Verify inference output

Check the output directory for generated files:

```bash
ls -lh /gpfs/.../LandSim/cnp_inference_knox_initial/cnp_predictions/
```

You should see:
- `predictions_scalar.csv`
- `pft_1d_predictions/` directory
- `soil_2d_predictions/` directory
- `predictions.pkl` file

## Step 3: Convert AI Predictions to NetCDF

Convert the AI predictions to NetCDF format for visualization and further processing:

```bash
cd /gpfs/.../LandSim/scripts
python ai_predictions_to_netcdf.py \
  --ai-predictions /gpfs/.../LandSim/cnp_inference_knox_initial/cnp_predictions \
  --output /gpfs/.../LandSim/cnp_inference_knox_initial/ai_predictions_knox.nc \
  --wrap-longitude
```

**Parameters:**
- `--ai-predictions`: Directory containing the AI predictions from Step 2
- `--output`: Output NetCDF file path
- `--wrap-longitude`: Convert longitudes to -180–180 range for compatibility with plotting utilities

The resulting NetCDF file (`ai_predictions_knox.nc`) can be used for:
- Visualization with plotting scripts
- Comparison with model outputs
- Further analysis

## Step 4: Create AI-Adjusted Restart File

### 4.1 Update restart file with AI predictions

Run the restart updater to create an AI-adjusted restart file. Only variables listed in the variable list file will be replaced with AI predictions; all other variables remain unchanged:

```bash
cd /gpfs/.../LandSim/scripts
python ai_predictions_to_restart.py \
  --ai-predictions /gpfs/.../LandSim/cnp_inference_knox_initial/ai_predictions_knox.nc \
  --restart-file /gpfs/.../kmELM/e3sm_runs/uELM_knox_I1850CNPRDCTCBC/original_uELM_knox_I1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc \
  --variable-list /gpfs/.../LandSim/CNP_IO_updated14_xfer.txt \
  --output /gpfs/.../LandSim/cnp_inference_knox_initial/uELM_knox_AIrestart.elm.r.0021-01-01-00000.nc
```

**Parameters:**
- `--ai-predictions`: NetCDF file with AI predictions (from Step 3)
- `--restart-file`: Original restart file from uELM accelerated spinup
- `--variable-list`: Path to variable list file (e.g., `CNP_IO_updated14_xfer.txt`) that specifies which variables to update
- `--output`: Path for the new AI-adjusted restart file


### 4.2 Use AI-adjusted restart for final spinup

The AI-adjusted restart file can now be used in the `tes_aoi_release` workflow for final spinup. See the `tes_aoi_release` README for instructions on using the restart file.


