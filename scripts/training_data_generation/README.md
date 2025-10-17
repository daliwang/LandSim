# Training Data Generation

Simple workflow to extract TVA forcing data from monthly NetCDF files.

## Setup

### 1. Create Virtual Environment
```bash
python3.11 -m venv venv_py311
```

### 2. Activate Virtual Environment
```bash
source venv_py311/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Complete Workflow

1. **Generate Forcing NetCDF Files**
```bash
./bash_script/run_all_forcing_extraction.sh
```
Generates 6 NetCDF files in `output/forcing_netcdf/`:
- TVA_FLDS_1980-1999.nc (Longwave radiation)
- TVA_FSDS_1980-1999.nc (Shortwave radiation)
- TVA_PSRF_1980-1999.nc (Surface pressure)
- TVA_QBOT_1980-1999.nc (Specific humidity)
- TVA_PRECTmms_1980-1999.nc (Precipitation)
- TVA_TBOT_1980-1999.nc (Air temperature)

2. **Validate NetCDF Data Accuracy**
```bash
./bash_script/run_forcing_netcdf_validation.sh
```
Compares generated NetCDF files with reference files to ensure correctness.

3. **Generate PKL Files (Training Ready)**
```bash
./bash_script/run_forcing_pkl_generation.sh
```
Creates optimized PKL files in `output/forcing_hourly_pkl/` for machine learning:
- 12 batch files (TVA_forcing_batch_01.pkl to TVA_forcing_batch_12.pkl)
- Each batch contains 1000 gridcells (except last batch: 357)
- 9 variables per gridcell: landfrac, lat, lon, 6 forcing variables
- 58,400 time steps (3-hour resolution, 20 years)
- **Automatically converts forcing variables to list format for training compatibility**

4. **Validate PKL Data Accuracy**
```bash
./bash_script/run_forcing_pkl_validation.sh
```
Validates PKL files against generated NetCDF files using sequential gridcell mapping:
- Validates first 5 PKL batches (5000 gridcells total)
- Each batch corresponds to sequential NetCDF gridcells (0-999, 1000-1999, etc.)
- Ensures PKL data matches NetCDF data with 100% accuracy
- **PKL files are already in list format and ready for training**

5. **Generate Complete Training Dataset (Monthly Averaged)**
```bash
./bash_script/run_incomplete_training_dataset.sh
```
- Integrates ecosystem variables with forcing data
- Applies monthly averaging to forcing variables (240 values for 20 years)
- Output: `output/training_dataset_pkl/monthly_training_data_batch_XX.pkl`
- Automatically removes original PKL files

6. **Generate Enhanced Dataset**
```bash
./bash_script/run_enhanced_dataset_generation.sh
```
- Adds pool variables (cpool, npool, ppool, xsmrpool) from restart files
- Adds 38 transfer variables and corresponding Y variables
- Output: `output/enhanced_training_dataset/enhanced_monthly_training_data_batch_XX.pkl`
- Automatically removes intermediate files

7. **Add PFT Variables**
```bash
./bash_script/run_adding_pft_variables.sh
```
- Adds PFT (Plant Functional Type) variables from `clm_params_c211124.nc`
- Removes unwanted variables (fire-related, unnecessary PFT variables, SCALARAVG_vr)
- **Final dataset ready for machine learning training**

## Data Validation

### When to Validate Your Data

**After Step 1 (Forcing NetCDF Generation):**
```bash
./validation/forcing_netcdf_validation.py
```
- Validates generated NetCDF files against reference files
- Ensures 6 forcing variables are correctly processed

**After Step 4 (Forcing PKL Generation):**
```bash
./bash_script/run_forcing_pkl_validation.sh
```
- Validates PKL files against generated NetCDF files
- Sequential gridcell mapping validation (first 5 batches)

**After Step 7 (Final Enhanced Dataset):**
```bash
./bash_script/run_comprehensive_validation.sh
```
- Complete validation of the final enhanced dataset
- Includes monthly averaging, data consistency, spatial mapping, and scientific validity checks
- **This is the most important validation** - run after completing all processing steps


## Configuration

Edit `config.py` to modify paths and settings.
