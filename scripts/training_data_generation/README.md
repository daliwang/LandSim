# Training Data Generation

Simple workflow to generate training datasets for machine learning using two main scripts.

## Setup

### 1. Create Virtual Environment
```bash
# Create a virtual environment in the current directory
python3 -m venv venv
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Scripts Overview

### 1. `construct_forcing_20years.py`
Generates processed forcing NetCDF files from raw monthly data.

### 2. `enhanced_training_dataset.py`
Generates three different types of training datasets for machine learning.

## Usage

### Step 1: Generate Forcing NetCDF Files

First, create the forcing NetCDF files from raw monthly data:

```bash
python python_scripts/construct_forcing_20years.py
```

**Command Line Options:**
```bash
python python_scripts/construct_forcing_20years.py \
    --input-dir /path/to/raw/forcing/data \
    --output-dir /path/to/output/directory \
    --start-year 1980 \
    --end-year 1999
```

**Output**: `output/forcing_netcdf/`
- `FLDS_1980-1999.nc` (Downward longwave radiation)
- `FSDS_1980-1999.nc` (Downward shortwave radiation)
- `PRECTmms_1980-1999.nc` (Precipitation rate)
- `PSRF_1980-1999.nc` (Surface pressure)
- `QBOT_1980-1999.nc` (Specific humidity)
- `TBOT_1980-1999.nc` (Air temperature)

### Step 2: Generate Training Datasets

Use the unified script with three different modes:

```bash
python python_scripts/enhanced_training_dataset.py --[mode]
```

#### Mode 1: Forcing-Only Dataset (`--forcing_only`)
- **Purpose**: Raw meteorological forcing data (no monthly averaging)
- **Variables**: 9 variables (landfrac, lat, lon, 6 forcing variables)
- **Data**: Raw 3-hour time series (58,400 time steps)
- **Output**: `output/forcing_only_dataset/`
- **Files**: `forcing_data_batch_XX.pkl`

#### Mode 2: Enhanced Dataset (`--enhanced_dataset`)
- **Purpose**: Complete ecosystem training dataset with all variables
- **Variables**: 291 variables (initial conditions + Y_variables + PFT variables)
- **Data**: Monthly averaged time series + ecosystem state variables
- **Output**: `output/enhanced_training_dataset/`
- **Files**: `enhanced_monthly_training_data_batch_XX.pkl`
- **Intermediate**: `output/training_dataset_pkl/` (Step 1 output)

#### Mode 3: Initial-Only Dataset (`--initial_only`)
- **Purpose**: Initial conditions without simulation results (no Y_variables)
- **Variables**: 195 variables (initial conditions + PFT variables, NO Y_variables)
- **Data**: Initial state variables only
- **Output**: `output/initial_condition_dataset/`
- **Files**: `enhanced_monthly_training_data_batch_XX.pkl`

## Output Directory Structure

```
output/
├── forcing_netcdf/                     # Step 1: Forcing NetCDF files
│   ├── FLDS_1980-1999.nc
│   ├── FSDS_1980-1999.nc
│   └── ... (6 NetCDF files)
│
├── forcing_only_dataset/               # Mode 1: Forcing-only
│   ├── forcing_data_batch_01.pkl
│   └── ... (batch files)
│
├── training_dataset_pkl/               # Mode 2: Intermediate (Step 1)
│   ├── training_data_batch_01.pkl
│   └── ... (batch files)
│
├── enhanced_training_dataset/           # Mode 2: Final output
│   ├── enhanced_monthly_training_data_batch_01.pkl
│   └── ... (batch files)
│
└── initial_condition_dataset/          # Mode 3: Initial-only
    ├── enhanced_monthly_training_data_batch_01.pkl
    └── ... (batch files)
```

## Quick Start

1. **Generate forcing files**:
   ```bash
   python python_scripts/construct_forcing_20years.py
   ```

2. **Choose your dataset mode**:
   ```bash
   # For complete ecosystem data (291 variables)
   python python_scripts/enhanced_training_dataset.py --enhanced_dataset
   
   # For initial conditions only (195 variables)
   python python_scripts/enhanced_training_dataset.py --initial_only
   
   # For forcing data only (9 variables, raw time series)
   python python_scripts/enhanced_training_dataset.py --forcing_only
   ```

3. **Find your results** in the corresponding `output/` subdirectory.

## Configuration

**All file paths and settings can be modified in `config.py`:**

- **Input data paths**: Raw forcing data, surface data, restart files
- **Output directories**: Where to save generated datasets
- **File patterns**: How to find input files
- **Variable definitions**: Which variables to include (from CNP_IO file)

**Key configuration sections:**
- `forcing_raw_data_path`: Raw monthly forcing data directory
- `forcing_netcdf_output_dir`: Processed forcing NetCDF output
- `output_dir`: Base output directory for all datasets
- `surface_data_files`: Surface data NetCDF files
- `ad_spinup_*_files`: Initial spinup files
- `final_spinup_*_files`: Final spinup files (for Y_variables)
- `clm_params_nc_path`: CLM parameters file for PFT variables

**Example configuration changes:**
```python
# Change input data directory
forcing_raw_data_path = '/path/to/your/raw/data'

# Change output directory
output_dir = './your_output'

# Change dataset files
surface_data_files = ['/path/to/your/surface.nc']
```

