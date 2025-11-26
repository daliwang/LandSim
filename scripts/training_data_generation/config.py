"""
Configuration for Training Data Generation - TVA Dataset
"""

import os

try:
    from cnp_io_parse import parse_cnp_io_list
except Exception:
    parse_cnp_io_list = None

# =============================================================================
# DATA PATHS CONFIGURATION
# =============================================================================

# Raw forcing data directory (contains monthly NetCDF files)
forcing_raw_data_path = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/forcing'

# Base output directory (use absolute path)
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'output')

# Processed forcing NetCDF files output directory
# Updated to point to the actual location of processed forcing files
forcing_netcdf_output_dir = '/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/Dataset_test/TVA/TVA_Forcing_netcdf'

# Forcing PKL files output directory
forcing_pkl_output_dir = os.path.join(output_dir, 'forcing_hourly_pkl')

# Training dataset PKL files output directory
training_dataset_pkl_output_dir = os.path.join(output_dir, 'training_dataset_pkl')

# CLM parameters NetCDF file path
clm_params_nc_path = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/clm_params_c211124.nc'

# =============================================================================
# INPUT FILES CONFIGURATION
# =============================================================================

# Surface data files
surface_data_files = [
    '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/domain_surfdata/TVA_surfdata.TES_SE.4km.1d.NLCD.c241219.nc'
]

# AD-SPINUP files (initial spinup)
ad_spinup_history_files = [
    '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_adspinref.elm.h0.0021-01-01-00000.nc'
]

ad_spinup_restart_files = [
    '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_adspinref.elm.r.0021-01-01-00000.nc'
]

# FINAL-SPINUP files (final spinup)
final_spinup_history_files = [
    '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_runs/uELM_TVA_finalspinref/run/uELM_TVA_finalspinref.elm.h0.0781-01.nc'
]

final_spinup_restart_files = [
    '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc'
]

# Special P input NetCDF file
#special_p_input_nc = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc'

# =============================================================================
# 37_DATASET CONFIGURATION (Enhanced Dataset Generation)
# =============================================================================

class Config:
    # Input paths for enhanced dataset generation
    INPUT_GLOB = os.path.join(output_dir, "training_dataset_pkl", "monthly_training_data_batch_*.pkl")
    OUTPUT_DIR = os.path.join(output_dir, "enhanced_training_dataset")
    ENHANCED_PREFIX = "enhanced_"
    POOL_VARS = ["cpool", "npool", "ppool", "xsmrpool"]
    
    # Special P input NetCDF file
    #SPECIAL_P_INPUT_NC = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc"
    SPECIAL_P_VARS = []

    # CNP IO configuration file (can be changed to any CNP_IO file)
    CNP_IO_FILE = os.path.join(os.path.dirname(__file__), "python_scripts", "CNP_IO_updated14_xfer.txt")
    
    # Alternative CNP_IO files (uncomment to use different files)
    # CNP_IO_FILE = os.path.join(os.path.dirname(__file__), "python_scripts", "CNP_IO_alternative.txt")
    # CNP_IO_FILE = os.path.join(os.path.dirname(__file__), "python_scripts", "CNP_IO_custom.txt")

    # Dynamic variable lists (populated from CNP_IO file)
    # These will be automatically populated by apply_cnp_io_overrides()
    X_LIST_COLUMNS_1D = []
    X_LIST_COLUMNS_2D = []
    Y_LIST_COLUMNS_1D = []
    Y_LIST_COLUMNS_2D = []
    COLS_TO_DROP = []

    WATER_VARIABLES = []
    Y_WATER_VARIABLES = []

    VARS_TO_RESHAPE = ['cwdp', 'totcolp', 'totlitc', 'Y_cwdp', 'Y_totcolp', 'Y_totlitc']

    # Additional dataset variables
    dataset_new_1D_PFT_VARIABLES: list = []
    dataset_new_Water_variables: list = []
    dataset_new_TIME_SERIES_VARIABLES: list = []
    dataset_new_SURFACE_PROPERTIES: list = []
    dataset_new_PFT_PARAMETERS: list = []
    dataset_new_SCALAR_VARIABLES: list = []
    dataset_new_2D_VARIABLES: list = []
    dataset_new_RESTART_COL_1D_VARS: list = []

    @classmethod
    def apply_cnp_io_overrides(cls) -> None:
        try:
            if parse_cnp_io_list is None or not os.path.exists(cls.CNP_IO_FILE):
                return

            parsed = parse_cnp_io_list(cls.CNP_IO_FILE)

            new_1d_vars = list(dict.fromkeys(parsed.get('pft_1d_variables', []) or []))
            new_2d_vars = list(dict.fromkeys(parsed.get('variables_2d_soil', []) or []))
            new_water_vars = list(dict.fromkeys(parsed.get('water_variables', []) or []))
            new_cols_to_drop = list(dict.fromkeys(parsed.get('cols_to_drop', []) or []))

            cls.dataset_new_1D_PFT_VARIABLES = list(dict.fromkeys(
                (parsed.get('dataset_new_1D_PFT_VARIABLES') or parsed.get('pft_1d_variables') or [])
            ))
            cls.dataset_new_Water_variables = list(dict.fromkeys(
                (parsed.get('dataset_new_Water_variables') or parsed.get('water_variables') or [])
            ))
            cls.dataset_new_TIME_SERIES_VARIABLES = list(dict.fromkeys(
                (parsed.get('dataset_new_TIME_SERIES_VARIABLES') or [])
            ))
            cls.dataset_new_SURFACE_PROPERTIES = list(dict.fromkeys(
                (parsed.get('dataset_new_SURFACE_PROPERTIES') or [])
            ))
            cls.dataset_new_PFT_PARAMETERS = list(dict.fromkeys(
                (parsed.get('dataset_new_PFT_PARAMETERS') or [])
            ))
            cls.dataset_new_SCALAR_VARIABLES = list(dict.fromkeys(
                (parsed.get('dataset_new_SCALAR_VARIABLES') or [])
            ))
            cls.dataset_new_2D_VARIABLES = list(dict.fromkeys(
                (parsed.get('dataset_new_2D_VARIABLES') or parsed.get('variables_2d_soil') or [])
            ))
            cls.dataset_new_RESTART_COL_1D_VARS = list(dict.fromkeys(
                (parsed.get('dataset_new_RESTART_COL_1D_VARS') or [])
            ))

            if new_1d_vars:
                cls.POOL_VARS = new_1d_vars

            if new_1d_vars:
                cls.X_LIST_COLUMNS_1D = list(dict.fromkeys(list(cls.X_LIST_COLUMNS_1D) + new_1d_vars))
            if new_2d_vars:
                cls.X_LIST_COLUMNS_2D = list(dict.fromkeys(list(cls.X_LIST_COLUMNS_2D) + new_2d_vars))
            if cls.dataset_new_RESTART_COL_1D_VARS:
                cls.X_LIST_COLUMNS_2D = list(dict.fromkeys(list(cls.X_LIST_COLUMNS_2D) + cls.dataset_new_RESTART_COL_1D_VARS))
            
            if new_water_vars:
                cls.WATER_VARIABLES = new_water_vars
                cls.X_LIST_COLUMNS_2D = list(dict.fromkeys(list(cls.X_LIST_COLUMNS_2D) + new_water_vars))
            
            if new_cols_to_drop:
                cls.COLS_TO_DROP = list(dict.fromkeys(list(cls.COLS_TO_DROP) + new_cols_to_drop))

            cls.Y_LIST_COLUMNS_1D = [f"Y_{name}" for name in cls.X_LIST_COLUMNS_1D]
            cls.Y_LIST_COLUMNS_2D = [f"Y_{name}" for name in cls.X_LIST_COLUMNS_2D]
            cls.Y_WATER_VARIABLES = [f"Y_{name}" for name in cls.WATER_VARIABLES]

            if cls.WATER_VARIABLES:
                keep_set = set(cls.WATER_VARIABLES) | set(cls.Y_WATER_VARIABLES)
                cls.COLS_TO_DROP = [c for c in cls.COLS_TO_DROP if c not in keep_set]

            cls.VARIABLES_1D = cls.X_LIST_COLUMNS_1D.copy()
            cls.VARIABLES_2D = cls.X_LIST_COLUMNS_2D.copy()
            
            if cls.dataset_new_RESTART_COL_1D_VARS:
                restart_vars = cls.dataset_new_RESTART_COL_1D_VARS
                restart_y_vars = [f"Y_{var}" for var in restart_vars]
                cls.VARS_TO_RESHAPE = list(dict.fromkeys(cls.VARS_TO_RESHAPE + restart_vars + restart_y_vars))
        except Exception:
            return

# Apply CNP IO overrides
Config.apply_cnp_io_overrides()

Config.num_all_columns_2D = len(Config.X_LIST_COLUMNS_2D)
Config.num_all_columns_1D = len(Config.X_LIST_COLUMNS_1D)


