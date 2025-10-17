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

# Base output directory
output_dir = './output'

# Processed forcing NetCDF files output directory
forcing_netcdf_output_dir = './output/forcing_netcdf'

# Forcing PKL files output directory
forcing_pkl_output_dir = './output/forcing_hourly_pkl'

# Training dataset PKL files output directory
training_dataset_pkl_output_dir = './output/training_dataset_pkl'

# CLM parameters NetCDF file path
clm_params_nc_path = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/clm_params_c211124.nc'

# =============================================================================
# 37_DATASET CONFIGURATION (Enhanced Dataset Generation)
# =============================================================================

class Config:
    # Input paths for 37_dataset.py
    INPUT_GLOB = "./output/training_dataset_pkl/monthly_training_data_batch_*.pkl"
    OUTPUT_DIR = "./output/enhanced_training_dataset"
    ENHANCED_PREFIX = "enhanced_"
    POOL_VARS = ["cpool", "npool", "ppool", "xsmrpool"]
    
    # Special P input NetCDF file
    SPECIAL_P_INPUT_NC = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_SE_dataset/TVA/history_restart_files/uELM_TVA_finalspinref.elm.r.0781-01-01-00000.nc"
    SPECIAL_P_VARS = []

    # CNP IO configuration file
    CNP_IO_FILE = os.path.join(os.path.dirname(__file__), "python_scripts", "CNP_IO_updated14_xfer.txt")

    # Columns to drop during processing
    COLS_TO_DROP = [
        'H2OSFC', 'H2OSNO', 'H2OSOI_LIQ', 'H2OSOI_ICE', 'LAKE_SOILC', 'H2OCAN',
        'TH2OSFC', 'T_GRND', 'T_GRND_R', 'T_GRND_U', 'T_LAKE', 'T_SOISNO',
        'TS_TOPO', 'taf', 'T_VEG', 'T10_VALUE',
        'Y_H2OSFC', 'Y_H2OSNO', 'Y_H2OSOI_LIQ', 'Y_H2OSOI_ICE', 'Y_LAKE_SOILC', 'Y_H2OCAN',
        'Y_TH2OSFC', 'Y_T_GRND', 'Y_T_GRND_R', 'Y_T_GRND_U', 'Y_T_LAKE', 'Y_T_SOISNO',
        'Y_TS_TOPO', 'Y_taf', 'Y_T_VEG', 'Y_T10_VALUE',
        'annsum_npp', 'avail_retransn', 'avail_retransp', 'cannsum_npp', 
        'Y_annsum_npp', 'Y_avail_retransn', 'Y_avail_retransp', 'Y_cannsum_npp',
        'leafc_xfer', 'frootc_xfer', 'livestemc_xfer', 'deadstemc_xfer', 'livecrootc_xfer', 'deadcrootc_xfer',
        'gresp_xfer', 'leafn_xfer', 'frootn_xfer', 'livestemn_xfer', 'deadstemn_xfer', 'livecrootn_xfer',
        'deadcrootn_xfer', 'leafp_xfer', 'frootp_xfer', 'livestemp_xfer', 'deadstemp_xfer', 'livecrootp_xfer', 'deadcrootp_xfer', 
        'retransn', 'retransp', 'gresp_storage',
        'Y_leafc_xfer', 'Y_frootc_xfer', 'Y_livestemc_xfer', 'Y_deadstemc_xfer', 'Y_livecrootc_xfer', 'Y_deadcrootc_xfer',
        'Y_gresp_xfer', 'Y_leafn_xfer', 'Y_frootn_xfer', 'Y_livestemn_xfer', 'Y_deadstemn_xfer', 'Y_livecrootn_xfer',
        'Y_deadcrootn_xfer', 'Y_leafp_xfer', 'Y_frootp_xfer', 'Y_livestemp_xfer', 'Y_deadstemp_xfer', 'Y_livecrootp_xfer', 'Y_deadcrootp_xfer', 
        'Y_retransn', 'Y_retransp', 'Y_gresp_storage',
        'labilep_vr', 'occlp_vr', 'primp_vr',
        'Y_labilep_vr', 'Y_occlp_vr', 'Y_primp_vr',
        'cpool', 'npool', 'ppool', 'xsmrpool',
        'Y_cpool', 'Y_npool', 'Y_ppool', 'Y_xsmrpool',
        'FH2OSFC',
        'Y_FH2OSFC',
        'secondp_vr',
        'Y_secondp_vr'
    ]
    
    # List columns configuration
    X_LIST_COLUMNS_2D = [
        'soil3c_vr', 'soil4c_vr', 'cwdc_vr', 'cwdn_vr', 'secondp_vr', 'cwdp', 'totcolp', 'totlitc', 'cwdp_vr',
        'soil1c_vr', 'soil1n_vr', 'soil1p_vr',
        'soil2c_vr', 'soil2n_vr', 'soil2p_vr',
        'soil3n_vr', 'soil3p_vr',
        'soil4n_vr', 'soil4p_vr',
        'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
        'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
        'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
        'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
    ]

    X_LIST_COLUMNS_1D = [
        'deadcrootc', 'deadstemc', 'tlai', 'totvegc', 'deadstemn', 'deadcrootn', 'deadstemp', 'deadcrootp',
        'leafc', 'leafc_storage', 'frootc', 'frootc_storage',
        'leafn', 'leafn_storage', 'frootn', 'frootn_storage',
        'leafp', 'leafp_storage', 'frootp', 'frootp_storage',
        'livestemc', 'livestemc_storage', 'livestemn', 'livestemn_storage',
        'livestemp', 'livestemp_storage', 'deadcrootc_storage', 'deadstemc_storage',
        'livecrootc', 'livecrootc_storage', 'deadcrootn_storage', 'deadstemn_storage',
        'livecrootn', 'livecrootn_storage', 'deadcrootp_storage', 'deadstemp_storage',
        'livecrootp', 'livecrootp_storage',
    ]

    Y_LIST_COLUMNS_2D = [f"Y_{name}" for name in X_LIST_COLUMNS_2D]
    Y_LIST_COLUMNS_1D = [f"Y_{name}" for name in X_LIST_COLUMNS_1D]

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


