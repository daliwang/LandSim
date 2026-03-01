"""
Configuration module for model training.

This module centralizes all training parameters and configurations,
making it easy to modify training settings without changing the core code.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import re
import ast


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_paths: List[str] = field(default_factory=lambda: [
        "/global/cfs/cdirs/m4814/daweigao/0_trendy_case/dataset"
    ])
    
    # File patterns
    file_pattern: str = "1_training_data_batch_*.pkl"
    # Optional per-dataset file patterns keyed by absolute path
    dataset_file_patterns: Dict[str, str] = field(default_factory=dict)
    
    # Columns to drop
    columns_to_drop: List[str] = field(default_factory=lambda: [
        'Y_OCCLUDED_P', 'Y_SECONDARY_P', 'Y_LABILE_P', 'Y_APATITE_P', 'H2OSOI_10CM'
    ])
    
    # Time series columns (can be modified for different inputs)
    time_series_columns: List[str] = field(default_factory=lambda: [
        'FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT'
    ])
    
    # Fixed static columns to ensure consistency across datasets
    static_columns: List[str] = field(default_factory=lambda: [
        'lat', 'lon', 'area', 'landfrac'
    ])
    
    # PFT parameter columns
    pft_param_columns: List[str] = field(default_factory=lambda: [
        'pft_deadwdcn', 'pft_frootcn'
    ])

    # Reorganized input groups for enhanced model
    static_surface_columns: List[str] = field(default_factory=list)
    water_group_columns: List[str] = field(default_factory=list)
    temperature_group_columns: List[str] = field(default_factory=list)
    
    # 2D input features (can be modified for different inputs)
    x_list_columns_2d: List[str] = field(default_factory=lambda: [
        'soil3c_vr', 'soil4c_vr', 'cwdc_vr'
    ])
    
    # 2D target features (can be modified for different outputs)
    y_list_columns_2d: List[str] = field(default_factory=lambda: [
        'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr'
    ])
    
    # 1D input features (can be modified for different inputs)
    x_list_columns_1d: List[str] = field(default_factory=lambda: [
        'deadcrootc', 'deadstemc', 'tlai'
    ])
    
    # 1D target features (can be modified for different outputs)
    y_list_columns_1d: List[str] = field(default_factory=lambda: [
        'Y_deadcrootc', 'Y_deadstemc', 'Y_tlai'
    ])
    
    # 1D scalar input features (optional, for separate scalar group)
    x_list_scalar_columns: List[str] = field(default_factory=lambda: [
      'NPP', 'GPP',  'HR', 'AR'
    ])
    # 1D scalar output features (optional, for separate scalar group)
    y_list_scalar_columns: List[str] = field(default_factory=lambda: [
      'Y_NPP', 'Y_GPP', 'Y_HR', 'Y_AR'
    ])  


    # Data preprocessing settings
    time_series_length: int = 240
    max_time_series_length: int = 1476
    max_1d_length: int = 16
    max_2d_rows: int = 2  # Updated: soil2D now has 2 columns instead of 18
    max_2d_cols: int = 15  # Updated: soil2D now has 15 layers instead of 10
    
    # Data splitting
    train_split: float = 0.8
    test_split: Optional[float] = None
    random_state: int = 42
    # Tropical-only filtering (apply before train/test split)
    tropical_only: bool = False
    tropical_lat_range: Tuple[float, float] = (-23.5, 23.5)
    tropical_lat_column: Optional[str] = None
    # Natveg-only filtering: keep only gridcells with PCT_NATVEG > 0 and PCT_NAT_PFT_0 < 100
    natveg_only: bool = False
    # When True: filter to natveg before shuffle/split (legacy). When False: split on full data, then
    # filter only training set to natveg, so test set is the same as no-filter run (natveg test ⊂ no-filter test).
    # Default True preserves legacy behavior when key is missing.
    natveg_filter_before_split: bool = True

    # File loading limits (for testing)
    max_files: Optional[int] = None  # Maximum number of files to load (None = all files)
    
    # New parameter for filtering NaN in time series
    filter_time_series_nan: bool = False
    filter_column: Optional[str] = None # Added for CNP model
    
    # Longitude filtering - list of longitude values to drop from dataset
    longitudes_to_drop: List[float] = field(default_factory=list)




@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Core dimensions (Dual Stream Architecture)
    embed_dim: int = 256
    patch_size: int = 60

    # LSTM parameters (Legacy / Stream 1 variant)
    lstm_hidden_size: int = 64
    
    # Fully connected layers
    fc_hidden_size: int = 32
    static_fc_size: int = 64
    
    # CNN parameters for 2D data
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    conv_kernel_size: int = 3
    conv_padding: int = 1
    
    # Transformer parameters
    num_tokens: int = 4
    token_dim: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    # Global dropout probability (set to 0.0 for bit-for-bit reproducibility)
    dropout_p: float = 0.1
    
    # Output dimensions
    scalar_output_size: int = 5
    vector_output_size: int = 3
    vector_length: int = 16
    matrix_output_size: int = 3
    matrix_rows: int = 1  # Updated: effective soil2D columns after slicing to first column
    matrix_cols: int = 10  # Updated: we only want top 10 layers out of 15

    # FC for PFT parameters (44 variables)
    # CNN for PFT parameters (44 variables)
    pft_param_cnn_channels: List[int] = field(default_factory=lambda: [32, 64])
    pft_param_cnn_kernel_size: int = 3
    pft_param_cnn_padding: int = 1
    # FC for water variables (6 variables)
    water_fc_size: int = 64
    # FC for scalar variables (4 variables)
    scalar_fc_size: int = 64
    # FC for 1D PFT variables (14 variables)
    pft_1d_fc_size: int = 64
    num_pfts: int = 17  # Number of PFTs (default/fallback)
    use_cnn_for_pft_param: bool = False  # Whether to use CNN for PFT parameters
    # PFT1D activation control
    pft1d_activation: str = 'abs'  # 'abs', 'relu', 'softplus', 'linear'
    pft1d_activation_overrides: Dict[str, str] = field(default_factory=dict)
    
    # Multihead/Multimode configuration
    use_multihead_attention: bool = False  # Enable multi-head attention for mode-specific feature extraction
    multihead_num_heads: int = 4  # Number of attention heads for multihead attention
    use_mode_specific_heads: bool = False  # Enable separate feature extractors for each output mode


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training parameters
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.0001
    
    # Loss weights (can be adjusted for different output priorities)
    scalar_loss_weight: float = 1.0
    vector_loss_weight: float = 1.0
    matrix_loss_weight: float = 1.0
    # Specific variable weights
    xsmrpool_loss_weight: float = 10.0
    # Litter pools (2D soil) weights
    litter_c_loss_weight: float = 1.0  # applies to litr1c_vr, litr2c_vr, litr3c_vr
    litter_n_loss_weight: float = 1.0  # applies to litr1n_vr, litr2n_vr, litr3n_vr
    litter_p_loss_weight: float = 1.0  # applies to litr1p_vr, litr2p_vr, litr3p_vr
    
    # Optimizer
    optimizer_type: str = 'adam'  # 'adam', 'sgd', 'adamw'
    weight_decay: float = 0.0
    
    # Learning rate scheduler
    use_scheduler: bool = False
    scheduler_type: str = 'step'  # 'step', 'cosine', 'plateau'
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    
    # Early stopping
    use_early_stopping: bool = False
    patience: int = 10
    min_delta: float = 0.001
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # GPU Optimization
    use_mixed_precision: bool = True  # Use mixed precision training for faster training
    use_amp: bool = True  # Use Automatic Mixed Precision
    use_grad_scaler: bool = True  # Use gradient scaler for mixed precision
    pin_memory: bool = True  # Pin memory for faster data transfer to GPU
    num_workers: int = 0  # Number of data loading workers (0 to avoid GPU context issues)
    prefetch_factor: int = 2  # Number of batches to prefetch
    persistent_workers: bool = False  # Keep workers alive between epochs
    
    # GPU Memory Optimization
    empty_cache_freq: int = 10  # Empty GPU cache every N batches
    max_memory_usage: float = 0.9  # Maximum GPU memory usage (0.9 = 90%)
    memory_efficient_attention: bool = True  # Use memory efficient attention if available
    
    # GPU Monitoring
    log_gpu_memory: bool = False  # Log GPU memory usage
    log_gpu_utilization: bool = False  # Log GPU utilization
    gpu_monitor_interval: int = 100  # Log GPU stats every N batches
    
    # Logging and saving
    save_model: bool = True
    model_save_path: str = "LSTM_model.pt"
    save_losses: bool = True
    losses_save_path: str = "training_validation_losses.csv"
    save_predictions: bool = True
    predictions_dir: str = "predictions"
    
    # Validation
    validation_frequency: int = 1  # Validate every N epochs
    
    # Fair comparison settings
    random_seed: int = 42  # Fixed random seed
    deterministic: bool = False  # Relax strict determinism by default

    # Learnable loss weighting for CNP model
    use_learnable_loss_weights: bool = False

    # PFT sparsity regularization (encourage zero predictions where targets are zero)
    pft_zero_sparsity_weight: float = 0.0  # default disabled; set >0 to enable
    pft_zero_threshold: float = 1e-8       # threshold in normalized target space for zero mask
    pft_zero_sparsity_weights: Dict[str, float] = field(default_factory=dict)
    # Tail-aware loss for heavy-tailed PFT1D variables
    tail_aware_vars: List[str] = field(default_factory=list)
    tail_aware_loss: str = 'log1p_mse'  # 'log1p_mse', 'log1p_huber', 'log1p_quantile', or 'mse'
    tail_aware_epsilon: float = 1e-8
    tail_aware_weight: float = 1.0
    tail_aware_weights: Dict[str, float] = field(default_factory=dict)
    tail_aware_huber_delta: float = 1.0
    tail_aware_quantile_tau: float = 0.9

    # Mask predictions for absent PFTs using PCT_NAT_PFT (PFT0 ignored)
    mask_absent_pfts: bool = False
    # Min PFT percent for training-only mask (0 = pct>0; e.g. 2.0 = pct>=2%). Inference always uses pct>0.
    pft_presence_threshold: float = 0.0
    
    # Variable-specific loss weights from JSON file
    variable_weights_json: Optional[str] = None  # Path to JSON file with pft1d_weights, soil2d_weights, scalar_weights

    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.device)
    
    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get the optimizer based on configuration."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def get_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get the learning rate scheduler based on configuration."""
        if not self.use_scheduler:
            return None
        
        if self.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
            )
        elif self.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        elif self.scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.scheduler_gamma, patience=self.scheduler_step_size
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Normalization methods
    time_series_normalization: str = 'minmax'  # 'minmax', 'standard', 'robust'
    static_normalization: str = 'minmax'
    scalar_normalization: str = 'minmax'
    target_normalization: str = 'minmax'
    list_1d_normalization: str = 'minmax'
    list_2d_normalization: str = 'minmax'
    
    # Normalization ranges
    minmax_range: tuple = (0, 1)
    
    # Data type
    data_type: torch.dtype = torch.float32
    
    # Memory management
    memory_save_threshold: int = 50  # Save to disk every N variables

    # PFT presence mask: min percent for "present" during training only (0 = pct>0; e.g. 2.0 = pct>=2%).
    # Used only when normalize_data_individual(transform_only=False). Inference uses pct>0.
    pft_presence_threshold: float = 0.0

class TrainingConfigManager:
    def __init__(self):
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.preprocessing_config = PreprocessingConfig()
    
    def update_data_config(self, **kwargs):
        """Update data configuration."""
        for key, value in kwargs.items():
            if hasattr(self.data_config, key):
                setattr(self.data_config, key, value)
            else:
                raise ValueError(f"Unknown data config parameter: {key}")
    
    def update_model_config(self, **kwargs):
        """Update model configuration."""
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
            else:
                raise ValueError(f"Unknown model config parameter: {key}")
    
    def update_training_config(self, **kwargs):
        """Update training configuration."""
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
            else:
                raise ValueError(f"Unknown training config parameter: {key}")
        # Sync training-only PFT mask threshold to preprocessing (for data loader)
        if 'pft_presence_threshold' in kwargs:
            self.preprocessing_config.pft_presence_threshold = self.training_config.pft_presence_threshold
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary."""
        return {
            'data_config': self.data_config,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'preprocessing_config': self.preprocessing_config
        }

# Only keep get_default_config, get_cnp_model_config, get_cnp_combined_config, and config classes for release 

def get_default_config() -> TrainingConfigManager:
    config = TrainingConfigManager()
    # Minimal/simple config for quick start
    config.data_config.data_paths = [os.path.join(os.path.dirname(__file__), '../data/example_dataset')]
    config.data_config.file_pattern = "mini_data_*.pkl"
    config.data_config.max_files = 3
    config.data_config.train_split = 0.8
    config.data_config.time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
    config.data_config.static_columns = ['Latitude', 'Longitude']
    config.data_config.pft_param_columns = ['pft_leafcn']
    pft_1d_list = ['deadcrootc']
    config.data_config.x_list_columns_1d = pft_1d_list
    config.data_config.y_list_columns_1d = [f'Y_{col}' for col in pft_1d_list]
    # Add minimal scalar variables for demonstration
    config.data_config.x_list_scalar_columns = ['GPP', 'NPP']
    config.data_config.y_list_scalar_columns = ['Y_GPP', 'Y_NPP']
    config.data_config.x_list_columns_2d = ['soil1c_vr']
    config.data_config.y_list_columns_2d = ['Y_soil1c_vr']
    # No filtering for quick start
    config.data_config.filter_column = None
    
    # Update model config to match the reduced variable counts
    config.model_config.scalar_output_size = len(config.data_config.y_list_scalar_columns)  # 2 outputs
    config.model_config.matrix_output_size = len(config.data_config.y_list_columns_2d)     # 1 output
    config.model_config.vector_output_size = len(config.data_config.y_list_columns_1d)    # 1 output
    config.model_config.pft_param_size = 1
    config.model_config.num_pfts = 17
    config.model_config.use_cnn_for_pft_param = False  # Use FC for mini/simple model
    
    # Update training config for simpler, more reliable training
    config.training_config.use_mixed_precision = False  # Disable for simplicity
    config.training_config.use_amp = False              # Disable for simplicity
    config.training_config.use_grad_scaler = False      # Disable for simplicity
    config.training_config.num_epochs = 10              # Fewer epochs for quick demo
    config.training_config.batch_size = 32              # Reasonable batch size
    
    return config


def get_cnp_model_config(include_water: bool = False, max_files: Optional[int] = None, use_trendy1: Optional[bool] = None, use_trendy05: Optional[bool] = None) -> TrainingConfigManager:
    """
    Get CNP model configuration for Trendy_1_data_CNP and/or Trendy_05_data_CNP, optionally including water variables.
    Args:
        include_water: Whether to include water variables
        max_files: Maximum number of files to use from the dataset
        use_trendy1: Whether to include Trendy_1_data_CNP
        use_trendy05: Whether to include Trendy_05_data_CNP
    Returns:
        TrainingConfigManager with configuration for selected datasets
    """
    # Default logic: if neither is specified, use Trendy_1 only (for backward compatibility)
    if use_trendy1 is None and use_trendy05 is None:
        use_trendy1_val = True
        use_trendy05_val = False
    else:
        use_trendy1_val = bool(use_trendy1)
        use_trendy05_val = bool(use_trendy05)
    config = get_cnp_combined_config(use_trendy1=use_trendy1_val, use_trendy05=use_trendy05_val, max_files=max_files, include_water=include_water)
    return config


def get_cnp_model_config_with_water() -> TrainingConfigManager:
    """
    Get CNP model configuration without water variables.
    
    Returns:
        TrainingConfigManager with CNP model configuration (no water)
    """
    return get_cnp_model_config(include_water=True) 

def parse_cnp_io_list(filename):
    """
    Parse a CNP IO variable list file into a dictionary of variable groups.
    Args:
        filename (str): Path to the variable list file (e.g., CNP_IO_list_general.txt)
    Returns:
        dict: Mapping of variable group keys to lists of variable names.
              Also includes 'longitudes_to_drop' key if specified in the file.
    """
    # Map section titles to config keys
    section_map = {
        'LONGITUDE FILTERING': 'longitudes_to_drop',
        'TIME SERIES VARIABLES': 'time_series_variables',
        'SURFACE PROPERTIES': 'surface_properties',
        'PFT PARAMETERS': 'pft_parameters',
        'WATER VARIABLES': 'water_variables',
        'TEMPERATURE VARIABLES': 'temperature_variables',
        'SCALAR VARIABLES': 'scalar_variables',
        '1D PFT VARIABLES': 'pft_1d_variables',
        '2D VARIABLES': 'variables_2d_soil',
        'DATA PATHS': 'data_paths'
    }
    # Prepare result dict
    result = {v: [] for v in section_map.values()}
    # Additional single-value keys for dataset configuration
    result.update({
        'trendy1_path': None,
        'trendy05_path': None,
        'tva4km_path': None,
        'file_pattern': None,
        'trendy1_file_pattern': None,
        'trendy05_file_pattern': None,
        'tva4km_file_pattern': None,
        'max_files': None,
        'ai_predictions_default': None,
        'model_default': None,
        'comparison_output_dir': None,
        'csv_predictions_default': None,
        'ai_restart_default': None,
        'fallback_data_dir': None,
        'fallback_reference_file': None,
        'fallback_reference_filename': None
    })
    current_section = None

    with open(filename) as f:
        for line in f:
            line = line.strip()
            # Skip full-line comments and blanks (support //, #, ;)
            if not line or line.startswith('#') or line.startswith('//') or line.startswith(';'):
                continue
            # Remove inline comments introduced by '#'; avoid '//' inline to not break paths
            hash_idx = line.find('#')
            if hash_idx != -1:
                line = line[:hash_idx].strip()
                if not line:
                    continue
            
            # Section header detection
            for section_title, key in section_map.items():
                if line.startswith(section_title):
                    current_section = key
                    break
            else:
                # If line is a variable line (starts with • or comma-separated list)
                if current_section and line.startswith('•'):
                    # Remove bullet and split by comma, filter out empty strings
                    vars_ = [v.strip() for v in line[1:].split(',') if v.strip()]
                    
                    # Special handling for longitude filtering - convert to floats
                    if current_section == 'longitudes_to_drop':
                        try:
                            longitudes = [float(x) for x in vars_]
                            result[current_section].extend(longitudes)
                            logging.info(f"Parsed longitudes to drop: {longitudes}")
                        except Exception as e:
                            logging.warning(f"Failed to parse longitudes to drop: {e}")
                    else:
                        result[current_section].extend(vars_)
                # Some variables are listed as comma-separated after a bullet
                elif current_section and ',' in line and not line.startswith('['):
                    vars_ = [v.strip('• ').strip() for v in line.split(',') if v.strip('• ').strip()]
                    result[current_section].extend(vars_)
                # Some variables are listed as single words (rare, but just in case)
                elif current_section and line and not line.startswith('[') and not line.startswith('#'):
                    # Only add if it's not a description or exclusion
                    if re.match(r'^[A-Za-z0-9_]+$', line):
                        result[current_section].append(line)
                # Outside of a section or in any section, allow key=value dataset config
                # e.g., TRENDY1_PATH: /path/to/trendy1
                #       TRENDY05_PATH = /path/to/trendy05
                #       FILE_PATTERN: enhanced_1_training_data_batch_*.pkl
                #       DATA_PATHS: /p1,/p2
                #       MAX_FILES: 3
                if line and not line.startswith('#'):
                    kv_match = re.match(r'(?i)^(trendy1_path|trendy05_path|tva4km_path|file_pattern|trendy1_file_pattern|trendy05_file_pattern|tva4km_file_pattern|data_paths|max_files|ai_predictions_default|model_default|comparison_output_dir|csv_predictions_default|ai_restart_default|fallback_data_dir|fallback_reference_file|fallback_reference_filename)\s*[:=]\s*(.+)$', line)
                    if kv_match:
                        key = kv_match.group(1).lower()
                        val = kv_match.group(2).strip()
                        if key == 'data_paths':
                            # Support comma-separated list
                            paths = [p.strip() for p in val.split(',') if p.strip()]
                            result['data_paths'].extend(paths)
                        elif key == 'file_pattern':
                            result['file_pattern'] = val
                        elif key == 'max_files':
                            try:
                                result['max_files'] = int(val)
                            except ValueError:
                                logging.warning(f"Invalid MAX_FILES value '{val}', expected integer. Ignoring.")
                        elif key == 'trendy1_path':
                            result['trendy1_path'] = val
                        elif key == 'trendy05_path':
                            result['trendy05_path'] = val
                        elif key == 'tva4km_path':
                            result['tva4km_path'] = val
                        elif key == 'trendy1_file_pattern':
                            result['trendy1_file_pattern'] = val
                        elif key == 'trendy05_file_pattern':
                            result['trendy05_file_pattern'] = val
                        elif key == 'tva4km_file_pattern':
                            result['tva4km_file_pattern'] = val
                        elif key == 'ai_predictions_default':
                            result['ai_predictions_default'] = val
                        elif key == 'model_default':
                            result['model_default'] = val
                        elif key == 'comparison_output_dir':
                            result['comparison_output_dir'] = val
                        elif key == 'csv_predictions_default':
                            result['csv_predictions_default'] = val
                        elif key == 'ai_restart_default':
                            result['ai_restart_default'] = val
                        elif key == 'fallback_data_dir':
                            result['fallback_data_dir'] = val
                        elif key == 'fallback_reference_file':
                            result['fallback_reference_file'] = val
                        elif key == 'fallback_reference_filename':
                            result['fallback_reference_filename'] = val
    return result

def parse_cnp_model_config(filename: str) -> Dict[str, Any]:
    """
    Parse a simple text-based model configuration file into a dictionary of
    ModelConfig overrides. The format supports lines of the form:

      key = value

    Where value can be:
      - int/float (e.g., 64, 0.1)
      - bool (true/false/yes/no/on/off)
      - comma-separated list (e.g., 16,32,64)
      - Python-like list (e.g., [16, 32, 64])
      - quoted string ("text" or 'text')

    Blank lines and lines starting with # or ; are ignored. Section headers
    like [ENCODERS] are allowed and ignored.

    Returns a dict of parsed keys to values. Unknown keys will be filtered out
    by the caller when applying to ModelConfig.
    """
    overrides: Dict[str, Any] = {}
    if filename is None:
        return overrides
    try:
        with open(filename, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#') or line.startswith(';'):
                    continue
                # Ignore section headers like [SECTION]
                if line.startswith('[') and line.endswith(']'):
                    continue
                # Support key: value or key = value
                if ':' in line or '=' in line:
                    if ':' in line and '=' in line:
                        # take first delimiter appearance
                        idx_colon = line.find(':')
                        idx_eq = line.find('=')
                        idx = min(idx_colon, idx_eq)
                        key = line[:idx].strip()
                        value_str = line[idx+1:].strip()
                    else:
                        parts = re.split(r'[:=]', line, maxsplit=1)
                        if len(parts) != 2:
                            continue
                        key, value_str = parts[0].strip(), parts[1].strip()
                    if not key:
                        continue
                    # Normalize booleans
                    low = value_str.lower()
                    if low in ('true', 'yes', 'on'):
                        overrides[key] = True
                        continue
                    if low in ('false', 'no', 'off'):
                        overrides[key] = False
                        continue
                    # Try Python literal (list, int, float, string)
                    parsed_val: Any = None
                    try:
                        parsed_val = ast.literal_eval(value_str)
                    except Exception:
                        # Fallback: comma-separated list without brackets
                        if ',' in value_str:
                            items = [v.strip() for v in value_str.split(',') if v.strip()]
                            # Try to cast each item to int/float where possible
                            cast_items: List[Any] = []
                            for it in items:
                                try:
                                    cast_items.append(int(it))
                                    continue
                                except Exception:
                                    pass
                                try:
                                    cast_items.append(float(it))
                                    continue
                                except Exception:
                                    pass
                                cast_items.append(it)
                            parsed_val = cast_items
                        else:
                            # Try number cast
                            try:
                                parsed_val = int(value_str)
                            except Exception:
                                try:
                                    parsed_val = float(value_str)
                                except Exception:
                                    # Strip quotes if present
                                    parsed_val = value_str.strip('"\'')
                    overrides[key] = parsed_val
    except FileNotFoundError:
        logging.warning(f"Model config file not found: {filename}. Using defaults.")
    except Exception as e:
        logging.warning(f"Failed to parse model config file {filename}: {e}")
    return overrides

def get_cnp_combined_config(
    use_trendy1: bool = True,
    use_trendy05: bool = True,
    use_tva4km: bool = False,
    max_files: Optional[int] = None,
    include_water: bool = False,
    variable_list_path: Optional[str] = None,
    model_config_path: Optional[str] = None
) -> TrainingConfigManager:
    """
    Get CNP model configuration for Trendy_1_data_CNP, Trendy_05_data_CNP, or both.
    Args:
        use_trendy1: Whether to include Trendy_1_data_CNP
        use_trendy05: Whether to include Trendy_05_data_CNP
        max_files: Maximum number of files to use from each dataset
        variable_list_path: Path to a custom variable list file (e.g., CNP_IO_4Plist.txt)
    Returns:
        TrainingConfigManager with combined configuration
    """
    config = TrainingConfigManager()
    data_paths = []
    file_pattern = None
    dataset_file_patterns: Dict[str, str] = {}
    # If a variable list is provided, prefer dataset paths from it
    parsed = None
    if variable_list_path is not None:
        try:
            parsed = parse_cnp_io_list(variable_list_path)
        except Exception as e:
            logging.warning(f"Failed to parse variable list for data paths: {e}")
    if parsed is not None:
        # Collect from any or all of: data_paths, trendy1_path, trendy05_path, tva4km_path
        if parsed.get('data_paths'):
            data_paths.extend([p for p in parsed['data_paths'] if p])
        if parsed.get('trendy1_path') and use_trendy1:
            p = parsed['trendy1_path']
            data_paths.append(p)
            if parsed.get('trendy1_file_pattern'):
                dataset_file_patterns[p] = parsed['trendy1_file_pattern']
        if parsed.get('trendy05_path') and use_trendy05:
            p = parsed['trendy05_path']
            data_paths.append(p)
            if parsed.get('trendy05_file_pattern'):
                dataset_file_patterns[p] = parsed['trendy05_file_pattern']
        if parsed.get('tva4km_path') and use_tva4km:
            p = parsed['tva4km_path']
            data_paths.append(p)
            if parsed.get('tva4km_file_pattern'):
                dataset_file_patterns[p] = parsed['tva4km_file_pattern']
        if parsed.get('file_pattern'):
            file_pattern = parsed['file_pattern']
        # Apply max_files from CNP_IO if specified (CLI max_files parameter takes precedence)
        if parsed.get('max_files') is not None and max_files is None:
            max_files = parsed['max_files']
            logging.info(f"Using MAX_FILES from CNP_IO file: {max_files}")
    # Fallback to defaults if none provided via CNP_IO
    if not data_paths:
        if use_trendy1:
            data_paths.append("/global/cfs/cdirs/m4814/daweigao/14_Code/all_dataset_1_degree")
        if use_trendy05:
            data_paths.append("/mnt/proj-shared/AI4BGC_7xw/TrainingData/Trendy_05_data_CNP")
        if use_tva4km:
            # Prefer environment variable if provided
            env_tva = os.environ.get('TVA4KM_PATH')
            if env_tva:
                data_paths.append(env_tva)
                env_pat = os.environ.get('TVA4KM_FILE_PATTERN')
                if env_pat:
                    dataset_file_patterns[env_tva] = env_pat
    if file_pattern is None:
        file_pattern = "enhanced_1_training_data_batch_*.pkl"

    config.update_data_config(
        data_paths=data_paths,
        file_pattern=file_pattern,
        dataset_file_patterns=dataset_file_patterns,
        max_files=max_files,
        train_split=0.8,
        filter_column=None,
        time_series_length=240,
        max_time_series_length=240,
        max_1d_length=16,
        max_2d_rows=2, # Updated: soil2D now has 2 columns instead of 18
        max_2d_cols=15, # Updated: soil2D now has 15 layers instead of 10
    )

    # Defaults
    default_time_series = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
    default_surface = [
        'Latitude', 'Longitude', 'AREA', 'landfrac', 'LANDFRAC_PFT', 'PCT_NATVEG',
        'OCCLUDED_P', 'SECONDARY_P', 'LABILE_P', 'APATITE_P',
        'SOIL_COLOR', 'SOIL_ORDER',
        'PCT_NAT_PFT_0', 'PCT_NAT_PFT_1', 'PCT_NAT_PFT_2', 'PCT_NAT_PFT_3', 'PCT_NAT_PFT_4', 
        'PCT_NAT_PFT_5', 'PCT_NAT_PFT_6', 'PCT_NAT_PFT_7', 'PCT_NAT_PFT_8', 'PCT_NAT_PFT_9', 
        'PCT_NAT_PFT_10', 'PCT_NAT_PFT_11', 'PCT_NAT_PFT_12', 'PCT_NAT_PFT_13', 'PCT_NAT_PFT_14', 
        'PCT_NAT_PFT_15', 'PCT_NAT_PFT_16',
        'PCT_CLAY_0', 'PCT_CLAY_1', 'PCT_CLAY_2', 'PCT_CLAY_3', 'PCT_CLAY_4', 
        'PCT_CLAY_5', 'PCT_CLAY_6', 'PCT_CLAY_7', 'PCT_CLAY_8', 'PCT_CLAY_9',
        'PCT_SAND_0', 'PCT_SAND_1', 'PCT_SAND_2', 'PCT_SAND_3', 'PCT_SAND_4', 
        'PCT_SAND_5', 'PCT_SAND_6', 'PCT_SAND_7', 'PCT_SAND_8', 'PCT_SAND_9'
    ]

    default_pft_parameters = [
        'pft_deadwdcn', 'pft_frootcn', 'pft_leafcn', 'pft_lflitcn', 'pft_livewdcn', 'pft_c3psn', 'pft_croot_stem', 'pft_crop', 'pft_dleaf', 'pft_dsladlai', 'pft_evergreen', 'pft_fcur', 'pft_flivewd', 'pft_flnr', 'pft_fr_fcel', 'pft_fr_flab', 'pft_fr_flig', 'pft_froot_leaf', 'pft_grperc', 'pft_grpnow', 'pft_leaf_long', 'pft_lf_fcel', 'pft_lf_flab', 'pft_lf_flig', 'pft_rholnir', 'pft_rholvis', 'pft_rhosnir', 'pft_rhosvis', 'pft_roota_par', 'pft_rootb_par', 'pft_rootprof_beta', 'pft_season_decid', 'pft_slatop', 'pft_smpsc', 'pft_smpso', 'pft_stem_leaf', 'pft_stress_decid', 'pft_taulnir', 'pft_taulvis', 'pft_tausnir', 'pft_tausvis', 'pft_woody', 'pft_xl', 'pft_z0mr'
    ]
    default_water = ['H2OCAN', 'H2OSFC', 'H2OSNO', 'TH2OSFC', 'H2OSOI_LIQ', 'H2OSOI_ICE']
    default_scalar = ['GPP', 'NPP', 'AR', 'HR']
    default_pft_1d = [
        'deadcrootc', 'deadcrootn', 'deadcrootp', 'deadstemc', 'deadstemn', 'deadstemp',
        'frootc', 'frootc_storage', 'leafc', 'leafc_storage', 'totvegc', 'tlai'
    ]
    default_2d_soil = [
        'cwdc_vr', 'cwdn_vr', 'cwdp_vr',
        'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
        'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
        'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
        'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
        'soil1c_vr', 'soil1n_vr', 'soil1p_vr', 
        'soil2c_vr', 'soil2n_vr', 'soil2p_vr', 
        'soil3c_vr', 'soil3n_vr', 'soil3p_vr', 
        'soil4c_vr', 'soil4n_vr', 'soil4p_vr'
    #        'secondp_vr' # this is not in the list, but it is in the data  
    ]

    # If a variable list file is provided, use it for variable groups; otherwise use defaults
    longitudes_to_drop = []
    if variable_list_path is not None:
        if parsed is None:
            try:
                parsed = parse_cnp_io_list(variable_list_path)
            except Exception as _e:
                logging.warning(f"Failed to parse variable list at {variable_list_path}: {_e}")
                parsed = None
        if parsed is not None:
            time_series_columns = parsed.get('time_series_variables', default_time_series)
            surface_properties = parsed.get('surface_properties', default_surface)
            pft_parameters = parsed.get('pft_parameters', default_pft_parameters)
            water_variables = parsed.get('water_variables', default_water)
            scalar_variables = parsed.get('scalar_variables', default_scalar)
            pft_1d_variables = parsed.get('pft_1d_variables', default_pft_1d)
            variables_2d_soil = parsed.get('variables_2d_soil', default_2d_soil)
            longitudes_to_drop = parsed.get('longitudes_to_drop', [])
            try:
                logging.info(f"Applied variable groups from {variable_list_path}: "
                             f"ts={len(time_series_columns)}, static={len(surface_properties)}, pft_params={len(pft_parameters)}, "
                             f"scalar={len(scalar_variables)}, pft1d={len(pft_1d_variables)}, soil2d={len(variables_2d_soil)}")
            except Exception:
                pass
        else:
            time_series_columns = default_time_series
            surface_properties = default_surface
            pft_parameters = default_pft_parameters
            water_variables = default_water
            scalar_variables = default_scalar
            pft_1d_variables = default_pft_1d
            variables_2d_soil = default_2d_soil
    else:
        time_series_columns = default_time_series
        surface_properties = default_surface
        pft_parameters = default_pft_parameters
        water_variables = default_water
        scalar_variables = default_scalar
        pft_1d_variables = default_pft_1d
        variables_2d_soil = default_2d_soil

    data_config_kwargs = dict(
        time_series_columns=time_series_columns,
        static_columns=surface_properties,
        pft_param_columns=pft_parameters,
        x_list_scalar_columns=scalar_variables,
        x_list_columns_1d=pft_1d_variables,
        x_list_columns_2d=variables_2d_soil,
        longitudes_to_drop=longitudes_to_drop
    )
    if include_water:
        data_config_kwargs['x_list_water_columns'] = water_variables

    config.update_data_config(**data_config_kwargs)

    # Outputs
    output_scalar = ['Y_' + v for v in scalar_variables]
    output_1d_pft = ['Y_' + v for v in pft_1d_variables]
    output_2d = ['Y_' + v for v in variables_2d_soil]

    config.update_data_config(
        y_list_scalar_columns=output_scalar,
        y_list_columns_1d=output_1d_pft,
        y_list_columns_2d=output_2d
    )
    
    # Model configuration for CNP architecture (defaults)
    config.update_model_config(
        # LSTM for time series (6 variables, 20 years)
        lstm_hidden_size=64,  # Reduced from 128
        
        # FC for surface properties (31 variables - soil variables moved to 2D)
        static_fc_size=64,  # Reduced from 128
        
        # FC for PFT parameters (44 variables) - separate from surface properties
        # CNN for PFT parameters (44 variables)
        pft_param_cnn_channels=[32, 64],  # Reduced from [32, 64, 128]
        pft_param_cnn_kernel_size=3,
        pft_param_cnn_padding=1,

        # FC for water variables (6 variables) - separate from surface properties
        water_fc_size=64 if include_water else 0,  # Reduced from 128

        # FC for scalar variables (4 variables) - separate from surface properties
        scalar_fc_size=64,  # Reduced from 128

        # FC for 1D PFT variables (14 variables) - separate from surface properties
        pft_1d_fc_size=64,  # Reduced from 128
        num_pfts=17,

        # CNN for 2D soil variables (28 input variables, 28 output variables)
        conv_channels=[16, 32, 64],  # Reduced from [32, 64, 128, 256]
        conv_kernel_size=3,
        conv_padding=1,
        
        # Transformer parameters
        num_tokens=8,  # Reduced from 8
        token_dim=128,  # Reduced from 128
        transformer_layers=4,  # Reduced from 4
        transformer_heads=8,  # Reduced from 8
        
        # Output dimensions - align soil2D head to selected vars and 1x10 slice
        scalar_output_size=4,
        vector_output_size=14,
        vector_length=16,
        matrix_output_size=len(output_2d),
        matrix_rows=1,   # first column only
        matrix_cols=10,  # top 10 layers
        
        # Temperature output excluded for first experiments
    )
    config.model_config.pft_param_size = len(pft_parameters)  # which is 44
    config.model_config.use_cnn_for_pft_param = True  # Use CNN for CNPCombinedModel

    # Optional: override model architecture from a text config file
    if model_config_path is not None:
        try:
            overrides = parse_cnp_model_config(model_config_path)
            if overrides:
                # Only apply keys that exist on ModelConfig
                applicable = {}
                for k, v in overrides.items():
                    if hasattr(config.model_config, k):
                        applicable[k] = v
                    else:
                        logging.info(f"Ignoring unknown ModelConfig key in {model_config_path}: {k}")
                if applicable:
                    config.update_model_config(**applicable)
                    logging.info(f"Applied {len(applicable)} ModelConfig overrides from {model_config_path}")
                    # Record metadata for downstream verification
                    try:
                        setattr(config, 'model_config_overrides_keys', sorted(list(applicable.keys())))
                        setattr(config, 'model_config_source', model_config_path)
                    except Exception:
                        pass
        except Exception as e:
            logging.warning(f"Could not apply model config overrides from {model_config_path}: {e}")
    
    # Training configuration
    config.update_training_config(
        num_epochs=10,  # Reduced for testing with single file
        batch_size=128,  # Further reduced batch size for GPU memory
        learning_rate=0.0001,
        
        # Loss weights for different output types
        scalar_loss_weight=1.0,
        matrix_loss_weight=1.0,
        
        # Optimizer
        optimizer_type='adamw',
        weight_decay=0.01,
        
        # Learning rate scheduler
        use_scheduler=True,
        scheduler_type='cosine',
        
        # Early stopping
        use_early_stopping=False,
        patience=3,  # Reduced for testing
        min_delta=0.001,
        
        # Device and optimization
        device='cpu',  # Force CPU for testing to avoid GPU memory issues
        use_mixed_precision=False,  # Disable mixed precision for CPU
        use_amp=False,  # Disable AMP for CPU
        use_grad_scaler=False,  # Disable grad scaler for CPU
        
        # GPU Memory Optimization
        empty_cache_freq=5,  # Empty GPU cache more frequently
        max_memory_usage=0.7,  # Use less GPU memory (70% instead of 90%)
        memory_efficient_attention=False,
        
        # DataLoader settings for CPU
        num_workers=0,  # No multiprocessing for CPU
        prefetch_factor=None,  # No prefetching for CPU
        persistent_workers=False,  # No persistent workers for CPU
        pin_memory=False,  # Disable pin_memory for CPU
        
        # Logging
        save_model=True,
        model_save_path="cnp_model_test.pt",  # Different name for test
        save_losses=True,
        losses_save_path="cnp_training_losses_test.csv",  # Different name for test
        save_predictions=True,
        predictions_dir="cnp_predictions_test"  # Different name for test
    )
    
    # Set filter_column for CNP model config
    config.data_config.filter_column = 'H2OSOI_10CM'

    # Set up other config fields as needed for CNP model
    return config 