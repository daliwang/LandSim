#!/usr/bin/env python3
"""
CNP Model Inference Script - FIXED VERSION

This script runs inference with the trained CNP model over the entire dataset.
It has been fixed to automatically load the exact training configuration to avoid
model size mismatches that commonly occur when the inference configuration differs
from the training configuration.

FIXED ISSUES:
- Model size mismatches due to different variable lists between training and inference
- Automatically loads training configuration from cnp_config.json
- Ensures exact same data dimensions and model architecture as training
- Provides CLI flags to control the fix behavior

Usage:
    python run_inference_all.py --model path/to/model.pth [options]
    
Key flags:
    --use-training-config: Use exact training configuration (default: True)
    --strict-loading: Use strict model loading to catch mismatches early (default: True)
    --variable-list: Path to variable list file (optional, will auto-detect from config)
"""

import argparse
import logging
import sys
from pathlib import Path
import shutil
import torch
import numpy as np
import os
import pandas as pd

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.trainer import ModelTrainer
from models.cnp_combined_model import CNPCombinedModel

# Individual normalization support (lazy import when needed)
try:
    from data.data_loader_individual import DataLoaderIndividual
except Exception:
    DataLoaderIndividual = None

from data.data_loader_pandas import PandasDataLoader
from config.training_config import get_cnp_model_config, parse_cnp_io_list, get_cnp_combined_config

def _load_training_variables_from_config(model_path: Path) -> dict:
    """Load variable lists from cnp_config.json in the training run directory."""
    try:
        # Look for cnp_config.json in the model directory
        model_dir = Path(model_path).parent
        config_path = model_dir / 'cnp_config.json'
        
        if config_path.exists():
            logging.info(f"Found cnp_config.json at {config_path}")
            return _extract_variables_from_config(config_path)
        
        # Also check parent directories (for cases where model is in subdirectory)
        for parent in model_dir.parents:
            config_path = parent / 'cnp_config.json'
            if config_path.exists():
                logging.info(f"Found cnp_config.json at {config_path}")
                return _extract_variables_from_config(config_path)
        
        logging.warning("No cnp_config.json found in training run directory")
        return None
        
    except Exception as e:
        logging.warning(f"Error discovering config: {e}")
        return None

def _extract_variables_from_config(config_path: Path) -> dict:
    """Extract variable lists from cnp_config.json and return in parse_cnp_io_list format."""
    import json
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'data_info' not in config:
            logging.warning("No data_info found in training configuration")
            return None
            
        data_info = config['data_info']
        
        # Convert from training config format to parse_cnp_io_list format
        variables = {
            'time_series_variables': data_info.get('time_series_columns', []),
            'surface_properties': data_info.get('static_columns', []),
            'pft_parameters': data_info.get('pft_param_columns', []),
            'scalar_variables': data_info.get('x_list_scalar_columns', []),
            'pft_1d_variables': data_info.get('variables_1d_pft', []),
            'variables_2d_soil': data_info.get('x_list_columns_2d', [])
        }
        
        logging.info(f"Extracted variables from training config:")
        logging.info(f"  Time series: {len(variables['time_series_variables'])} variables")
        logging.info(f"  Surface properties: {len(variables['surface_properties'])} variables")
        logging.info(f"  PFT parameters: {len(variables['pft_parameters'])} variables")
        logging.info(f"  Scalar variables: {len(variables['scalar_variables'])} variables")
        logging.info(f"  PFT 1D variables: {len(variables['pft_1d_variables'])} variables")
        logging.info(f"  2D soil variables: {len(variables['variables_2d_soil'])} variables")
        
        return variables
        
    except Exception as e:
        logging.warning(f"Could not extract variables from config: {e}")
        return None


# New: load model_config from cnp_config.json for exact architecture reuse
def _load_training_model_config_from_config(model_path: Path) -> dict:
    """Load model_config dict from cnp_config.json in the training run directory."""
    import json
    try:
        model_dir = Path(model_path).parent
        # Search current and parent directories for cnp_config.json
        candidate_paths = [model_dir / 'cnp_config.json'] + [p / 'cnp_config.json' for p in model_dir.parents]
        for config_path in candidate_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        cfg = json.load(f)
                    if isinstance(cfg, dict) and 'model_config' in cfg and isinstance(cfg['model_config'], dict):
                        logging.info(f"Loaded model_config from {config_path}")
                        return cfg['model_config']
                except Exception as e:
                    logging.warning(f"Failed reading model_config from {config_path}: {e}")
                break
        logging.warning("No model_config found in cnp_config.json near model path")
    except Exception as e:
        logging.warning(f"Error discovering model_config: {e}")
    return None



# Load training model architecture from cnp_config.json
def _load_training_model_config(model_path: Path) -> dict:
    """Load model_config (architecture) from cnp_config.json in the training run directory."""
    import json
    try:
        model_dir = Path(model_path).parent
        # Search model dir then parents
        search_dirs = [model_dir] + list(model_dir.parents)
        for d in search_dirs:
            config_path = d / 'cnp_config.json'
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        cfg = json.load(f)
                    mc = cfg.get('model_config')
                    if isinstance(mc, dict) and mc:
                        logging.info(f"Loaded training model_config from {config_path}")
                        return mc
                    else:
                        logging.warning(f"model_config missing in {config_path}")
                        return None
                except Exception as e:
                    logging.warning(f"Failed reading model_config from {config_path}: {e}")
                    return None
        logging.warning("No cnp_config.json found to load model_config (searched model dir and parents)")
        return None
    except Exception as e:
        logging.warning(f"Error discovering training model_config: {e}")
        return None









# Add this function to verify locations
def verify_locations(df: pd.DataFrame, context: str) -> None:
    """Verify and print sample locations in a DataFrame."""
    if 'Longitude' in df.columns and 'Latitude' in df.columns:
        sample_locs = df[['Longitude', 'Latitude']].drop_duplicates().head(5)
        print(f"{context} sample locations (first 5 unique):\n{sample_locs}")
    else:
        print(f"Warning: Location columns missing in {context} data")

def run_inference_all(
    model_path: str,
    data_paths: str,
    file_pattern: str,
    output_dir: str,
    variable_list: str = None,
    model_config: str = None,
    scalers_dir: str = None,
    use_training_config: bool = True,
    strict_loading: bool = True,
    debug_vars: bool = False,
    loader: str = 'auto',
    mask_pft_with_gt: bool = False
) -> Path:
    """Run inference with the trained CNP model over the entire dataset.
    
    This function:
    1. Loads the trained model and existing scalers (no retraining)
    2. Loads and preprocesses new data for inference
    3. Uses the existing scalers for normalization during inference
    4. Runs predictions and saves results
    """
    
    # Set fixed random seed for reproducible results
    import random
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Handle variable list: either use provided CNP_IO file or auto-detect from training config
    variables = None
    
    if variable_list is not None:
        # User provided a CNP_IO file - parse it using the existing function
        logging.info(f"Using provided variable list: {variable_list}")
        variables = parse_cnp_io_list(variable_list)
    else:
        # No variable list provided - auto-detect from training configuration
        logging.info("No variable list provided, auto-detecting from training configuration...")
        variables = _load_training_variables_from_config(Path(model_path))
        
        if variables:
            logging.info("Successfully loaded variable lists from training configuration")
        else:
            raise ValueError("--variable-list not provided and could not auto-detect from training config")
    
    # Get configuration using the same method as train_cnp_model.py, with optional overrides
    config = get_cnp_combined_config(
        use_trendy1=True,
        use_trendy05=False,
        include_water=False,
        variable_list_path=variable_list,
        model_config_path=model_config
    )
    if model_config is not None and use_training_config:
        logging.warning("--model-config provided along with --use-training-config; training config will still govern variables and scalers. Model overrides only affect architecture sizing.")
    
    # Apply model_config from training cnp_config.json if requested
    json_model_config = None
    if use_training_config:
        json_model_config = _load_training_model_config_from_config(Path(model_path))
        if isinstance(json_model_config, dict):
            applied = 0
            for k, v in json_model_config.items():
                if hasattr(config.model_config, k):
                    try:
                        setattr(config.model_config, k, v)
                        applied += 1
                    except Exception as e:
                        logging.warning(f"Failed applying model_config.{k} from JSON: {e}")
                else:
                    # Some fields may be added dynamically later; log and skip
                    logging.info(f"Ignoring unknown ModelConfig key in JSON: {k}")
            logging.info(f"Applied {applied} model_config fields from training JSON")

    # CRITICAL FIX: Apply the loaded variable configuration to ensure model compatibility
    if variables is not None:
        logging.info("Applying variable configuration to model config...")
        
        # Update the config with the exact variable lists
        config.data_config.time_series_columns = variables.get('time_series_variables', config.data_config.time_series_columns)
        config.data_config.static_columns = variables.get('surface_properties', config.data_config.static_columns)
        config.data_config.pft_param_columns = variables.get('pft_parameters', config.data_config.pft_param_columns)
        config.data_config.x_list_scalar_columns = variables.get('scalar_variables', config.data_config.x_list_scalar_columns)
        config.data_config.x_list_columns_1d = variables.get('pft_1d_variables', config.data_config.x_list_columns_1d)
        config.data_config.x_list_columns_2d = variables.get('variables_2d_soil', config.data_config.x_list_columns_2d)
        
        # Generate output variable lists (Y_ prefixed versions)
        config.data_config.y_list_scalar_columns = ['Y_' + v for v in variables.get('scalar_variables', [])]
        config.data_config.y_list_columns_1d = ['Y_' + v for v in variables.get('pft_1d_variables', [])]
        config.data_config.y_list_columns_2d = ['Y_' + v for v in variables.get('variables_2d_soil', [])]
        
        logging.info(f"Applied variable configuration:")
        logging.info(f"  Time series: {len(config.data_config.time_series_columns)} variables")
        logging.info(f"  Surface properties: {len(config.data_config.static_columns)} variables")
        logging.info(f"  PFT parameters: {len(config.data_config.pft_param_columns)} variables")
        logging.info(f"  Scalar variables: {len(config.data_config.x_list_scalar_columns)} variables")
        logging.info(f"  PFT 1D variables: {len(config.data_config.x_list_columns_1d)} variables")
        logging.info(f"  2D soil variables: {len(config.data_config.x_list_columns_2d)} variables")
        
        # Update model configuration for output dimensions only if not provided by JSON
        if not (isinstance(json_model_config, dict) and 'scalar_output_size' in json_model_config):
            config.model_config.scalar_output_size = len(config.data_config.x_list_scalar_columns)
        if not (isinstance(json_model_config, dict) and 'vector_output_size' in json_model_config):
            config.model_config.vector_output_size = len(config.data_config.x_list_columns_1d)
        if not (isinstance(json_model_config, dict) and 'matrix_output_size' in json_model_config):
            config.model_config.matrix_output_size = len(config.data_config.x_list_columns_2d)
        
        logging.info(f"Updated model output dimensions:")
        logging.info(f"  Scalar output size: {config.model_config.scalar_output_size}")
        logging.info(f"  Vector output size: {config.model_config.vector_output_size}")
        logging.info(f"  Matrix output size: {config.model_config.matrix_output_size}")
    else:
        logging.info("Using default configuration - no variable override applied")
    
    # CRITICAL FIX: Apply training architecture (model_config) so weights match exactly
    if use_training_config:
        training_model_cfg = _load_training_model_config(Path(model_path))
        if training_model_cfg:
            logging.info("Applying training model_config (architecture) from cnp_config.json...")
            # Set attributes present in config.model_config
            for key, value in training_model_cfg.items():
                try:
                    if hasattr(config.model_config, key):
                        setattr(config.model_config, key, value)
                except Exception as e:
                    logging.warning(f"Failed to apply model_config key '{key}': {e}")
            # Re-log a few critical dimensions
            try:
                logging.info(
                    f"Architecture summary: lstm_hidden_size={getattr(config.model_config, 'lstm_hidden_size', 'NA')}, "
                    f"pft_1d_fc_size={getattr(config.model_config, 'pft_1d_fc_size', 'NA')}, "
                    f"transformer_layers={getattr(config.model_config, 'transformer_layers', 'NA')}"
                )
            except Exception:
                pass
    
    # Update data config with provided paths and pattern
    config.data_config.data_paths = [data_paths]
    config.data_config.file_pattern = file_pattern
    
    # CRITICAL FIX: Use the EXACT same data processing as training
    # During training: data was shuffled with random_state=42, then split 80/20
    # For inference: we want to process the entire dataset but in the SAME order
    # This ensures the test portion (20%) produces identical predictions
    config.data_config.train_split = 0.0  # All data goes to test for inference
    config.data_config.random_state = 42  # Same shuffle as training
    
    # Decide normalization pipeline by inspecting training run scalers next to model_path
    model_dir = Path(model_path).parent
    
    # Use provided scalers_dir if available, otherwise auto-detect
    if scalers_dir is not None:
        scalers_dir = Path(scalers_dir)
        logging.info(f"Using provided scalers directory: {scalers_dir}")
    else:
        scalers_dir = model_dir / 'scalers'
        logging.info(f"Auto-detected scalers directory: {scalers_dir}")
    
    uses_individual = False
    try:
        if scalers_dir.exists():
            # Heuristics: presence of any file/dir indicating individual scalers or metadata
            has_individual_files = any(p.name.startswith('individual_') for p in scalers_dir.iterdir() if p.is_file())
            has_metadata = any(p.name == 'metadata.json' for p in scalers_dir.iterdir() if p.is_file())
            
            if has_metadata:
                # Check metadata for normalization type
                import json
                metadata_path = scalers_dir / 'metadata.json'
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    uses_individual = metadata.get('normalization_type', '').startswith('individual')
                except:
                    pass
            
            if not uses_individual and has_individual_files:
                uses_individual = True
                
        # Also check cnp_config.json for normalization method
        config_path = model_dir / 'cnp_config.json'
        if config_path.exists():
            import json
            try:
                with open(config_path, 'r') as f:
                    cnp_config = json.load(f)
                if cnp_config.get('normalization_method') == 'individual':
                    uses_individual = True
            except:
                pass
                
    except Exception as e:
        logging.warning(f"Error detecting normalization type: {e}")
    
    # Load existing scalers from training run (no need to recreate them)
    if not scalers_dir.exists():
        raise ValueError(f"Scalers directory not found at {scalers_dir}. Please ensure you have the trained scalers.")
    
    logging.info(f"Loading existing scalers from {scalers_dir}")
    
    # Load the scalers that were saved during training
    # FIXED: Prioritize individual scalers to match training normalization
    scalers = {}
    try:
        import pickle  # local import to avoid polluting module scope
        
        # First pass: load individual scalers (priority)
        individual_scalers = {}
        for scaler_file in scalers_dir.glob('individual_*.pkl'):
            try:
                with open(scaler_file, 'rb') as f:
                    obj = pickle.load(f)
                # Normalize key name: strip 'individual_' prefix and '_scaler' suffix
                key = scaler_file.stem
                if key.startswith('individual_'):
                    key = key[len('individual_'):]
                if key.endswith('_scaler'):
                    key = key[:-len('_scaler')]
                individual_scalers[key] = obj
                logging.info(f"Loaded individual scaler: {scaler_file.name} -> {key}")
            except Exception as e:
                logging.warning(f"Failed loading individual scaler {scaler_file.name}: {e}")
        
        # Second pass: load group scalers only if individual not available
        group_scalers = {}
        # 1) Load consolidated group scalers if present
        group_file = scalers_dir / 'group_scalers.pkl'
        if group_file.exists():
            try:
                with open(group_file, 'rb') as f:
                    grp = pickle.load(f)
                if isinstance(grp, dict):
                    group_scalers.update(grp)
                    logging.info(f"Loaded group_scalers.pkl with keys: {list(grp.keys())}")
            except Exception as e:
                logging.warning(f"Failed loading group_scalers.pkl: {e}")
        
        # 2) Load any other standalone scaler pickles (non-individual)
        for scaler_file in scalers_dir.glob('*.pkl'):
            if scaler_file.name == 'group_scalers.pkl' or scaler_file.name.startswith('individual_'):
                continue
            try:
                with open(scaler_file, 'rb') as f:
                    obj = pickle.load(f)
                # Normalize key name: strip '_scaler' suffix
                key = scaler_file.stem
                if key.endswith('_scaler'):
                    key = key[:-len('_scaler')]
                group_scalers[key] = obj
            except Exception as e:
                logging.warning(f"Failed loading group scaler {scaler_file.name}: {e}")
        
        # Merge: individual scalers take priority
        scalers.update(group_scalers)  # Load group first
        scalers.update(individual_scalers)  # Override with individual
        
        logging.info(f"Scaler priority: individual={len(individual_scalers)}, group={len(group_scalers)}")
        
    except Exception as e:
        raise ValueError(f"Failed to load scalers from {scalers_dir}: {e}")
    
    logging.info(f"Loaded {len(scalers)} scalers successfully")
    try:
        logging.info(f"Available scaler keys: {sorted(list(scalers.keys()))}")
    except Exception:
        pass
    logging.info(f"Detected normalization type: {'individual' if uses_individual else 'group'}")
    
    # Get data info for model construction (we only need the structure, not the actual data)
    # Create a minimal data_info with just the variable lists we need
    # IMPORTANT: Use the variables from training config, not the default config
    # CRITICAL: The CNPCombinedModel expects specific key names that match the training data loader
    data_info = {
        'time_series_columns': variables.get('time_series_variables', []),
        'static_columns': variables.get('surface_properties', []),
        'pft_param_columns': variables.get('pft_parameters', []),
        'x_list_scalar_columns': variables.get('scalar_variables', []),
        'y_list_scalar_columns': ['Y_' + v for v in variables.get('scalar_variables', [])],
        'variables_1d_pft': variables.get('pft_1d_variables', []),  # CRITICAL: Model expects this key name
        'y_list_columns_1d': ['Y_' + v for v in variables.get('pft_1d_variables', [])],
        'x_list_columns_2d': variables.get('variables_2d_soil', []),
        'y_list_columns_2d': ['Y_' + v for v in variables.get('variables_2d_soil', [])]
    }
    
    # Log the data_info for debugging
    logging.info(f"Created data_info for model construction:")
    logging.info(f"  PFT 1D variables: {len(data_info['variables_1d_pft'])} variables")  # Use correct key
    logging.info(f"  2D soil variables: {len(data_info['x_list_columns_2d'])} variables")
    logging.info(f"  Scalar variables: {len(data_info['x_list_scalar_columns'])} variables")
    logging.info(f"  PFT parameters: {len(data_info['pft_param_columns'])} variables")
    
    # Create CNP model using the same method as train_cnp_model.py
    model = CNPCombinedModel(
        config.model_config,
        data_info,
        include_water=False,
        use_learnable_loss_weights=config.training_config.use_learnable_loss_weights
    )
    
    # Load the trained model weights
    logging.info(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Log model architecture details for debugging
    logging.info(f"Model architecture details:")
    logging.info(f"  PFT 1D input size: {model.pft_1d_input_size}")
    logging.info(f"  Soil 2D channels: {model.actual_2d_channels}")
    logging.info(f"  Scalar input size: {model.scalar_input_size}")
    logging.info(f"  PFT param input size: {model.pft_param_input_size}")
    
    # Check for size mismatches before loading
    if strict_loading:
        for name, param in model.state_dict().items():
            if name in state_dict:
                expected_shape = param.shape
                actual_shape = state_dict[name].shape
                if expected_shape != actual_shape:
                    logging.error(f"Size mismatch for {name}: expected {expected_shape}, got {actual_shape}")
                    raise ValueError(f"Model size mismatch: {name} expected {expected_shape}, got {actual_shape}")
        
        model.load_state_dict(state_dict, strict=True)
        logging.info("Model weights loaded successfully with strict=True")
    else:
        model.load_state_dict(state_dict, strict=False)
        logging.info("Model weights loaded successfully with strict=False")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # SIMPLER APPROACH: Direct inference without data splitting
    # Load and preprocess data, then apply existing scalers directly
    logging.info("Loading data for inference...")
    # Select loader based on normalization and user preference
    selected_loader_name = 'PandasDataLoader'
    if loader not in ('auto', 'pandas', 'individual'):
        logging.warning(f"Unknown loader option '{loader}', defaulting to 'auto'")
        loader = 'auto'
    if loader == 'individual' or (loader == 'auto' and uses_individual and DataLoaderIndividual is not None):
        if DataLoaderIndividual is None:
            logging.warning("Requested individual loader but DataLoaderIndividual is unavailable; using PandasDataLoader")
            _loader = PandasDataLoader(config.data_config, config.preprocessing_config)
        else:
            _loader = DataLoaderIndividual(config.data_config, config.preprocessing_config)
            selected_loader_name = 'DataLoaderIndividual'
    else:
        _loader = PandasDataLoader(config.data_config, config.preprocessing_config)
        selected_loader_name = 'PandasDataLoader'
    logging.info(f"Selected data loader: {selected_loader_name}")
    
    # Load raw data
    raw_data = _loader.load_data()
    logging.info("Data loaded successfully")
    
    # Verify locations in raw data
    if hasattr(_loader, 'df') and isinstance(_loader.df, pd.DataFrame):
        verify_locations(_loader.df, "Raw data after loading")
    else:
        logging.warning("Cannot verify raw data locations: loader does not have a DataFrame attribute 'df'")
  

    # Use EXACT same data flow as train_cnp_model.py for compatibility
    logging.info("Preprocessing data using training-compatible method...")
    preprocessed_data = _loader.preprocess_data()
    # Retry with individual loader if auto-selected pandas failed and individual is indicated
    if preprocessed_data is None and uses_individual and selected_loader_name == 'PandasDataLoader' and DataLoaderIndividual is not None:
        logging.info("Preprocessing returned None with PandasDataLoader; retrying with DataLoaderIndividual...")
        try:
            _loader = DataLoaderIndividual(config.data_config, config.preprocessing_config)
            selected_loader_name = 'DataLoaderIndividual'
            preprocessed_data = _loader.preprocess_data()
        except Exception as e:
            logging.error(f"Retry with DataLoaderIndividual failed: {e}")
    if preprocessed_data is None:
        logging.error("preprocess_data() returned None. Cannot proceed to normalization.")
        logging.error(f"Loader class: {type(_loader).__name__}")
        logging.error(f"Data paths: {config.data_config.data_paths}")
        logging.error(f"File pattern: {config.data_config.file_pattern}")
        logging.error(f"Variable lists - scalar: {len(config.data_config.x_list_scalar_columns)}, 1d: {len(config.data_config.x_list_columns_1d)}, 2d: {len(config.data_config.x_list_columns_2d)}")
        logging.error(f"Normalization type: {'individual' if uses_individual else 'group'}; scalers_dir: {scalers_dir}")
        if debug_vars:
            logging.info(f"Scalar vars: {config.data_config.x_list_scalar_columns}")
            logging.info(f"PFT 1D vars: {config.data_config.x_list_columns_1d}")
            logging.info(f"Soil 2D vars: {config.data_config.x_list_columns_2d}")
        raise ValueError("Preprocessing produced None. See logs above for context.")
    logging.info("Data preprocessed successfully.")
    if debug_vars:
        try:
            # If dict-like, log keys and shapes; else log DataFrame shape and some columns
            if isinstance(preprocessed_data, dict):
                keys = list(preprocessed_data.keys())
                logging.info(f"Preprocessed data keys: {keys}")
                for k in ['time_series_data','static_data','pft_param_data','scalar_data','variables_1d_pft','variables_2d_soil']:
                    if k in preprocessed_data and hasattr(preprocessed_data[k], 'shape'):
                        logging.info(f"  {k}: shape={preprocessed_data[k].shape}")
            else:
                # Assume DataFrame
                try:
                    shape = preprocessed_data.shape if hasattr(preprocessed_data, 'shape') else None
                    cols = list(preprocessed_data.columns)[:10] if hasattr(preprocessed_data, 'columns') else None
                    logging.info(f"Preprocessed DataFrame shape: {shape}, sample columns: {cols}")
                except Exception:
                    pass
        except Exception:
            pass
    
    # Get data_info for model construction
    data_info = _loader.get_data_info()
    
    # Helper: Normalize using training scalers without refit
    def _normalize_with_training_scalers() -> dict:
        logging.info("Normalizing data using saved training scalers (no refit)...")
        df = _loader.df
        dtype = torch.float32
        
        def _apply_scalar_manager(manager, data_mat, variable_names):
            try:
                for i, var in enumerate(variable_names):
                    key = f'scalar_{var}'
                    if hasattr(manager, 'scalers') and key in manager.scalers and hasattr(manager.scalers[key], 'transform'):
                        data_mat[:, i:i+1] = manager.scalers[key].transform(data_mat[:, i:i+1])
                    else:
                        logging.warning(f"Scalar per-variable scaler missing for {var}; leaving as-is")
            except Exception as e:
                logging.warning(f"Failed applying scalar IndividualScalerManager: {e}")
            return data_mat
        
        def _apply_pft1d_manager(manager, data_3d, variable_names):
            try:
                # data_3d: (n, vars, pfts)
                n, v, p = data_3d.shape
                # Detect pft names stored in manager
                stored_keys = list(getattr(manager, 'scalers', {}).keys())
                pft_name_mode = 'PFT0' if any(k.startswith('pft1d_PFT0_') for k in stored_keys) else ('PFT1' if any(k.startswith('pft1d_PFT1_') for k in stored_keys) else None)
                for var_idx, var in enumerate(variable_names):
                    for pft_idx in range(p):
                        if pft_name_mode == 'PFT0':
                            pft_name = f'PFT{pft_idx}'
                        elif pft_name_mode == 'PFT1':
                            pft_name = f'PFT{pft_idx+1}'
                        else:
                            # Fallback: try both
                            candidates = [f'pft1d_PFT{pft_idx}_{var}', f'pft1d_PFT{pft_idx+1}_{var}']
                            sel_key = next((k for k in candidates if k in manager.scalers), None)
                            if sel_key and hasattr(manager.scalers[sel_key], 'transform'):
                                slice_data = data_3d[:, var_idx:var_idx+1, pft_idx:pft_idx+1].reshape(n, 1)
                                data_3d[:, var_idx:var_idx+1, pft_idx:pft_idx+1] = manager.scalers[sel_key].transform(slice_data).reshape(n, 1, 1)
                            continue
                        key = f'pft1d_{pft_name}_{var}'
                        if key in manager.scalers and hasattr(manager.scalers[key], 'transform'):
                            slice_data = data_3d[:, var_idx:var_idx+1, pft_idx:pft_idx+1].reshape(n, 1)
                            data_3d[:, var_idx:var_idx+1, pft_idx:pft_idx+1] = manager.scalers[key].transform(slice_data).reshape(n, 1, 1)
                        else:
                            logging.warning(f"PFT1D per-variable scaler missing for {pft_name}_{var}; leaving as-is")
            except Exception as e:
                logging.warning(f"Failed applying PFT1D IndividualScalerManager: {e}")
            return data_3d
        
        def _apply_soil2d_manager(manager, data_4d, variable_names):
            try:
                # data_4d: (n, vars, 1, layers)
                n, v, _, L = data_4d.shape
                for var_idx, var in enumerate(variable_names):
                    for layer_idx in range(L):
                        key = f'soil2d_{var}_layer{layer_idx}'
                        if hasattr(manager, 'scalers') and key in manager.scalers and hasattr(manager.scalers[key], 'transform'):
                            slice_data = data_4d[:, var_idx, :, layer_idx].reshape(n, 1)
                            data_4d[:, var_idx, :, layer_idx] = manager.scalers[key].transform(slice_data).reshape(n, 1)
                        else:
                            logging.warning(f"Soil2D per-layer scaler missing for {var} layer {layer_idx}; leaving as-is")
            except Exception as e:
                logging.warning(f"Failed applying Soil2D IndividualScalerManager: {e}")
            return data_4d
        # Time series
        ts_cols = config.data_config.time_series_columns
        time_steps = config.data_config.time_series_length
        ts_list = []
        for col in ts_cols:
            if col in df.columns:
                col_vals = [np.array(x, dtype=np.float32) if isinstance(x, (list, np.ndarray)) else np.zeros(time_steps, dtype=np.float32) for x in df[col].values]
                arr = np.vstack(col_vals)
                if arr.shape[1] != time_steps:
                    arr = (arr[:, :time_steps] if arr.shape[1] > time_steps else np.hstack([arr, np.repeat(arr[:, -1:], time_steps - arr.shape[1], axis=1)]))
            else:
                logging.warning(f"Time series column {col} missing; filling zeros")
                arr = np.zeros((len(df), time_steps), dtype=np.float32)
            ts_list.append(arr)
        time_series = np.stack(ts_list, axis=-1)  # (n, T, F)
        if 'time_series' in scalers and hasattr(scalers['time_series'], 'transform'):
            flat = time_series.reshape(-1, len(ts_cols))
            flat_norm = scalers['time_series'].transform(flat)
            time_series = flat_norm.reshape(len(df), time_steps, len(ts_cols))
        else:
            logging.warning("Training scaler 'time_series' not found; leaving time_series unnormalized")
        time_series_t = torch.tensor(time_series, dtype=dtype)

        # Static
        static_cols = config.data_config.static_columns
        for col in static_cols:
            if col not in df.columns:
                logging.warning(f"Static column {col} missing; filling zeros")
        static_mat = df.reindex(columns=static_cols).fillna(0).values
        if 'static' in scalers and hasattr(scalers['static'], 'transform'):
            static_mat = scalers['static'].transform(static_mat)
        else:
            logging.warning("Training scaler 'static' not found; leaving static unnormalized")
        static_t = torch.tensor(static_mat, dtype=dtype)

        # PFT param
        pp_cols = config.data_config.pft_param_columns
        num_pfts = 17
        pp_list = []
        for _, row in df.iterrows():
            row_vecs = []
            for col in pp_cols:
                val = row[col] if col in df.columns else None
                if isinstance(val, (list, np.ndarray)) and len(val) == num_pfts:
                    row_vecs.append(np.asarray(val, dtype=np.float32))
                else:
                    row_vecs.append(np.zeros(num_pfts, dtype=np.float32))
            pp_list.append(np.stack(row_vecs, axis=0))
        pft_param = np.stack(pp_list, axis=0)  # (n, len(pp_cols), 17)
        if 'pft_param' in scalers and hasattr(scalers['pft_param'], 'transform'):
            flat = pft_param.reshape(len(df), -1)
            flat_norm = scalers['pft_param'].transform(flat)
            pft_param = flat_norm.reshape(pft_param.shape)
        else:
            logging.warning("Training scaler 'pft_param' not found; leaving pft_param unnormalized")
        pft_param_t = torch.tensor(pft_param, dtype=dtype)

        # Scalar X
        x_scalar_cols = config.data_config.x_list_scalar_columns
        scalar_mat = df.reindex(columns=x_scalar_cols).fillna(0).values
        if 'scalar' in scalers:
            if hasattr(scalers['scalar'], 'transform'):
                scalar_mat = scalers['scalar'].transform(scalar_mat)
            elif hasattr(scalers['scalar'], 'scalers'):
                scalar_mat = _apply_scalar_manager(scalers['scalar'], scalar_mat, x_scalar_cols)
            else:
                logging.warning("Training scaler 'scalar' present but unsupported type; leaving unnormalized")
        else:
            logging.warning("Training scaler 'scalar' not found; leaving scalar unnormalized")
        scalar_t = torch.tensor(scalar_mat, dtype=dtype)

        # Y Scalar
        y_scalar_cols = config.data_config.y_list_scalar_columns
        y_scalar_mat = df.reindex(columns=y_scalar_cols).fillna(0).values
        if 'y_scalar' in scalers:
            if hasattr(scalers['y_scalar'], 'transform'):
                y_scalar_mat = scalers['y_scalar'].transform(y_scalar_mat)
            elif hasattr(scalers['y_scalar'], 'scalers'):
                y_scalar_mat = _apply_scalar_manager(scalers['y_scalar'], y_scalar_mat, y_scalar_cols)
            else:
                logging.warning("Training scaler 'y_scalar' present but unsupported type; leaving unnormalized")
        else:
            logging.warning("Training scaler 'y_scalar' not found; leaving y_scalar unnormalized")
        y_scalar_t = torch.tensor(y_scalar_mat, dtype=dtype)

        # 1D PFT X (ensure PFT0 is dropped and length=16)
        x1d_cols = config.data_config.x_list_columns_1d
        col_data = []
        for col in x1d_cols:
            if col in df.columns:
                raw = df[col].values
                mat = []
                for v in raw:
                    arr = np.array(v, dtype=np.float32) if isinstance(v, (list, np.ndarray)) else np.zeros(16, dtype=np.float32)
                    if arr.ndim == 1:
                        # If 17 (has PFT0), drop the first; if 16 (PFT1..PFT16), keep as-is; otherwise pad to 16
                        if arr.shape[0] >= 17:
                            arr = arr[1:17]
                        elif arr.shape[0] == 16:
                            arr = arr
                        else:
                            arr = np.pad(arr, (0, 16 - arr.shape[0]), mode='constant')
                    elif arr.ndim > 1:
                        arr = arr.flatten()
                        if arr.shape[0] >= 17:
                            arr = arr[1:17]
                        elif arr.shape[0] == 16:
                            arr = arr
                        else:
                            arr = np.pad(arr, (0, 16 - arr.shape[0]), mode='constant')
                    mat.append(arr)
                mat = np.vstack(mat)
            else:
                mat = np.zeros((len(df), 16), dtype=np.float32)
            col_data.append(mat)
        pft1d = np.stack(col_data, axis=1)  # (n, vars, 16)
        if 'pft_1d' in scalers:
            if hasattr(scalers['pft_1d'], 'transform'):
                flat = pft1d.reshape(len(df), -1)
                flat_norm = scalers['pft_1d'].transform(flat)
                pft1d = flat_norm.reshape(pft1d.shape)
            elif hasattr(scalers['pft_1d'], 'scalers'):
                pft1d = _apply_pft1d_manager(scalers['pft_1d'], pft1d, x1d_cols)
            else:
                logging.warning("Training scaler 'pft_1d' present but unsupported type; leaving unnormalized (group)")
        else:
            logging.warning("Training scaler 'pft_1d' not found; leaving pft_1d unnormalized (group)")
        variables_1d_pft_t = torch.tensor(pft1d, dtype=dtype)

        # 1D PFT Y (ensure PFT0 is dropped and length=16)
        y1d_cols = config.data_config.y_list_columns_1d
        y_col_data = []
        for col in y1d_cols:
            if col in df.columns:
                raw = df[col].values
                mat = []
                for v in raw:
                    arr = np.array(v, dtype=np.float32) if isinstance(v, (list, np.ndarray)) else np.zeros(16, dtype=np.float32)
                    if arr.ndim == 1:
                        if arr.shape[0] >= 17:
                            arr = arr[1:17]
                        elif arr.shape[0] == 16:
                            arr = arr
                        else:
                            arr = np.pad(arr, (0, 16 - arr.shape[0]), mode='constant')
                    elif arr.ndim > 1:
                        arr = arr.flatten()
                        if arr.shape[0] >= 17:
                            arr = arr[1:17]
                        elif arr.shape[0] == 16:
                            arr = arr
                        else:
                            arr = np.pad(arr, (0, 16 - arr.shape[0]), mode='constant')
                    mat.append(arr)
                mat = np.vstack(mat)
            else:
                mat = np.zeros((len(df), 16), dtype=np.float32)
            y_col_data.append(mat)
        y_pft1d = np.stack(y_col_data, axis=1)
        if 'y_pft_1d' in scalers:
            if hasattr(scalers['y_pft_1d'], 'transform'):
                flat = y_pft1d.reshape(len(df), -1)
                flat_norm = scalers['y_pft_1d'].transform(flat)
                y_pft1d = flat_norm.reshape(y_pft1d.shape)
            elif hasattr(scalers['y_pft_1d'], 'scalers'):
                y_pft1d = _apply_pft1d_manager(scalers['y_pft_1d'], y_pft1d, y1d_cols)
            else:
                logging.warning("Training scaler 'y_pft_1d' present but unsupported type; leaving unnormalized (group)")
        else:
            logging.warning("Training scaler 'y_pft_1d' not found; leaving y_pft_1d unnormalized (group)")
        y_pft_1d_t = torch.tensor(y_pft1d, dtype=dtype)

        # 2D Soil X (first column, top 10 layers)
        x2d_cols = config.data_config.x_list_columns_2d
        soil_list = []
        for col in x2d_cols:
            vals = df[col].values if col in df.columns else [None] * len(df)
            samples = []
            for v in vals:
                try:
                    arr = np.array(v)
                    if arr.ndim == 2 and arr.shape[0] >= 1:
                        take = min(10, arr.shape[1])
                        out = np.zeros((1, 10), dtype=np.float32)
                        out[:, :take] = arr[0:1, :take]
                    else:
                        out = np.zeros((1, 10), dtype=np.float32)
                except Exception:
                    out = np.zeros((1, 10), dtype=np.float32)
                samples.append(out)
            soil_list.append(np.stack(samples))
        soil2d = np.stack(soil_list, axis=1)  # (n, vars, 1, 10)
        if 'variables_2d_soil' in scalers:
            if hasattr(scalers['variables_2d_soil'], 'transform'):
                flat = soil2d.reshape(len(df), -1)
                flat_norm = scalers['variables_2d_soil'].transform(flat)
                soil2d = flat_norm.reshape(soil2d.shape)
            elif hasattr(scalers['variables_2d_soil'], 'scalers'):
                soil2d = _apply_soil2d_manager(scalers['variables_2d_soil'], soil2d, x2d_cols)
            else:
                logging.warning("Training scaler 'variables_2d_soil' present but unsupported type; leaving unnormalized (group)")
        else:
            logging.warning("Training scaler 'variables_2d_soil' not found; leaving soil_2d unnormalized (group)")
        variables_2d_soil_t = torch.tensor(soil2d, dtype=dtype)

        # 2D Soil Y
        y2d_cols = config.data_config.y_list_columns_2d
        y_soil_list = []
        for col in y2d_cols:
            vals = df[col].values if col in df.columns else [None] * len(df)
            samples = []
            for v in vals:
                try:
                    arr = np.array(v)
                    if arr.ndim == 2 and arr.shape[0] >= 1:
                        take = min(10, arr.shape[1])
                        out = np.zeros((1, 10), dtype=np.float32)
                        out[:, :take] = arr[0:1, :take]
                    else:
                        out = np.zeros((1, 10), dtype=np.float32)
                except Exception:
                    out = np.zeros((1, 10), dtype=np.float32)
                samples.append(out)
            y_soil_list.append(np.stack(samples))
        y_soil2d = np.stack(y_soil_list, axis=1)
        if 'y_soil_2d' in scalers:
            if hasattr(scalers['y_soil_2d'], 'transform'):
                flat = y_soil2d.reshape(len(df), -1)
                flat_norm = scalers['y_soil_2d'].transform(flat)
                y_soil2d = flat_norm.reshape(y_soil2d.shape)
            elif hasattr(scalers['y_soil_2d'], 'scalers'):
                y_soil2d = _apply_soil2d_manager(scalers['y_soil_2d'], y_soil2d, y2d_cols)
            else:
                logging.warning("Training scaler 'y_soil_2d' present but unsupported type; leaving unnormalized (group)")
        else:
            logging.warning("Training scaler 'y_soil_2d' not found; leaving y_soil_2d unnormalized (group)")
        y_soil_2d_t = torch.tensor(y_soil2d, dtype=dtype)

        return {
            'time_series_data': time_series_t,
            'static_data': static_t,
            'pft_param_data': pft_param_t,
            'scalar_data': scalar_t,
            'variables_1d_pft': variables_1d_pft_t,
            'variables_2d_soil': variables_2d_soil_t,
            'y_scalar': y_scalar_t,
            'y_pft_1d': y_pft_1d_t,
            'y_soil_2d': y_soil_2d_t,
            'water': None,
            'y_water': None,
        }

    # Normalize using training scalers by default (no refit), or refit if requested
    logging.info("Normalizing data using training-compatible method...")
    try:
        use_refit = getattr(config, 'refit_normalization', False)
    except Exception:
        use_refit = False
    # Use transform-only mode with pre-fitted training scalers
    if uses_individual and hasattr(_loader, 'individual_scalers'):
        logging.info("Replacing loader scalers with loaded training scalers")
        
        # Replace loader scalers with training scalers
        _loader.individual_scalers = scalers
        
        # Use transform-only mode (no fitting, just transform with existing scalers)
        logging.info("Using exact training normalization method with transform-only mode")
        normalized_data = _loader.normalize_data_individual(transform_only=True)
    else:
        logging.info("ERROR: Cannot use transform-only mode - falling back to broken method")
        normalized_data = _normalize_with_training_scalers()
    
    # Now use the EXACT same split_data method as training (but with train_split=0.0)
    logging.info("Splitting data using training-compatible method (all data goes to test)...")
    split_data = _loader.split_data(normalized_data)
    logging.info("Data split successfully.")
    
    # For inference, we use the test data (which contains all data when train_split=0.0)
    test_data = split_data['test']
    logging.info(f"Using test data for inference: {len(test_data)} data types")
    
    # Convert test_data to model_inputs format expected by the model
    model_inputs = {}
    
    # Map test_data keys to model input keys
    key_mapping = {
        'time_series': 'time_series',
        'static': 'static', 
        'pft_param': 'pft_param',
        'scalar': 'scalar',
        'variables_1d_pft': 'variables_1d_pft',
        'variables_2d_soil': 'variables_2d_soil'
    }
    
    # Verify data normalization completed successfully
    logging.info("Data normalization completed successfully")
    
    for test_key, model_key in key_mapping.items():
        if test_key in test_data:
            model_inputs[model_key] = test_data[test_key]
            logging.info(f"Model input {model_key}: {model_inputs[model_key].shape}")
    
    # CRITICAL DEBUG: Detailed tensor comparison for specific location
    if debug_vars:
        logging.info("=== DEEP DEBUGGING NORMALIZED TENSORS FOR 110.0,12.722513 ===")
        
        # First, find the location in raw data and track its processing
        target_location_found = False
        target_row_idx = None
        
        if hasattr(_loader, 'df') and isinstance(_loader.df, pd.DataFrame):
            for idx, row in _loader.df.iterrows():
                if abs(row['Longitude'] - 110.0) < 0.001 and abs(row['Latitude'] - 12.722513) < 0.001:
                    target_location_found = True
                    target_row_idx = idx
                    logging.info(f"=== FOUND TARGET LOCATION AT RAW DATA INDEX {idx} ===")
                    logging.info(f"Coordinates: Lon={row['Longitude']}, Lat={row['Latitude']}")
                    
                    # Log the raw input features for this exact row
                    for col in ['tlai', 'deadcrootc', 'deadstemc']:
                        if col in row:
                            raw_data = np.array(row[col]) if hasattr(row[col], '__len__') else [row[col]]
                            logging.info(f"Raw {col}: {raw_data}")
                    
                    # Now find this same location in the model inputs
                    # We need to map from raw DataFrame index to model input index
                    # This is tricky because of shuffling, but let's try to find it
                    
                    # Check all model input samples to find matching coordinates
                    for tensor_idx in range(model_inputs['static'].shape[0]):
                        # Get the denormalized static data for comparison
                        static_norm = model_inputs['static'][tensor_idx].cpu().numpy()
                        
                        # Try to denormalize the static data to find coordinates
                        # We'll need to use the static scaler for this
                        if 'static' in scalers and hasattr(scalers['static'], 'inverse_transform'):
                            try:
                                static_denorm = scalers['static'].inverse_transform(static_norm.reshape(1, -1))[0]
                                # Debug: show first few coordinates for all samples
                                if tensor_idx < 3:  # Only show first 3 to avoid spam
                                    logging.info(f"Sample {tensor_idx} denormalized coords: [{static_denorm[0]:.3f}, {static_denorm[1]:.3f}]")
                                # Check if this matches our target coordinates
                                # Try both Lon,Lat and Lat,Lon orders
                                if len(static_denorm) >= 2:
                                    coord_match = False
                                    if abs(static_denorm[0] - 110.0) < 0.001 and abs(static_denorm[1] - 12.722513) < 0.001:
                                        coord_match = True
                                        coord_order = "Lon,Lat"
                                    elif abs(static_denorm[1] - 110.0) < 0.001 and abs(static_denorm[0] - 12.722513) < 0.001:
                                        coord_match = True  
                                        coord_order = "Lat,Lon"
                                    
                                    if coord_match:
                                        logging.info(f"=== FOUND TARGET IN MODEL INPUTS AT INDEX {tensor_idx} ===")
                                        logging.info(f"Denormalized coordinates: Lon={static_denorm[0]}, Lat={static_denorm[1]}")
                                        
                                        # Log ALL normalized input tensors for this location
                                        logging.info("NORMALIZED MODEL INPUTS:")
                                        if 'variables_1d_pft' in model_inputs:
                                            pft_tensor = model_inputs['variables_1d_pft'][tensor_idx].cpu().numpy()
                                            logging.info(f"variables_1d_pft shape: {pft_tensor.shape}")
                                            for var_idx, var_name in enumerate(['tlai', 'deadcrootc', 'deadstemc']):
                                                logging.info(f"  {var_name}: {pft_tensor[var_idx]}")
                                        
                                        if 'scalar' in model_inputs:
                                            scalar_tensor = model_inputs['scalar'][tensor_idx].cpu().numpy()
                                            logging.info(f"scalar: {scalar_tensor}")
                                        
                                        if 'static' in model_inputs:
                                            logging.info(f"static (normalized): {static_norm[:10]} (first 10)")
                                            logging.info(f"static (denormalized): {static_denorm[:10]} (first 10)")
                                        
                                        break
                            except Exception as e:
                                logging.warning(f"Failed to denormalize static data: {e}")
                    break
        
        if not target_location_found:
            logging.warning("Target location 110.0,12.722513 not found in raw data!")
    
    # Additional debug summaries before inference
    if debug_vars:
        logging.info("Preview of model inputs before inference (first 3 rows, up to 5 cols):")
        def _preview(group_key: str, names: list):
            if group_key in model_inputs and model_inputs[group_key] is not None:
                tensor = model_inputs[group_key]
                try:
                    sub = tensor[:3, :min(5, tensor.shape[1])]
                    # If higher than 2D, aggregate across trailing dims for a scalar preview
                    if sub.dim() > 2:
                        reduce_dims = tuple(range(2, sub.dim()))
                        sub = sub.float().mean(dim=reduce_dims)
                    arr = sub.detach().cpu().numpy()
                except Exception:
                    return
                cols = min(arr.shape[1], len(names)) if names is not None else arr.shape[1]
                use_names = (names[:cols] if names else [f"c{i}" for i in range(cols)])
                logging.info(f"{group_key}: shape={tuple(tensor.shape)}")
                for r in range(arr.shape[0]):
                    line = ", ".join([f"{use_names[c]}={float(arr[r, c]):.4f}" for c in range(cols)])
                    logging.info(f"  row{r}: {line}")
        _preview('scalar', config.data_config.x_list_scalar_columns)
        _preview('variables_1d_pft', config.data_config.x_list_columns_1d)
        _preview('variables_2d_soil', config.data_config.x_list_columns_2d)
        _preview('time_series', config.data_config.time_series_columns)
        _preview('static', config.data_config.static_columns)
        _preview('pft_param', config.data_config.pft_param_columns)
    
    # Run inference directly on all data
    logging.info("Running inference on entire dataset...")
    
    # CRITICAL: Ensure model is in evaluation mode and disable all randomness
    model.eval()
    torch.set_grad_enabled(False)
    
    # Additional determinism for batch norm and dropout
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False
    
    # Move inputs to device
    for k in list(model_inputs.keys()):
        if isinstance(model_inputs[k], torch.Tensor):
            model_inputs[k] = model_inputs[k].to(device)
    
    with torch.no_grad():
        # CRITICAL FIX: Use AMP if CUDA is available to match training evaluation
        use_amp = torch.cuda.is_available()
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                # Support optional water if present
                if 'water' in model_inputs:
                    predictions = model(
                        model_inputs.get('time_series'),
                        model_inputs.get('static'),
                        model_inputs.get('pft_param'),
                        model_inputs.get('scalar'),
                        model_inputs.get('variables_1d_pft'),
                        model_inputs.get('variables_2d_soil'),
                        model_inputs.get('water')
                    )
                else:
                    predictions = model(
                        model_inputs.get('time_series'),
                        model_inputs.get('static'),
                        model_inputs.get('pft_param'),
                        model_inputs.get('scalar'),
                        model_inputs.get('variables_1d_pft'),
                        model_inputs.get('variables_2d_soil')
                    )
        else:
            # Support optional water if present
            if 'water' in model_inputs:
                predictions = model(
                    model_inputs.get('time_series'),
                    model_inputs.get('static'),
                    model_inputs.get('pft_param'),
                    model_inputs.get('scalar'),
                    model_inputs.get('variables_1d_pft'),
                    model_inputs.get('variables_2d_soil'),
                    model_inputs.get('water')
                )
            else:
                predictions = model(
                    model_inputs.get('time_series'),
                    model_inputs.get('static'),
                    model_inputs.get('pft_param'),
                    model_inputs.get('scalar'),
                    model_inputs.get('variables_1d_pft'),
                    model_inputs.get('variables_2d_soil')
                )
    
    logging.info("Inference completed successfully")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save predictions
    predictions_path = output_path / "predictions.pkl"
    with open(predictions_path, 'wb') as f:
        import pickle
        pickle.dump(predictions, f)
    
    logging.info(f"Predictions saved to: {predictions_path}")
    
    # Save denormalized predictions and ground truth similar to training outputs
    try:
        predictions_dir = output_path / 'cnp_predictions'
        predictions_dir.mkdir(exist_ok=True)

        # Prepare location vectors (Longitude/Latitude) for joining to CSVs
        longitude_values = None
        latitude_values = None
        try:
            # Preferred: derive from denormalized static features (computed below as well)
            # We'll compute once here if possible; otherwise fall back to loader DataFrame
            maybe_static = None
            maybe_static_cols = None
            if isinstance(test_data, dict) and 'static' in test_data and isinstance(scalers, dict) and 'static' in scalers and hasattr(scalers['static'], 'inverse_transform'):
                _static_np = test_data['static'].detach().cpu().numpy()
                _static_denorm = scalers['static'].inverse_transform(_static_np)
                _static_cols = data_info.get('static_columns', [])
                if not _static_cols or len(_static_cols) < _static_denorm.shape[1]:
                    _static_cols = [f'static_{i}' for i in range(_static_denorm.shape[1])]
                maybe_static = _static_denorm
                maybe_static_cols = _static_cols
            # Extract coordinates from static if present
            if maybe_static is not None and maybe_static_cols is not None:
                # Try common longitude/latitude column names
                lon_keys = ['Longitude', 'longitude', 'lon', 'LON']
                lat_keys = ['Latitude', 'latitude', 'lat', 'LAT']
                lon_name = next((k for k in lon_keys if k in maybe_static_cols), None)
                lat_name = next((k for k in lat_keys if k in maybe_static_cols), None)
                if lon_name is not None and lat_name is not None:
                    lon_idx = maybe_static_cols.index(lon_name)
                    lat_idx = maybe_static_cols.index(lat_name)
                    longitude_values = maybe_static[:, lon_idx]
                    latitude_values = maybe_static[:, lat_idx]
            # Stronger preference: if the loader DataFrame has explicit coordinates, use those.
            # This ensures we respect the exact site(s) selected for inference, even if static inverse lacks or mislabels coords.
            if hasattr(_loader, 'df') and isinstance(_loader.df, pd.DataFrame):
                if 'Longitude' in _loader.df.columns and 'Latitude' in _loader.df.columns:
                    if isinstance(test_data, dict) and 'static' in test_data:
                        n = len(test_data['static'])
                        longitude_values = _loader.df['Longitude'].values[-n:]
                        latitude_values = _loader.df['Latitude'].values[-n:]
                    else:
                        longitude_values = _loader.df['Longitude'].values
                        latitude_values = _loader.df['Latitude'].values
        except Exception as _e_loc:
            logging.warning(f"Failed to prepare location vectors: {_e_loc}")

        # 1) Save scalar predictions and ground truth
        try:
            if 'scalar' in predictions and hasattr(predictions['scalar'], 'numel') and predictions['scalar'].numel() > 0:
                preds_scalar_np = predictions['scalar'].detach().cpu().numpy()
                scalar_cols = data_info.get('y_list_scalar_columns', [])
                if not scalar_cols or len(scalar_cols) < preds_scalar_np.shape[1]:
                    scalar_cols = [f'Y_scalar_{i}' for i in range(preds_scalar_np.shape[1])]
                # Inverse transform using saved scaler when available
                try:
                    if isinstance(scalers, dict) and 'y_scalar' in scalers and hasattr(scalers['y_scalar'], 'inverse_transform'):
                        preds_scalar_denorm = scalers['y_scalar'].inverse_transform(preds_scalar_np)
                        logging.info("Applied inverse transformation to scalar predictions")
                    else:
                        preds_scalar_denorm = preds_scalar_np
                        logging.warning("Training scaler 'y_scalar' not found; saving normalized scalar predictions")
                except Exception as e:
                    logging.warning(f"Failed inverse transform for scalar predictions: {e}")
                    preds_scalar_denorm = preds_scalar_np
                _pred_scalar_df = pd.DataFrame(preds_scalar_denorm, columns=scalar_cols)
                # Prepend location columns if available
                if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(_pred_scalar_df):
                    _pred_scalar_df.insert(0, 'Longitude', longitude_values)
                    _pred_scalar_df.insert(1, 'Latitude', latitude_values)
                _pred_scalar_df.to_csv(predictions_dir / 'predictions_scalar.csv', index=False)
                # Ground truth if present
                if isinstance(test_data, dict) and 'y_scalar' in test_data and hasattr(test_data['y_scalar'], 'numel') and test_data['y_scalar'].numel() > 0:
                    gt_scalar_np = test_data['y_scalar'].detach().cpu().numpy()
                    try:
                        if isinstance(scalers, dict) and 'y_scalar' in scalers and hasattr(scalers['y_scalar'], 'inverse_transform'):
                            gt_scalar_denorm = scalers['y_scalar'].inverse_transform(gt_scalar_np)
                            logging.info("Applied inverse transformation to scalar ground truth")
                        else:
                            gt_scalar_denorm = gt_scalar_np
                            logging.warning("Training scaler 'y_scalar' not found; saving normalized scalar ground truth")
                    except Exception as e:
                        logging.warning(f"Failed inverse transform for scalar ground truth: {e}")
                        gt_scalar_denorm = gt_scalar_np
                    _gt_scalar_df = pd.DataFrame(gt_scalar_denorm, columns=scalar_cols)
                    if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(_gt_scalar_df):
                        _gt_scalar_df.insert(0, 'Longitude', longitude_values)
                        _gt_scalar_df.insert(1, 'Latitude', latitude_values)
                    _gt_scalar_df.to_csv(predictions_dir / 'ground_truth_scalar.csv', index=False)
            else:
                logging.info("No scalar predictions present; skipping scalar CSV export")
        except Exception as e:
            logging.warning(f"Error while saving scalar CSVs: {e}")
        
        if 'pft_1d' in predictions and hasattr(predictions['pft_1d'], 'numel') and predictions['pft_1d'].numel() > 0:
            predictions_pft_1d_np = predictions['pft_1d'].detach().cpu().numpy()
            n_samples = predictions_pft_1d_np.shape[0]
            # Ensure (samples, variables, pfts)
            if predictions_pft_1d_np.ndim == 2:
                predictions_pft_1d_np = predictions_pft_1d_np.reshape(n_samples, -1, 16)
            num_variables = predictions_pft_1d_np.shape[1]
            num_pfts = predictions_pft_1d_np.shape[2]
            # Variable names from data_info
            var_names_all = data_info.get('y_list_columns_1d', []) if isinstance(data_info, dict) else []
            if not var_names_all or len(var_names_all) < num_variables:
                var_names = [f'pft_1d_var_{i}' for i in range(num_variables)]
            else:
                var_names = list(var_names_all)[:num_variables]
            
            pft_1d_dir = predictions_dir / 'pft_1d_predictions'
            pft_1d_dir.mkdir(exist_ok=True)
            
            # Optionally prepare GT mask for PFTs (to zero-out predictions where GT is zero)
            gt_mask_per_var = None
            if mask_pft_with_gt and isinstance(test_data, dict) and 'y_pft_1d' in test_data and hasattr(test_data['y_pft_1d'], 'numel') and test_data['y_pft_1d'].numel() > 0:
                try:
                    gt_arr = test_data['y_pft_1d'].detach().cpu().numpy()
                    if gt_arr.ndim == 2:
                        gt_arr = gt_arr.reshape(n_samples, -1, num_pfts)
                    gt_mask_per_var = (gt_arr > 0)
                except Exception:
                    gt_mask_per_var = None

            # Write predictions per variable (denormalized when possible)
            for v in range(num_variables):
                var_name = var_names[v]
                var_predictions = predictions_pft_1d_np[:, v, :]
                var_predictions_original = var_predictions
                try:
                    scaler_key = 'y_pft_1d'
                    if isinstance(scalers, dict) and scaler_key in scalers and hasattr(scalers[scaler_key], 'inverse_transform_pft_1d'):
                        per_var_3d = var_predictions.reshape(n_samples, num_pfts, 1)
                        canonical_pft_names = [f'PFT{i}' for i in range(1, num_pfts + 1)]
                        var_predictions_denorm = scalers[scaler_key].inverse_transform_pft_1d(
                            per_var_3d, canonical_pft_names, [var_name]
                        )
                        var_predictions_original = var_predictions_denorm[:, :, 0]
                        logging.info(f"Applied inverse transformation to PFT 1D predictions for {var_name}")
                    else:
                        # Fallback: sklearn-like scaler
                        if isinstance(scalers, dict) and scaler_key in scalers and hasattr(scalers[scaler_key], 'inverse_transform'):
                            if hasattr(scalers[scaler_key], 'n_features_in_') and getattr(scalers[scaler_key], 'n_features_in_', None) == var_predictions.shape[1]:
                                var_predictions_original = scalers[scaler_key].inverse_transform(var_predictions)
                                logging.info(f"Applied fallback inverse transform to PFT 1D predictions for {var_name}")
                            else:
                                logging.warning(f"PFT 1D scaler incompatible for {var_name}; saving normalized values")
                        else:
                            logging.warning(f"No suitable PFT 1D scaler for {var_name}; saving normalized values")
                except Exception as e:
                    logging.warning(f"Failed inverse transform for PFT 1D predictions {var_name}: {e}")
                    var_predictions_original = var_predictions
                # Apply optional GT-based mask
                try:
                    if gt_mask_per_var is not None and v < gt_mask_per_var.shape[1]:
                        mask_v = gt_mask_per_var[:, v, :]
                        var_predictions_original = var_predictions_original * mask_v.astype(var_predictions_original.dtype)
                except Exception:
                    pass
                
                # Optional dump before saving predictions
                try:
                    if os.getenv('DUMP_ALL_PFT_SOIL', '0') == '1':
                        for i in range(var_predictions_original.shape[0]):
                            logging.info(f"[pre-save PRED] {var_name} row{i}: {','.join([f'{x:.6g}' for x in var_predictions_original[i]])}")
                except Exception as _e:
                    logging.warning(f"Pre-save PFT1D pred dump failed for {var_name}: {_e}")

                columns = [f'{var_name}_pft{p+1}' for p in range(num_pfts)]
                _pft_pred_df = pd.DataFrame(var_predictions_original, columns=columns)
                if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(_pft_pred_df):
                    _pft_pred_df.insert(0, 'Longitude', longitude_values)
                    _pft_pred_df.insert(1, 'Latitude', latitude_values)
                _pft_pred_df.to_csv(pft_1d_dir / f'predictions_{var_name}.csv', index=False)
            
            # Write ground truth if present
            if isinstance(test_data, dict) and 'y_pft_1d' in test_data and hasattr(test_data['y_pft_1d'], 'numel') and test_data['y_pft_1d'].numel() > 0:
                ground_truth_pft_1d_np = test_data['y_pft_1d'].detach().cpu().numpy()
                if ground_truth_pft_1d_np.ndim == 2:
                    ground_truth_pft_1d_np = ground_truth_pft_1d_np.reshape(n_samples, -1, num_pfts)
                pft_1d_gt_dir = predictions_dir / 'pft_1d_ground_truth'
                pft_1d_gt_dir.mkdir(exist_ok=True)
                for v in range(num_variables):
                    var_name = var_names[v]
                    var_gt = ground_truth_pft_1d_np[:, v, :]
                    var_gt_original = var_gt
                    try:
                        scaler_key = 'y_pft_1d'
                        if isinstance(scalers, dict) and scaler_key in scalers and hasattr(scalers[scaler_key], 'inverse_transform_pft_1d'):
                            per_var_3d = var_gt.reshape(n_samples, num_pfts, 1)
                            canonical_pft_names = [f'PFT{i}' for i in range(1, num_pfts + 1)]
                            var_gt_denorm = scalers[scaler_key].inverse_transform_pft_1d(
                                per_var_3d, canonical_pft_names, [var_name]
                            )
                            var_gt_original = var_gt_denorm[:, :, 0]
                            logging.info(f"Applied inverse transformation to PFT 1D ground truth for {var_name}")
                        else:
                            if isinstance(scalers, dict) and scaler_key in scalers and hasattr(scalers[scaler_key], 'inverse_transform'):
                                if hasattr(scalers[scaler_key], 'n_features_in_') and getattr(scalers[scaler_key], 'n_features_in_', None) == var_gt.shape[1]:
                                    var_gt_original = scalers[scaler_key].inverse_transform(var_gt)
                                    logging.info(f"Applied fallback inverse transform to PFT 1D ground truth for {var_name}")
                                else:
                                    logging.warning(f"PFT 1D scaler incompatible for GT {var_name}; saving normalized values")
                            else:
                                logging.warning(f"No suitable PFT 1D scaler for GT {var_name}; saving normalized values")
                    except Exception as e:
                        logging.warning(f"Failed inverse transform for PFT 1D ground truth {var_name}: {e}")
                        var_gt_original = var_gt
                    # Guard against all-zeros after inverse
                    try:
                        if int((var_gt != 0).sum()) > 0 and int((var_gt_original != 0).sum()) == 0:
                            logging.warning(f"Inverse yielded all-zeros for PFT1D {var_name}; using normalized GT")
                            var_gt_original = var_gt
                    except Exception:
                        pass
                    # Optional dump before saving GT
                    try:
                        if os.getenv('DUMP_ALL_PFT_SOIL', '0') == '1':
                            for i in range(var_gt_original.shape[0]):
                                logging.info(f"[pre-save GT] {var_name} row{i}: {','.join([f'{x:.6g}' for x in var_gt_original[i]])}")
                    except Exception as _e:
                        logging.warning(f"Pre-save PFT1D GT dump failed for {var_name}: {_e}")
                    columns = [f'{var_name}_pft{p+1}' for p in range(num_pfts)]
                    _pft_gt_df = pd.DataFrame(var_gt_original, columns=columns)
                    if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(_pft_gt_df):
                        _pft_gt_df.insert(0, 'Longitude', longitude_values)
                        _pft_gt_df.insert(1, 'Latitude', latitude_values)
                    _pft_gt_df.to_csv(pft_1d_gt_dir / f'ground_truth_{var_name}.csv', index=False)
            logging.info("pft_1d predictions and ground truth saved under cnp_predictions/")
        else:
            logging.info("No pft_1d predictions present; skipping CSV export")
        # 2) Save soil_2d predictions and ground truth (first column, top 10 layers preserved in normalization)
        try:
            if 'soil_2d' in predictions and hasattr(predictions['soil_2d'], 'numel') and predictions['soil_2d'].numel() > 0:
                soil_pred = predictions['soil_2d'].detach().cpu().numpy()
                n_samples = soil_pred.shape[0]
                # Determine columns/layers using test targets when available
                num_columns = 18
                num_layers = 10
                if isinstance(test_data, dict) and 'y_soil_2d' in test_data and hasattr(test_data['y_soil_2d'], 'shape'):
                    tgt_shape = tuple(test_data['y_soil_2d'].shape)
                    if len(tgt_shape) >= 3:
                        num_columns = tgt_shape[2]
                        num_layers = tgt_shape[3] if len(tgt_shape) > 3 else 1
                # Ensure shape (n, vars, cols, layers)
                if soil_pred.ndim == 4:
                    num_variables = soil_pred.shape[1]
                elif soil_pred.ndim in (2, 3):
                    flat_per_sample = soil_pred.shape[-1]
                    per_var = num_columns * num_layers
                    if flat_per_sample % per_var != 0:
                        raise ValueError(f"Unexpected soil_2d width {flat_per_sample} not divisible by cols*layers {per_var}")
                    num_variables = flat_per_sample // per_var
                    soil_pred = soil_pred.reshape(n_samples, num_variables, num_columns, num_layers)
                else:
                    raise ValueError(f"Unsupported soil_2d prediction ndim: {soil_pred.ndim}")
                var_names_all = data_info.get('y_list_columns_2d', []) if isinstance(data_info, dict) else []
                var_names = (list(var_names_all)[:num_variables] if var_names_all and len(var_names_all) >= num_variables else [f'soil_2d_var_{i}' for i in range(num_variables)])
                soil_dir = predictions_dir / 'soil_2d_predictions'
                soil_dir.mkdir(exist_ok=True)
                for v in range(num_variables):
                    var_name = var_names[v]
                    var_pred = soil_pred[:, v, :, :]
                    try:
                        if isinstance(scalers, dict) and 'y_soil_2d' in scalers and hasattr(scalers['y_soil_2d'], 'inverse_transform_soil_2d'):
                            per_var_tensor = var_pred.reshape(n_samples, 1, num_columns, num_layers)
                            denorm = scalers['y_soil_2d'].inverse_transform_soil_2d(per_var_tensor, [var_name], num_layers)
                            var_pred_orig = denorm[:, 0, :, :]
                            logging.info(f"Applied inverse transformation to soil 2D predictions for {var_name}")
                        else:
                            var_pred_orig = var_pred
                            logging.warning(f"No soil 2D scaler for {var_name}; saving normalized values")
                    except Exception as e:
                        logging.warning(f"Failed inverse transform for soil 2D predictions {var_name}: {e}")
                        var_pred_orig = var_pred
                    flat = var_pred_orig.reshape(n_samples, num_columns * num_layers)
                    columns = [f'{var_name}_col{c+1}_layer{l+1}' for c in range(num_columns) for l in range(num_layers)]
                    _soil_pred_df = pd.DataFrame(flat, columns=columns)
                    if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(_soil_pred_df):
                        _soil_pred_df.insert(0, 'Longitude', longitude_values)
                        _soil_pred_df.insert(1, 'Latitude', latitude_values)
                    _soil_pred_df.to_csv(soil_dir / f'predictions_{var_name}.csv', index=False)
                # Ground truth if present
                if isinstance(test_data, dict) and 'y_soil_2d' in test_data and hasattr(test_data['y_soil_2d'], 'numel') and test_data['y_soil_2d'].numel() > 0:
                    soil_gt = test_data['y_soil_2d'].detach().cpu().numpy()
                    if soil_gt.ndim == 4:
                        pass
                    elif soil_gt.ndim == 3:
                        soil_gt = soil_gt.reshape(n_samples, soil_gt.shape[1], num_columns, num_layers)
                    elif soil_gt.ndim == 2:
                        flat_per_sample = soil_gt.shape[1]
                        per_var = num_columns * num_layers
                        if flat_per_sample % per_var != 0:
                            raise ValueError(f"Unexpected y_soil_2d width {flat_per_sample} not divisible by cols*layers {per_var}")
                        gt_num_variables = flat_per_sample // per_var
                        if gt_num_variables != num_variables:
                            m = min(num_variables, gt_num_variables)
                            num_variables = m
                            var_names = var_names[:m]
                        soil_gt = soil_gt.reshape(n_samples, num_variables, num_columns, num_layers)
                    else:
                        raise ValueError(f"Unsupported y_soil_2d ndim: {soil_gt.ndim}")
                    soil_gt_dir = predictions_dir / 'soil_2d_ground_truth'
                    soil_gt_dir.mkdir(exist_ok=True)
                    for v in range(num_variables):
                        var_name = var_names[v]
                        var_gt = soil_gt[:, v, :, :]
                        try:
                            if isinstance(scalers, dict) and 'y_soil_2d' in scalers and hasattr(scalers['y_soil_2d'], 'inverse_transform_soil_2d'):
                                per_var_tensor = var_gt.reshape(n_samples, 1, num_columns, num_layers)
                                denorm = scalers['y_soil_2d'].inverse_transform_soil_2d(per_var_tensor, [var_name], num_layers)
                                var_gt_orig = denorm[:, 0, :, :]
                                logging.info(f"Applied inverse transformation to soil 2D ground truth for {var_name}")
                            else:
                                var_gt_orig = var_gt
                                logging.warning(f"No soil 2D scaler for {var_name}; saving normalized GT")
                        except Exception as e:
                            logging.warning(f"Failed inverse transform for soil 2D GT {var_name}: {e}")
                            var_gt_orig = var_gt
                        flat = var_gt_orig.reshape(n_samples, num_columns * num_layers)
                        columns = [f'{var_name}_col{c+1}_layer{l+1}' for c in range(num_columns) for l in range(num_layers)]
                        _soil_gt_df = pd.DataFrame(flat, columns=columns)
                        if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(_soil_gt_df):
                            _soil_gt_df.insert(0, 'Longitude', longitude_values)
                            _soil_gt_df.insert(1, 'Latitude', latitude_values)
                        _soil_gt_df.to_csv(soil_gt_dir / f'ground_truth_{var_name}.csv', index=False)
                logging.info("soil_2d predictions and ground truth saved under cnp_predictions/")
            else:
                logging.info("No soil_2d predictions present; skipping soil_2d CSV export")
        except Exception as e:
            logging.warning(f"Error while saving Soil 2D CSVs: {e}")
    except Exception as e:
        logging.warning(f"Error while saving prediction CSVs: {e}")

    # Save basic test metrics similar to training (overall RMSE/MSE)
    try:
        metrics = {}
        def _safe_rmse_mse(pred, targ):
            try:
                pred_np = np.asarray(pred, dtype=np.float64)
                targ_np = np.asarray(targ, dtype=np.float64)
                mask = ~np.isnan(pred_np) & ~np.isnan(targ_np)
                if mask.sum() == 0:
                    return (float('nan'), float('nan'))
                diff = pred_np[mask] - targ_np[mask]
                mse = float(np.mean(diff * diff))
                rmse = float(np.sqrt(mse))
                return (rmse, mse)
            except Exception:
                return (float('nan'), float('nan'))
        # Scalar
        if 'scalar' in predictions and isinstance(test_data, dict) and 'y_scalar' in test_data:
            s_pred = predictions['scalar'].detach().cpu().numpy()
            s_targ = test_data['y_scalar'].detach().cpu().numpy()
            r, m = _safe_rmse_mse(s_pred, s_targ)
            metrics['scalar_rmse'] = r
            metrics['scalar_mse'] = m
        # PFT 1D
        if 'pft_1d' in predictions and isinstance(test_data, dict) and 'y_pft_1d' in test_data and predictions['pft_1d'].numel() > 0 and test_data['y_pft_1d'].numel() > 0:
            p_pred = predictions['pft_1d'].detach().cpu().numpy()
            p_targ = test_data['y_pft_1d'].detach().cpu().numpy()
            r, m = _safe_rmse_mse(p_pred.reshape(p_pred.shape[0], -1), p_targ.reshape(p_targ.shape[0], -1))
            metrics['pft_1d_rmse'] = r
            metrics['pft_1d_mse'] = m
        # Soil 2D
        if 'soil_2d' in predictions and isinstance(test_data, dict) and 'y_soil_2d' in test_data and predictions['soil_2d'].numel() > 0 and test_data['y_soil_2d'].numel() > 0:
            z_pred = predictions['soil_2d'].detach().cpu().numpy()
            z_targ = test_data['y_soil_2d'].detach().cpu().numpy()
            r, m = _safe_rmse_mse(z_pred.reshape(z_pred.shape[0], -1), z_targ.reshape(z_targ.shape[0], -1))
            metrics['soil_2d_rmse'] = r
            metrics['soil_2d_mse'] = m
        if metrics:
            pd.DataFrame([metrics]).to_csv(predictions_dir / 'test_metrics.csv', index=False)
            logging.info(f"Metrics saved to {predictions_dir / 'test_metrics.csv'}")
        else:
            logging.info("No metrics computed (missing targets)")
    except Exception as e:
        logging.warning(f"Failed to compute/save test metrics: {e}")

    # Save inverse-transformed static features for test set (coordinates for mapping)
    try:
        if isinstance(test_data, dict) and 'static' in test_data and isinstance(scalers, dict) and 'static' in scalers and hasattr(scalers['static'], 'inverse_transform'):
            static_np = test_data['static'].detach().cpu().numpy()
            static_denorm = scalers['static'].inverse_transform(static_np)
            static_cols = data_info.get('static_columns', [])
            if not static_cols or len(static_cols) < static_denorm.shape[1]:
                static_cols = [f'static_{i}' for i in range(static_denorm.shape[1])]
            static_df = pd.DataFrame(static_denorm, columns=static_cols)
            static_df.to_csv(predictions_dir / 'test_static_inverse.csv', index=False)
            logging.info(f"Inverse-transformed test static features saved to {predictions_dir / 'test_static_inverse.csv'}")

            # Verify locations in static inverse
            try:
                verify_locations(static_df, "Static inverse (denormalized)")
            except Exception:
                pass
        else:
            logging.info("Static scaler or data not available; skipping test_static_inverse.csv")
    except Exception as e:
        logging.warning(f"Failed to save inverse-transformed static features: {e}")
    logging.info("Inference completed successfully!")
    
    return Path(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Run CNP model inference over entire dataset")
    parser.add_argument("--model", default='./cnp_predictions/model.pth', help="Path to trained model (.pth)")
    parser.add_argument("--data-paths", default='/mnt/proj-shared/AI4BGC_7xw/TrainingData/Trendy_1_data_CNP', help="Data directories containing PKL batches for inference")
    parser.add_argument("--file-pattern", default='enhanced_*1_training_data_batch_*.pkl', help="Glob pattern for PKL files")
    parser.add_argument("--output-dir", default='cnp_inference_entire_dataset', help="Output directory for results")
    parser.add_argument("--variable-list", help="Path to variable list file (optional, will auto-detect from config)")
    parser.add_argument("--scalers-dir", help="Path to scalers directory (optional, will auto-detect from model directory)")
    parser.add_argument("--model-config", help="Path to model config text file to override architecture (use with caution)")
    parser.add_argument("--use-training-config", action='store_true', default=True, help="Use exact training configuration to avoid model size mismatches (default: True)")
    parser.add_argument("--strict-loading", action='store_true', default=True, help="Use strict model loading to catch size mismatches early (default: True)")
    parser.add_argument("--debug-vars", action='store_true', help="Print detailed variable names and sample values during preprocessing/inference")
    parser.add_argument("--loader", choices=['auto','pandas','individual'], default='auto', help="Data loader to use (default: auto)")
    parser.add_argument("--mask-pft-with-gt", action='store_true', default=False, help="Mask PFT1D predictions by GT non-zero mask when available")
    parser.add_argument("--refit-normalization", action='store_true', default=False, help="Refit scalers on inference data (default: False; use training scalers)")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        output_path = run_inference_all(
            model_path=args.model,
            data_paths=args.data_paths,
            file_pattern=args.file_pattern,
            output_dir=args.output_dir,
            variable_list=args.variable_list,
            model_config=args.model_config,
            scalers_dir=args.scalers_dir,
            use_training_config=args.use_training_config,
            strict_loading=args.strict_loading
            , debug_vars=args.debug_vars
            , loader=args.loader
            , mask_pft_with_gt=args.mask_pft_with_gt
        )
        print(f"Inference completed successfully. Results saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
