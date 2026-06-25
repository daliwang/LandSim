#!/usr/bin/env python3
"""
CNP Model Training Script

This script trains the CNP (Carbon-Nitrogen-Phosphorus) model based on the 
CNP_IO_list1.txt structure with the following architecture:

- LSTM for 6 time-series variables (20 years)
- FC for surface properties (geographic, soil texture, P forms, PFT coverage)
- FC for 44 PFT characteristics parameters
- FC for water variables (optional)
- FC for scalar variables
- FC for 1D variables
- CNN for 2D variables
- Transformer encoder for feature fusion
- Multi-task perceptrons for separate predictions

Usage:
    python train_cnp_model.py [--with-water] [--variable-list CNP_IO_*.txt] [--model-config CNP_model_config_*.txt]

Examples:
    # Default variables with compact architecture overrides
    python train_cnp_model.py \
        --variable-list CNP_IO_LiterP.txt \
        --model-config CNP_model_config_v01.txt \
        --epochs 5 --batch-size 128 --learning-rate 1e-4

    # Full variable list with larger 27M-like architecture
    python train_cnp_model.py \
        --variable-list CNP_IO_list1.txt \
        --model-config CNP_model_config_27M.txt \
        --epochs 300 --batch-size 128 --learning-rate 1e-4
"""

import sys
import os
import json
import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import pandas as pd

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from config.training_config import get_cnp_model_config
from data.data_loader_individual import DataLoaderIndividual
from data.preprocessed_cache import (
    cache_is_valid,
    cache_paths,
    compute_cache_fingerprint,
    env_flag_enabled,
    list_data_files,
    load_preprocessed_cache,
    resolve_cache_dir,
    save_preprocessed_cache,
)
from models.cnp_combined_model import CNPCombinedModel
from training.trainer import ModelTrainer

from scripts.run_inference_all import verify_locations

def setup_logging(log_file: str, level: str = 'INFO') -> None:
    """Set up logging configuration with a specific log file."""
    level_value = getattr(logging, level.upper(), logging.INFO)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
    try:
        logging.basicConfig(level=level_value, format=fmt, handlers=handlers, force=True)
    except TypeError:
        logging.basicConfig(level=level_value, format=fmt, handlers=handlers)


def set_global_determinism(seed: int) -> None:
    """Enable strict determinism across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For deterministic cuBLAS GEMM; no-op on CPU
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Disable TF32 for exact reproducibility on Ampere/Hopper
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Highest precision for matmul to avoid TF32 paths (PyTorch 2+)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not enable full deterministic algorithms: {e}")


# Add detailed logging for soil2D data
def log_data_details(data, label, logger):
    logger.info(f"{label} data details:")
    for key, value in data.items():
        if 'soil' in key.lower() or '2d' in key.lower():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                logger.info(f"  {key}: shape={value.shape}, min={value.min()}, max={value.max()}, mean={value.mean()}")
                if value.shape[0] > 0:
                    logger.info(f"  {key} sample (first few elements): {value[0, :5]}")
                # Check for non-zero values
                non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                logger.info(f"  {key} non-zero count: {non_zero_count}")
            else:
                logger.info(f"  {key}: type={type(value)}, value={value}")


def main():
    """Main training function for CNP model."""
    parser = argparse.ArgumentParser(description='CNP Model Training')
    # turn off water for now
    # parser.add_argument(
    #     '--with-water',
    #     action='store_true',
    #     help='Include water variables in both input and output (default: no water)'
    # )
    parser.add_argument(
        '--log-level', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--output-dir',
        default='cnp_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--output-dir-suffix',
        default=None,
        type=str,
        metavar='SUFFIX',
        help='Optional suffix for the run folder name (e.g. natveg_improved -> run_YYYYMMDD_HHMMSS_natveg_improved)'
    )
    parser.add_argument(
        '--epochs', '--epoch',
        dest='epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: use value from --training-config-json, or 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    # Determinism and reproducibility controls
    parser.add_argument(
        '--strict-determinism',
        action='store_true',
        help='Enable strict deterministic settings (seed all RNGs, disable TF32, deterministic algorithms)'
    )
    parser.add_argument(
        '--dropout-p',
        type=float,
        default=None,
        help='Override global dropout probability in the model (e.g., 0.0 to disable)'
    )

    # Shuffling controls
    parser.add_argument(
        '--random-shuffle',
        action='store_true',
        help='Enable random shuffling for dataset rows and DataLoader (default: fixed seed shuffling)'
    )
    parser.add_argument(
        '--shuffle-seed',
        type=int,
        default=None,
        help='Optional seed to use when --random-shuffle is enabled (default: no fixed seed)'
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=None,
        help='Seed for train/val split shuffle (default: 42). Use a different value to get a different train/validation split.'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=None,
        metavar='RATIO',
        help='Fraction of data for training, 0-1 (default: 0.8). Remainder is validation.'
    )

    parser.add_argument(
        '--use-trendy1',
        action='store_true',
        help='Include Trendy_1_data_CNP dataset'
    )
    parser.add_argument(
        '--use-trendy05',
        action='store_true',
        help='Include Trend_05_data_CNP dataset'
    )
    parser.add_argument(
        '--use-tva4km',
        action='store_true',
        help='Include TVA4km dataset'
    )
    parser.add_argument(
        '--variable-list',
        type=str,
        default=None,
        help='Path to variable list file (e.g., CNP_IO_list_default.txt) for dynamic configuration'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to model config text file (e.g., CNP_model_config.txt) to override encoders/transformer/MLPs'
    )
    parser.add_argument(
        '--data-paths',
        type=str,
        default=None,
        help='Comma-separated list of directories containing training .pkl batches (overrides variable list)'
    )
    parser.add_argument(
        '--file-pattern',
        type=str,
        default=None,
        help='Glob pattern for training files (e.g., enhanced_1_training_data_batch_*.pkl)'
    )
    parser.add_argument(
        '--use-preprocessed-cache',
        action='store_true',
        help='Load/save preprocessed train/val tensors to skip pickle load+preprocess+normalize on reruns'
    )
    parser.add_argument(
        '--preprocessed-cache-dir',
        type=str,
        default=None,
        help='Directory for preprocessed tensor cache (default: <data_path>/.preprocessed_cache)'
    )
    parser.add_argument(
        '--rebuild-preprocessed-cache',
        action='store_true',
        help='Force rebuild of preprocessed cache even when a valid cache exists'
    )
    parser.add_argument(
        '--preprocessed-cache-only',
        action='store_true',
        help='Build or verify preprocessed cache and exit before model training'
    )
    parser.add_argument(
        '--tropical-only',
        action='store_true',
        help='Filter dataset to tropical latitude band before train/test split'
    )
    parser.add_argument(
        '--tropical-lat-range',
        type=str,
        default=None,
        help='Latitude range for tropical filter, format "min,max" (default: -23.5,23.5)'
    )
    parser.add_argument(
        '--tropical-lat-column',
        type=str,
        default=None,
        help='Latitude column name override (default: auto-detect from static columns)'
    )
    parser.add_argument(
        '--longitudes-to-drop',
        type=str,
        default=None,
        help='Comma-separated longitudes to drop from training (e.g. "0,358.75"). Overrides config/CNP_IO.'
    )
    parser.add_argument(
        '--natveg-only',
        action='store_true',
        help='Keep only gridcells with natural vegetation (PCT_NATVEG>0 and PCT_NAT_PFT_0<100). Overrides config.'
    )
    parser.add_argument(
        '--no-natveg-filter-before-split',
        action='store_true',
        help='With --natveg-only: split on full data then filter only training set to natveg, so test set matches no-filter run. Default: filter before split (legacy).'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to load for testing (default: all files)'
    )
    parser.add_argument(
        '--normalization',
        choices=['group', 'individual', 'hybrid'],
        default='individual',
        help='Normalization method: group, individual (per-variable, default), or hybrid (selective)'
    )
    parser.add_argument(
        '--xsmrpool-loss-weight',
        type=float,
        default=None,
        help='Extra loss weight applied to xsmrpool (non-positive pool)'
    )
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='Exit after building configuration; do not load data or train (for CI)'
    )
    parser.add_argument(
        '--litter-c-loss-weight',
        type=float,
        default=None,
        help='Extra loss weight multiplier for litter carbon vars (litr1/2/3c_vr)'
    )
    parser.add_argument(
        '--litter-n-loss-weight',
        type=float,
        default=None,
        help='Extra loss weight multiplier for litter nitrogen vars (litr1/2/3n_vr)'
    )
    parser.add_argument(
        '--pft-zero-sparsity-weight',
        type=float,
        default=None,
        help='Penalty weight for non-zero PFT1D predictions where target is zero (default: disabled)'
    )
    parser.add_argument(
        '--pft-zero-threshold',
        type=float,
        default=None,
        help='Threshold in normalized target space to treat PFT1D target as zero (default: 1e-8)'
    )
    parser.add_argument(
        '--pft-zero-sparsity-weights-json',
        type=str,
        default=None,
        help='Path to JSON mapping of per-variable sparsity weights (keys like cpool or Y_cpool)'
    )
    parser.add_argument(
        '--pft1d-activation',
        type=str,
        default=None,
        choices=['abs', 'relu', 'softplus', 'linear'],
        help='Default activation for PFT1D outputs'
    )
    parser.add_argument(
        '--pft1d-activation-overrides-json',
        type=str,
        default=None,
        help='Path to JSON mapping of per-variable PFT1D activations (keys like cpool or Y_cpool)'
    )
    parser.add_argument(
        '--tail-aware-vars',
        type=str,
        default=None,
        help='Comma-separated list of PFT1D variables to use tail-aware loss (e.g., cpool,deadstemc)'
    )
    parser.add_argument(
        '--tail-aware-vars-json',
        type=str,
        default=None,
        help='Path to JSON list of PFT1D variables to use tail-aware loss'
    )
    parser.add_argument(
        '--tail-aware-loss',
        type=str,
        default=None,
        choices=['log1p_mse', 'log1p_huber', 'log1p_quantile', 'mse'],
        help='Tail-aware loss type (default: log1p_mse)'
    )
    parser.add_argument(
        '--tail-aware-eps',
        type=float,
        default=None,
        help='Epsilon for tail-aware log1p loss'
    )
    parser.add_argument(
        '--tail-aware-weight',
        type=float,
        default=None,
        help='Base weight for tail-aware variables (multiplier)'
    )
    parser.add_argument(
        '--tail-aware-weights-json',
        type=str,
        default=None,
        help='Path to JSON mapping of per-variable tail-aware weights'
    )
    parser.add_argument(
        '--variable-weights-json',
        type=str,
        default=None,
        help='Path to JSON file with variable-specific loss weights. Expected format: {"pft1d_weights": {...}, "soil2d_weights": {...}, "scalar_weights": {...}}'
    )
    parser.add_argument(
        '--training-config-json',
        type=str,
        default=None,
        help='Path to unified JSON config file with all training settings. Expected format: {"variable_weights": {...}, "tail_aware_weights": {...}, "pft_zero_sparsity_weights": {...}, "pft1d_activation_overrides": {...}}. Individual JSON files take precedence if both are specified.'
    )
    parser.add_argument(
        '--tail-aware-huber-delta',
        type=float,
        default=None,
        help='Huber delta (beta) for log1p_huber loss'
    )
    parser.add_argument(
        '--tail-aware-quantile-tau',
        type=float,
        default=None,
        help='Quantile tau for log1p_quantile loss'
    )
    parser.add_argument(
        '--litter-p-loss-weight',
        type=float,
        default=None,
        help='Extra loss weight multiplier for litter phosphorus vars (litr1/2/3p_vr)'
    )
    parser.add_argument(
        '--mask-absent-pfts',
        dest='mask_absent_pfts',
        action='store_true',
        help='Zero predictions where PCT_NAT_PFT_k == 0 and exclude from loss'
    )
    parser.add_argument(
        '--no-mask-absent-pfts',
        dest='mask_absent_pfts',
        action='store_false',
        help='Disable masking of absent PFTs'
    )
    parser.set_defaults(mask_absent_pfts=True)
    parser.add_argument(
        '--pft-presence-threshold',
        dest='pft_presence_threshold',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Min PFT percent for training mask only (0 = pct>0; e.g. 2.0 = pct>=2%%). Inference always uses pct>0.'
    )
    args = parser.parse_args()
    
    # Create output directory with timestamp (optional suffix)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = (args.output_dir_suffix or '').strip().replace(' ', '_').replace('/', '_').strip('_')
    run_name = f"run_{timestamp}" + (f"_{suffix}" if suffix else "")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with timestamped log file in output directory (default INFO so log file is populated)
    log_file = output_dir / f"cnp_training_{timestamp}.log"
    setup_logging(str(log_file), args.log_level or 'INFO')
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Normalization method: {args.normalization}")
    if args.normalization == 'group':
        logger.info("Using group normalization (same as original system)")
    elif args.normalization == 'individual':
        logger.info("Using individual variable normalization (new method)")
    elif args.normalization == 'hybrid':
        logger.info("Using hybrid normalization (selective individual/group)")
    
    if not args.use_trendy1 and not args.use_trendy05 and not getattr(args, 'use_tva4km', False):
        args.use_trendy1 = True
        args.use_trendy05 = False

    try:
        # turn off water for now
        include_water = False
        # include_water = args.with_water
        logger.info(f"Water variables included: {include_water}")
        
        # Get configuration (support variable list and optional model-config overrides)
        from config.training_config import get_cnp_combined_config
        config = get_cnp_combined_config(
            use_trendy1=args.use_trendy1,
            use_trendy05=args.use_trendy05,
            use_tva4km=args.use_tva4km,
            max_files=args.max_files,
            include_water=include_water,
            variable_list_path=args.variable_list,
            model_config_path=args.model_config
        )
        # Optional overrides via CLI/env to avoid fixed variable list files
        env_data_paths = os.environ.get('DATA_PATHS') or os.environ.get('CNP_DATA_PATHS')
        env_file_pattern = os.environ.get('FILE_PATTERN') or os.environ.get('CNP_FILE_PATTERN')
        final_data_paths = args.data_paths if args.data_paths else env_data_paths
        final_file_pattern = args.file_pattern if args.file_pattern else env_file_pattern
        if final_data_paths or final_file_pattern:
            update_kwargs = {}
            if final_data_paths:
                # Support comma-separated paths
                update_kwargs['data_paths'] = [p.strip() for p in str(final_data_paths).split(',') if p.strip()]
            if final_file_pattern:
                update_kwargs['file_pattern'] = str(final_file_pattern).strip()
            try:
                config.update_data_config(**update_kwargs)
                logger.info(f"Applied data overrides: {update_kwargs}")
            except Exception as e:
                logger.warning(f"Failed to apply data overrides: {e}")
        load_workers_env = os.environ.get('LOAD_WORKERS')
        if load_workers_env is not None:
            try:
                config.update_data_config(load_workers=int(load_workers_env))
                logger.info(f"Applied parallel data load workers: {int(load_workers_env)}")
            except Exception as e:
                logger.warning(f"Failed to apply LOAD_WORKERS={load_workers_env}: {e}")
        # Optional tropical-only filtering
        if args.tropical_only:
            tropical_kwargs = {'tropical_only': True}
            if args.tropical_lat_range:
                try:
                    parts = [p.strip() for p in str(args.tropical_lat_range).split(',')]
                    if len(parts) == 2:
                        tropical_kwargs['tropical_lat_range'] = (float(parts[0]), float(parts[1]))
                    else:
                        logger.warning("Invalid --tropical-lat-range; expected format 'min,max'. Using default.")
                except Exception:
                    logger.warning("Failed to parse --tropical-lat-range; using default.")
            if args.tropical_lat_column:
                tropical_kwargs['tropical_lat_column'] = str(args.tropical_lat_column).strip()
            try:
                config.update_data_config(**tropical_kwargs)
                logger.info(f"Enabled tropical filtering: {tropical_kwargs}")
            except Exception as e:
                logger.warning(f"Failed to apply tropical filtering config: {e}")
        # Optional longitude filtering (CLI overrides config/CNP_IO)
        if args.longitudes_to_drop is not None:
            try:
                parts = [p.strip() for p in str(args.longitudes_to_drop).split(',') if p.strip()]
                longitudes = [float(x) for x in parts]
                config.update_data_config(longitudes_to_drop=longitudes)
                logger.info(f"Longitudes to drop (CLI): {longitudes}")
            except Exception as e:
                logger.warning(f"Failed to parse --longitudes-to-drop: {e}")
        # Optional natveg-only filtering (CLI overrides config)
        if args.natveg_only:
            config.update_data_config(natveg_only=True)
            logger.info("Enabled natveg-only filtering (PCT_NATVEG>0 and PCT_NAT_PFT_0<100).")
        if getattr(args, 'no_natveg_filter_before_split', False):
            config.update_data_config(natveg_filter_before_split=False)
            logger.info("Natveg filter applied after split (test set will match no-filter run).")
        if args.variable_list is not None:
            logger.info(f"Using CNP configuration from variable list file: {args.variable_list}")
        else:
            logger.info(f"Using default CNP variable configuration{' with water' if include_water else ' without water'}")
        if args.model_config is not None:
            logger.info(f"Applied model architecture overrides from: {args.model_config}")
        if args.pft_zero_sparsity_weight is not None or args.pft_zero_threshold is not None:
            update_kwargs = {}
            if args.pft_zero_sparsity_weight is not None:
                update_kwargs['pft_zero_sparsity_weight'] = float(args.pft_zero_sparsity_weight)
            if args.pft_zero_threshold is not None:
                update_kwargs['pft_zero_threshold'] = float(args.pft_zero_threshold)
            try:
                config.update_training_config(**update_kwargs)
                logger.info(f"Applied PFT zero sparsity settings: {update_kwargs}")
            except Exception as e:
                logger.warning(f"Failed to apply PFT zero sparsity settings: {e}")
        if args.pft_zero_sparsity_weights_json is not None:
            try:
                with open(args.pft_zero_sparsity_weights_json, 'r') as f:
                    weights = json.load(f)
                if isinstance(weights, dict):
                    config.update_training_config(pft_zero_sparsity_weights=weights)
                    logger.info(f"Applied per-variable PFT sparsity weights from: {args.pft_zero_sparsity_weights_json}")
            except Exception as e:
                logger.warning(f"Failed to load per-variable sparsity weights: {e}")
        if args.pft1d_activation is not None:
            try:
                config.update_model_config(pft1d_activation=str(args.pft1d_activation).lower())
                logger.info(f"Applied default PFT1D activation: {args.pft1d_activation}")
            except Exception as e:
                logger.warning(f"Failed to set default PFT1D activation: {e}")
        if args.pft1d_activation_overrides_json is not None:
            try:
                with open(args.pft1d_activation_overrides_json, 'r') as f:
                    overrides = json.load(f)
                if isinstance(overrides, dict):
                    config.update_model_config(pft1d_activation_overrides=overrides)
                    logger.info(f"Applied PFT1D activation overrides from: {args.pft1d_activation_overrides_json}")
            except Exception as e:
                logger.warning(f"Failed to load PFT1D activation overrides: {e}")
        if (args.tail_aware_vars or args.tail_aware_vars_json or args.tail_aware_loss or
            args.tail_aware_eps or args.tail_aware_weight or args.tail_aware_weights_json or
            args.tail_aware_huber_delta or args.tail_aware_quantile_tau):
            try:
                tail_vars = []
                if args.tail_aware_vars:
                    tail_vars = [v.strip() for v in str(args.tail_aware_vars).split(',') if v.strip()]
                if args.tail_aware_vars_json:
                    with open(args.tail_aware_vars_json, 'r') as f:
                        payload = json.load(f)
                    if isinstance(payload, list):
                        tail_vars = payload
                update_kwargs = {}
                if tail_vars:
                    update_kwargs['tail_aware_vars'] = tail_vars
                if args.tail_aware_loss is not None:
                    update_kwargs['tail_aware_loss'] = str(args.tail_aware_loss).lower()
                if args.tail_aware_eps is not None:
                    update_kwargs['tail_aware_epsilon'] = float(args.tail_aware_eps)
                if args.tail_aware_weight is not None:
                    update_kwargs['tail_aware_weight'] = float(args.tail_aware_weight)
                if args.tail_aware_weights_json is not None:
                    with open(args.tail_aware_weights_json, 'r') as f:
                        weights = json.load(f)
                    if isinstance(weights, dict):
                        update_kwargs['tail_aware_weights'] = weights
                if args.tail_aware_huber_delta is not None:
                    update_kwargs['tail_aware_huber_delta'] = float(args.tail_aware_huber_delta)
                if args.tail_aware_quantile_tau is not None:
                    update_kwargs['tail_aware_quantile_tau'] = float(args.tail_aware_quantile_tau)
                if update_kwargs:
                    config.update_training_config(**update_kwargs)
                    logger.info(f"Applied tail-aware loss settings: {update_kwargs}")
            except Exception as e:
                logger.warning(f"Failed to apply tail-aware loss settings: {e}")
        
        # Load unified training config JSON file if provided
        unified_config = None
        if args.training_config_json is not None:
            try:
                with open(args.training_config_json, 'r') as f:
                    unified_config = json.load(f)
                if isinstance(unified_config, dict):
                    logger.info(f"Loaded unified training config from: {args.training_config_json}")
                    
                    # Extract variable_weights section
                    if 'variable_weights' in unified_config:
                        config.update_training_config(variable_weights_json=args.training_config_json)
                        logger.info("Applied variable_weights from unified config")
                    
                    # Extract tail_aware_config section (loss type and epsilon)
                    if 'tail_aware_config' in unified_config:
                        tail_config = unified_config['tail_aware_config']
                        if isinstance(tail_config, dict):
                            update_kwargs = {}
                            # Set loss type if not already set via command line
                            if 'loss' in tail_config and args.tail_aware_loss is None:
                                update_kwargs['tail_aware_loss'] = str(tail_config['loss']).lower()
                            # Set epsilon if not already set via command line
                            if 'epsilon' in tail_config and args.tail_aware_eps is None:
                                update_kwargs['tail_aware_epsilon'] = float(tail_config['epsilon'])
                            if 'huber_delta' in tail_config and getattr(args, 'tail_aware_huber_delta', None) is None:
                                update_kwargs['tail_aware_huber_delta'] = float(tail_config['huber_delta'])
                            if update_kwargs:
                                config.update_training_config(**update_kwargs)
                                logger.info(f"Applied tail_aware_config from unified config: {update_kwargs}")
                    
                    # Extract tail_aware_weights section (only if not already set via individual file)
                    if 'tail_aware_weights' in unified_config and args.tail_aware_weights_json is None:
                        tail_weights = unified_config['tail_aware_weights']
                        if isinstance(tail_weights, dict):
                            # Merge with existing tail-aware settings
                            update_kwargs = {}
                            # Extract tail vars from weights keys if tail_aware_vars not set via command line
                            if not args.tail_aware_vars and not args.tail_aware_vars_json:
                                update_kwargs['tail_aware_vars'] = list(tail_weights.keys())
                            update_kwargs['tail_aware_weights'] = tail_weights
                            config.update_training_config(**update_kwargs)
                            logger.info("Applied tail_aware_weights from unified config")
                    
                    # Extract pft_zero_sparsity_config section (weight and threshold)
                    if 'pft_zero_sparsity_config' in unified_config:
                        sparsity_config = unified_config['pft_zero_sparsity_config']
                        if isinstance(sparsity_config, dict):
                            update_kwargs = {}
                            # Set weight if not already set via command line
                            if 'weight' in sparsity_config and args.pft_zero_sparsity_weight is None:
                                update_kwargs['pft_zero_sparsity_weight'] = float(sparsity_config['weight'])
                            # Set threshold if not already set via command line
                            if 'threshold' in sparsity_config and args.pft_zero_threshold is None:
                                update_kwargs['pft_zero_threshold'] = float(sparsity_config['threshold'])
                            if update_kwargs:
                                config.update_training_config(**update_kwargs)
                                logger.info(f"Applied pft_zero_sparsity_config from unified config: {update_kwargs}")
                    
                    # Extract pft_zero_sparsity_weights section (only if not already set)
                    if 'pft_zero_sparsity_weights' in unified_config and args.pft_zero_sparsity_weights_json is None:
                        sparsity_weights = unified_config['pft_zero_sparsity_weights']
                        if isinstance(sparsity_weights, dict):
                            config.update_training_config(pft_zero_sparsity_weights=sparsity_weights)
                            logger.info("Applied pft_zero_sparsity_weights from unified config")
                    
                    # Extract pft1d_activation_overrides section (only if not already set)
                    if 'pft1d_activation_overrides' in unified_config and args.pft1d_activation_overrides_json is None:
                        activation_overrides = unified_config['pft1d_activation_overrides']
                        if isinstance(activation_overrides, dict):
                            config.update_model_config(pft1d_activation_overrides=activation_overrides)
                            logger.info("Applied pft1d_activation_overrides from unified config")
                    
                    # Extract training_hyperparameters section (epochs, batch_size, learning_rate, optimizer, scheduler, loss weights)
                    if 'training_hyperparameters' in unified_config:
                        hyperparams = unified_config['training_hyperparameters']
                        if isinstance(hyperparams, dict):
                            update_kwargs = {}
                            # Basic training params (CLI takes precedence)
                            if 'num_epochs' in hyperparams and args.epochs is None:
                                update_kwargs['num_epochs'] = int(hyperparams['num_epochs'])
                            if 'batch_size' in hyperparams and args.batch_size is None:
                                update_kwargs['batch_size'] = int(hyperparams['batch_size'])
                            if 'learning_rate' in hyperparams and args.learning_rate is None:
                                update_kwargs['learning_rate'] = float(hyperparams['learning_rate'])
                            
                            # Optimizer settings
                            if 'optimizer_type' in hyperparams:
                                update_kwargs['optimizer_type'] = str(hyperparams['optimizer_type']).lower()
                            if 'weight_decay' in hyperparams:
                                update_kwargs['weight_decay'] = float(hyperparams['weight_decay'])
                            
                            # Scheduler settings
                            if 'use_scheduler' in hyperparams:
                                update_kwargs['use_scheduler'] = bool(hyperparams['use_scheduler'])
                            if 'scheduler_type' in hyperparams:
                                update_kwargs['scheduler_type'] = str(hyperparams['scheduler_type']).lower()
                            if 'scheduler_step_size' in hyperparams:
                                update_kwargs['scheduler_step_size'] = int(hyperparams['scheduler_step_size'])
                            if 'scheduler_gamma' in hyperparams:
                                update_kwargs['scheduler_gamma'] = float(hyperparams['scheduler_gamma'])
                            
                            # Loss weights (CLI takes precedence; use getattr so missing CLI args don't crash)
                            if 'scalar_loss_weight' in hyperparams and getattr(args, 'scalar_loss_weight', None) is None:
                                update_kwargs['scalar_loss_weight'] = float(hyperparams['scalar_loss_weight'])
                            if 'vector_loss_weight' in hyperparams and getattr(args, 'vector_loss_weight', None) is None:
                                update_kwargs['vector_loss_weight'] = float(hyperparams['vector_loss_weight'])
                            if 'matrix_loss_weight' in hyperparams and getattr(args, 'matrix_loss_weight', None) is None:
                                update_kwargs['matrix_loss_weight'] = float(hyperparams['matrix_loss_weight'])
                            if 'xsmrpool_loss_weight' in hyperparams and getattr(args, 'xsmrpool_loss_weight', None) is None:
                                update_kwargs['xsmrpool_loss_weight'] = float(hyperparams['xsmrpool_loss_weight'])
                            if 'litter_c_loss_weight' in hyperparams and getattr(args, 'litter_c_loss_weight', None) is None:
                                update_kwargs['litter_c_loss_weight'] = float(hyperparams['litter_c_loss_weight'])
                            if 'litter_n_loss_weight' in hyperparams and getattr(args, 'litter_n_loss_weight', None) is None:
                                update_kwargs['litter_n_loss_weight'] = float(hyperparams['litter_n_loss_weight'])
                            if 'litter_p_loss_weight' in hyperparams and getattr(args, 'litter_p_loss_weight', None) is None:
                                update_kwargs['litter_p_loss_weight'] = float(hyperparams['litter_p_loss_weight'])
                            
                            if update_kwargs:
                                config.update_training_config(**update_kwargs)
                                logger.info(f"Applied training_hyperparameters from unified config: {update_kwargs}")
                    
                    # Extract reproducibility_config section (random_seed, deterministic, train_split, normalization, dropout_p)
                    if 'reproducibility_config' in unified_config:
                        repro_config = unified_config['reproducibility_config']
                        if isinstance(repro_config, dict):
                            update_training_kwargs = {}
                            update_data_kwargs = {}
                            update_model_kwargs = {}
                            
                            # Random seed (CLI takes precedence)
                            if 'random_seed' in repro_config and args.split_seed is None and not args.random_shuffle:
                                seed_val = int(repro_config['random_seed'])
                                update_training_kwargs['random_seed'] = seed_val
                                update_data_kwargs['random_state'] = seed_val
                            
                            # Deterministic mode (CLI takes precedence)
                            if 'strict_determinism' in repro_config and not args.strict_determinism:
                                update_training_kwargs['deterministic'] = bool(repro_config['strict_determinism'])
                            
                            # Train split (CLI takes precedence)
                            if 'train_split' in repro_config and args.train_split is None:
                                update_data_kwargs['train_split'] = float(repro_config['train_split'])
                            
                            # Normalization - store for later application (CLI takes precedence)
                            # Note: normalization is applied via args.normalization later, so we store it
                            # and will check if args.normalization is default before applying
                            if 'normalization' in repro_config:
                                norm_val = str(repro_config['normalization']).lower()
                                if norm_val in ['group', 'individual', 'hybrid']:
                                    # Store in a way that can override args.normalization if it's still default
                                    # We'll apply this after checking if normalization was explicitly set via CLI
                                    repro_config['_normalization_from_config'] = norm_val
                            
                            # Dropout (CLI takes precedence; skip if null in JSON)
                            if 'dropout_p' in repro_config and args.dropout_p is None and repro_config['dropout_p'] is not None:
                                update_model_kwargs['dropout_p'] = float(repro_config['dropout_p'])
                            
                            if update_training_kwargs:
                                config.update_training_config(**update_training_kwargs)
                            if update_data_kwargs:
                                config.update_data_config(**update_data_kwargs)
                            if update_model_kwargs:
                                config.update_model_config(**update_model_kwargs)
                            
                            if update_training_kwargs or update_data_kwargs or update_model_kwargs:
                                logger.info(f"Applied reproducibility_config from unified config: training={update_training_kwargs}, data={update_data_kwargs}, model={update_model_kwargs}")
                            
                            # Store normalization for later application (if CLI didn't explicitly set it)
                            if 'normalization' in repro_config and args.normalization == 'individual':  # Default value
                                norm_val = str(repro_config['normalization']).lower()
                                if norm_val in ['group', 'individual', 'hybrid']:
                                    # Store on args for later use
                                    args._normalization_from_config = norm_val
                    
                    # Extract data_filtering_config section (tropical_only, tropical_lat_range)
                    if 'data_filtering_config' in unified_config:
                        filter_config = unified_config['data_filtering_config']
                        if isinstance(filter_config, dict):
                            update_kwargs = {}
                            
                            # Tropical filtering (CLI takes precedence)
                            if 'tropical_only' in filter_config and not args.tropical_only:
                                update_kwargs['tropical_only'] = bool(filter_config['tropical_only'])
                            if 'tropical_lat_range' in filter_config and args.tropical_lat_range is None:
                                lat_range = filter_config['tropical_lat_range']
                                if isinstance(lat_range, list) and len(lat_range) == 2:
                                    update_kwargs['tropical_lat_range'] = (float(lat_range[0]), float(lat_range[1]))
                                elif isinstance(lat_range, str):
                                    # Parse "min,max" format
                                    parts = [p.strip() for p in lat_range.split(',')]
                                    if len(parts) == 2:
                                        update_kwargs['tropical_lat_range'] = (float(parts[0]), float(parts[1]))
                            
                            # Longitude filtering: drop samples at these longitudes (config overrides CNP_IO)
                            if 'longitudes_to_drop' in filter_config and args.longitudes_to_drop is None:
                                lon_drop = filter_config['longitudes_to_drop']
                                if isinstance(lon_drop, list):
                                    update_kwargs['longitudes_to_drop'] = [float(x) for x in lon_drop]
                                elif isinstance(lon_drop, str):
                                    update_kwargs['longitudes_to_drop'] = [float(x.strip()) for x in lon_drop.split(',') if x.strip()]
                            
                            # Natveg-only: keep only PCT_NATVEG>0 and PCT_NAT_PFT_0<100 (CLI takes precedence)
                            if 'natveg_only' in filter_config and not args.natveg_only:
                                update_kwargs['natveg_only'] = bool(filter_config['natveg_only'])
                            if 'natveg_filter_before_split' in filter_config and not getattr(args, 'no_natveg_filter_before_split', False):
                                update_kwargs['natveg_filter_before_split'] = bool(filter_config['natveg_filter_before_split'])
                            if 'region_boxes' in filter_config:
                                boxes = filter_config['region_boxes']
                                if isinstance(boxes, (list, tuple)) and len(boxes) > 0:
                                    parsed = []
                                    for b in boxes:
                                        if isinstance(b, (list, tuple)) and len(b) >= 4:
                                            parsed.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                                    if parsed:
                                        update_kwargs['region_boxes'] = parsed
                            if update_kwargs:
                                config.update_data_config(**update_kwargs)
                                logger.info(f"Applied data_filtering_config from unified config: {update_kwargs}")
                    
                    # Extract pft_mask_config section (mask_absent_pfts and pft_presence_threshold)
                    if 'pft_mask_config' in unified_config:
                        mask_config = unified_config['pft_mask_config']
                        if isinstance(mask_config, dict):
                            update_kwargs = {}
                            # Set mask_absent_pfts if not already set via command line
                            if 'mask_absent_pfts' in mask_config and not hasattr(args, 'mask_absent_pfts') or args.mask_absent_pfts is None:
                                update_kwargs['mask_absent_pfts'] = bool(mask_config['mask_absent_pfts'])
                            # Set pft_presence_threshold if not already set via command line
                            if 'pft_presence_threshold' in mask_config and getattr(args, 'pft_presence_threshold', None) is None:
                                update_kwargs['pft_presence_threshold'] = float(mask_config['pft_presence_threshold'])
                            if update_kwargs:
                                config.update_training_config(**update_kwargs)
                                logger.info(f"Applied pft_mask_config from unified config: {update_kwargs}")
            except Exception as e:
                logger.warning(f"Failed to load unified training config: {e}")
        
        # Load variable weights from individual JSON file if provided (takes precedence over unified config)
        if args.variable_weights_json is not None:
            try:
                config.update_training_config(variable_weights_json=args.variable_weights_json)
                logger.info(f"Variable weights JSON file specified: {args.variable_weights_json}")
                # So that loss scale is consistent: if this JSON contains tail_aware_weights but we did
                # not load them via --training-config-json, apply them here (avoids ~18x loss difference).
                if args.training_config_json is None or args.training_config_json != args.variable_weights_json:
                    try:
                        with open(args.variable_weights_json, 'r') as f:
                            vw_data = json.load(f)
                        if isinstance(vw_data, dict):
                            if 'tail_aware_weights' in vw_data and isinstance(vw_data['tail_aware_weights'], dict):
                                tw = vw_data['tail_aware_weights']
                                update_kwargs = {
                                    'tail_aware_vars': list(tw.keys()),
                                    'tail_aware_weights': tw,
                                }
                                if 'tail_aware_config' in vw_data and isinstance(vw_data['tail_aware_config'], dict):
                                    tc = vw_data['tail_aware_config']
                                    if 'loss' in tc:
                                        update_kwargs['tail_aware_loss'] = str(tc['loss']).lower()
                                    if 'epsilon' in tc:
                                        update_kwargs['tail_aware_epsilon'] = float(tc['epsilon'])
                                # CNP ratio constraints from same JSON (e.g. use_cnp_ratio_constraints, cnp_ratio_constraint_weight)
                                for key in ('use_cnp_ratio_constraints', 'cnp_ratio_constraint_weight', 'cnp_ratio_tolerance'):
                                    if key in vw_data and hasattr(config.training_config, key):
                                        if key == 'use_cnp_ratio_constraints':
                                            update_kwargs[key] = bool(vw_data[key])
                                        elif key == 'cnp_ratio_constraint_weight':
                                            update_kwargs[key] = float(vw_data[key])
                                        elif key == 'cnp_ratio_tolerance':
                                            update_kwargs[key] = float(vw_data[key])
                                config.update_training_config(**update_kwargs)
                                logger.info("Applied tail_aware_vars, tail_aware_weights, and CNP ratio settings from variable_weights JSON (consistent loss scale)")
                            
                            # Also apply pft_zero_sparsity_config and weights if present
                            if 'pft_zero_sparsity_config' in vw_data and isinstance(vw_data['pft_zero_sparsity_config'], dict):
                                sparsity_config = vw_data['pft_zero_sparsity_config']
                                sparsity_update = {}
                                if 'weight' in sparsity_config:
                                    sparsity_update['pft_zero_sparsity_weight'] = float(sparsity_config['weight'])
                                if 'threshold' in sparsity_config:
                                    sparsity_update['pft_zero_threshold'] = float(sparsity_config['threshold'])
                                if sparsity_update:
                                    config.update_training_config(**sparsity_update)
                                    logger.info(f"Applied pft_zero_sparsity_config from variable_weights JSON: {sparsity_update}")
                            
                            if 'pft_zero_sparsity_weights' in vw_data and isinstance(vw_data['pft_zero_sparsity_weights'], dict):
                                config.update_training_config(pft_zero_sparsity_weights=vw_data['pft_zero_sparsity_weights'])
                                logger.info("Applied pft_zero_sparsity_weights from variable_weights JSON")
                    except Exception as e2:
                        logger.debug(f"Could not apply tail_aware from variable_weights JSON: {e2}")
            except Exception as e:
                logger.warning(f"Failed to set variable weights JSON path: {e}")
        
        # Set train/validation split (CLI takes precedence, but may have been set from config)
        train_split = args.train_split if args.train_split is not None else config.data_config.train_split
        config.update_data_config(train_split=train_split)
        if args.train_split is not None:
            logger.info(f"Train/validation split ratio: {train_split} (from --train-split)")
        else:
            logger.info(f"Train/validation split ratio: {train_split}")
        # Prefer GPU when available, otherwise CPU
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.update_training_config(device=device_str)
        # Apply max_files from CLI if provided (takes precedence over CNP_IO file)
        if args.max_files is not None:
            config.update_data_config(max_files=args.max_files)
            logger.info(f"Using MAX_FILES from CLI: {args.max_files}")
        elif config.data_config.max_files is not None:
            logger.info(f"Using MAX_FILES from CNP_IO file: {config.data_config.max_files}")
        # Turn off GPU monitoring and debug logging
        config.update_training_config(log_gpu_memory=False, log_gpu_utilization=False)
        # Override training parameters if specified (CLI overrides config; if CLI not set, use config)
        effective_lr = args.learning_rate if args.learning_rate is not None else config.training_config.learning_rate
        effective_epochs = args.epochs if args.epochs is not None else config.training_config.num_epochs
        config.update_training_config(
            num_epochs=effective_epochs,
            batch_size=args.batch_size,
            learning_rate=effective_lr,
            model_save_path=str(output_dir / "cnp_model.pt"),
            losses_save_path=str(output_dir / "cnp_training_losses.csv"),
            predictions_dir=str(output_dir / "cnp_predictions"),
            use_early_stopping=False
        )
        # Apply PFT mask settings (CLI takes precedence over config)
        if args.mask_absent_pfts:
            try:
                config.update_training_config(mask_absent_pfts=True)
                logger.info("Masking absent PFTs enabled (using PCT_NAT_PFT_1..16)")
            except Exception as e:
                logger.warning(f"Failed to enable mask_absent_pfts: {e}")
        elif hasattr(args, 'mask_absent_pfts') and args.mask_absent_pfts is False:
            # Explicitly disable if --no-mask-absent-pfts was used
            try:
                config.update_training_config(mask_absent_pfts=False)
                logger.info("Masking absent PFTs disabled")
            except Exception as e:
                logger.warning(f"Failed to disable mask_absent_pfts: {e}")
        # pft_presence_threshold: CLI takes precedence
        if getattr(args, 'pft_presence_threshold', None) is not None:
            try:
                config.update_training_config(pft_presence_threshold=float(args.pft_presence_threshold))
                logger.info("PFT presence threshold for training mask: pct >= %s (inference still uses pct > 0)", args.pft_presence_threshold)
            except Exception as e:
                logger.warning(f"Failed to set pft_presence_threshold: {e}")
        # apply xsmrpool loss weight from CLI if provided
        if args.xsmrpool_loss_weight is not None:
            try:
                config.update_training_config(xsmrpool_loss_weight=float(args.xsmrpool_loss_weight))
                logger.info(f"Using xsmrpool loss weight: {config.training_config.xsmrpool_loss_weight}")
            except Exception as e:
                logger.warning(f"Failed to set xsmrpool loss weight: {e}")
        # apply litter weights if provided
        try:
            if args.litter_c_loss_weight is not None:
                config.update_training_config(litter_c_loss_weight=float(args.litter_c_loss_weight))
                logger.info(f"Using litter C loss weight: {config.training_config.litter_c_loss_weight}")
            if args.litter_n_loss_weight is not None:
                config.update_training_config(litter_n_loss_weight=float(args.litter_n_loss_weight))
                logger.info(f"Using litter N loss weight: {config.training_config.litter_n_loss_weight}")
            if args.litter_p_loss_weight is not None:
                config.update_training_config(litter_p_loss_weight=float(args.litter_p_loss_weight))
                logger.info(f"Using litter P loss weight: {config.training_config.litter_p_loss_weight}")
        except Exception as e:
            logger.warning(f"Failed to set litter loss weights: {e}")
        logger.info(f"Effective learning rate for this run: {effective_lr}")

        # Shuffling policy: split-seed (train/val split), then fixed vs random for reproducibility
        if args.split_seed is not None:
            config.update_data_config(random_state=int(args.split_seed))
            config.update_training_config(random_seed=int(args.split_seed))
            logger.info(f"Train/validation split seed: {args.split_seed} (different split than default 42)")
        elif args.random_shuffle:
            # Use provided shuffle seed or system randomness
            if args.shuffle_seed is not None:
                config.update_data_config(random_state=int(args.shuffle_seed))
                config.update_training_config(random_seed=int(args.shuffle_seed))
                logger.info(f"Random shuffling enabled with shuffle_seed={args.shuffle_seed}")
            else:
                # Remove fixed seeds to allow non-deterministic shuffling
                # Keep a log message for provenance
                logger.info("Random shuffling enabled with no fixed seed (non-deterministic shuffles)")
                # Use a time-based seed for DataLoader generator consistency per run
                import time
                dyn_seed = int(time.time()) % (2**31 - 1)
                config.update_data_config(random_state=dyn_seed)
                config.update_training_config(random_seed=dyn_seed)
        else:
            # Keep fixed seeds for fair comparisons (default random_state=42)
            logger.info("Fixed shuffling (seeded) enabled for fair comparison")

        # If only validating configuration, exit early before heavy work (used in CI)
        if args.config_only:
            # Validate 2D output alignment invariant
            assert config.data_config.y_list_columns_2d == ['Y_' + v for v in config.data_config.x_list_columns_2d], \
                f"2D columns not aligned!\nX: {config.data_config.x_list_columns_2d}\nY: {config.data_config.y_list_columns_2d}"
            # Log a brief summary and exit
            logger.info("Configuration-only mode: built training/model/data configs successfully.")
            logger.info(f"Data paths: {config.data_config.data_paths}")
            logger.info(f"File pattern: {config.data_config.file_pattern}")
            logger.info(f"Epochs: {config.training_config.num_epochs}, Batch size: {config.training_config.batch_size}")
            return {
                'status': 'ok',
                'config_only': True
            }

        # Optional strict determinism (opt-in via CLI)
        if args.strict_determinism:
            seed = getattr(config.training_config, 'random_seed', 42)
            set_global_determinism(seed)
            logger.info(f"Strict determinism enabled with seed={seed}")
        else:
            logger.info("Strict determinism disabled (default). Running with standard PyTorch settings.")

        # Optional global dropout override for reproducibility testing
        if args.dropout_p is not None:
            try:
                old_p = getattr(config.model_config, 'dropout_p', None)
                config.update_model_config(dropout_p=float(args.dropout_p))
                logger.info(f"Global model dropout overridden: {old_p} -> {config.model_config.dropout_p}")
            except Exception as e:
                logger.warning(f"Failed to apply --dropout-p override: {e}")

        # --- FORCE 2D COLUMN ALIGNMENT FOR SAFETY ---
        #aligned_2d_vars = [
        #    'cwdc_vr', 'cwdn_vr', 'secondp_vr', 'cwdp_vr',
        #    'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
        #    'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
        #    'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
        #    'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
        #    'soil1c_vr', 'soil2c_vr', 'soil3c_vr', 'soil4c_vr',
        #    'soil1n_vr', 'soil2n_vr', 'soil3n_vr', 'soil4n_vr',
        #    'soil1p_vr', 'soil2p_vr', 'soil3p_vr', 'soil4p_vr'
        #]
        #config.data_config.x_list_columns_2d = aligned_2d_vars
        #config.data_config.y_list_columns_2d = ['Y_' + v for v in aligned_2d_vars]
        # print('x_list_columns_2d (forced):', config.data_config.x_list_columns_2d)
        # print('y_list_columns_2d (forced):', config.data_config.y_list_columns_2d)
        assert config.data_config.y_list_columns_2d == ['Y_' + v for v in config.data_config.x_list_columns_2d], \
            f"2D columns not aligned!\nX: {config.data_config.x_list_columns_2d}\nY: {config.data_config.y_list_columns_2d}"

        normalization_method = args.normalization
        if hasattr(args, '_normalization_from_config') and args._normalization_from_config:
            normalization_method = args._normalization_from_config
            logger.info(f"Using normalization from config: {normalization_method}")

        use_preprocessed_cache = (
            args.use_preprocessed_cache
            or args.preprocessed_cache_only
            or env_flag_enabled('USE_PREPROCESSED_CACHE')
        )
        rebuild_preprocessed_cache = (
            args.rebuild_preprocessed_cache
            or args.preprocessed_cache_only
            or env_flag_enabled('REBUILD_PREPROCESSED_CACHE')
        )
        cache_dir_arg = args.preprocessed_cache_dir or os.environ.get('PREPROCESSED_CACHE_DIR')
        pft_presence_threshold = float(
            getattr(config.preprocessing_config, 'pft_presence_threshold', None)
            if getattr(config.preprocessing_config, 'pft_presence_threshold', None) is not None
            else getattr(config.training_config, 'pft_presence_threshold', 0.0)
            or 0.0
        )

        cache_fingerprint = None
        cache_manifest = None
        cache_file_paths = None
        cached_bundle = None
        if use_preprocessed_cache:
            data_files = list_data_files(config.data_config)
            cache_fingerprint, cache_manifest = compute_cache_fingerprint(
                config.data_config,
                config.preprocessing_config,
                normalization_method=normalization_method,
                pft_presence_threshold=pft_presence_threshold,
                files=data_files,
            )
            cache_dir = resolve_cache_dir(cache_dir_arg, config.data_config)
            cache_file_paths = cache_paths(cache_dir, cache_fingerprint)
            logger.info(
                "Preprocessed cache enabled (fingerprint=%s..., dir=%s)",
                cache_fingerprint[:12],
                cache_dir,
            )
            if (
                cache_is_valid(cache_file_paths, cache_fingerprint)
                and not rebuild_preprocessed_cache
            ):
                cached_bundle = load_preprocessed_cache(cache_file_paths)
            elif rebuild_preprocessed_cache:
                logger.info("Rebuilding preprocessed cache (forced)")
            else:
                logger.info("Preprocessed cache miss; running full data pipeline")

        logger.info("Using DataLoaderIndividual for consistent PFT indexing")
        data_loader = DataLoaderIndividual(
            config.data_config,
            config.preprocessing_config
        )

        split_data = None
        normalized_scalers = None
        data_info = None

        if cached_bundle is not None:
            split_data = {
                'train': cached_bundle['train'],
                'test': cached_bundle['test'],
                'train_size': cached_bundle.get('train_size'),
                'test_size': cached_bundle.get('test_size'),
            }
            normalized_scalers = cached_bundle['scalers']
            data_info = cached_bundle['data_info']
            logger.info(
                "Skipped pickle load, preprocess, normalize, and split (preprocessed cache hit)"
            )
        else:
            logger.info("Loading data...")
            raw_data = data_loader.load_data()

            if hasattr(data_loader, 'df') and isinstance(data_loader.df, pd.DataFrame):
                logger.info("=" * 80)
                logger.info("训练数据集变量列表 (Training Dataset Variables):")
                logger.info("=" * 80)
                logger.info(f"总变量数: {len(data_loader.df.columns)}")
                logger.info(f"数据集形状: {data_loader.df.shape}")
                logger.info("\n所有变量列表 (All Variables):")
                for i, col in enumerate(sorted(data_loader.df.columns), 1):
                    logger.info(f"  {i:4d}. {col}")
                logger.info("=" * 80)

            logger.info("Checking raw data for soil2D variables...")
            for key, value in raw_data.items():
                if 'soil' in key.lower() and '2d' in key.lower():
                    if isinstance(value, pd.Series):
                        non_zero_count = 0
                        total_items = len(value)
                        for item in value:
                            if isinstance(item, (list, tuple, np.ndarray)):
                                if isinstance(item, np.ndarray):
                                    non_zero_count += np.count_nonzero(item)
                                else:
                                    for sub_item in item:
                                        if isinstance(sub_item, (list, tuple, np.ndarray)):
                                            if isinstance(sub_item, np.ndarray):
                                                non_zero_count += np.count_nonzero(sub_item)
                                            else:
                                                non_zero_count += sum(1 for x in sub_item if x != 0)
                                        else:
                                            non_zero_count += 1 if sub_item != 0 else 0
                            else:
                                non_zero_count += 1 if item != 0 else 0
                        logger.info(f"  Raw {key}: total items={total_items}, non-zero count={non_zero_count}")
                    else:
                        logger.info(f"  Raw {key}: type={type(value)}")

            if hasattr(data_loader, 'df') and isinstance(data_loader.df, pd.DataFrame):
                verify_locations(data_loader.df, "All the data")

            logger.info("Preprocessing data...")
            preprocessed_data = data_loader.preprocess_data()
            logger.info("Data preprocessed successfully.")
            data_info = data_loader.get_data_info()
            logger.info("Checking preprocessed data for soil2D variables...")
            if preprocessed_data is not None:
                for key, value in preprocessed_data.items():
                    if 'soil' in key.lower() and '2d' in key.lower():
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                            logger.info(f"  Preprocessed {key}: shape={value.shape}, non-zero count={non_zero_count}")
                        else:
                            logger.info(f"  Preprocessed {key}: type={type(value)}")
            else:
                logger.warning("Preprocessed data is None, skipping check.")

            logger.info("Normalizing data...")
            if normalization_method == 'group':
                normalized_data = data_loader.normalize_data()
                logger.info("Applied group normalization (same as original system)")
            elif normalization_method == 'individual':
                normalized_data = data_loader.normalize_data_individual()
                logger.info("Applied individual normalization to all variables")
                logger.info("Checking data after individual normalization for soil2D variables...")
                for key, value in normalized_data.items():
                    if 'soil' in key.lower() and '2d' in key.lower():
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                            logger.info(f"  After Individual Normalization {key}: shape={value.shape}, non-zero count={non_zero_count}")
                        else:
                            logger.info(f"  After Individual Normalization {key}: type={type(value)}")
            else:
                normalized_data = data_loader.normalize_data_hybrid(
                    use_individual_for=['scalar', 'pft_1d', 'soil_2d', 'y_scalar', 'y_pft_1d', 'y_soil_2d'],
                    group_soil_vars=['sminn_vr', 'smin_no3_vr', 'smin_nh4_vr']
                )
                logger.info("Applied hybrid normalization: individual for most variables, group for soil minerals variables")
                if 'xsmrpool_loss_weight' not in config.__dict__:
                    config.xsmrpool_loss_weight = 2.0
                if 'soil_mineral_loss_weight' not in config.__dict__:
                    config.soil_mineral_loss_weight = 1.5

            logger.info("Data normalized successfully.")
            logger.info("Checking normalized data for soil2D variables...")
            for key, value in normalized_data.items():
                if 'soil' in key.lower() and '2d' in key.lower():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                        logger.info(f"  Normalized {key}: shape={value.shape}, non-zero count={non_zero_count}")
                    else:
                        logger.info(f"  Normalized {key}: type={type(value)}")

            logger.info("Splitting data into train and test sets...")
            split_data = data_loader.split_data(normalized_data)
            normalized_scalers = normalized_data['scalers']
            logger.info("Data split successfully.")

            if hasattr(data_loader, 'df') and isinstance(data_loader.df, pd.DataFrame):
                train_indices = data_loader.train_indices
                test_indices = data_loader.test_indices
                train_df = data_loader.df.iloc[train_indices]
                val_df = data_loader.df.iloc[test_indices]
                verify_locations(train_df, "Training data")
                verify_locations(val_df, "Validation data")
            else:
                logger.warning("Cannot verify training/validation locations: loader does not have a DataFrame attribute 'df'")

            logger.info("Checking train data for soil2D variables...")
            for key, value in split_data['train'].items():
                if 'soil' in key.lower() and '2d' in key.lower():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                        logger.info(f"  Train {key}: shape={value.shape}, non-zero count={non_zero_count}")
                    else:
                        logger.info(f"  Train {key}: type={type(value)}")
            logger.info("Checking test data for soil2D variables...")
            for key, value in split_data['test'].items():
                if 'soil' in key.lower() and '2d' in key.lower():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                        logger.info(f"  Test {key}: shape={value.shape}, non-zero count={non_zero_count}")
                    else:
                        logger.info(f"  Test {key}: type={type(value)}")

            if use_preprocessed_cache and cache_file_paths is not None and cache_fingerprint is not None:
                save_preprocessed_cache(
                    cache_file_paths,
                    fingerprint=cache_fingerprint,
                    manifest=cache_manifest,
                    split_data=split_data,
                    scalers=normalized_scalers,
                    data_info=data_info,
                )

        if args.preprocessed_cache_only or env_flag_enabled('PREPROCESSED_CACHE_ONLY'):
            if use_preprocessed_cache and cache_file_paths is not None:
                logger.info(
                    "Preprocessed cache ready at %s",
                    cache_file_paths['tensors'],
                )
            else:
                logger.warning(
                    "preprocessed-cache-only requested but cache is not enabled; "
                    "use --use-preprocessed-cache or USE_PREPROCESSED_CACHE=1"
                )
            return {
                'status': 'ok',
                'preprocessed_cache_only': True,
                'cache_tensors': str(cache_file_paths['tensors']) if cache_file_paths else None,
                'train_samples': int(split_data['train']['time_series'].shape[0]),
                'test_samples': int(split_data['test']['time_series'].shape[0]),
            }

        # Create CNP model
        logger.info("Creating CNP model...")

        model = CNPCombinedModel(
            config.model_config,
            data_info,
            include_water=include_water,
            use_learnable_loss_weights=config.training_config.use_learnable_loss_weights
        )
        
        # Ensure proper initialization for all layers
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Lower gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # After preprocessing and before normalization
        # data_info = data_loader.get_data_info() # This line is moved up
        # Initialize trainer with data_info
        trainer = ModelTrainer(
            config.training_config,
            model,
            split_data['train'],
            split_data['test'],
            normalized_scalers,
            data_info
        )
        
        # Run training pipeline
        logger.info("Starting CNP model training pipeline...")
        results = trainer.run_training_pipeline()
        
        logger.info("CNP model training completed successfully!")
        logger.info(f"Final metrics: {results['metrics']}")
        
        # Save results to output directory

        with open(output_dir / "cnp_metrics.json", "w") as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save model configuration
        with open(output_dir / "cnp_config.json", "w") as f:
            # Derive per-group counts for quick comparison with CNP_IO.txt
            di = data_info if isinstance(data_info, dict) else {}
            time_series_cols = di.get('time_series_columns', []) or []
            static_cols = di.get('static_columns', []) or []
            pft_param_cols = di.get('pft_param_columns', []) or []
            scalar_in_cols = di.get('x_list_scalar_columns', []) or []
            scalar_out_cols = di.get('y_list_scalar_columns', []) or []
            pft1d_in_vars = di.get('variables_1d_pft', []) or []
            pft1d_out_vars = di.get('y_list_columns_1d', []) or []
            soil2d_in_vars = di.get('x_list_columns_2d', []) or []
            soil2d_out_vars = di.get('y_list_columns_2d', []) or []

            group_counts = {
                'time_series_variables': len(time_series_cols),
                'static_columns': len(static_cols),
                'pft_param_columns': len(pft_param_cols),
                'scalar_variables_in': len(scalar_in_cols),
                'scalar_variables_out': len(scalar_out_cols),
                'pft_1d_variables_in': len(pft1d_in_vars),
                'pft_1d_variables_out': len(pft1d_out_vars),
                'soil_2d_variables_in': len(soil2d_in_vars),
                'soil_2d_variables_out': len(soil2d_out_vars),
                'total_predicted_variables': len(scalar_out_cols) + len(pft1d_out_vars) + len(soil2d_out_vars)
            }

            # Expanded prediction element counts (PFTs and Soil layers)
            # PFTs per variable use model_config.vector_length (expected 16: PFT1..PFT16)
            pfts_per_var = int(getattr(config.model_config, 'vector_length', 16) or 16)
            soil_rows_per_var = int(getattr(config.model_config, 'matrix_rows', 1) or 1)
            soil_layers_per_var = int(getattr(config.model_config, 'matrix_cols', 10) or 10)
            prediction_element_counts = {
                'pft_1d': {
                    'variables_out': len(pft1d_out_vars),
                    'pfts_per_variable': pfts_per_var,
                    'total_elements': len(pft1d_out_vars) * pfts_per_var
                },
                'soil_2d': {
                    'variables_out': len(soil2d_out_vars),
                    'columns_per_variable': soil_rows_per_var,
                    'layers_per_variable': soil_layers_per_var,
                    'total_elements': len(soil2d_out_vars) * soil_rows_per_var * soil_layers_per_var
                },
                'scalar_1d': {
                    'variables_out': len(scalar_out_cols)
                }
            }

            # Save data_config (paths and pattern) so inference can use the same data as training
            data_cfg = getattr(config, 'data_config', None)
            data_config_snapshot = None
            if data_cfg is not None:
                tr = getattr(data_cfg, 'tropical_lat_range', (-23.5, 23.5))
                if tr is None:
                    tr = (-23.5, 23.5)
                data_config_snapshot = {
                    'data_paths': list(getattr(data_cfg, 'data_paths', []) or []),
                    'file_pattern': getattr(data_cfg, 'file_pattern', None) or 'enhanced_1_training_data_batch_*.pkl',
                    'dataset_file_patterns': dict(getattr(data_cfg, 'dataset_file_patterns', None) or {}),
                    'longitudes_to_drop': list(getattr(data_cfg, 'longitudes_to_drop', None) or []),
                    'natveg_only': bool(getattr(data_cfg, 'natveg_only', False)),
                    'natveg_filter_before_split': bool(getattr(data_cfg, 'natveg_filter_before_split', True)),
                    'tropical_only': bool(getattr(data_cfg, 'tropical_only', False)),
                    'tropical_lat_range': [float(tr[0]), float(tr[1])],
                }
            config_dict = {
                'include_water': include_water,
                'normalization_method': args.normalization,
                'data_info': data_info,
                'data_counts': group_counts,
                'prediction_element_counts': prediction_element_counts,
                'model_config': config.model_config.__dict__,
                'training_config': config.training_config.__dict__,
                'data_config': data_config_snapshot,
                # Model-config provenance for verification
                'model_config_source': getattr(config, 'model_config_source', None),
                'model_config_overrides_keys': getattr(config, 'model_config_overrides_keys', None)
            }
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"CNP model results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"CNP model training failed: {e}")
        raise


if __name__ == "__main__":
    main() 