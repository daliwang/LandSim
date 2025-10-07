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
from models.cnp_combined_model import CNPCombinedModel
from training.trainer import ModelTrainer

from scripts.run_inference_all import verify_locations

def setup_logging(log_file: str, level: str = 'INFO') -> None:
    """Set up logging configuration with a specific log file."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


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
        '--epochs',
        type=int,
        default=150,
        help='Number of training epochs'
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
        '--litter-p-loss-weight',
        type=float,
        default=None,
        help='Extra loss weight multiplier for litter phosphorus vars (litr1/2/3p_vr)'
    )
    parser.add_argument(
        '--mask-absent-pfts',
        action='store_true',
        help='Zero predictions where PCT_NAT_PFT_k == 0 and exclude from loss'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with timestamped log file in output directory
    log_file = output_dir / f"cnp_training_{timestamp}.log"
    setup_logging(str(log_file), args.log_level if args.log_level else 'WARNING')
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Normalization method: {args.normalization}")
    if args.normalization == 'group':
        logger.info("Using group normalization (same as original system)")
    elif args.normalization == 'individual':
        logger.info("Using individual variable normalization (new method)")
    elif args.normalization == 'hybrid':
        logger.info("Using hybrid normalization (selective individual/group)")
    
    if not args.use_trendy1 and not args.use_trendy05:
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
            max_files=args.max_files,
            include_water=include_water,
            variable_list_path=args.variable_list,
            model_config_path=args.model_config
        )
        if args.variable_list is not None:
            logger.info(f"Using CNP configuration from variable list file: {args.variable_list}")
        else:
            logger.info(f"Using default CNP variable configuration{' with water' if include_water else ' without water'}")
        if args.model_config is not None:
            logger.info(f"Applied model architecture overrides from: {args.model_config}")
        # Set train/validation split to 50/50
        config.update_data_config(train_split=0.8)
        # Ensure GPU and all files
        config.update_training_config(device='cuda')
        config.update_data_config(max_files=args.max_files)
        # Turn off GPU monitoring and debug logging
        config.update_training_config(log_gpu_memory=False, log_gpu_utilization=False)
        # Override training parameters if specified
        effective_lr = args.learning_rate if args.learning_rate is not None else config.training_config.learning_rate
        config.update_training_config(
            num_epochs=args.epochs,  
            batch_size=args.batch_size,
            learning_rate=effective_lr,
            model_save_path=str(output_dir / "cnp_model.pt"),
            losses_save_path=str(output_dir / "cnp_training_losses.csv"),
            predictions_dir=str(output_dir / "cnp_predictions"),
            use_early_stopping=False
        )
        if args.mask_absent_pfts:
            try:
                config.update_training_config(mask_absent_pfts=True)
                logger.info("Masking absent PFTs enabled (using PCT_NAT_PFT_1..16)")
            except Exception as e:
                logger.warning(f"Failed to enable mask_absent_pfts: {e}")
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

        # Initialize data loader
        logger.info("Loading data...")
        logger.info("Using DataLoaderIndividual for consistent PFT indexing")
        data_loader = DataLoaderIndividual(
            config.data_config,
            config.preprocessing_config
        )
        # Check raw data for non-zero values after loading
        raw_data = data_loader.load_data()
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

        # After loading data
        if hasattr(data_loader, 'df') and isinstance(data_loader.df, pd.DataFrame):
            verify_locations(data_loader.df, "All the data")

        # Proceed with preprocessing and normalization
        logger.info("Preprocessing data...")
        preprocessed_data = data_loader.preprocess_data()
        logger.info("Data preprocessed successfully.")
        data_info = data_loader.get_data_info()
        # Check preprocessed data for soil2D variables
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
        # Normalize data
        logger.info("Normalizing data...")
        if args.normalization == 'group':
            normalized_data = data_loader.normalize_data()
            logger.info("Applied group normalization (same as original system)")
        elif args.normalization == 'individual':
            normalized_data = data_loader.normalize_data_individual()
            logger.info("Applied individual normalization to all variables")
            # Log after individual normalization for soil2D
            logger.info("Checking data after individual normalization for soil2D variables...")
            for key, value in normalized_data.items():
                if 'soil' in key.lower() and '2d' in key.lower():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                        logger.info(f"  After Individual Normalization {key}: shape={value.shape}, non-zero count={non_zero_count}")
                    else:
                        logger.info(f"  After Individual Normalization {key}: type={type(value)}")
        else:  # hybrid
            normalized_data = data_loader.normalize_data_hybrid(
                use_individual_for=['scalar', 'pft_1d', 'soil_2d', 'y_scalar', 'y_pft_1d', 'y_soil_2d'],
                group_soil_vars=['sminn_vr', 'smin_no3_vr', 'smin_nh4_vr']
            )
            logger.info("Applied hybrid normalization: individual for most variables, group for soil minerals variables")

            # Add xsmrpool-specific loss weighting
            if 'xsmrpool_loss_weight' not in config.__dict__:
                config.xsmrpool_loss_weight = 2.0  # Increase weight for xsmrpool
            if 'soil_mineral_loss_weight' not in config.__dict__:
                config.soil_mineral_loss_weight = 1.5  # Increase weight for soil minerals
        logger.info("Data normalized successfully.")
        # Log details of normalized data for soil2D variables
        logger.info("Checking normalized data for soil2D variables...")
        for key, value in normalized_data.items():
            if 'soil' in key.lower() and '2d' in key.lower():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    non_zero_count = np.count_nonzero(value) if isinstance(value, np.ndarray) else torch.count_nonzero(value).item()
                    logger.info(f"  Normalized {key}: shape={value.shape}, non-zero count={non_zero_count}")
                else:
                    logger.info(f"  Normalized {key}: type={type(value)}")
        # Split data
        logger.info("Splitting data into train and test sets...")
        split_data = data_loader.split_data(normalized_data)
        logger.info("Data split successfully.")

        # Verify locations in training and validation data
        if hasattr(data_loader, 'df') and isinstance(data_loader.df, pd.DataFrame):
            # Get indices used for training and validation
            train_indices = data_loader.train_indices
            test_indices = data_loader.test_indices
            
            # Extract training and validation dataframes
            train_df = data_loader.df.iloc[train_indices]
            val_df = data_loader.df.iloc[test_indices]
            
            # Verify locations
            verify_locations(train_df, "Training data")
            verify_locations(val_df, "Validation data")
        else:
            logger.warning("Cannot verify training/validation locations: loader does not have a DataFrame attribute 'df'")

        # Log details of split data for soil2D variables
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
        
        # print(f"Training samples: {train_data['time_series'].shape[0]}")
        # print(f"Validation/Evaluation samples: {test_data['time_series'].shape[0]}")
        
        # Debug print for pft_param
        # if 'pft_param' in train_data:
        #     print('[DEBUG] pft_param shape (train):', train_data['pft_param'].shape)
        # else:
        #     print('[DEBUG] pft_param not found in train split!')
        # if 'pft_param' in test_data:
        #     print('[DEBUG] pft_param shape (test):', test_data['pft_param'].shape)
        # else:
        #     print('[DEBUG] pft_param not found in test split!')

        # Debug: Print train_data keys and check for y_scalar
        # print('DEBUG: train_data keys:', train_data.keys())
        # print('DEBUG: y_scalar in train_data:', 'y_scalar' in train_data)

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
            normalized_data['scalers'],
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
            config_dict = {
                'include_water': include_water,
                'normalization_method': args.normalization,
                'data_info': data_info,
                'model_config': config.model_config.__dict__,
                'training_config': config.training_config.__dict__
            }
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"CNP model results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"CNP model training failed: {e}")
        raise


if __name__ == "__main__":
    main() 