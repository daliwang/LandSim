"""
Flexible trainer module for model training.

This module provides a flexible training framework that can handle
different model configurations, data types, and training strategies.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import mean_squared_error
from pathlib import Path
import time
from tqdm import tqdm
import warnings
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
import json

# from config.training_config import TrainingConfig  # Uncomment if TrainingConfig is defined
from models.combined_model import CombinedModel, FlexibleCombinedModel
from models.cnp_combined_model import CNPCombinedModel
from config.variable_weights import get_pft1d_variable_weights, get_soil2d_variable_weights, get_scalar_variable_weights

# Import GPU monitoring
from utils.gpu_monitor import GPUMonitor, log_memory_usage

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Flexible trainer for climate model training.
    
    This class handles the complete training pipeline including
    data preparation, model training, validation, and saving results.
    """
    
    def __init__(self, training_config: Any, model: nn.Module, 
                 train_data: Dict[str, Any], test_data: Dict[str, Any],
                 scalers: Dict[str, Any], data_info: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            training_config: Training configuration
            model: Model to train
            train_data: Training data
            test_data: Test data
            scalers: Data scalers for inverse transformation
            data_info: Information about the data structure
        """
        self.config = training_config
        self.device = self.config.get_device()
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.scalers = scalers
        self.data_info = data_info

        # --- ENFORCE COLUMN ORDER CONSISTENCY ---
        # These lists should be used everywhere for model input/output order
        self.scalar_columns = self.data_info.get('x_list_scalar_columns', [])
        self.y_scalar_columns = self.data_info.get('y_list_scalar_columns', [])
        self.variables_1d_pft_columns = self.data_info.get('variables_1d_pft', [])  # Canonical 1D PFT variable list
        self.y_pft_1d_columns = self.data_info.get('y_list_columns_1d', [])
        self.variables_2d_soil_columns = self.data_info.get('x_list_columns_2d', [])
        self.y_soil_2d_columns = self.data_info.get('y_list_columns_2d', [])
        self.pft_param_columns = self.data_info.get('pft_param_columns', [])
        
        # Set random seeds for reproducibility
        if hasattr(self.config, 'random_seed'):
            torch.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            logger.info(f"Random seed set to {self.config.random_seed}")
        
        # Set deterministic behavior if requested
        if hasattr(self.config, 'deterministic') and self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Deterministic behavior enabled")
        
        # Ensure all arrays in train/test splits have the same number of samples
        final_tensor_keys = [
            'time_series', 'static', 'pft_param', 'scalar',
            'variables_1d_pft', 'variables_2d_soil',
            'y_scalar', 'y_pft_1d', 'y_soil_2d',
            'water', 'y_water'
        ]

        # DataLoader generator for deterministic shuffling
        self._torch_generator = torch.Generator(device='cpu')
        if hasattr(self.config, 'random_seed'):
            self._torch_generator.manual_seed(self.config.random_seed)
        
        # --- DEBUG: Force small batch size and minimal DataLoader workers for OOM debugging ---
        # print the shape of the tensors in the train and test data
        # print(f"[DEBUG] train_data['time_series'].shape: {self.train_data['time_series'].shape}")
        # print(f"[DEBUG] train_data['static'].shape: {self.train_data['static'].shape}")
        # print(f"[DEBUG] train_data['pft_param'].shape: {self.train_data['pft_param'].shape}")
        # print(f"[DEBUG] train_data['scalar'].shape: {self.train_data['scalar'].shape}")
        # print(f"[DEBUG] train_data['variables_1d_pft'].shape: {self.train_data['variables_1d_pft'].shape}")
        # print(f"[DEBUG] train_data['variables_2d_soil'].shape: {self.train_data['variables_2d_soil'].shape}")

        # print(f"[DEBUG] test_data['time_series'].shape: {self.test_data['time_series'].shape}")
        # print(f"[DEBUG] test_data['static'].shape: {self.test_data['static'].shape}")
        # print(f"[DEBUG] test_data['pft_param'].shape: {self.test_data['pft_param'].shape}")
        # print(f"[DEBUG] test_data['scalar'].shape: {self.test_data['scalar'].shape}")
        # print(f"[DEBUG] test_data['variables_1d_pft'].shape: {self.test_data['variables_1d_pft'].shape}")
        # print(f"[DEBUG] test_data['variables_2d_soil'].shape: {self.test_data['variables_2d_soil'].shape}")

        # --- DEBUG: Force small batch size and minimal DataLoader workers for OOM debugging ---

        for split_name in ['train', 'test']:
            split = getattr(self, f'{split_name}_data')
            keys_present = [k for k in final_tensor_keys if k in split and isinstance(split[k], torch.Tensor)]
            if not keys_present:
                continue
            min_length = min(split[k].shape[0] for k in keys_present)
            for k in keys_present:
                if split[k].shape[0] != min_length:
                    split[k] = split[k][:min_length]

        # Move all final tensors to device (for both train and test splits)
        for split_name in ['train', 'test']:
            split = getattr(self, f'{split_name}_data')
            for key in final_tensor_keys:
                if key in split:
                    split[key] = split[key].to(self.device)

        # Setup device for model
        self.model.to(self.device)
        try:
            first_param = next(self.model.parameters())
            logger.info(f"Runtime device check: device={self.device}, cuda_available={torch.cuda.is_available()}, model_param_device={first_param.device}")
            if torch.cuda.is_available() and self.device.type == 'cuda':
                current_idx = torch.cuda.current_device()
                logger.info(f"CUDA device index={current_idx}, name={torch.cuda.get_device_name(current_idx)}")
        except StopIteration:
            logger.warning("Model has no parameters to check device placement")
        
        # Initialize GPU monitoring
        self.gpu_monitor = GPUMonitor(self.device)
        
        # Setup mixed precision training
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Automatic Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision disabled for fair comparison")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        # Loss weights from config (defaults)
        self.scalar_loss_weight = getattr(self.config, 'scalar_loss_weight', 1.0)
        self.vector_loss_weight = getattr(self.config, 'vector_loss_weight', 1.0)
        self.matrix_loss_weight = getattr(self.config, 'matrix_loss_weight', 1.0)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Ensure all tensors have the same batch size for TensorDataset
        self._ensure_consistent_batch_sizes()
        
        # Log initial GPU stats
        if self.config.log_gpu_memory:
            self.gpu_monitor.log_gpu_stats("Initial ")
            
        # Initialize variable-specific weights
        self.use_variable_weights = getattr(self.config, 'use_variable_weights', True)
        self._initialize_variable_weights()
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _initialize_variable_weights(self):
        """Initialize variable-specific weights for loss calculation."""
        self.pft1d_var_weights = None
        self.soil2d_var_weights = None
        self.scalar_var_weights = None
        
        if not self.use_variable_weights:
            logger.info("Variable-specific weights disabled")
            return
            
        # Get variable names from data_info if available
        if hasattr(self, 'data_info'):
            # PFT1D variables
            if 'variables_1d_pft' in self.data_info:
                pft1d_vars = self.data_info.get('variables_1d_pft', [])
                self.pft1d_var_weights = get_pft1d_variable_weights(pft1d_vars)
                logger.info(f"Initialized PFT1D variable weights: {self.pft1d_var_weights}")
                
            # Soil2D variables
            if 'x_list_columns_2d' in self.data_info:
                soil2d_vars = [var.replace('Y_', '') for var in self.data_info.get('y_list_columns_2d', [])]                
                self.soil2d_var_weights = get_soil2d_variable_weights(soil2d_vars)
                logger.info(f"Initialized Soil2D variable weights: {self.soil2d_var_weights}")
                
            # Scalar variables
            if 'x_list_scalar_columns' in self.data_info:
                scalar_vars = self.data_info.get('x_list_scalar_columns', [])
                self.scalar_var_weights = get_scalar_variable_weights(scalar_vars)
                logger.info(f"Initialized scalar variable weights: {self.scalar_var_weights}")
    
        # Use learnable loss weights if specified in config
        self.use_learnable_loss_weights = getattr(self.config, 'use_learnable_loss_weights', False)

    def _ensure_consistent_batch_sizes(self):
        """Ensure all final tensors in train_data and test_data have the same batch size."""
        final_tensor_keys = [
            'time_series', 'static', 'pft_param', 'scalar',
            'variables_1d_pft', 'variables_2d_soil',
            'y_scalar', 'y_pft_1d', 'y_soil_2d',
            'water', 'y_water'
        ]
        for split_name in ['train', 'test']:
            split = getattr(self, f'{split_name}_data')
            keys_present = [k for k in final_tensor_keys if k in split and isinstance(split[k], torch.Tensor)]
            if not keys_present:
                continue
            min_length = min(split[k].shape[0] for k in keys_present)
            for k in keys_present:
                if split[k].shape[0] != min_length:
                    split[k] = split[k][:min_length]
    
    def _concat_list_columns(self, list_dict, col_names):
        """Concatenate 1D list columns into a tensor."""
        tensors = [list_dict[col] for col in col_names if col in list_dict]
        if tensors:
            return torch.cat(tensors, dim=1)
        else:
            # Return empty tensor with correct batch size from other data sources
            # Use the batch size from time_series data if available
            if hasattr(self, 'train_data') and 'time_series' in self.train_data:
                batch_size = self.train_data['time_series'].shape[0]
            elif hasattr(self, 'test_data') and 'time_series' in self.test_data:
                batch_size = self.test_data['time_series'].shape[0]
            else:
                batch_size = 0
            return torch.empty((batch_size, 0), device=self.device)

    def _concat_list_columns_2d(self, list_dict, col_names):
        """Concatenate 2D list columns into a tensor."""
        tensors = [list_dict[col].unsqueeze(1) for col in col_names if col in list_dict]
        if tensors:
            return torch.cat(tensors, dim=1)
        else:
            # Return empty tensor with correct batch size from other data sources
            # Use the batch size from time_series data if available
            if hasattr(self, 'train_data') and 'time_series' in self.train_data:
                batch_size = self.train_data['time_series'].shape[0]
            elif hasattr(self, 'test_data') and 'time_series' in self.test_data:
                batch_size = self.test_data['time_series'].shape[0]
            else:
                batch_size = 0
            return torch.empty((batch_size, 0, 0, 0), device=self.device)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Debug: Check tensor sizes before creating TensorDataset
        tensors_to_check = [
            self.train_data['time_series'],
            self.train_data['static'],
            self.train_data['pft_param'],
            self.train_data['variables_1d_pft'],
            self.train_data['variables_2d_soil'],
            self.train_data['y_pft_1d'],
            self.train_data['y_soil_2d']
        ]
        
        tensor_names = ['time_series', 'static', 'pft_param', 'variables_1d_pft', 'variables_2d_soil', 'y_pft_1d', 'y_soil_2d']
        batch_sizes = [t.shape[0] for t in tensors_to_check]
        
        logger.info(f"Training tensor batch sizes: {dict(zip(tensor_names, batch_sizes))}")
        
        if len(set(batch_sizes)) > 1:
            logger.error(f"Tensor batch sizes are inconsistent: {dict(zip(tensor_names, batch_sizes))}")
            raise ValueError(f"Tensor batch sizes must be consistent. Found: {dict(zip(tensor_names, batch_sizes))}")
        
        # Add scalar to tensors_to_check and tensor_names if present
        if 'scalar' in self.train_data:
            tensors_to_check.append(self.train_data['scalar'])
            tensor_names.append('scalar')
        if 'y_scalar' in self.train_data:
            tensors_to_check.append(self.train_data['y_scalar'])
            tensor_names.append('y_scalar')
        # Add water to tensors_to_check and tensor_names if present
        if 'water' in self.train_data:
            tensors_to_check.append(self.train_data['water'])
            tensor_names.append('water')
        if 'y_water' in self.train_data:
            tensors_to_check.append(self.train_data['y_water'])
            tensor_names.append('y_water')
        
        # Create data loader with GPU optimizations
        # Build dataset based on presence of scalar and water
        dataset_tensors = [
            self.train_data['time_series'],
            self.train_data['static'],
            self.train_data['pft_param'],
        ]
        if 'scalar' in self.train_data:
            dataset_tensors.append(self.train_data['scalar'])
        dataset_tensors.extend([
            self.train_data['variables_1d_pft'],
            self.train_data['variables_2d_soil'],
        ])
        if 'y_scalar' in self.train_data:
            dataset_tensors.append(self.train_data['y_scalar'])
        dataset_tensors.extend([
            self.train_data['y_pft_1d'],
            self.train_data['y_soil_2d'],
        ])
        if 'water' in self.train_data and 'y_water' in self.train_data:
            dataset_tensors.extend([
                self.train_data['water'],
                self.train_data['y_water'],
            ])
        if 'pft_presence_mask' in self.train_data:
            dataset_tensors.append(self.train_data['pft_presence_mask'])
        train_dataset = TensorDataset(*dataset_tensors)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=self._torch_generator,
            pin_memory=self.config.pin_memory and self.device.type == 'cpu',  # Only pin memory for CPU tensors
            num_workers=self.config.num_workers,
            prefetch_factor=(2 if self.config.num_workers > 0 else None),
            persistent_workers=False
        )
        progress_bar = tqdm(train_loader, desc="Training")
        
        def get_loss_value(loss):
            return loss.item() if hasattr(loss, 'item') else loss

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch based on presence of scalar, water, and pft_presence_mask
            idx = 0
            time_series = batch[idx]; idx += 1
            static = batch[idx]; idx += 1
            pft_param = batch[idx]; idx += 1
            scalar = batch[idx] if 'scalar' in self.train_data else None; idx += 1 if 'scalar' in self.train_data else 0
            variables_1d_pft = batch[idx]; idx += 1
            variables_2d_soil = batch[idx]; idx += 1
            y_scalar = batch[idx] if 'y_scalar' in self.train_data else None; idx += 1 if 'y_scalar' in self.train_data else 0
            y_pft_1d = batch[idx]; idx += 1
            y_soil_2d = batch[idx]; idx += 1
            water = batch[idx] if 'water' in self.train_data and 'y_water' in self.train_data else None; idx += 1 if 'water' in self.train_data and 'y_water' in self.train_data else 0
            y_water = batch[idx] if 'water' in self.train_data and 'y_water' in self.train_data else None; idx += 1 if 'water' in self.train_data and 'y_water' in self.train_data else 0
            pft_presence_mask = batch[idx] if 'pft_presence_mask' in self.train_data else None
            # --- DEBUG: Print tensor shapes and device before model call ---
            # print(f"[DEBUG] Batch {batch_idx} tensor shapes and device:")
            # print(f"  time_series: {time_series.shape}, device: {time_series.device}")
            # print(f"  static: {static.shape}, device: {static.device}")
            # print(f"  pft_param: {pft_param.shape}, device: {pft_param.device}")
            # print(f"  scalar: {scalar.shape}, device: {scalar.device}")
            # print(f"  variables_1d_pft: {variables_1d_pft.shape}, device: {variables_1d_pft.device}")
            # print(f"  variables_2d_soil: {variables_2d_soil.shape}, device: {variables_2d_soil.device}")
            # print(f"  y_scalar: {y_scalar.shape}, device: {y_scalar.device}")
            # print(f"  y_pft_1d: {y_pft_1d.shape}, device: {y_pft_1d.device}")
            # print(f"  y_soil_2d: {y_soil_2d.shape}, device: {y_soil_2d.device}")
            # if 'water' in self.train_data and 'y_water' in self.train_data:
            #     print(f"  water: {water.shape}, device: {water.device}")
            #     print(f"  y_water: {y_water.shape}, device: {y_water.device}")
            # Move data to device and ensure contiguous
            time_series = time_series.to(self.device, non_blocking=True).contiguous()
            static = static.to(self.device, non_blocking=True).contiguous()
            variables_1d_pft = variables_1d_pft.to(self.device, non_blocking=True).contiguous()
            variables_2d_soil = variables_2d_soil.to(self.device, non_blocking=True).contiguous()
            pft_param = pft_param.to(self.device, non_blocking=True).contiguous()
            if scalar is not None:
                scalar = scalar.to(self.device, non_blocking=True).contiguous()
            if y_scalar is not None:
                y_scalar = y_scalar.to(self.device, non_blocking=True).contiguous()
            y_pft_1d = y_pft_1d.to(self.device, non_blocking=True).contiguous()                     
            y_soil_2d = y_soil_2d.to(self.device, non_blocking=True).contiguous()
            if 'water' in self.train_data and 'y_water' in self.train_data:
                water = water.to(self.device, non_blocking=True).contiguous()
                y_water = y_water.to(self.device, non_blocking=True).contiguous()
            # Presence mask to device if provided
            if 'pft_presence_mask' in self.train_data:
                pft_presence_mask = pft_presence_mask.to(self.device, non_blocking=True).contiguous()

            # print(f"[DEBUG] variables_1d_pft shape before model: {variables_1d_pft.shape}")
            # if variables_1d_pft.dim() == 2 and variables_1d_pft.shape[1] == 224:
            #     variables_1d_pft = variables_1d_pft.view(-1, 14, 16)
            #     print(f"[DEBUG] variables_1d_pft reshaped to: {variables_1d_pft.shape}")

            self.optimizer.zero_grad()
            
            # Forward pass (with or without mixed precision)
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    if water is not None:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                    else:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
            else:
                if water is not None:
                    outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                else:
                    outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)

            # Optionally apply PFT presence mask to predictions before loss
            if getattr(self.config, 'mask_absent_pfts', False) and 'pft_1d' in outputs and 'pft_presence_mask' in self.train_data:
                try:
                    vec = outputs['pft_1d']
                    varnames = list(self.model.data_info.get('variables_1d_pft', [])) if hasattr(self.model, 'data_info') else None
                    n_vars = len(varnames) if varnames is not None and len(varnames) > 0 else self.train_data['y_pft_1d'].size(1)
                    n_pfts = 16
                    if vec.dim() == 2 and vec.size(1) == n_vars * n_pfts:
                        vec = vec.view(vec.size(0), n_vars, n_pfts)
                    if pft_presence_mask.dim() == 2 and pft_presence_mask.size(1) == n_pfts:
                        mask = pft_presence_mask.view(pft_presence_mask.size(0), 1, n_pfts)
                        vec = vec * mask
                        outputs['pft_1d'] = vec.view(vec.size(0), -1)
                except Exception:
                    pass

            # Compute loss with variable-specific weights for scalar variables
            loss = 0.0
            if 'scalar' in outputs and y_scalar is not None:
                if self.use_variable_weights and hasattr(self, 'scalar_var_weights') and self.scalar_var_weights:
                    # Apply variable-specific weights to scalar variables
                    scalar_loss = 0.0
                    scalar_pred = outputs['scalar']
                    
                    # Get variable names
                    scalar_vars = self.data_info.get('x_list_scalar_columns', [])
                    
                    for i, var_name in enumerate(scalar_vars):
                        if i < scalar_pred.size(1):  # Ensure index is within bounds
                            var_weight = self.scalar_var_weights.get(var_name, 1.0)
                            var_loss = self._compute_loss(scalar_pred[:, i:i+1], y_scalar[:, i:i+1])
                            scalar_loss += var_weight * var_loss
                            
                    # Normalize by number of variables to maintain scale
                    scalar_loss = scalar_loss / max(1, len(scalar_vars))
                    loss = self.scalar_loss_weight * scalar_loss
                else:
                    # Use standard loss calculation
                    loss = self.scalar_loss_weight * self._compute_loss(outputs['scalar'], y_scalar)

            # Vector (PFT1D): Apply differential weighting specifically to xsmrpool
            vector_pred = outputs['pft_1d']  # shape: [batch, n_vars*n_pfts] or [batch, n_vars, n_pfts]
            vector_targ = y_pft_1d         # expected shape: [batch, n_vars, n_pfts]

            xsmrpool_weight = getattr(self.config, 'xsmrpool_loss_weight', 1.0)

            try:
                # Determine variable list and reshape predictions if needed
                varnames = None
                if hasattr(self.model, 'data_info') and 'variables_1d_pft' in self.model.data_info:
                    varnames = list(self.model.data_info['variables_1d_pft'])
                n_vars = len(varnames) if varnames is not None else vector_targ.size(1)
                n_pfts = vector_targ.size(2)

                if vector_pred.dim() == 2:
                    # reshape flat predictions to [batch, n_vars, n_pfts]
                    vector_pred_reshaped = vector_pred.view(vector_pred.size(0), n_vars, n_pfts)
                else:
                    vector_pred_reshaped = vector_pred

                # Identify xsmrpool index reliably
                if varnames is not None and 'xsmrpool' in varnames:
                    x_idx = varnames.index('xsmrpool')
                else:
                    # fallback to conventional index (cpool,npool,ppool,xsmrpool,tlai)
                    x_idx = 3

                # Split xsmrpool vs others
                x_pred = vector_pred_reshaped[:, x_idx, :]
                x_targ = vector_targ[:, x_idx, :]
                other_pred = torch.cat([vector_pred_reshaped[:, :x_idx, :], vector_pred_reshaped[:, x_idx+1:, :]], dim=1)
                other_targ = torch.cat([vector_targ[:, :x_idx, :], vector_targ[:, x_idx+1:, :]], dim=1)

                # Apply variable-specific weights for PFT1D variables
                if self.use_variable_weights and hasattr(self, 'pft1d_var_weights') and self.pft1d_var_weights:
                    # Get variable names
                    pft1d_vars = list(self.data_info.get('variables_1d_pft', []))
                    pft1d_loss = 0.0
                    
                    # Process each variable separately (excluding xsmrpool which is handled specially)
                    for i in range(other_pred.size(1)):
                        # Map the index back to the original variable name
                        var_idx = i if i < x_idx else i + 1  # Account for removed xsmrpool
                        if var_idx < len(pft1d_vars):
                            var_name = pft1d_vars[var_idx]
                            var_weight = self.pft1d_var_weights.get(var_name, 1.0)
                            
                            # Extract this variable across all PFTs
                            var_pred = other_pred[:, i:i+1, :].reshape(other_pred.size(0), -1)
                            var_targ = other_targ[:, i:i+1, :].reshape(other_targ.size(0), -1)
                            
                            # Apply weighted loss
                            var_loss = self._compute_loss(var_pred, var_targ)
                            pft1d_loss += var_weight * var_loss
                    
                    # Add normalized loss
                    loss += self.vector_loss_weight * pft1d_loss / max(1, other_pred.size(1))
                else:
                    # Apply standard loss for other variables
                    loss += self.vector_loss_weight * self._compute_loss(
                        other_pred.view(other_pred.size(0), -1),
                        other_targ.view(other_targ.size(0), -1)
                    )

                # Weighted MSE for xsmrpool
                x_pred_flat = x_pred.view(x_pred.size(0), -1)
                x_targ_flat = x_targ.view(x_targ.size(0), -1)
                with torch.no_grad():
                    nz_mask = (x_targ_flat < 0).float()
                base_w = 1.0
                extra = max(1.0, xsmrpool_weight) - 1.0
                weights = base_w + extra * nz_mask
                se = (x_pred_flat - x_targ_flat) ** 2
                weighted_mse = (se * weights).mean()
                loss += self.vector_loss_weight * weighted_mse
            except Exception:
                # Fallback: original aggregate loss
                loss += self.vector_loss_weight * self._compute_loss(
                    vector_pred.view(vector_pred.size(0), -1),
                    vector_targ.view(vector_targ.size(0), -1)
                )
            # Optional sparsity regularization: penalize non-zero predictions where target is zero
            if getattr(self.config, 'pft_zero_sparsity_weight', 0.0) > 0.0:
                with torch.no_grad():
                    zero_mask = (vector_targ.abs() <= getattr(self.config, 'pft_zero_threshold', 1e-8))
                # Reshape predictions to match target shape if needed
                try:
                    pred_for_penalty = (vector_pred if vector_pred.shape == vector_targ.shape
                                        else vector_pred.view_as(vector_targ))
                    sparsity_penalty = (pred_for_penalty.abs() * zero_mask).mean()
                    loss = loss + self.config.pft_zero_sparsity_weight * sparsity_penalty
                except Exception:
                    pass
            # Matrix (Soil2D) with variable-specific weights
            if self.use_variable_weights and hasattr(self, 'soil2d_var_weights') and self.soil2d_var_weights:
                # Apply variable-specific weights to soil2D variables
                soil2d_loss = 0.0
                soil2d_pred = outputs['soil_2d']
                
                # Get variable names (remove 'Y_' prefix)
                soil2d_vars = [var.replace('Y_', '') for var in self.data_info.get('y_list_columns_2d', [])]
                
                # Reshape predictions and targets for per-variable processing
                n_vars = len(soil2d_vars)
                batch_size = soil2d_pred.size(0)
                
                # Reshape to [batch, n_vars, ...] if needed
                if soil2d_pred.dim() == 4:  # [batch, n_vars, rows, cols]
                    soil2d_pred_reshaped = soil2d_pred
                    soil2d_targ_reshaped = y_soil_2d
                else:  # Need to reshape
                    rows = y_soil_2d.size(2) if y_soil_2d.dim() >= 3 else 1
                    cols = y_soil_2d.size(3) if y_soil_2d.dim() >= 4 else 1
                    soil2d_pred_reshaped = soil2d_pred.view(batch_size, n_vars, rows, cols)
                    soil2d_targ_reshaped = y_soil_2d
                
                # Calculate weighted loss for each variable, applying litter overrides if provided
                litter_c_names = {'litr1c_vr', 'litr2c_vr', 'litr3c_vr'}
                litter_n_names = {'litr1n_vr', 'litr2n_vr', 'litr3n_vr'}
                litter_p_names = {'litr1p_vr', 'litr2p_vr', 'litr3p_vr'}
                litter_c_w = getattr(self.config, 'litter_c_loss_weight', 1.0)
                litter_n_w = getattr(self.config, 'litter_n_loss_weight', 1.0)
                litter_p_w = getattr(self.config, 'litter_p_loss_weight', 1.0)

                for i, var_name in enumerate(soil2d_vars):
                    if i < soil2d_pred_reshaped.size(1):  # Ensure index is within bounds
                        base_weight = self.soil2d_var_weights.get(var_name, 1.0)
                        # Apply litter overrides to base weight (multiplicative)
                        if var_name in litter_c_names:
                            var_weight = base_weight * litter_c_w
                        elif var_name in litter_n_names:
                            var_weight = base_weight * litter_n_w
                        elif var_name in litter_p_names:
                            var_weight = base_weight * litter_p_w
                        else:
                            var_weight = base_weight
                        var_pred = soil2d_pred_reshaped[:, i:i+1].reshape(batch_size, -1)
                        var_targ = soil2d_targ_reshaped[:, i:i+1].reshape(batch_size, -1)
                        var_loss = self._compute_loss(var_pred, var_targ)
                        soil2d_loss += var_weight * var_loss
                
                # Normalize by number of variables
                loss += self.matrix_loss_weight * soil2d_loss / max(1, n_vars)
            else:
                # Use standard loss calculation
                loss += self.matrix_loss_weight * self._compute_loss(
                    outputs['soil_2d'].view(y_soil_2d.size(0), -1),
                    y_soil_2d.view(y_soil_2d.size(0), -1)
                )
            if 'water' in self.train_data and 'y_water' in self.train_data and 'water' in outputs:
                loss += self._compute_loss(outputs['water'], y_water)

            # Backward and optimizer step
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            loss_value = get_loss_value(loss)
            total_loss += loss_value
            num_batches += 1
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{(total_loss/num_batches):.4f}'
            })
            
            # GPU memory management
            if batch_idx % self.config.empty_cache_freq == 0:
                self.gpu_monitor.empty_cache()
            
            # GPU monitoring
            if (batch_idx % self.config.gpu_monitor_interval == 0 and 
                self.config.log_gpu_memory):
                self.gpu_monitor.log_gpu_stats(f"Batch {batch_idx} ")
            
            # Check memory threshold
            if self.gpu_monitor.check_memory_threshold(self.config.max_memory_usage):
                logger.warning(f"GPU memory usage exceeds {self.config.max_memory_usage*100}%")
                self.gpu_monitor.empty_cache()
            # --- DEBUG: Call torch.cuda.empty_cache() after each batch ---
            torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Debug: Check tensor sizes before creating TensorDataset
        tensors_to_check = [
            self.test_data['time_series'],
            self.test_data['static'],
            self.test_data['variables_1d_pft'],
            self.test_data['variables_2d_soil'],
            self.test_data['y_soil_2d'],
            self.test_data['pft_param'],
            self.test_data['y_pft_1d']
        ]
        
        tensor_names = ['time_series', 'static', 'variables_1d_pft', 'variables_2d_soil', 'y_soil_2d', 'pft_param', 'y_pft_1d']
        batch_sizes = [t.shape[0] for t in tensors_to_check]
        
        logger.info(f"Validation tensor batch sizes: {dict(zip(tensor_names, batch_sizes))}")
        
        # Check if we have any validation data
        if all(size == 0 for size in batch_sizes):
            logger.warning("No validation data available. Skipping validation.")
            return float('inf')  # Return infinity to indicate no validation
        
        if len(set(batch_sizes)) > 1:
            logger.error(f"Validation tensor batch sizes are inconsistent: {dict(zip(tensor_names, batch_sizes))}")
            raise ValueError(f"Validation tensor batch sizes must be consistent. Found: {dict(zip(tensor_names, batch_sizes))}")
        
        # Add scalar to tensors_to_check and tensor_names if present
        if 'scalar' in self.test_data:
            tensors_to_check.append(self.test_data['scalar'])
            tensor_names.append('scalar')
        if 'y_scalar' in self.test_data:
            tensors_to_check.append(self.test_data['y_scalar'])
            tensor_names.append('y_scalar')
        # Add water to tensors_to_check and tensor_names if present
        if 'water' in self.test_data:
            tensors_to_check.append(self.test_data['water'])
            tensor_names.append('water')
        if 'y_water' in self.test_data:
            tensors_to_check.append(self.test_data['y_water'])
            tensor_names.append('y_water')
        
        # Create data loader with GPU optimizations
        # Build dataset based on presence of scalar and water
        dataset_tensors = [
            self.test_data['time_series'],
            self.test_data['static'],
            self.test_data['pft_param'],
        ]
        if 'scalar' in self.test_data:
            dataset_tensors.append(self.test_data['scalar'])
        dataset_tensors.extend([
            self.test_data['variables_1d_pft'],
            self.test_data['variables_2d_soil'],
        ])
        if 'y_scalar' in self.test_data:
            dataset_tensors.append(self.test_data['y_scalar'])
        dataset_tensors.extend([
            self.test_data['y_pft_1d'],
            self.test_data['y_soil_2d'],
        ])
        if 'water' in self.test_data and 'y_water' in self.test_data:
            dataset_tensors.extend([
                self.test_data['water'],
                self.test_data['y_water'],
            ])
        if 'pft_presence_mask' in self.test_data:
            dataset_tensors.append(self.test_data['pft_presence_mask'])
        val_dataset = TensorDataset(*dataset_tensors)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            generator=self._torch_generator,
            pin_memory=self.config.pin_memory and self.device.type == 'cpu',  # Only pin memory for CPU tensors
            num_workers=self.config.num_workers,
            prefetch_factor=(self.config.prefetch_factor if self.config.num_workers > 0 else None),
            persistent_workers=self.config.persistent_workers
        )
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            def get_loss_value(loss):
                return loss.item() if hasattr(loss, 'item') else loss
            for batch_idx, batch in enumerate(progress_bar):
                # Unpack batch based on presence of scalar, water, and pft_presence_mask
                idx = 0
                time_series = batch[idx]; idx += 1
                static = batch[idx]; idx += 1
                pft_param = batch[idx]; idx += 1
                scalar = batch[idx] if 'scalar' in self.test_data else None; idx += 1 if 'scalar' in self.test_data else 0
                variables_1d_pft = batch[idx]; idx += 1
                variables_2d_soil = batch[idx]; idx += 1
                y_scalar = batch[idx] if 'y_scalar' in self.test_data else None; idx += 1 if 'y_scalar' in self.test_data else 0
                y_pft_1d = batch[idx]; idx += 1
                y_soil_2d = batch[idx]; idx += 1
                water = batch[idx] if 'water' in self.test_data and 'y_water' in self.test_data else None; idx += 1 if 'water' in self.test_data and 'y_water' in self.test_data else 0
                y_water = batch[idx] if 'water' in self.test_data and 'y_water' in self.test_data else None; idx += 1 if 'water' in self.test_data and 'y_water' in self.test_data else 0
                pft_presence_mask = batch[idx] if 'pft_presence_mask' in self.test_data else None
                # Move data to device and ensure contiguous
                time_series = time_series.to(self.device, non_blocking=True).contiguous()
                static = static.to(self.device, non_blocking=True).contiguous()
                variables_1d_pft = variables_1d_pft.to(self.device, non_blocking=True).contiguous()
                variables_2d_soil = variables_2d_soil.to(self.device, non_blocking=True).contiguous()
                pft_param = pft_param.to(self.device, non_blocking=True).contiguous()
                if scalar is not None:
                    scalar = scalar.to(self.device, non_blocking=True).contiguous()
                if y_scalar is not None:
                    y_scalar = y_scalar.to(self.device, non_blocking=True).contiguous()
                y_pft_1d = y_pft_1d.to(self.device, non_blocking=True).contiguous()
                y_soil_2d = y_soil_2d.to(self.device, non_blocking=True).contiguous()
                if water is not None:
                    water = water.to(self.device, non_blocking=True).contiguous()
                if y_water is not None:
                    y_water = y_water.to(self.device, non_blocking=True).contiguous()
                if pft_presence_mask is not None:
                    pft_presence_mask = pft_presence_mask.to(self.device, non_blocking=True).contiguous()

                # print(f"[DEBUG] variables_1d_pft shape before model (val): {variables_1d_pft.shape}")
                # if variables_1d_pft.dim() == 2 and variables_1d_pft.shape[1] == 224:
                #     variables_1d_pft = variables_1d_pft.view(-1, 14, 16)
                #     print(f"[DEBUG] variables_1d_pft reshaped to: {variables_1d_pft.shape}")

                # Forward pass (with or without mixed precision)
                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        if 'water' in self.test_data and 'y_water' in self.test_data:
                            outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                        else:
                            outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
                else:
                    if 'water' in self.test_data and 'y_water' in self.test_data:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                    else:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)

                # Apply mask in validation as well for consistency
                if getattr(self.config, 'mask_absent_pfts', False) and 'pft_1d' in outputs and 'pft_presence_mask' in self.test_data:
                    try:
                        vec = outputs['pft_1d']
                        varnames = list(self.model.data_info.get('variables_1d_pft', [])) if hasattr(self.model, 'data_info') else None
                        n_vars = len(varnames) if varnames is not None and len(varnames) > 0 else y_pft_1d.size(1)
                        n_pfts = 16
                        if vec.dim() == 2 and vec.size(1) == n_vars * n_pfts:
                            vec = vec.view(vec.size(0), n_vars, n_pfts)
                        if pft_presence_mask.dim() == 2 and pft_presence_mask.size(1) == n_pfts:
                            mask = pft_presence_mask.view(pft_presence_mask.size(0), 1, n_pfts)
                            vec = vec * mask
                            outputs['pft_1d'] = vec.view(vec.size(0), -1)
                    except Exception:
                        pass

                # Compute loss
                loss = 0.0
                if 'scalar' in outputs and y_scalar is not None:
                    loss = self._compute_loss(outputs['scalar'], y_scalar)
                # Vector (PFT1D): base MSE
                vector_pred = outputs['pft_1d']
                vector_targ = y_pft_1d
                loss += self._compute_loss(vector_pred.view(vector_pred.size(0), -1), vector_targ.view(vector_targ.size(0), -1))
                # Optional sparsity regularization at validation (reporting only)
                if getattr(self.config, 'pft_zero_sparsity_weight', 0.0) > 0.0:
                    with torch.no_grad():
                        zero_mask = (vector_targ.abs() <= getattr(self.config, 'pft_zero_threshold', 1e-8))
                        try:
                            pred_for_penalty = (vector_pred if vector_pred.shape == vector_targ.shape
                                                else vector_pred.view_as(vector_targ))
                            sparsity_penalty = (pred_for_penalty.abs() * zero_mask).mean()
                            loss = loss + self.config.pft_zero_sparsity_weight * sparsity_penalty
                        except Exception:
                            pass
                # Matrix (Soil2D)
                loss += self._compute_loss(outputs['soil_2d'].view(y_soil_2d.size(0), -1), y_soil_2d.view(y_soil_2d.size(0), -1))
                if 'water' in self.test_data and 'y_water' in self.test_data and 'water' in outputs:
                    loss += self._compute_loss(outputs['water'], y_water)
                
                loss_value = get_loss_value(loss)
                total_loss += loss_value
                num_batches += 1
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{(total_loss/num_batches):.4f}'
                })
        
        # Handle the case where no batches were processed
        if num_batches == 0:
            logger.warning("No validation batches processed. Returning infinity.")
            return float('inf')
        
        return total_loss / num_batches
    
    def _prepare_batch_data(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Prepare batch data for training/validation using only final model-ready tensors, in config order."""
        batch_data = {}
        batch_size = end_idx - start_idx
        device = data['time_series'].device
        
        # Required inputs (order enforced by config)
        batch_data['time_series'] = data['time_series'][start_idx:end_idx]
        batch_data['static'] = data['static'][start_idx:end_idx]
        batch_data['pft_param'] = data['pft_param'][start_idx:end_idx]
        if 'scalar' in data:
            batch_data['scalar'] = data['scalar'][start_idx:end_idx]
        batch_data['variables_1d_pft'] = data['variables_1d_pft'][start_idx:end_idx]
        batch_data['variables_2d_soil'] = data['variables_2d_soil'][start_idx:end_idx]
        if 'y_scalar' in data:
            batch_data['y_scalar'] = data['y_scalar'][start_idx:end_idx]
        batch_data['y_pft_1d'] = data['y_pft_1d'][start_idx:end_idx]
        batch_data['y_soil_2d'] = data['y_soil_2d'][start_idx:end_idx]

        # Optional water variables
        if 'water' in data:
            batch_data['water'] = data['water'][start_idx:end_idx]
        if 'y_water' in data:
            batch_data['y_water'] = data['y_water'][start_idx:end_idx]

        # Assert batch sizes for all present keys
        for k, v in batch_data.items():
            assert v.shape[0] == batch_size, f"{k} batch size mismatch: {v.shape[0]} vs {batch_size}"

        # Assert feature order for key tensors (optional, for debugging)
        # Example: assert batch_data['scalar'].shape[1] == len(self.scalar_columns)
        # Example: assert batch_data['variables_1d_pft'].shape[1] == len(self.variables_1d_pft_columns)
        
        return batch_data
    
    def _compute_loss(self, scalar_pred, target, **kwargs):
        """Compute only scalar loss for quick test or full loss with learnable weights if enabled."""
        # If using CNPCombinedModel and learnable loss weights, use log_sigma weighting
        if isinstance(self.model, CNPCombinedModel) and self.use_learnable_loss_weights:
            # Assume outputs and targets are dicts with keys: scalar, matrix, water, pft_1d
            outputs = kwargs.get('outputs', None)
            targets = kwargs.get('targets', None)
            if outputs is None or targets is None:
                # Fallback to scalar only
                return self.criterion(scalar_pred, target)
            loss = 0.0
            # Scalar
            if 'scalar' in outputs and 'scalar' in targets and self.model.log_sigma_scalar is not None:
                loss += (torch.exp(-2 * self.model.log_sigma_scalar) * self.criterion(outputs['scalar'], targets['scalar']) + self.model.log_sigma_scalar)
            # Matrix
            if 'matrix' in outputs and 'matrix' in targets and self.model.log_sigma_matrix is not None:
                loss += (torch.exp(-2 * self.model.log_sigma_matrix) * self.criterion(outputs['matrix'], targets['matrix']) + self.model.log_sigma_matrix)
            # Water
            if hasattr(self.model, 'log_sigma_water') and self.model.log_sigma_water is not None and 'water' in outputs and 'water' in targets:
                loss += (torch.exp(-2 * self.model.log_sigma_water) * self.criterion(outputs['water'], targets['water']) + self.model.log_sigma_water)
            # PFT 1D
            if hasattr(self.model, 'log_sigma_pft_1d') and self.model.log_sigma_pft_1d is not None and 'pft_1d' in outputs and 'pft_1d' in targets:
                loss += (torch.exp(-2 * self.model.log_sigma_pft_1d) * self.criterion(outputs['pft_1d'], targets['pft_1d']) + self.model.log_sigma_pft_1d)
            return loss
        else:
            return self.criterion(scalar_pred, target)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Returns:
            Dictionary containing training and validation losses
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        logger.info("=" * 60)
        
        # Print training summary
        print(f"\n{'='*20} TRAINING SUMMARY {'='*20}")
        print(f"📋 Configuration:")
        print(f"   • Total epochs: {self.config.num_epochs}")
        print(f"   • Batch size: {self.config.batch_size}")
        print(f"   • Learning rate: {self.config.learning_rate}")
        print(f"   • Device: {self.device}")
        print(f"   • Mixed precision: {'Enabled' if self.use_amp else 'Disabled'}")
        
        # Calculate approximate data info
        train_samples = self.train_data['time_series'].shape[0]
        test_samples = self.test_data['time_series'].shape[0]
        total_batches_per_epoch = (train_samples + self.config.batch_size - 1) // self.config.batch_size
        
        print(f"📊 Data Info:")
        print(f"   • Training samples: {train_samples:,}")
        print(f"   • Test samples: {test_samples:,}")
        print(f"   • Batches per epoch: ~{total_batches_per_epoch}")
        print(f"   • Total training batches: ~{total_batches_per_epoch * self.config.num_epochs:,}")
        
        print(f"🎯 Training Goals:")
        print(f"   • Target: Minimize combined loss (scalar + matrix)")
        print(f"   • Early stopping: {'Enabled' if self.config.use_early_stopping else 'Disabled'}")
        if self.config.use_early_stopping:
            print(f"   • Patience: {self.config.patience} epochs")
        print(f"   • Validation frequency: Every {self.config.validation_frequency} epoch(s)")
        
        print(f"{'='*60}")
        
        for epoch in range(self.config.num_epochs):
            # Check for NaNs/Infs in training data at the start of the epoch
            def count_nans_infs(name, tensor):
                n_nan = torch.isnan(tensor).sum().item() if torch.is_tensor(tensor) else 0
                n_inf = torch.isinf(tensor).sum().item() if torch.is_tensor(tensor) else 0
                print(f"Epoch {epoch+1}: {name} - NaNs: {n_nan}, Infs: {n_inf}")
            count_nans_infs('time_series', self.train_data['time_series'])
            count_nans_infs('static', self.train_data['static'])
            if 'list_1d' in self.train_data:
                for k, v in self.train_data['list_1d'].items():
                    count_nans_infs(f'list_1d[{k}]', v)
            if 'list_2d' in self.train_data:
                for k, v in self.train_data['list_2d'].items():
                    count_nans_infs(f'list_2d[{k}]', v)
            if 'list_scalar' in self.train_data:
                count_nans_infs('list_scalar', self.train_data['list_scalar'])
            if 'pft_param' in self.train_data:
                count_nans_infs('pft_param', self.train_data['pft_param'])
            if 'y_pft_1d' in self.train_data:
                count_nans_infs('y_pft_1d', self.train_data['y_pft_1d'])
            if 'y_scalar' in self.train_data:
                count_nans_infs('y_scalar', self.train_data['y_scalar'])
            # Print epoch header
            # print(f"\n{'='*20} EPOCH {epoch+1}/{self.config.num_epochs} {'='*20}")
            logger.info(f"Starting Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training
            # print(f"Training...")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % self.config.validation_frequency == 0:
                # print(f"Validating...")
                val_loss = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # Enhanced progress logging
                if val_loss == float('inf'):
                    progress_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: N/A (no validation data)"
                    print(f"✓ {progress_msg}")
                    logger.info(progress_msg)
                else:
                    progress_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    print(f"✓ {progress_msg}")
                    logger.info(progress_msg)
                    
                    # Show improvement indicator
                    if val_loss < self.best_val_loss:
                        improvement = self.best_val_loss - val_loss
                        print(f"🎉 New best validation loss! Improved by {improvement:.4f}")
                        logger.info(f"New best validation loss! Improved by {improvement:.4f}")
                
                # Log loss weights
                loss_weights = self.model.get_loss_weights()
                log_msg = f"Loss weights - "
                if 'scalar' in loss_weights:
                    log_msg += f"Scalar: {loss_weights['scalar']:.4f}, "
                log_msg += f"Matrix: {loss_weights['matrix']:.4f}"
                logger.info(log_msg)
                
                # Show progress percentage
                progress_pct = ((epoch + 1) / self.config.num_epochs) * 100
                print(f"📊 Progress: {progress_pct:.1f}% complete ({epoch+1}/{self.config.num_epochs} epochs)")
                
                # Early stopping (only if we have validation data)
                if self.config.use_early_stopping and val_loss != float('inf'):
                    if self._check_early_stopping(val_loss):
                        logger.info("Early stopping triggered")
                        break
                
                # Learning rate scheduling (only if we have validation data)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loss != float('inf'):
                            old_lr = self.optimizer.param_groups[0]['lr']
                            self.scheduler.step(val_loss)
                            new_lr = self.optimizer.param_groups[0]['lr']
                            if new_lr != old_lr:
                                print(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                                logger.info(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                    else:
                        self.scheduler.step()
            else:
                # For epochs without validation, still show training progress
                progress_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {train_loss:.4f}"
                print(f"✓ {progress_msg}")
                logger.info(progress_msg)
                
                # Show progress percentage
                progress_pct = ((epoch + 1) / self.config.num_epochs) * 100
                print(f"📊 Progress: {progress_pct:.1f}% complete ({epoch+1}/{self.config.num_epochs} epochs)")
        
        print(f"\n{'='*20} TRAINING COMPLETED {'='*20}")
        logger.info("Training completed")
        
        # Print final summary
        print(f" Training completed successfully!")
        print(f"📈 Final Results:")
        print(f"   • Total epochs completed: {len(self.train_losses)}")
        print(f"   • Final training loss: {self.train_losses[-1]:.4f}")
        
        if self.val_losses:
            print(f"   • Final validation loss: {self.val_losses[-1]:.4f}")
            print(f"   • Best validation loss: {min(self.val_losses):.4f}")
            
            # Show improvement
            if len(self.train_losses) > 1:
                train_improvement = self.train_losses[0] - self.train_losses[-1]
                print(f"   • Training loss improvement: {train_improvement:.4f}")
            
            if len(self.val_losses) > 1:
                val_improvement = self.val_losses[0] - min(self.val_losses)
                print(f"   • Validation loss improvement: {val_improvement:.4f}")
        
        print(f"💾 Results saved to:")
        print(f"   • Training log: training.log")
        print(f"   • Loss curves: training_validation_losses.csv")
        print(f"   • Model predictions: predictions/")
        
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def evaluate(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Evaluate the model and return predictions and metrics."""
        self.model.eval()
        
        # Check if we have any test data
        test_batch_sizes = [
            self.test_data['time_series'].shape[0],
            self.test_data['static'].shape[0],
            self.test_data['pft_param'].shape[0],
            self.test_data['variables_1d_pft'].shape[0],
            self.test_data['variables_2d_soil'].shape[0],
            self.test_data['y_pft_1d'].shape[0],
            self.test_data['y_soil_2d'].shape[0]
        ]
        if all(size == 0 for size in test_batch_sizes):
            logger.warning("No test data available. Skipping evaluation.")
            empty_predictions = {
                'scalar': torch.empty(0, 0),
                'pft_1d': torch.empty(0, 0),
                'soil_2d': torch.empty(0, 0, 0, 0)
            }
            default_metrics = {
                'scalar_rmse': 0.0,
                'scalar_mse': 0.0,
                'pft_1d_rmse': 0.0,
                'pft_1d_mse': 0.0,
                'soil_2d_rmse': 0.0,
                'soil_2d_mse': 0.0
            }
            return empty_predictions, default_metrics
        
        # Create evaluation data loader (optionally include presence mask)
        # Build dataset based on presence of scalar
        dataset_tensors = [
            self.test_data['time_series'],
            self.test_data['static'],
            self.test_data['pft_param'],
        ]
        if 'scalar' in self.test_data:
            dataset_tensors.append(self.test_data['scalar'])
        dataset_tensors.extend([
            self.test_data['variables_1d_pft'],
            self.test_data['variables_2d_soil'],
        ])
        if 'y_scalar' in self.test_data:
            dataset_tensors.append(self.test_data['y_scalar'])
        dataset_tensors.extend([
            self.test_data['y_pft_1d'],
            self.test_data['y_soil_2d'],
        ])
        if 'pft_presence_mask' in self.test_data:
            dataset_tensors.append(self.test_data['pft_presence_mask'])
        eval_dataset = TensorDataset(*dataset_tensors)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            generator=self._torch_generator,
            pin_memory=self.config.pin_memory and self.device.type == 'cpu',  # Only pin memory for CPU tensors
            num_workers=self.config.num_workers
        )
        all_predictions = {
            'pft_1d': [],
            'soil_2d': []
        }
        all_targets = {
            'y_pft_1d': [],
            'y_soil_2d': []
        }
        if 'scalar' in self.test_data:
            all_predictions['scalar'] = []
        if 'y_scalar' in self.test_data:
            all_targets['y_scalar'] = []
        with torch.no_grad():
            for batch in eval_loader:
                idx = 0
                time_series = batch[idx]; idx += 1
                static = batch[idx]; idx += 1
                pft_param = batch[idx]; idx += 1
                scalar = batch[idx] if 'scalar' in self.test_data else None; idx += 1 if 'scalar' in self.test_data else 0
                variables_1d_pft = batch[idx]; idx += 1
                variables_2d_soil = batch[idx]; idx += 1
                y_scalar = batch[idx] if 'y_scalar' in self.test_data else None; idx += 1 if 'y_scalar' in self.test_data else 0
                y_pft_1d = batch[idx]; idx += 1
                y_soil_2d = batch[idx]; idx += 1
                pft_presence_mask = batch[idx] if 'pft_presence_mask' in self.test_data else None; idx += 1 if 'pft_presence_mask' in self.test_data else 0
                # Move to device
                time_series = time_series.to(self.device, non_blocking=True)
                static = static.to(self.device, non_blocking=True)
                pft_param = pft_param.to(self.device, non_blocking=True)
                if scalar is not None:
                    scalar = scalar.to(self.device, non_blocking=True)
                variables_1d_pft = variables_1d_pft.to(self.device, non_blocking=True)
                variables_2d_soil = variables_2d_soil.to(self.device, non_blocking=True)
                if y_scalar is not None:
                    y_scalar = y_scalar.to(self.device, non_blocking=True)
                y_pft_1d = y_pft_1d.to(self.device, non_blocking=True)
                y_soil_2d = y_soil_2d.to(self.device, non_blocking=True)
                if pft_presence_mask is not None:
                    pft_presence_mask = pft_presence_mask.to(self.device, non_blocking=True)
                # Forward pass
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        if scalar is not None:
                            outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
                        else:
                            outputs = self.model(time_series, static, pft_param, variables_1d_pft, variables_2d_soil)
                else:
                    if scalar is not None:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
                    else:
                        outputs = self.model(time_series, static, pft_param, variables_1d_pft, variables_2d_soil)
                # Apply presence mask to predictions if enabled
                if getattr(self.config, 'mask_absent_pfts', False) and 'pft_1d' in outputs and pft_presence_mask is not None:
                    try:
                        vec = outputs['pft_1d']
                        # Determine n_vars and reshape
                        varnames = list(self.model.data_info.get('variables_1d_pft', [])) if hasattr(self.model, 'data_info') else None
                        n_vars = len(varnames) if varnames is not None and len(varnames) > 0 else y_pft_1d.size(1)
                        n_pfts = 16
                        if vec.dim() == 2 and vec.size(1) == n_vars * n_pfts:
                            vec = vec.view(vec.size(0), n_vars, n_pfts)
                        if pft_presence_mask.dim() == 2 and pft_presence_mask.size(1) == n_pfts:
                            mask = pft_presence_mask.view(pft_presence_mask.size(0), 1, n_pfts)
                            vec = vec * mask
                            outputs['pft_1d'] = vec.view(vec.size(0), -1)
                    except Exception:
                        pass
                if 'scalar' in outputs:
                    all_predictions['scalar'].append(outputs['scalar'].cpu())
                all_predictions['pft_1d'].append(outputs['pft_1d'].cpu())
                all_predictions['soil_2d'].append(outputs['soil_2d'].cpu())
                if y_scalar is not None:
                    all_targets['y_scalar'].append(y_scalar.cpu())
                all_targets['y_pft_1d'].append(y_pft_1d.cpu())
                all_targets['y_soil_2d'].append(y_soil_2d.cpu())
        # Concatenate all batches (only if list is not empty)
        predictions = {}
        targets = {}
        for k, v in all_predictions.items():
            if v:
                predictions[k] = torch.cat(v, dim=0)
            # Don't add empty tensors to predictions/targets - they will be handled in _calculate_metrics
        for k, v in all_targets.items():
            if v:
                targets[k] = torch.cat(v, dim=0)
        
        # Debug: Print shapes to identify the issue
        print(f"[DEBUG] Evaluation - predictions shapes:")
        for k, v in predictions.items():
            print(f"  {k}: {v.shape}")
        print(f"[DEBUG] Evaluation - targets shapes:")
        for k, v in targets.items():
            print(f"  {k}: {v.shape}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        return predictions, metrics
    
    def _calculate_metrics(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        # Scalar (only if present and has valid shape)
        if 'scalar' in predictions and 'y_scalar' in targets:
            pred_scalar = predictions['scalar']
            target_scalar = targets['y_scalar']
            # Check if tensors are not empty and have correct dimensions
            if pred_scalar.numel() > 0 and target_scalar.numel() > 0 and pred_scalar.ndim >= 2 and target_scalar.ndim >= 2:
                pred_scalar_full = pred_scalar.cpu().numpy()
                target_scalar_full = target_scalar.cpu().numpy()
                # Overall scalar metrics
                mask_all = ~np.isnan(pred_scalar_full) & ~np.isnan(target_scalar_full)
                # Ensure arrays have the same shape before flattening
                if pred_scalar_full.shape != target_scalar_full.shape:
                    print(f"Warning: Shape mismatch - pred: {pred_scalar_full.shape}, target: {target_scalar_full.shape}")
                    # Use the minimum shape to avoid indexing errors
                    if len(pred_scalar_full.shape) >= 2 and len(target_scalar_full.shape) >= 2:
                        min_shape = (min(pred_scalar_full.shape[0], target_scalar_full.shape[0]), 
                                    min(pred_scalar_full.shape[1], target_scalar_full.shape[1]))
                        pred_scalar_full = pred_scalar_full[:min_shape[0], :min_shape[1]]
                        target_scalar_full = target_scalar_full[:min_shape[0], :min_shape[1]]
                        mask_all = ~np.isnan(pred_scalar_full) & ~np.isnan(target_scalar_full)
                
                # Flatten arrays and mask for overall metrics
                pred_flat = pred_scalar_full.flatten()
                target_flat = target_scalar_full.flatten()
                mask_flat = mask_all.flatten()
                if np.sum(mask_flat) > 0:
                    mse_scalar_all = mean_squared_error(target_flat[mask_flat], pred_flat[mask_flat])
                    metrics['scalar_rmse'] = np.sqrt(mse_scalar_all)
                    metrics['scalar_mse'] = mse_scalar_all
                    # Per-scalar metrics with names
                    if len(pred_scalar_full.shape) >= 2:
                        num_scalar = pred_scalar_full.shape[1]
                        scalar_names = self.data_info.get('y_list_scalar_columns', [f'scalar_{i}' for i in range(num_scalar)])
                        if len(scalar_names) != num_scalar:
                            scalar_names = [f'scalar_{i}' for i in range(num_scalar)]
                        for s in range(num_scalar):
                            pred_s = pred_scalar_full[:, s]
                            targ_s = target_scalar_full[:, s]
                            mask_s = ~np.isnan(pred_s) & ~np.isnan(targ_s)
                            if np.sum(mask_s) > 0:
                                mse_s = mean_squared_error(targ_s[mask_s], pred_s[mask_s])
                                rmse_s = np.sqrt(mse_s)
                                mean_s = np.mean(targ_s[mask_s])
                                nrmse_s = rmse_s / mean_s if mean_s != 0 else float('inf')
                                r2_s = r2_score(targ_s[mask_s], pred_s[mask_s])
                                name_s = scalar_names[s]
                                metrics[f'{name_s}_mse'] = mse_s
                                metrics[f'{name_s}_rmse'] = rmse_s
                                metrics[f'{name_s}_nrmse'] = nrmse_s
                                metrics[f'{name_s}_r2'] = r2_s
                else:
                    metrics['scalar_rmse'] = 0.0
                    metrics['scalar_mse'] = 0.0
            else:
                metrics['scalar_rmse'] = 0.0
                metrics['scalar_mse'] = 0.0
        else:
            metrics['scalar_rmse'] = 0.0
            metrics['scalar_mse'] = 0.0
        # PFT 1D - Detailed metrics per variable and PFT
        pred_pft_1d = predictions['pft_1d'].cpu().numpy()
        target_pft_1d = targets['y_pft_1d'].cpu().numpy()
        # Check if tensors are valid
        if pred_pft_1d.ndim == 0 or pred_pft_1d.shape[0] == 0:
            metrics['pft_1d_rmse'] = 0.0
            metrics['pft_1d_mse'] = 0.0
        else:
            # Ensure shapes are (samples, num_variables, num_pfts)
            if pred_pft_1d.ndim == 2:
                pred_pft_1d = pred_pft_1d.reshape(pred_pft_1d.shape[0], -1, 16)
                target_pft_1d = target_pft_1d.reshape(target_pft_1d.shape[0], -1, 16)
            if pred_pft_1d.ndim < 3:
                metrics['pft_1d_rmse'] = 0.0
                metrics['pft_1d_mse'] = 0.0
            else:
                num_variables = pred_pft_1d.shape[1]
                num_pfts = pred_pft_1d.shape[2]
                # Get variable names from data_info if available
                var_names = self.data_info.get('y_list_columns_1d', [f'pft_1d_var_{i}' for i in range(num_variables)])
                if len(var_names) != num_variables:
                    var_names = [f'pft_1d_var_{i}' for i in range(num_variables)]
                # Overall metrics for pft_1d
                mask = ~np.isnan(pred_pft_1d) & ~np.isnan(target_pft_1d)
                # Ensure arrays have the same shape before flattening
                if pred_pft_1d.shape != target_pft_1d.shape:
                    print(f"Warning: PFT 1D shape mismatch - pred: {pred_pft_1d.shape}, target: {target_pft_1d.shape}")
                    # Use the minimum shape to avoid indexing errors
                    min_shape = tuple(min(pred_pft_1d.shape[i], target_pft_1d.shape[i]) for i in range(len(pred_pft_1d.shape)))
                    pred_pft_1d = pred_pft_1d[:min_shape[0], :min_shape[1], :min_shape[2]]
                    target_pft_1d = target_pft_1d[:min_shape[0], :min_shape[1], :min_shape[2]]
                    mask = ~np.isnan(pred_pft_1d) & ~np.isnan(target_pft_1d)
                
                # Flatten arrays and mask for overall metrics
                pred_flat = pred_pft_1d.flatten()
                target_flat = target_pft_1d.flatten()
                mask_flat = mask.flatten()
                mse_pft_1d = mean_squared_error(target_flat[mask_flat], pred_flat[mask_flat])
                metrics['pft_1d_rmse'] = np.sqrt(mse_pft_1d)
                metrics['pft_1d_mse'] = mse_pft_1d
                # Detailed metrics per variable and PFT
                for v in range(num_variables):
                    var_name = var_names[v]
                    for p in range(num_pfts):
                        mask_vp = ~np.isnan(pred_pft_1d[:, v, p]) & ~np.isnan(target_pft_1d[:, v, p])
                        if np.sum(mask_vp) > 0:
                            mse_vp = mean_squared_error(target_pft_1d[:, v, p][mask_vp], pred_pft_1d[:, v, p][mask_vp])
                            rmse_vp = np.sqrt(mse_vp)
                            target_mean = np.mean(target_pft_1d[:, v, p][mask_vp])
                            nrmse_vp = rmse_vp / target_mean if target_mean != 0 else float('inf')
                            r2_vp = r2_score(target_pft_1d[:, v, p][mask_vp], pred_pft_1d[:, v, p][mask_vp])
                            pft_idx = p + 1  # Use 1..16 in keys
                            metrics[f'{var_name}_pft{pft_idx}_mse'] = mse_vp
                            metrics[f'{var_name}_pft{pft_idx}_rmse'] = rmse_vp
                            metrics[f'{var_name}_pft{pft_idx}_nrmse'] = nrmse_vp
                            metrics[f'{var_name}_pft{pft_idx}_r2'] = r2_vp
        # Soil 2D
        pred_soil_2d = predictions['soil_2d'].cpu().numpy()
        target_soil_2d = targets['y_soil_2d'].cpu().numpy()
        # Ensure shapes match (samples, variables, columns, layers)
        if pred_soil_2d.ndim == 0 or pred_soil_2d.shape[0] == 0:
            metrics['soil_2d_rmse'] = 0.0
            metrics['soil_2d_mse'] = 0.0
            return metrics
        n_samples = pred_soil_2d.shape[0]
        if pred_soil_2d.ndim == 2:
            # If 2D, assume it's flattened and reshape to match target dimensions
            if target_soil_2d.ndim == 4:
                num_vars, num_cols, num_layers = target_soil_2d.shape[1:]
                pred_soil_2d = pred_soil_2d.reshape(n_samples, num_vars, num_cols, num_layers)
            elif target_soil_2d.ndim == 3:
                num_vars, num_cols = target_soil_2d.shape[1:]
                pred_soil_2d = pred_soil_2d.reshape(n_samples, num_vars, num_cols)
        elif pred_soil_2d.ndim == 3 and target_soil_2d.ndim == 4:
            num_vars, num_cols, num_layers = target_soil_2d.shape[1:]
            pred_soil_2d = pred_soil_2d.reshape(n_samples, num_vars, num_cols, num_layers)
        # Flatten both for overall metrics calculation
        pred_soil_2d_flat = pred_soil_2d.reshape(n_samples, -1)
        target_soil_2d_flat = target_soil_2d.reshape(n_samples, -1)
        mask = ~np.isnan(pred_soil_2d_flat) & ~np.isnan(target_soil_2d_flat)
        
        # Ensure arrays have the same shape before flattening
        if pred_soil_2d_flat.shape != target_soil_2d_flat.shape:
            print(f"Warning: Soil 2D shape mismatch - pred: {pred_soil_2d_flat.shape}, target: {target_soil_2d_flat.shape}")
            # Use the minimum shape to avoid indexing errors
            min_shape = (min(pred_soil_2d_flat.shape[0], target_soil_2d_flat.shape[0]), 
                        min(pred_soil_2d_flat.shape[1], target_soil_2d_flat.shape[1]))
            pred_soil_2d_flat = pred_soil_2d_flat[:min_shape[0], :min_shape[1]]
            target_soil_2d_flat = target_soil_2d_flat[:min_shape[0], :min_shape[1]]
            mask = ~np.isnan(pred_soil_2d_flat) & ~np.isnan(target_soil_2d_flat)
        
        # Flatten arrays and mask for overall metrics
        pred_flat = pred_soil_2d_flat.flatten()
        target_flat = target_soil_2d_flat.flatten()
        mask_flat = mask.flatten()
        mse_soil_2d = mean_squared_error(target_flat[mask_flat], pred_flat[mask_flat])
        metrics['soil_2d_rmse'] = np.sqrt(mse_soil_2d)
        metrics['soil_2d_mse'] = mse_soil_2d
        # Per-variable per-layer metrics (aggregated across columns)
        if pred_soil_2d.ndim < 2:
            metrics['soil_2d_rmse'] = 0.0
            metrics['soil_2d_mse'] = 0.0
            return metrics
        num_variables_soil = pred_soil_2d.shape[1]
        num_layers_soil = pred_soil_2d.shape[3] if pred_soil_2d.ndim == 4 else 1
        soil_var_names = self.data_info.get('y_list_columns_2d', [f'soil_2d_var_{i}' for i in range(num_variables_soil)])
        if len(soil_var_names) != num_variables_soil:
            soil_var_names = [f'soil_2d_var_{i}' for i in range(num_variables_soil)]
        if pred_soil_2d.ndim == 4 and target_soil_2d.ndim == 4:
            for v in range(num_variables_soil):
                var_name = soil_var_names[v]
                for l in range(num_layers_soil):
                    pred_slice = pred_soil_2d[:, v, :, l].reshape(-1)
                    targ_slice = target_soil_2d[:, v, :, l].reshape(-1)
                    mask_slice = ~np.isnan(pred_slice) & ~np.isnan(targ_slice)
                    if np.sum(mask_slice) > 0:
                        mse_vl = mean_squared_error(targ_slice[mask_slice], pred_slice[mask_slice])
                        rmse_vl = np.sqrt(mse_vl)
                        mean_vl = np.mean(targ_slice[mask_slice])
                        nrmse_vl = rmse_vl / mean_vl if mean_vl != 0 else float('inf')
                        r2_vl = r2_score(targ_slice[mask_slice], pred_slice[mask_slice])
                        layer_idx = l + 1  # Use 1..num_layers
                        metrics[f'{var_name}_layer{layer_idx}_mse'] = mse_vl
                        metrics[f'{var_name}_layer{layer_idx}_rmse'] = rmse_vl
                        metrics[f'{var_name}_layer{layer_idx}_nrmse'] = nrmse_vl
                        metrics[f'{var_name}_layer{layer_idx}_r2'] = r2_vl
        return metrics
    
    def save_results(self, predictions: Dict[str, np.ndarray], metrics: Dict[str, float]):
        """Save training results and predictions."""
        # Save losses (independent of predictions)
        if self.config.save_losses:
            losses_df = pd.DataFrame({
                'Epoch': list(range(1, len(self.train_losses) + 1)),
                'Train Loss': self.train_losses,
                'Validation Loss': self.val_losses
            })
            losses_df.to_csv(self.config.losses_save_path, index=False)
            logger.info(f"Losses saved to {self.config.losses_save_path}")
        
        # Save predictions (only if enabled)
        if self.config.save_predictions:
            # Create predictions directory
            predictions_dir = Path(self.config.predictions_dir)
            predictions_dir.mkdir(exist_ok=True)
        
        # Save predictions
        self._save_predictions(predictions, predictions_dir)
        
        # Save scalers for future inverse transformation
        self._save_scalers(predictions_dir)
        
        # Save model
        model_path = predictions_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scripted model (commented out due to TorchScript limitations with custom objects)
        # scripted_model = torch.jit.script(self.model)
        # scripted_model_path = predictions_dir / "model_scripted.pt"
        # scripted_model.save(str(scripted_model_path))
        # logger.info(f"Scripted model saved to {scripted_model_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(predictions_dir / "test_metrics.csv", index=False)
        logger.info(f"Metrics saved to {predictions_dir / 'test_metrics.csv'}")

        # Save inverse-transformed static features for test set to enable geospatial mapping
        try:
            if 'static' in self.test_data and 'static' in self.scalers and 'static_columns' in self.data_info:
                static_np = self.test_data['static'].cpu().numpy()
                static_inv = self.scalers['static'].inverse_transform(static_np)
                static_cols = self.data_info['static_columns']
                df_static = pd.DataFrame(static_inv, columns=static_cols)
                df_static.to_csv(predictions_dir / 'test_static_inverse.csv', index=False)
                logger.info(f"Inverse-transformed test static features saved to {predictions_dir / 'test_static_inverse.csv'}")
        except Exception as e:
            logger.warning(f"Failed to save inverse-transformed static features: {e}")
    
    def _save_predictions(self, predictions: Dict[str, np.ndarray], predictions_dir: Path):
        """Save predictions with inverse transformation."""

        for key, tensor in predictions.items():
            if hasattr(tensor, 'shape'):
                logger.info(f"  {key}: {tensor.shape}")
            else:
                logger.info(f"  {key}: {type(tensor)}")

        # Prepare location vectors (Longitude/Latitude) for joining to CSVs
        longitude_values = None
        latitude_values = None
        try:
            if 'static' in self.test_data and 'static' in self.scalers and hasattr(self.scalers['static'], 'inverse_transform'):
                _static_np = self.test_data['static'].cpu().numpy()
                _static_denorm = self.scalers['static'].inverse_transform(_static_np)
                _static_cols = self.data_info.get('static_columns', [])
                if not _static_cols or len(_static_cols) < _static_denorm.shape[1]:
                    _static_cols = [f'static_{i}' for i in range(_static_denorm.shape[1])]
                lon_keys = ['Longitude', 'longitude', 'lon', 'LON']
                lat_keys = ['Latitude', 'latitude', 'lat', 'LAT']
                lon_name = next((k for k in lon_keys if k in _static_cols), None)
                lat_name = next((k for k in lat_keys if k in _static_cols), None)
                if lon_name is not None and lat_name is not None:
                    lon_idx = _static_cols.index(lon_name)
                    lat_idx = _static_cols.index(lat_name)
                    longitude_values = _static_denorm[:, lon_idx]
                    latitude_values = _static_denorm[:, lat_idx]
        except Exception as _e_loc:
            logger.warning(f"Failed to prepare location vectors: {_e_loc}")

        # Prepare scalar column names (used by both predictions and ground truth)
        num_expected_scalars = len(self.data_info.get('y_list_scalar_columns', []))
        
        # Save scalar predictions with inverse transformation (only if available)
        if 'scalar' in predictions and predictions['scalar'].numel() > 0:
            predictions_scalar_np = predictions['scalar'].cpu().numpy()
            
            # Handle shape mismatch - only use the first scalar output if model outputs more than expected
            if num_expected_scalars > 0 and predictions_scalar_np.shape[1] > num_expected_scalars:
                print(f"Warning: Model outputs {predictions_scalar_np.shape[1]} scalars but only {num_expected_scalars} expected. Using first {num_expected_scalars}.")
                predictions_scalar_np = predictions_scalar_np[:, :num_expected_scalars]
            
            scalar_cols = self.data_info.get('y_list_scalar_columns', [f'scalar_{i}' for i in range(predictions_scalar_np.shape[1])])
            if len(scalar_cols) != predictions_scalar_np.shape[1]:
                scalar_cols = [f'scalar_{i}' for i in range(predictions_scalar_np.shape[1])]
            
            # Apply inverse transformation to convert from normalized to original units
            try:
                if 'y_scalar' in self.scalers and self.scalers['y_scalar'] is not None:
                    predictions_scalar_original = self.scalers['y_scalar'].inverse_transform(predictions_scalar_np)
                    logger.info("Applied inverse transformation to scalar predictions")
                else:
                    predictions_scalar_original = predictions_scalar_np
                    logger.warning("No scalar scaler found, saving normalized values")
            except Exception as e:
                logger.warning(f"Failed to apply inverse transformation to scalar predictions: {e}")
                predictions_scalar_original = predictions_scalar_np
            
            predictions_df = pd.DataFrame(predictions_scalar_original, columns=scalar_cols)
            if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(predictions_df):
                predictions_df.insert(0, 'Longitude', longitude_values)
                predictions_df.insert(1, 'Latitude', latitude_values)
            predictions_df.to_csv(os.path.join(predictions_dir, 'predictions_scalar.csv'), index=False)
            logger.info("Scalar predictions saved successfully")
        else:
            logger.warning("Scalar predictions not available, skipping scalar prediction save")
        
        # Save ground truth scalar with inverse transformation if available
        if 'y_scalar' in self.test_data:
            ground_truth_scalar_np = self.test_data['y_scalar'].cpu().numpy()
            
            # Get scalar column names for ground truth
            scalar_cols = self.data_info.get('y_list_scalar_columns', [f'scalar_{i}' for i in range(ground_truth_scalar_np.shape[1])])
            if len(scalar_cols) != ground_truth_scalar_np.shape[1]:
                scalar_cols = [f'scalar_{i}' for i in range(ground_truth_scalar_np.shape[1])]
            
            # Handle shape mismatch - ensure ground truth matches expected scalar count
            if num_expected_scalars > 0 and ground_truth_scalar_np.shape[1] > num_expected_scalars:
                print(f"Warning: Ground truth has {ground_truth_scalar_np.shape[1]} scalars but only {num_expected_scalars} expected. Using first {num_expected_scalars}.")
                ground_truth_scalar_np = ground_truth_scalar_np[:, :num_expected_scalars]
                scalar_cols = scalar_cols[:num_expected_scalars]
            
            # Apply inverse transformation to ground truth as well
            try:
                if 'y_scalar' in self.scalers and self.scalers['y_scalar'] is not None:
                    ground_truth_scalar_original = self.scalers['y_scalar'].inverse_transform(ground_truth_scalar_np)
                    logger.info("Applied inverse transformation to scalar ground truth")
                else:
                    ground_truth_scalar_original = ground_truth_scalar_np
                    logger.warning("No scalar scaler found, saving normalized ground truth values")
            except Exception as e:
                logger.warning(f"Failed to apply inverse transformation to scalar ground truth: {e}")
                ground_truth_scalar_original = ground_truth_scalar_np
            
            ground_truth_scalar_df = pd.DataFrame(ground_truth_scalar_original, columns=scalar_cols)
            if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(ground_truth_scalar_df):
                ground_truth_scalar_df.insert(0, 'Longitude', longitude_values)
                ground_truth_scalar_df.insert(1, 'Latitude', latitude_values)
            ground_truth_scalar_df.to_csv(os.path.join(predictions_dir, 'ground_truth_scalar.csv'), index=False)

        # Save pft_1d predictions if available
        if 'pft_1d' in predictions and predictions['pft_1d'].numel() > 0:
            predictions_pft_1d_np = predictions['pft_1d'].cpu().numpy()
            n_samples = predictions_pft_1d_np.shape[0]
            # Ensure 3D shape (samples, variables, pfts)
            if predictions_pft_1d_np.ndim == 2:
                predictions_pft_1d_np = predictions_pft_1d_np.reshape(n_samples, -1, 16)
            num_variables = predictions_pft_1d_np.shape[1]
            num_pfts = predictions_pft_1d_np.shape[2]
            # Get variable names from data_info if available
            var_names = self.data_info.get('y_list_columns_1d', [f'pft_1d_var_{i}' for i in range(num_variables)])
            if len(var_names) != num_variables:
                var_names = [f'pft_1d_var_{i}' for i in range(num_variables)]
            # Create a directory for pft_1d predictions
            pft_1d_dir = os.path.join(predictions_dir, 'pft_1d_predictions')
            os.makedirs(pft_1d_dir, exist_ok=True)
            # Save predictions for each variable separately
            for v in range(num_variables):
                var_name = var_names[v]
                var_predictions = predictions_pft_1d_np[:, v, :]
                
                # Apply inverse transformation to convert from normalized to original units
                try:
                    if 'y_pft_1d' in self.scalers and self.scalers['y_pft_1d'] is not None:
                        # Use manager method expecting (samples, pfts, variables) and PFT1..PFT16
                        per_var_3d = var_predictions.reshape(n_samples, num_pfts, 1)
                        canonical_pft_names = [f'PFT{i}' for i in range(1, 17)]
                        var_predictions_denorm = self.scalers['y_pft_1d'].inverse_transform_pft_1d(
                            per_var_3d, canonical_pft_names, [var_name]
                        )
                        var_predictions_original = var_predictions_denorm[:, :, 0]
                        logger.info(f"Applied inverse transformation to PFT 1D predictions for {var_name}")
                        # Debug dump of normalized vs denorm xsmrpool
                        try:
                            if var_name.endswith('xsmrpool') and os.getenv('DUMP_XSMRPOOL_DEBUG', '0') == '1':
                                import numpy as _np
                                debug_dir = os.path.join(pft_1d_dir, 'debug')
                                os.makedirs(debug_dir, exist_ok=True)
                                _np.savetxt(os.path.join(debug_dir, 'xsmrpool_norm.csv'), var_predictions, delimiter=',')
                                _np.savetxt(os.path.join(debug_dir, 'xsmrpool_denorm.csv'), var_predictions_original, delimiter=',')
                                logger.info("Dumped xsmrpool normalized and denormalized predictions for debugging")
                        except Exception as _e:
                            logger.warning(f"Failed xsmrpool debug dump: {_e}")
                    else:
                        var_predictions_original = var_predictions
                        logger.warning(f"No PFT 1D scaler found for {var_name}, saving normalized values")
                except Exception as e:
                    logger.warning(f"Failed to apply inverse transformation to PFT 1D predictions for {var_name}: {e}")
                    var_predictions_original = var_predictions
                
                columns = [f'{var_name}_pft{p+1}' for p in range(num_pfts)]
                var_df = pd.DataFrame(var_predictions_original, columns=columns)
                if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(var_df):
                    var_df.insert(0, 'Longitude', longitude_values)
                    var_df.insert(1, 'Latitude', latitude_values)
                var_df.to_csv(os.path.join(pft_1d_dir, f'predictions_{var_name}.csv'), index=False)
            # Save ground truth if available
            if 'y_pft_1d' in self.test_data:
                ground_truth_pft_1d_np = self.test_data['y_pft_1d'].cpu().numpy()
                if ground_truth_pft_1d_np.ndim == 2:
                    ground_truth_pft_1d_np = ground_truth_pft_1d_np.reshape(n_samples, -1, 16)
                pft_1d_gt_dir = os.path.join(predictions_dir, 'pft_1d_ground_truth')
                os.makedirs(pft_1d_gt_dir, exist_ok=True)
                for v in range(num_variables):
                    var_name = var_names[v]
                    var_gt = ground_truth_pft_1d_np[:, v, :]
                    
                    # Apply inverse transformation to ground truth as well
                    try:
                        if 'y_pft_1d' in self.scalers and self.scalers['y_pft_1d'] is not None:
                            per_var_3d = var_gt.reshape(n_samples, num_pfts, 1)
                            canonical_pft_names = [f'PFT{i}' for i in range(1, 17)]
                            var_gt_denorm = self.scalers['y_pft_1d'].inverse_transform_pft_1d(
                                per_var_3d, canonical_pft_names, [var_name]
                            )
                            var_gt_original = var_gt_denorm[:, :, 0]
                            logger.info(f"Applied inverse transformation to PFT 1D ground truth for {var_name}")
                        else:
                            var_gt_original = var_gt
                            logger.warning(f"No PFT 1D scaler found for {var_name}, saving normalized ground truth values")
                    except Exception as e:
                        logger.warning(f"Failed to apply inverse transformation to PFT 1D ground truth for {var_name}: {e}")
                        var_gt_original = var_gt
                    # Fallback: if inverse produced all-zeros but raw had signal, use raw
                    try:
                        nonzeros_pre = int((var_gt != 0).sum())
                        nonzeros_post = int((var_gt_original != 0).sum())
                        if nonzeros_pre > 0 and nonzeros_post == 0:
                            logger.warning(f"Inverse transform yielded all-zeros for PFT1D {var_name}; falling back to normalized ground truth")
                            var_gt_original = var_gt
                    except Exception:
                        pass
                    # Optional dump before saving GT
                    try:
                        if os.getenv('DUMP_ALL_PFT_SOIL', '0') == '1':
                            for i in range(var_gt_original.shape[0]):
                                logger.info(f"[pre-save GT] {var_name} row{i}: {','.join([f'{x:.6g}' for x in var_gt_original[i]])}")
                    except Exception as _e:
                        logger.warning(f"Pre-save GT dump failed for {var_name}: {_e}")
                    
                    columns = [f'{var_name}_pft{p+1}' for p in range(num_pfts)]
                    var_gt_df = pd.DataFrame(var_gt_original, columns=columns)
                    if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(var_gt_df):
                        var_gt_df.insert(0, 'Longitude', longitude_values)
                        var_gt_df.insert(1, 'Latitude', latitude_values)
                    var_gt_df.to_csv(os.path.join(pft_1d_gt_dir, f'ground_truth_{var_name}.csv'), index=False)
            logger.info("pft_1d predictions and ground truth saved separately for each variable and PFT")

        # Save soil_2d predictions if available
        if 'soil_2d' in predictions and predictions['soil_2d'].numel() > 0:
            predictions_soil_2d_np = predictions['soil_2d'].cpu().numpy()
            n_samples = predictions_soil_2d_np.shape[0]
            # Infer columns/layers from targets when possible
            if 'y_soil_2d' in self.test_data and self.test_data['y_soil_2d'].ndim >= 3:
                tgt_shape = self.test_data['y_soil_2d'].shape
                # shapes: (n, vars, cols, layers) or (n, vars, cols)
                num_columns = tgt_shape[2]
                num_layers = tgt_shape[3] if len(tgt_shape) > 3 else 1
            else:
                num_columns = 18
                num_layers = 10
            # Ensure 4D shape (samples, variables, columns, layers)
            if predictions_soil_2d_np.ndim == 4:
                num_variables = predictions_soil_2d_np.shape[1]
            elif predictions_soil_2d_np.ndim == 3:
                # (n, ?, ?) -> assume (?, cols*layers)
                flat_per_sample = predictions_soil_2d_np.shape[2]
                per_var = num_columns * num_layers
                assert flat_per_sample % per_var == 0, f"Unexpected soil_2d width {flat_per_sample} not divisible by cols*layers {per_var}"
                num_variables = flat_per_sample // per_var
                predictions_soil_2d_np = predictions_soil_2d_np.reshape(n_samples, num_variables, num_columns, num_layers)
            elif predictions_soil_2d_np.ndim == 2:
                flat_per_sample = predictions_soil_2d_np.shape[1]
                per_var = num_columns * num_layers
                assert flat_per_sample % per_var == 0, f"Unexpected soil_2d width {flat_per_sample} not divisible by cols*layers {per_var}"
                num_variables = flat_per_sample // per_var
                predictions_soil_2d_np = predictions_soil_2d_np.reshape(n_samples, num_variables, num_columns, num_layers)
            else:
                raise ValueError(f"Unsupported soil_2d prediction ndim: {predictions_soil_2d_np.ndim}")
            # Get variable names from data_info and align length
            var_names_all = self.data_info.get('y_list_columns_2d', [])
            if not var_names_all or len(var_names_all) < num_variables:
                var_names = [f'soil_2d_var_{i}' for i in range(num_variables)]
            else:
                var_names = list(var_names_all)[:num_variables]
            # Create a directory for soil_2d predictions
            soil_2d_dir = os.path.join(predictions_dir, 'soil_2d_predictions')
            os.makedirs(soil_2d_dir, exist_ok=True)
            # Save predictions for each variable separately
            for v in range(num_variables):
                var_name = var_names[v]
                var_predictions = predictions_soil_2d_np[:, v, :, :]

                # Apply inverse transformation to convert from normalized to original units
                try:
                    if 'y_soil_2d' in self.scalers and self.scalers['y_soil_2d'] is not None:
                        # Use the individual scaler manager per variable without assuming a fixed channel count
                        scaler_mgr = self.scalers['y_soil_2d']
                        per_var_tensor = var_predictions.reshape(n_samples, 1, num_columns, num_layers)
                        per_var_denorm = scaler_mgr.inverse_transform_soil_2d(per_var_tensor, [var_name], num_layers)
                        var_predictions_original = per_var_denorm[:, 0, :, :]
                        logger.info(f"Applied inverse transformation to soil 2D predictions for {var_name}")
                        # Optional debug: dump a layer vector for minerals and primp
                        try:
                            if (var_name in ['Y_sminn_vr','Y_smin_no3_vr','Y_smin_nh4_vr','Y_primp_vr']) and os.getenv('DUMP_SOIL_DEBUG','0')=='1':
                                import numpy as _np
                                dbg_dir = os.path.join(soil_2d_dir, 'debug')
                                os.makedirs(dbg_dir, exist_ok=True)
                                _np.savetxt(os.path.join(dbg_dir, f'{var_name}_row0_layers.csv'), var_predictions_original[0,0,:], delimiter=',')
                        except Exception as _e:
                            logger.warning(f"Failed soil debug dump for {var_name}: {_e}")
                    else:
                        var_predictions_original = var_predictions
                        logger.warning(f"No soil 2D scaler found for {var_name}, saving normalized values")
                except Exception as e:
                    logger.warning(f"Failed to apply inverse transformation to soil 2D predictions for {var_name}: {e}")
                    var_predictions_original = var_predictions
                
                # Reshape to 2D for CSV (samples, columns*layers)
                var_predictions_2d = var_predictions_original.reshape(n_samples, num_columns * num_layers)
                columns = [f'{var_name}_col{c+1}_layer{l+1}' for c in range(num_columns) for l in range(num_layers)]
                var_df = pd.DataFrame(var_predictions_2d, columns=columns)
                if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(var_df):
                    var_df.insert(0, 'Longitude', longitude_values)
                    var_df.insert(1, 'Latitude', latitude_values)
                var_df.to_csv(os.path.join(soil_2d_dir, f'predictions_{var_name}.csv'), index=False)
            # Save ground truth if available
            if 'y_soil_2d' in self.test_data:
                ground_truth_soil_2d_np = self.test_data['y_soil_2d'].cpu().numpy()
                if ground_truth_soil_2d_np.ndim == 4:
                    pass  # already (n, vars, cols, layers)
                elif ground_truth_soil_2d_np.ndim == 3:
                    # (n, vars, cols) -> expand layers=1
                    ground_truth_soil_2d_np = ground_truth_soil_2d_np.reshape(n_samples, ground_truth_soil_2d_np.shape[1], num_columns, num_layers)
                elif ground_truth_soil_2d_np.ndim == 2:
                    flat_per_sample = ground_truth_soil_2d_np.shape[1]
                    per_var = num_columns * num_layers
                    assert flat_per_sample % per_var == 0, f"Unexpected y_soil_2d width {flat_per_sample} not divisible by cols*layers {per_var}"
                    gt_num_variables = flat_per_sample // per_var
                    # Align with predictions if mismatch
                    if gt_num_variables != num_variables:
                        logger.warning(f"Mismatch in soil_2d vars (pred={num_variables}, gt={gt_num_variables}); aligning to min")
                        m = min(num_variables, gt_num_variables)
                        num_variables = m
                        var_names = var_names[:m]
                    ground_truth_soil_2d_np = ground_truth_soil_2d_np.reshape(n_samples, num_variables, num_columns, num_layers)
                else:
                    raise ValueError(f"Unsupported y_soil_2d ndim: {ground_truth_soil_2d_np.ndim}")
                soil_2d_gt_dir = os.path.join(predictions_dir, 'soil_2d_ground_truth')
                os.makedirs(soil_2d_gt_dir, exist_ok=True)
                for v in range(num_variables):
                    var_name = var_names[v]
                    var_gt = ground_truth_soil_2d_np[:, v, :, :]

                    # Apply inverse transformation to ground truth as well
                    try:
                        if 'y_soil_2d' in self.scalers and self.scalers['y_soil_2d'] is not None:
                            scaler_mgr = self.scalers['y_soil_2d']
                            per_var_tensor = var_gt.reshape(n_samples, 1, num_columns, num_layers)
                            per_var_denorm = scaler_mgr.inverse_transform_soil_2d(per_var_tensor, [var_name], num_layers)
                            var_gt_original = per_var_denorm[:, 0, :, :]
                            logger.info(f"Applied inverse transformation to soil 2D ground truth for {var_name}")
                        else:
                            var_gt_original = var_gt
                            logger.warning(f"No soil 2D scaler found for {var_name}, saving normalized ground truth values")
                    except Exception as e:
                        logger.warning(f"Failed to apply inverse transformation to soil 2D ground truth for {var_name}: {e}")
                        var_gt_original = var_gt
                    # Fallback: if inverse produced all-zeros but raw had signal, use raw
                    try:
                        nonzeros_pre = int((var_gt != 0).sum())
                        nonzeros_post = int((var_gt_original != 0).sum())
                        if nonzeros_pre > 0 and nonzeros_post == 0:
                            logger.warning(f"Inverse transform yielded all-zeros for Soil2D {var_name}; falling back to normalized ground truth")
                            var_gt_original = var_gt
                    except Exception:
                        pass
                    # Optional dump before saving GT (flatten columns*layers for readability)
                    try:
                        if os.getenv('DUMP_ALL_PFT_SOIL', '0') == '1':
                            flat = var_gt_original.reshape(n_samples, -1)
                            for i in range(flat.shape[0]):
                                logger.info(f"[pre-save GT] {var_name} row{i}: {','.join([f'{x:.6g}' for x in flat[i]])}")
                    except Exception as _e:
                        logger.warning(f"Pre-save Soil2D GT dump failed for {var_name}: {_e}")
                    
                    var_gt_2d = var_gt_original.reshape(n_samples, num_columns * num_layers)
                    columns = [f'{var_name}_col{c+1}_layer{l+1}' for c in range(num_columns) for l in range(num_layers)]
                    var_gt_df = pd.DataFrame(var_gt_2d, columns=columns)
                    if longitude_values is not None and latitude_values is not None and len(longitude_values) == len(var_gt_df):
                        var_gt_df.insert(0, 'Longitude', longitude_values)
                        var_gt_df.insert(1, 'Latitude', latitude_values)
                    var_gt_df.to_csv(os.path.join(soil_2d_gt_dir, f'ground_truth_{var_name}.csv'), index=False)
            logger.info("soil_2d predictions and ground truth saved separately for each variable, column, and layer")

        logger.info("All predictions saved successfully")
    
    def _save_scalers(self, predictions_dir: Path):
        """Save scalers to disk for future inverse transformation."""
        
        try:
            scalers_dir = predictions_dir / "scalers"
            scalers_dir.mkdir(exist_ok=True)
            
            for scaler_name, scaler in self.scalers.items():
                if scaler is not None:
                    scaler_file = scalers_dir / f"{scaler_name}_scaler.pkl"
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(scaler, f)
                    logger.info(f"Saved {scaler_name} scaler to {scaler_file}")
            
            # Save scaler metadata
            scaler_info = {}
            for scaler_name, scaler in self.scalers.items():
                if scaler is not None:
                    if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
                        # StandardScaler or RobustScaler
                        scaler_info[scaler_name] = {
                            'type': type(scaler).__name__,
                            'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                            'mean_': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                            'var_': scaler.var_.tolist() if hasattr(scaler, 'var_') else None
                        }
                    elif hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
                        # MinMaxScaler
                        scaler_info[scaler_name] = {
                            'type': type(scaler).__name__,
                            'scale_': scaler.scale_.tolist(),
                            'min_': scaler.min_.tolist(),
                            'data_min_': scaler.data_min_.tolist(),
                            'data_max_': scaler.data_max_.tolist()
                        }
            
            metadata_file = scalers_dir / "scaler_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(scaler_info, f, indent=2)
            
            logger.info(f"Scalers saved to {scalers_dir}")
            
        except Exception as e:
            logger.warning(f"Could not save scalers: {e}")
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue', linewidth=2)
        
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1, self.config.validation_frequency)
            plt.plot(val_epochs, self.val_losses, label='Validation Loss', color='red', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {save_path}")
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        try:
            logger.info("Starting training pipeline...")
            
            # Log initial GPU stats
            if self.config.log_gpu_memory:
                self.gpu_monitor.log_gpu_stats("Training start - ")
            
            # Initialize loss lists in instance variables
            self.train_losses = []
            self.val_losses = []
            
            # Initialize early stopping variables
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            
            logger.info(f"Starting training for {self.config.num_epochs} epochs...")
            
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                
                # Train
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch results
                logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Time: {epoch_time:.2f}s")
                
                # Log GPU stats periodically
                if (epoch % 5 == 0 and self.config.log_gpu_memory):
                    self.gpu_monitor.log_gpu_stats(f"Epoch {epoch+1} - ")
                
                # Check for early stopping (only if enabled in config)
                if self.config.use_early_stopping:
                    if self._check_early_stopping(val_loss):
                        logger.info("Early stopping triggered")
                        break
            
            logger.info("Training completed")
            
            # Final GPU stats
            if self.config.log_gpu_memory:
                self.gpu_monitor.log_gpu_stats("Training end - ")
            
            # Evaluate and save results
            logger.info("Evaluating model on test data...")
            predictions, metrics = self.evaluate()
            
            # Save results
            self.save_results(predictions, metrics)
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'predictions': predictions,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise 