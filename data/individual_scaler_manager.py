"""
Individual Scaler Manager for CNP Model

This module provides individual variable normalization instead of group-level normalization
to handle variables with large ranges across different scales. This is particularly 
important for the CNP model where variables like GPP, NPP, AR, and HR can have 
vastly different ranges.
"""

import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class IndividualScalerManager:
    """
    Manages individual scalers for each variable to prevent range compression
    and ensure optimal normalization for each variable type.
    """
    
    def __init__(self, normalization_type: str = 'minmax', minmax_range: Tuple[float, float] = (0, 1)):
        """
        Initialize the IndividualScalerManager.
        
        Args:
            normalization_type: Type of normalization ('minmax', 'standard', 'robust')
            minmax_range: Range for MinMaxScaler (default: [0, 1])
        """
        self.normalization_type = normalization_type
        self.minmax_range = minmax_range
        self.scalers = {}
        self.scaler_info = {}
        
        # Validate normalization type
        if normalization_type not in ['minmax', 'standard', 'robust']:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")
    
    def _create_scaler(self) -> Union[MinMaxScaler, StandardScaler, RobustScaler]:
        """Create a new scaler instance based on the normalization type."""
        if self.normalization_type == 'minmax':
            return MinMaxScaler(feature_range=self.minmax_range)
        elif self.normalization_type == 'standard':
            return StandardScaler()
        elif self.normalization_type == 'robust':
            return RobustScaler()
    
    def fit_transform_scalar(self, data: np.ndarray, variable_names: List[str]) -> np.ndarray:
        """
        Fit and transform each scalar variable individually.
        
        Args:
            data: Input data of shape (samples, variables)
            variable_names: List of variable names
            
        Returns:
            Normalized data of same shape as input
        """
        if data.shape[1] != len(variable_names):
            raise ValueError(f"Data has {data.shape[1]} columns but {len(variable_names)} variable names provided")
        
        normalized_data = np.zeros_like(data)
        
        for i, var_name in enumerate(variable_names):
            scaler = self._create_scaler()
            var_data = data[:, i:i+1]  # Single column
            
            # Fit and transform
            normalized_data[:, i:i+1] = scaler.fit_transform(var_data)
            
            # Store scaler with descriptive key
            scaler_key = f'scalar_{var_name}'
            self.scalers[scaler_key] = scaler
            
            # Store metadata
            self.scaler_info[scaler_key] = {
                'type': self.normalization_type,
                'variable_name': var_name,
                'data_type': 'scalar',
                'shape': var_data.shape,
                'fitted': True
            }
            
            logger.info(f"Created individual scaler for scalar variable: {var_name}")
        
        return normalized_data
    
    def transform_scalar(self, data: np.ndarray, variable_names: List[str]) -> np.ndarray:
        """
        Transform scalar data using existing fitted scalers (no fitting).
        
        Args:
            data: Input data of shape (samples, variables)
            variable_names: List of variable names
            
        Returns:
            Normalized data of same shape as input
        """
        if data.shape[1] != len(variable_names):
            raise ValueError(f"Data has {data.shape[1]} columns but {len(variable_names)} variable names provided")
        
        normalized_data = np.zeros_like(data)
        
        for i, var_name in enumerate(variable_names):
            scaler_key = f'scalar_{var_name}'
            
            if scaler_key not in self.scalers:
                raise KeyError(f"Scaler not found for scalar: {var_name}. Available keys: {list(self.scalers.keys())}")
            
            scaler = self.scalers[scaler_key]
            var_data = data[:, i:i+1]  # Single column
            
            # Transform using existing scaler
            normalized_data[:, i:i+1] = scaler.transform(var_data)
        
        return normalized_data
    
    def inverse_transform_scalar(self, data: np.ndarray, variable_names: List[str]) -> np.ndarray:
        """
        Inverse transform each variable using its individual scaler.
        
        Args:
            data: Normalized data of shape (samples, variables)
            variable_names: List of variable names
            
        Returns:
            Denormalized data in original scale
        """
        if data.shape[1] != len(variable_names):
            raise ValueError(f"Data has {data.shape[1]} columns but {len(variable_names)} variable names provided")
        
        denormalized_data = np.zeros_like(data)
        
        for i, var_name in enumerate(variable_names):
            scaler_key = f'scalar_{var_name}'
            
            if scaler_key not in self.scalers:
                raise KeyError(f"Scaler not found for variable: {var_name}. Available scalers: {list(self.scalers.keys())}")
            
            scaler = self.scalers[scaler_key]
            var_data = data[:, i:i+1]
            
            # Inverse transform
            denormalized_data[:, i:i+1] = scaler.inverse_transform(var_data)
        
        return denormalized_data
    
    def fit_transform_pft_1d(self, data: np.ndarray, pft_names: List[str], variable_names: List[str]) -> np.ndarray:
        """
        Normalize each PFT variable individually.
        
        Args:
            data: Input data of shape (samples, pfts, variables)
            pft_names: List of PFT names
            variable_names: List of variable names
            
        Returns:
            Normalized data of same shape as input
        """
        if data.shape[1] != len(pft_names) or data.shape[2] != len(variable_names):
            raise ValueError(f"Data shape {data.shape} doesn't match PFT names ({len(pft_names)}) and variable names ({len(variable_names)})")
        
        normalized_data = np.zeros_like(data)
        
        for var_idx, var_name in enumerate(variable_names):
            # Special case: use GROUP StandardScaler across all PFTs for xsmrpool to avoid degenerate per-PFT scaling
            if 'xsmrpool' in var_name:
                from sklearn.preprocessing import StandardScaler
                group_scaler = StandardScaler()
                all_pfts_flat = data[:, :, var_idx:var_idx+1].reshape(-1, 1)
                _ = group_scaler.fit(all_pfts_flat)
                # Apply shared scaler to each PFT and store same scaler under each key
                for pft_idx, pft_name in enumerate(pft_names):
                    var_data = data[:, pft_idx, var_idx:var_idx+1]
                    normalized_data[:, pft_idx, var_idx:var_idx+1] = group_scaler.transform(var_data)
                    scaler_key = f'pft1d_{pft_name}_{var_name}'
                    self.scalers[scaler_key] = group_scaler
                    self.scaler_info[scaler_key] = {
                        'type': 'standard',
                        'pft_name': pft_name,
                        'variable_name': var_name,
                        'data_type': 'pft1d',
                        'shape': var_data.shape,
                        'fitted': True,
                        'shared_group': True
                    }
                logger.info(f"Created shared GROUP StandardScaler for PFT1D variable: {var_name} across all PFTs")
                continue

            # Default per-PFT individual scaling for other variables
            for pft_idx, pft_name in enumerate(pft_names):
                scaler = self._create_scaler()
                var_data = data[:, pft_idx, var_idx:var_idx+1]
                normalized_data[:, pft_idx, var_idx:var_idx+1] = scaler.fit_transform(var_data)
                scaler_key = f'pft1d_{pft_name}_{var_name}'
                self.scalers[scaler_key] = scaler
                self.scaler_info[scaler_key] = {
                    'type': self.normalization_type,
                    'pft_name': pft_name,
                    'variable_name': var_name,
                    'data_type': 'pft1d',
                    'shape': var_data.shape,
                    'fitted': True
                }
                logger.info(f"Created individual scaler for PFT1D: {pft_name}_{var_name}")
        
        return normalized_data
    
    def transform_pft_1d(self, data: np.ndarray, pft_names: List[str], variable_names: List[str]) -> np.ndarray:
        """
        Transform PFT1D data using existing fitted scalers (no fitting).
        
        Args:
            data: Input data of shape (samples, pfts, variables)
            pft_names: List of PFT names
            variable_names: List of variable names
            
        Returns:
            Normalized data of same shape as input
        """
        if data.shape[1] != len(pft_names) or data.shape[2] != len(variable_names):
            raise ValueError(f"Data shape {data.shape} doesn't match PFT names ({len(pft_names)}) and variable names ({len(variable_names)})")
        
        normalized_data = np.zeros_like(data)
        
        for pft_idx, pft_name in enumerate(pft_names):
            for var_idx, var_name in enumerate(variable_names):
                scaler_key = f'pft1d_{pft_name}_{var_name}'
                
                if scaler_key not in self.scalers:
                    raise KeyError(f"Scaler not found for PFT1D: {pft_name}_{var_name}. Available keys: {list(self.scalers.keys())}")
                
                scaler = self.scalers[scaler_key]
                
                # Extract data for this PFT-variable combination
                var_data = data[:, pft_idx, var_idx:var_idx+1]
                
                # Transform using existing scaler
                normalized_data[:, pft_idx, var_idx:var_idx+1] = scaler.transform(var_data)
        
        return normalized_data
    
    def inverse_transform_pft_1d(self, data: np.ndarray, pft_names: List[str], variable_names: List[str]) -> np.ndarray:
        """
        Inverse transform PFT1D variables using individual scalers.
        
        Args:
            data: Normalized data of shape (samples, pfts, variables)
            pft_names: List of PFT names
            variable_names: List of variable names
            
        Returns:
            Denormalized data in original scale
        """
        if data.shape[1] != len(pft_names) or data.shape[2] != len(variable_names):
            raise ValueError(f"Data shape {data.shape} doesn't match PFT names ({len(pft_names)}) and variable names ({len(variable_names)})")
        
        denormalized_data = np.zeros_like(data)
        
        for pft_idx, pft_name in enumerate(pft_names):
            for var_idx, var_name in enumerate(variable_names):
                scaler_key = f'pft1d_{pft_name}_{var_name}'
                
                if scaler_key not in self.scalers:
                    raise KeyError(f"Scaler not found for PFT1D: {pft_name}_{var_name}")
                
                scaler = self.scalers[scaler_key]
                var_data = data[:, pft_idx, var_idx:var_idx+1]
                
                # Inverse transform with zero-range guard
                try:
                    # Handle MinMaxScaler with zero or near-zero range
                    if hasattr(scaler, 'data_range_'):
                        # data_range_ can be array; use first element
                        rng = float(np.max(scaler.data_range_)) if np.ndim(scaler.data_range_) else float(scaler.data_range_)
                        if rng <= 1e-12:
                            # Fallback to data_min_ (constant) to avoid producing NaNs/garbage
                            const_val = float(np.max(scaler.data_min_)) if hasattr(scaler, 'data_min_') else 0.0
                            out = np.full_like(var_data, const_val, dtype=var_data.dtype)
                            logger.warning(f"Zero-range MinMax scaler for {scaler_key}; using constant {const_val:.6g}")
                            denormalized_data[:, pft_idx, var_idx:var_idx+1] = out
                            continue
                    # Handle StandardScaler/RobustScaler with near-zero scale
                    if hasattr(scaler, 'scale_'):
                        sc = float(np.max(np.abs(scaler.scale_)))
                        if sc <= 1e-12:
                            const_val = float(np.max(scaler.mean_)) if hasattr(scaler, 'mean_') else 0.0
                            out = np.full_like(var_data, const_val, dtype=var_data.dtype)
                            logger.warning(f"Zero-scale standard/robust scaler for {scaler_key}; using mean {const_val:.6g}")
                            denormalized_data[:, pft_idx, var_idx:var_idx+1] = out
                            continue
                    # Normal path
                    denormalized_data[:, pft_idx, var_idx:var_idx+1] = scaler.inverse_transform(var_data)
                except Exception as e:
                    logger.warning(f"inverse_transform failed for {scaler_key}: {e}; passing through normalized values")
                    denormalized_data[:, pft_idx, var_idx:var_idx+1] = var_data
        
        return denormalized_data
    
    def fit_transform_soil_2d(self, data: np.ndarray, variable_names: List[str], num_layers: int) -> np.ndarray:
        """
        Normalize each soil variable at each layer individually.
        
        Args:
            data: Input data of shape (samples, variables, columns, layers)
            variable_names: List of variable names
            num_layers: Number of soil layers
            
        Returns:
            Normalized data of same shape as input
        """
        if data.shape[1] != len(variable_names):
            raise ValueError(f"Data shape {data.shape} doesn't match variable names ({len(variable_names)})")
        
        # FIT_TRANSFORM: Data is already standardized to (samples, variables, 1, 10) by the DataLoader
        # Just verify the structure and use the actual number of layers
        if data.shape[2] == 1 and data.shape[3] == 10:
            logger.info(f"Processing standardized soil2D structure: {data.shape[2]} column, {data.shape[3]} layers")
            num_layers = data.shape[3]  # Use actual number of layers (10)
            logger.info(f"Using {num_layers} layers for processing")
        else:
            logger.warning(f"Unexpected soil2D structure: {data.shape}. Expected (samples, variables, 1, 10)")
            raise ValueError(f"Data shape {data.shape} doesn't match expected structure (samples, variables, 1, 10)")
        
        normalized_data = np.zeros_like(data)
        
        for var_idx, var_name in enumerate(variable_names):
            for layer_idx in range(num_layers):
                scaler = self._create_scaler()
                
                # Extract data for this variable-layer combination
                layer_data = data[:, var_idx, :, layer_idx:layer_idx+1]
                
                # Reshape for scaler (samples, features)
                layer_data_reshaped = layer_data.reshape(layer_data.shape[0], -1)
                
                # Fit and transform
                normalized_layer = scaler.fit_transform(layer_data_reshaped)
                # Quick fix: replace NaN/Inf produced by degenerate scaling with zeros
                normalized_layer = np.nan_to_num(normalized_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Reshape back to original shape
                normalized_data[:, var_idx, :, layer_idx:layer_idx+1] = normalized_layer.reshape(layer_data.shape)
                
                # Store scaler with descriptive key
                scaler_key = f'soil2d_{var_name}_layer{layer_idx}'
                self.scalers[scaler_key] = scaler
                
                # Store metadata
                self.scaler_info[scaler_key] = {
                    'type': self.normalization_type,
                    'variable_name': var_name,
                    'layer_idx': layer_idx,
                    'data_type': 'soil2d',
                    'shape': layer_data.shape,
                    'fitted': True
                }
                
                logger.info(f"Created individual scaler for Soil2D: {var_name}_layer{layer_idx}")
        
        return normalized_data
    
    def transform_soil_2d(self, data: np.ndarray, variable_names: List[str], num_layers: int) -> np.ndarray:
        """
        Transform Soil2D data using existing fitted scalers (no fitting).
        
        Args:
            data: Input data of shape (samples, variables, columns, layers)
            variable_names: List of variable names
            num_layers: Number of soil layers
            
        Returns:
            Normalized data of same shape as input
        """
        if data.shape[1] != len(variable_names):
            raise ValueError(f"Data has {data.shape[1]} variables but {len(variable_names)} variable names provided")
        
        if data.shape[3] != num_layers:
            raise ValueError(f"Data has {data.shape[3]} layers but {num_layers} layers expected")
        
        normalized_data = np.zeros_like(data)
        
        for var_idx, var_name in enumerate(variable_names):
            for layer_idx in range(num_layers):
                scaler_key = f'soil2d_{var_name}_layer{layer_idx}'
                
                if scaler_key not in self.scalers:
                    raise KeyError(f"Scaler not found for Soil2D: {var_name}_layer{layer_idx}. Available keys: {list(self.scalers.keys())}")
                
                scaler = self.scalers[scaler_key]
                
                # Extract data for this variable-layer combination
                layer_data = data[:, var_idx, :, layer_idx:layer_idx+1]  # Shape: (samples, columns, 1)
                original_shape = layer_data.shape
                layer_data_flat = layer_data.reshape(-1, 1)  # Flatten for scaler
                
                # Transform using existing scaler
                normalized_flat = scaler.transform(layer_data_flat)
                normalized_data[:, var_idx, :, layer_idx:layer_idx+1] = normalized_flat.reshape(original_shape)
        
        return normalized_data
    
    def inverse_transform_soil_2d(self, data: np.ndarray, variable_names: List[str], num_layers: int) -> np.ndarray:
        """
        Inverse transform Soil2D variables using individual scalers.
        
        Args:
            data: Normalized data of shape (samples, variables, columns, layers)
            variable_names: List of variable names
            num_layers: Number of soil layers
            
        Returns:
            Denormalized data in original scale
        """
        if data.shape[1] != len(variable_names):
            raise ValueError(f"Data shape {data.shape} doesn't match variable names ({len(variable_names)})")
        
        # INVERSE_TRANSFORM: Data is already standardized to (samples, variables, 1, 10) by the DataLoader
        # Just verify the structure and use the actual number of layers
        if data.shape[2] == 1 and data.shape[3] == 10:
            logger.info(f"Processing standardized soil2D structure: {data.shape[2]} column, {data.shape[3]} layers")
            num_layers = data.shape[3]  # Use actual number of layers (10)
            logger.info(f"Using {num_layers} layers for processing")
        else:
            logger.warning(f"Unexpected soil2D structure: {data.shape}. Expected (samples, variables, 1, 10)")
            raise ValueError(f"Data shape {data.shape} doesn't match expected structure (samples, variables, 1, 10)")
        
        denormalized_data = np.zeros_like(data)
        
        for var_idx, var_name in enumerate(variable_names):
            for layer_idx in range(num_layers):
                scaler_key = f'soil2d_{var_name}_layer{layer_idx}'
                
                if scaler_key not in self.scalers:
                    raise KeyError(f"Scaler not found for Soil2D: {var_name}_layer{layer_idx}")
                
                scaler = self.scalers[scaler_key]
                layer_data = data[:, var_idx, :, layer_idx:layer_idx+1]
                
                # Reshape for scaler (samples, features)
                layer_data_reshaped = layer_data.reshape(layer_data.shape[0], -1)
                
                # Inverse transform with zero-range guard
                try:
                    if hasattr(scaler, 'data_range_'):
                        rng = float(np.max(scaler.data_range_)) if np.ndim(scaler.data_range_) else float(scaler.data_range_)
                        if rng <= 1e-12:
                            const_val = float(np.max(scaler.data_min_)) if hasattr(scaler, 'data_min_') else 0.0
                            denormalized_layer = np.full_like(layer_data_reshaped, const_val)
                            logger.warning(f"Zero-range MinMax scaler for {scaler_key}; using constant {const_val:.6g}")
                        else:
                            denormalized_layer = scaler.inverse_transform(layer_data_reshaped)
                    elif hasattr(scaler, 'scale_'):
                        sc = float(np.max(np.abs(scaler.scale_)))
                        if sc <= 1e-12:
                            const_val = float(np.max(scaler.mean_)) if hasattr(scaler, 'mean_') else 0.0
                            denormalized_layer = np.full_like(layer_data_reshaped, const_val)
                            logger.warning(f"Zero-scale standard/robust scaler for {scaler_key}; using mean {const_val:.6g}")
                        else:
                            denormalized_layer = scaler.inverse_transform(layer_data_reshaped)
                    else:
                        denormalized_layer = scaler.inverse_transform(layer_data_reshaped)
                except Exception as e:
                    logger.warning(f"inverse_transform failed for {scaler_key}: {e}; passing through normalized values")
                    denormalized_layer = layer_data_reshaped
                # Safety: clean NaN/Inf
                denormalized_layer = np.nan_to_num(denormalized_layer, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Reshape back to original shape
                denormalized_data[:, var_idx, :, layer_idx:layer_idx+1] = denormalized_layer.reshape(layer_data.shape)
        
        return denormalized_data
    
    def get_scaler_info(self) -> Dict:
        """Get comprehensive information about all scalers."""
        return self.scaler_info.copy()
    
    def validate_scalers(self) -> bool:
        """Validate that all scalers are properly fitted."""
        if not self.scalers:
            return False
        
        for scaler_key, scaler in self.scalers.items():
            if not hasattr(scaler, 'scale_'):
                logger.warning(f"Scaler {scaler_key} is not properly fitted")
                return False
        
        return True
    
    def save_scalers(self, directory: Union[str, Path]) -> None:
        """
        Save all scalers to disk.
        
        Args:
            directory: Directory to save scalers
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save individual scalers
        for scaler_key, scaler in self.scalers.items():
            scaler_file = directory / f"{scaler_key}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved scaler: {scaler_file}")
        
        # Save metadata
        metadata_file = directory / "scaler_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.scaler_info, f, indent=2, default=str)
        logger.info(f"Saved scaler metadata: {metadata_file}")
    
    def load_scalers(self, directory: Union[str, Path]) -> None:
        """
        Load all scalers from disk.
        
        Args:
            directory: Directory containing saved scalers
        """
        directory = Path(directory)
        
        # Load metadata first
        metadata_file = directory / "scaler_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.scaler_info = json.load(f)
            logger.info(f"Loaded scaler metadata: {metadata_file}")
        
        # Load individual scalers
        for scaler_key in self.scaler_info.keys():
            scaler_file = directory / f"{scaler_key}.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.scalers[scaler_key] = pickle.load(f)
                logger.info(f"Loaded scaler: {scaler_file}")
            else:
                logger.warning(f"Scaler file not found: {scaler_file}")
    
    def inverse_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generic inverse transform method for backward compatibility.
        
        This method automatically detects the data type and calls the appropriate
        specific inverse transform method.
        
        Args:
            data: Input data to denormalize
            **kwargs: Additional arguments passed to specific methods
            
        Returns:
            Denormalized data in original scale
        """
        # Try to detect data type from shape and available scalers
        if data.ndim == 2:
            # Could be scalar or PFT1D data
            if data.shape[1] <= 10:  # Likely scalar data
                # Try to infer variable names from available scalers
                scalar_keys = [k for k in self.scalers.keys() if k.startswith('scalar_')]
                if scalar_keys:
                    # Extract variable names from scaler keys
                    variable_names = [k.replace('scalar_', '') for k in scalar_keys]
                    # Use the first few variables that match the data shape
                    variable_names = variable_names[:data.shape[1]]
                    logger.info(f"Auto-detected scalar data with variables: {variable_names}")
                    return self.inverse_transform_scalar(data, variable_names)
            
            # Check if it's PFT1D data (should have 16 or 48 features for 16 PFTs)
            if data.shape[1] in [16, 48]:  # 16 PFTs or 16 PFTs × 3 variables
                # Try to infer PFT and variable names
                pft_keys = [k for k in self.scalers.keys() if k.startswith('pft1d_')]
                if pft_keys:
                    # Extract PFT and variable names from first few keys
                    pft_names = [f'PFT{i}' for i in range(16)]
                    
                    # Extract variable names from the actual scaler keys
                    # Format is: pft1d_PFT0_Y_tlai, pft1d_PFT0_Y_deadstemc, etc.
                    # For data shape (samples, 16), we have 1 variable across 16 PFTs
                    # For data shape (samples, 48), we have 3 variables across 16 PFTs
                    
                    if data.shape[1] == 16:
                        # Single variable case - need to determine which variable
                        # Look at the first PFT0 scaler to get the variable name
                        pft0_keys = [k for k in pft_keys if k.startswith('pft1d_PFT0_')]
                        if pft0_keys:
                            # Extract variable name from first PFT0 key
                            # Key format: pft1d_PFT0_Y_tlai -> extract Y_tlai
                            var_name = pft0_keys[0].replace('pft1d_PFT0_', '')
                            variable_names = [var_name]
                            logger.info(f"Auto-detected PFT1D data with 16 PFTs and 1 variable: {variable_names}")
                            # Reshape to (samples, pfts, variables) = (samples, 16, 1)
                            data_reshaped = data.reshape(data.shape[0], 16, 1)
                        else:
                            raise ValueError("Could not determine variable name from PFT0 scalers")
                    elif data.shape[1] == 64:
                        # All 4 variables together: 4 variables × 16 PFTs
                        # Extract variable names from PFT0 keys
                        pft0_keys = [k for k in pft_keys if k.startswith('pft1d_PFT0_')]
                        variable_names = [k.replace('pft1d_PFT0_', '') for k in pft0_keys]
                        logger.info(f"Auto-detected PFT1D data with 16 PFTs and 4 variables: {variable_names}")
                        # Reshape to (samples, pfts, variables) = (samples, 16, 4)
                        data_reshaped = data.reshape(data.shape[0], 16, 4)
                    else:
                        # Multiple variables case (48 features = 16 PFTs × 3 variables)
                        variables = set()
                        for key in pft_keys:
                            parts = key.split('_')
                            if len(parts) >= 3:
                                var_name = '_'.join(parts[2:])  # Everything after PFT0, PFT1, etc.
                                variables.add(var_name)
                        
                        variable_names = list(variables)
                        logger.info(f"Auto-detected PFT1D data with 16 PFTs and {len(variable_names)} variables: {variable_names}")
                        data_reshaped = data.reshape(data.shape[0], 16, len(variable_names))
                    
                    return self.inverse_transform_pft_1d(data_reshaped, pft_names, variable_names)
            
            # Check if it's soil 2D data (samples, features) where features = variables × columns × layers
            # For new structure: 3 variables × 2 columns × 15 layers = 90 features (but we only use 1 column × 10 layers = 30 features)
            soil_keys = [k for k in self.scalers.keys() if k.startswith('soil2d_')]
            if soil_keys and data.shape[1] > 20:  # Likely soil 2D data (adjusted threshold)
                # Extract variable names and determine number of layers
                variables = set()
                layers = set()
                for key in soil_keys:
                    parts = key.split('_')
                    if len(parts) >= 3:
                        # Key format: soil2d_cwdc_vr_layer0 -> extract cwdc_vr
                        var_name = '_'.join(parts[1:-1])  # Everything between soil2d_ and layer
                        layer_part = parts[-1]
                        if layer_part.startswith('layer'):
                            layer_num = int(layer_part.replace('layer', ''))
                            variables.add(var_name)
                            layers.add(layer_num)
                
                variable_names = list(variables)
                num_layers = max(layers) + 1 if layers else 10
                
                # Calculate number of columns per variable
                # Total features = variables × columns × layers
                # For new structure: 30 = 3 × 1 × 10 (first column, top 10 layers)
                # For old structure: 540 = 3 × 18 × 10
                num_columns = data.shape[1] // (len(variable_names) * num_layers)
                
                logger.info(f"Auto-detected soil 2D data with variables: {variable_names}, {num_columns} columns, and {num_layers} layers")
                
                # Reshape data to (samples, variables, columns, layers)
                # Structure: samples × 3 variables × 1 column × 10 layers (new structure)
                data_reshaped = data.reshape(data.shape[0], len(variable_names), num_columns, num_layers)
                return self.inverse_transform_soil_2d(data_reshaped, variable_names, num_layers)
        
        elif data.ndim == 4:
            # Likely soil 2D data (samples, variables, columns, layers)
            soil_keys = [k for k in self.scalers.keys() if k.startswith('soil2d_')]
            if soil_keys:
                # Extract variable names and determine number of layers
                # Count unique variables
                variables = set()
                layers = set()
                for key in soil_keys:
                    parts = key.split('_')
                    if len(parts) >= 3:
                        # Key format: soil2d_cwdc_vr_layer0 -> extract cwdc_vr
                        var_name = '_'.join(parts[1:-1])  # Everything between soil2d_ and layer
                        layer_part = parts[-1]
                        if layer_part.startswith('layer'):
                            layer_num = int(layer_part.replace('layer', ''))
                            variables.add(var_name)
                            layers.add(layer_num)
                
                variable_names = list(variables)
                num_layers = max(layers) + 1 if layers else 10
                
                logger.info(f"Auto-detected soil 2D data with variables: {variable_names} and {num_layers} layers")
                return self.inverse_transform_soil_2d(data, variable_names, num_layers)
        
        # If we can't auto-detect, try to provide helpful error message
        available_methods = [method for method in dir(self) if method.startswith('inverse_transform_')]
        raise ValueError(
            f"Could not auto-detect data type for shape {data.shape}. "
            f"Available specific methods: {available_methods}. "
            f"Please use the appropriate specific method or provide additional context."
        )
    

    
    def __len__(self) -> int:
        """Return the number of scalers."""
        return len(self.scalers)
    
    def __contains__(self, key: str) -> bool:
        """Check if a scaler key exists."""
        return key in self.scalers
    
    def __getitem__(self, key: str):
        """Get a scaler by key."""
        return self.scalers[key]
    
    def keys(self):
        """Get all scaler keys."""
        return self.scalers.keys()
    
    def values(self):
        """Get all scalers."""
        return self.scalers.values()
    
    def items(self):
        """Get all scaler key-value pairs."""
        return self.scalers.items()
