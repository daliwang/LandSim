"""
Data loader module for model training with Individual Variable Normalization.

This module handles data loading, preprocessing, and preparation for training,
using individual scalers for each variable to prevent range compression.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import warnings
from pathlib import Path
import pickle

from config.training_config import DataConfig, PreprocessingConfig
from data.individual_scaler_manager import IndividualScalerManager

logger = logging.getLogger(__name__)


class DataLoaderIndividual:
    """
    Flexible data loader for climate model training with individual variable normalization.
    
    This class handles loading, preprocessing, and preparing data for training
    with different input and output configurations, using individual scalers
    for each variable to prevent range compression.
    """
    
    def __init__(self, data_config: DataConfig, preprocessing_config: PreprocessingConfig):
        """
        Initialize the data loader.
        
        Args:
            data_config: Data configuration
            preprocessing_config: Preprocessing configuration
        """
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config
        self.df = None
        self.scalers = {}
        
        # Initialize individual scaler managers
        scalar_norm = self.preprocessing_config.scalar_normalization if hasattr(self.preprocessing_config, 'scalar_normalization') else self.preprocessing_config.static_normalization
        list_1d_norm = self.preprocessing_config.list_1d_normalization
        list_2d_norm = self.preprocessing_config.list_2d_normalization
        
        self.individual_scalers = {
            'scalar': IndividualScalerManager(normalization_type=scalar_norm),
            'y_scalar': IndividualScalerManager(normalization_type=scalar_norm),
            'pft_1d': IndividualScalerManager(normalization_type=list_1d_norm),
            'y_pft_1d': IndividualScalerManager(normalization_type=list_1d_norm),
            'soil_2d': IndividualScalerManager(normalization_type=list_2d_norm),
            'y_soil_2d': IndividualScalerManager(normalization_type=list_2d_norm),
        }
        
        # Validate configurations
        self._validate_configs()
    
    def _validate_configs(self):
        """Validate data and preprocessing configurations."""
        if not self.data_config.data_paths:
            raise ValueError("Data paths cannot be empty")
        if not self.data_config.time_series_columns:
            raise ValueError("Time series columns cannot be empty")
        # Check for matching input/output pairs for 1D
        if len(self.data_config.x_list_columns_1d) != len(self.data_config.y_list_columns_1d):
            raise ValueError("Number of 1D input (x_list_columns_1d) and output columns must match")
        # Relaxed check for 2D: all outputs must be in inputs, but inputs can have extras
        def strip_y(col):
            return col[2:] if col.startswith('Y_') else col
        x2d_set = set(self.data_config.x_list_columns_2d)
        missing_outputs = [col for col in self.data_config.y_list_columns_2d if strip_y(col) not in x2d_set]
        if missing_outputs:
            raise ValueError(f"The following 2D output columns (after removing 'Y_') are not present in 2D input columns: {missing_outputs}")
        if len(self.data_config.x_list_columns_2d) != len(self.data_config.y_list_columns_2d):
            logger.warning(f"Number of 2D input columns ({len(self.data_config.x_list_columns_2d)}) does not match number of 2D output columns ({len(self.data_config.y_list_columns_2d)}). This is allowed if some 2D inputs are input-only.")
    
    def check_nans(self):
        """Check for NaN values in the loaded DataFrame and log the count per column."""
        if self.df is None:
            logger.warning("No data loaded to check for NaNs.")
            return
        nan_counts = self.df.isna().sum()
        total_nans = nan_counts.sum()
        logger.info(f"Total NaN values in DataFrame: {total_nans}")
        logger.info("NaN count per column:")
        for col, count in nan_counts.items():
            if count > 0:
                logger.info(f"  {col}: {count}")

    def load_data(self) -> pd.DataFrame:
        """Load data from configured paths and patterns."""
        df_list = []
        logger.info("Loading data from multiple paths...")
        for path in self.data_config.data_paths:
            # Resolve files matching pattern
            # Support per-dataset file patterns if provided
            try:
                per_dataset_patterns = getattr(self.data_config, 'dataset_file_patterns', {}) or {}
            except Exception:
                per_dataset_patterns = {}
            pattern = per_dataset_patterns.get(path, self.data_config.file_pattern)
            files = list(Path(path).glob(pattern))
            # Deterministic ordering for test runs
            if getattr(self.data_config, 'sort_file_list', True):
                files = sorted(files, key=lambda p: p.name)
            # Optional cap on number of files
            num_files = len(files)
            logger.info(f"Found {num_files} files in {path}")
            
            # Apply max_files limit if specified
            if hasattr(self.data_config, 'max_files') and self.data_config.max_files is not None:
                files = files[:self.data_config.max_files]
                logger.info(f"Limited to {len(files)} files due to max_files={self.data_config.max_files}")
            
            # Load each file
            for file_path in files:
                try:
                    # Check file extension and use appropriate loading method
                    if str(file_path).endswith('.pkl'):
                        df_chunk = pd.read_pickle(file_path)
                    elif str(file_path).endswith('.parquet'):
                        df_chunk = pd.read_parquet(file_path)
                    else:
                        # Try pickle first, then parquet as fallback
                        try:
                            df_chunk = pd.read_pickle(file_path)
                        except:
                            df_chunk = pd.read_parquet(file_path)
                    
                    # Load all files - zeros are valid data in soil science
                    df_list.append(df_chunk)
                    logger.debug(f"Loaded {len(df_chunk)} samples from {file_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue
        
        if not df_list:
            raise ValueError("No data files could be loaded")
        
        # Combine all dataframes
        self.df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Successfully loaded {len(self.df)} samples")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the loaded data."""
        logger.info("Starting data preprocessing...")
        
        # Filter samples by longitude if specified
        if hasattr(self.data_config, 'longitudes_to_drop') and self.data_config.longitudes_to_drop:
            if 'Longitude' in self.df.columns:
                original_size = len(self.df)
                longitudes_to_drop = self.data_config.longitudes_to_drop
                logger.info(f"Filtering samples with longitudes: {longitudes_to_drop}")
                
                # Create a mask for samples to keep (those NOT in the drop list)
                # Use a tolerance for floating point comparison
                tolerance = 0.01
                mask = ~self.df['Longitude'].apply(
                    lambda lon: any(abs(lon - drop_lon) < tolerance for drop_lon in longitudes_to_drop)
                )
                
                self.df = self.df[mask].reset_index(drop=True)
                filtered_size = len(self.df)
                dropped_count = original_size - filtered_size
                logger.info(f"Longitude filtering: {original_size} samples -> {filtered_size} samples (dropped {dropped_count} samples)")
            else:
                logger.warning("'Longitude' column not found in dataset. Cannot apply longitude filtering.")
        
        # Drop specified columns
        if hasattr(self.data_config, 'filter_columns') and self.data_config.filter_columns:
            for col in self.data_config.filter_columns:
                if col in self.df.columns:
                    self.df = self.df.drop(columns=[col])
                    logger.info(f"Dropped column: {col}")
                else:
                    logger.warning(f"Filter column '{col}' not found in dataset")
        
        # Process time series data
        logger.info("Processing time series data...")
        for col in self.data_config.time_series_columns:
            if col in self.df.columns:
                # Ensure time series data is properly formatted
                def _to_ts_and_truncate(x):
                    # Convert to numpy array (float32) and truncate/pad to configured time_series_length
                    target_len = int(getattr(self.data_config, 'time_series_length', 240))
                    if isinstance(x, (list, np.ndarray)):
                        arr = np.array(x, dtype=np.float32).flatten()
                        # Prefer earliest 20-year window as per repeated forcing spec
                        if arr.size >= target_len:
                            arr = arr[:target_len]
                        else:
                            # pad to target_len with zeros at the end
                            pad = target_len - arr.size
                            if pad > 0:
                                arr = np.pad(arr, (0, pad), mode='constant')
                        # ensure no NaN/Inf
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        return arr
                    # Fallback to zeros of target length
                    return np.zeros(target_len, dtype=np.float32)

                self.df[col] = self.df[col].apply(_to_ts_and_truncate)
        
        # Process list columns
        logger.info("Processing list columns...")
        list_columns = (
            self.data_config.x_list_columns_1d + 
            self.data_config.y_list_columns_1d +
            self.data_config.x_list_columns_2d + 
            self.data_config.y_list_columns_2d
        )
        
        for col in list_columns:
            if col in self.df.columns:
                if col in self.data_config.x_list_columns_1d or col in self.data_config.y_list_columns_1d:
                    # 1D list processing
                    self.df[col] = self.df[col].apply(
                        lambda x: self._pad_1d_array(x, self.data_config.max_1d_length)
                    )
                elif col in self.data_config.x_list_columns_2d or col in self.data_config.y_list_columns_2d:
                    # 2D list processing
                    self.df[col] = self.df[col].apply(
                        lambda x: self._pad_2d_array(x, self.data_config.max_2d_rows, self.data_config.max_2d_cols)
                    )
        
        # Shuffle data
        self._shuffle_data()
        logger.info("Data preprocessing completed")

        # Optional raw dump of PFT1D (17 PFTs inc. PFT0) and Soil2D (first col, top 10 layers)
        try:
            if os.getenv('DUMP_ALL_PFT_SOIL', '0') == '1':
                self._dump_raw_pft1d_soil2d()
        except Exception as _e:
            logger.warning(f"Raw dump failed: {_e}")
        return self.df

    def _dump_raw_pft1d_soil2d(self) -> None:
        """Print all PFT1D (17 elements incl. PFT0) and Soil2D (first col, top 10 layers) from raw DataFrame.
        Intended for small datasets (<=10 samples). Controlled by env DUMP_ALL_PFT_SOIL=1.
        """
        if self.df is None:
            return
        # PFT1D raw (expect length 17 with PFT0 first)
        pft1d_cols = list(self.data_config.x_list_columns_1d) + list(self.data_config.y_list_columns_1d)
        pft1d_cols = [c for c in pft1d_cols if c in getattr(self, 'df', pd.DataFrame()).columns]
        if pft1d_cols:
            logger.info("[RAW] PFT1D (17 incl. PFT0) per variable - all rows")
        for col in pft1d_cols:
            try:
                rows = []
                for v in self.df[col].values:
                    arr = np.array(v) if isinstance(v, (list, np.ndarray)) else np.zeros(17, dtype=float)
                    # Pad/truncate to 17
                    if arr.ndim == 1:
                        if arr.shape[0] < 17:
                            arr = np.pad(arr, (0, 17 - arr.shape[0]), mode='constant')
                        else:
                            arr = arr[:17]
                    else:
                        arr = np.zeros(17, dtype=float)
                    rows.append(arr)
                mat = np.stack(rows)
                logger.info(f"[RAW] {col}: shape={mat.shape}")
                for i in range(mat.shape[0]):
                    logger.info(f"  row{i}: {','.join([f'{x:.6g}' for x in mat[i]])}")
            except Exception as e:
                logger.warning(f"[RAW] Failed PFT1D dump for {col}: {e}")
        # Soil2D raw (first column, top 10 layers)
        soil2d_cols = list(self.data_config.x_list_columns_2d) + list(self.data_config.y_list_columns_2d)
        soil2d_cols = [c for c in soil2d_cols if c in getattr(self, 'df', pd.DataFrame()).columns]
        if soil2d_cols:
            logger.info("[RAW] Soil2D (first column, top 10 layers) per variable - all rows")
        for col in soil2d_cols:
            try:
                rows = []
                for v in self.df[col].values:
                    try:
                        arr = np.array(v)
                        if arr.ndim == 2 and arr.shape[0] >= 1:
                            take = min(10, arr.shape[1])
                            out = np.zeros(10, dtype=float)
                            out[:take] = arr[0, :take]
                        else:
                            out = np.zeros(10, dtype=float)
                    except Exception:
                        out = np.zeros(10, dtype=float)
                    rows.append(out)
                mat = np.stack(rows)
                logger.info(f"[RAW] {col}: shape={mat.shape}")
                for i in range(mat.shape[0]):
                    logger.info(f"  row{i}: {','.join([f'{x:.6g}' for x in mat[i]])}")
            except Exception as e:
                logger.warning(f"[RAW] Failed Soil2D dump for {col}: {e}")
    
    def _pad_1d_array(self, x: Any, target_length: int) -> np.ndarray:
        """Pad 1D array to target length."""
        if isinstance(x, (list, np.ndarray)):
            x_array = np.array(x)
            
            # For PFT data (target_length=16), drop PFT0 before truncation/padding
            # Raw data contains PFT0-PFT16, but model expects PFT1-PFT16
            if target_length == 16 and len(x_array) >= 17:
                x_array = x_array[1:]  # Drop PFT0, keep PFT1-PFT16
            
            if len(x_array) < target_length:
                result = np.pad(x_array, (0, target_length - len(x_array)), mode='constant')
            else:
                result = x_array[:target_length]
                    
            return result
        else:
            return np.zeros(target_length, dtype=np.float32)
    
    def _pad_2d_array(self, x: Any, target_rows: int, target_cols: int) -> np.ndarray:
        """Pad 2D array to target shape."""
        # Handle nested lists (column x layers) and arrays robustly.
        try:
            # Case 1: proper 2D ndarray
            if isinstance(x, np.ndarray) and x.ndim == 2:
                arr = x
            else:
                # Try to convert list/tuple to ndarray without ragged coercion
                if isinstance(x, (list, tuple)):
                    # If it's a list of lists (columns x layers), extract first column
                    if len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray)):
                        first_col = np.array(x[0], dtype=float).reshape(1, -1)
                        arr = first_col
                    else:
                        # Flat list: treat as 1xN layers
                        arr = np.array(x, dtype=float).reshape(1, -1)
                else:
                    arr = None

            if arr is None or arr.ndim != 2:
                return np.zeros((target_rows, target_cols), dtype=np.float32)

            # Ensure we only keep first row (first group/column)
            arr = arr[:1, :]
            # Pad/truncate to target_cols
            if arr.shape[1] < target_cols:
                arr = np.pad(arr, ((0, 0), (0, target_cols - arr.shape[1])), mode='constant')
            else:
                arr = arr[:, :target_cols]

            # Finally, pad rows to target_rows (usually 1)
            if arr.shape[0] < target_rows:
                arr = np.pad(arr, ((0, target_rows - arr.shape[0]), (0, 0)), mode='constant')
            else:
                arr = arr[:target_rows, :]

            return arr.astype(np.float32, copy=False)
        except Exception:
            return np.zeros((target_rows, target_cols), dtype=np.float32)
    
    def _shuffle_data(self):
        """Shuffle the dataset."""
        logger.info("Shuffling dataset...")
        self.df = shuffle(self.df, random_state=self.data_config.random_state).reset_index(drop=True)
    
    def normalize_data(self) -> Dict[str, Any]:
        """
        Normalize all data types using group normalization (default).
        Returns:
            Dictionary containing normalized data and scalers
        """
        logger.info("Normalizing data using group normalization...")

        # Time series (keep group normalization for now)
        time_series_data, time_series_scaler = self._normalize_time_series()
        
        # Static (keep group normalization for now)
        static_data, static_scaler = self._normalize_static(self.data_config.static_columns)

        # Scalar - Use group normalization
        scalar_data, scalar_scaler = self._normalize_scalar()
        y_scalar_data, y_scalar_scaler = self._normalize_y_scalar()

        # 1D PFT - Use group normalization
        pft_1d_data, pft_1d_scaler = self._normalize_list_1d(self.data_config.x_list_columns_1d)
        y_pft_1d_data, y_pft_1d_scaler = self._normalize_list_1d(self.data_config.y_list_columns_1d)

        # 2D Soil - Use group normalization
        variables_2d_soil, variables_2d_soil_scaler = self._normalize_list_2d(self.data_config.x_list_columns_2d)
        y_soil_2d, y_soil_2d_scaler = self._normalize_list_2d(self.data_config.y_list_columns_2d)

        # PFT param (keep group normalization for now)
        pft_param_data, pft_param_scaler = self._normalize_pft_param()

        # Water (if present)
        water_tensor = None
        y_water_tensor = None
        if hasattr(self.data_config, 'x_list_water_columns') and self.data_config.x_list_water_columns:
            water_tensor, water_scaler = self._normalize_list_1d(self.data_config.x_list_water_columns)
        if hasattr(self.data_config, 'y_list_water_columns') and self.data_config.y_list_water_columns:
            y_water_tensor, y_water_scaler = self._normalize_list_1d(self.data_config.y_list_water_columns)

        # Assert lists are not empty
        assert len(self.data_config.x_list_columns_1d) > 0, 'x_list_columns_1d list is empty!'
        assert pft_1d_data.shape[1] == len(self.data_config.x_list_columns_1d), 'Mismatch in 1D PFT variable count!'
        assert scalar_data.shape[1] == len(self.data_config.x_list_scalar_columns), 'Mismatch in scalar feature count!'
        assert variables_2d_soil.shape[1] == len(self.data_config.x_list_columns_2d), 'Mismatch in 2D soil feature count!'
        assert pft_param_data.shape[1] == len(self.data_config.pft_param_columns), 'Mismatch in PFT param feature count!'
        assert y_scalar_data.shape[1] == len(self.data_config.y_list_scalar_columns), 'Mismatch in y_scalar feature count!'
        assert y_pft_1d_data.shape[1] == len(self.data_config.y_list_columns_1d), 'Mismatch in y_pft_1d variable count!'
        assert y_soil_2d.shape[1] == len(self.data_config.y_list_columns_2d), 'Mismatch in y_soil_2d feature count!'

        # Store all scalers
        self.scalers = {
            'time_series': time_series_scaler,
            'static': static_scaler,
            'scalar': scalar_scaler,
            'y_scalar': y_scalar_scaler,
            'pft_1d': pft_1d_scaler,
            'y_pft_1d': y_pft_1d_scaler,
            'variables_2d_soil': variables_2d_soil_scaler,
            'y_soil_2d': y_soil_2d_scaler,
            'pft_param': pft_param_scaler,
            'water': water_scaler if 'water_scaler' in locals() else None,
            'y_water': y_water_scaler if 'y_water_scaler' in locals() else None,
        }
        
        ret = {
            'time_series_data': time_series_data,
            'static_data': static_data,
            'pft_param_data': pft_param_data,
            'scalar_data': scalar_data,
            'variables_1d_pft': pft_1d_data,
            'variables_2d_soil': variables_2d_soil,
            'y_scalar': y_scalar_data,
            'y_pft_1d': y_pft_1d_data,
            'y_soil_2d': y_soil_2d,
            'water': water_tensor,
            'y_water': y_water_tensor,
            'scalers': self.scalers
        }
        # Add per-sample PFT mask derived from raw PCT_NAT_PFT_1..16 (1 where >0, else 0)
        try:
            pct_cols = [f'PCT_NAT_PFT_{i}' for i in range(1, 17)]
            if all(c in self.df.columns for c in pct_cols):
                pct = self.df[pct_cols].values.astype(np.float32)
                mask = (pct > 0.0).astype(np.float32)  # shape [N,16]
                ret['pft_presence_mask'] = torch.tensor(mask, dtype=self.preprocessing_config.data_type)
            else:
                logger.warning("Some PCT_NAT_PFT_1..16 columns are missing; pft_presence_mask not created")
        except Exception as _e:
            logger.warning(f"Failed to create pft_presence_mask: {_e}")
        # Optional dump after normalization (group)
        if os.getenv('DUMP_ALL_PFT_SOIL', '0') == '1':
            try:
                self._dump_normalized_pft_soil(ret, stage='group_norm')
            except Exception as _e:
                logger.warning(f"Group-norm dump failed: {_e}")
        return ret

    def normalize_data_individual(self, transform_only: bool = False) -> Dict[str, Any]:
        """
        Normalize all data types using individual variable normalization.
        This method provides optimal normalization for each variable but uses more memory.
        
        Args:
            transform_only: If True, use existing scalers without fitting (for inference).
                           If False, fit new scalers (for training).
        
        Returns:
            Dictionary containing normalized data and individual scalers
        """
        if transform_only:
            logger.info("Normalizing data using existing individual scalers (transform-only mode)...")
        else:
            logger.info("Normalizing data using individual variable normalization (fit+transform mode)...")

        # Time series (keep group normalization for now)
        time_series_data, time_series_scaler = self._normalize_time_series()
        
        # Static (keep group normalization for now)
        static_data, static_scaler = self._normalize_static(self.data_config.static_columns)

        # Scalar - Use individual normalization
        scalar_data, scalar_scaler = self._normalize_scalar_individual(transform_only)
        y_scalar_data, y_scalar_scaler = self._normalize_y_scalar_individual(transform_only)

        # 1D PFT - Use individual normalization
        pft_1d_data, pft_1d_scaler = self._normalize_list_1d_individual(self.data_config.x_list_columns_1d, transform_only)
        y_pft_1d_data, y_pft_1d_scaler = self._normalize_list_1d_individual(self.data_config.y_list_columns_1d, transform_only)

        # 2D Soil - Use individual normalization
        variables_2d_soil, variables_2d_soil_scaler = self._normalize_list_2d_individual(self.data_config.x_list_columns_2d, transform_only)
        y_soil_2d, y_soil_2d_scaler = self._normalize_list_2d_individual(self.data_config.y_list_columns_2d, transform_only)

        # PFT param (keep group normalization for now)
        pft_param_data, pft_param_scaler = self._normalize_pft_param()

        # Water (if present)
        water_tensor = None
        y_water_tensor = None
        if hasattr(self.data_config, 'x_list_water_columns') and self.data_config.x_list_water_columns:
            water_tensor, water_scaler = self._normalize_list_1d(self.data_config.x_list_water_columns)
        if hasattr(self.data_config, 'y_list_water_columns') and self.data_config.y_list_water_columns:
            y_water_tensor, y_water_scaler = self._normalize_list_1d(self.data_config.y_list_water_columns)

        # Assert lists are not empty
        assert len(self.data_config.x_list_columns_1d) > 0, 'x_list_columns_1d list is empty!'
        assert pft_1d_data.shape[1] == len(self.data_config.x_list_columns_1d), 'Mismatch in 1D PFT variable count!'
        assert scalar_data.shape[1] == len(self.data_config.x_list_scalar_columns), 'Mismatch in scalar feature count!'
        assert variables_2d_soil.shape[1] == len(self.data_config.x_list_columns_2d), 'Mismatch in 2D soil feature count!'
        assert pft_param_data.shape[1] == len(self.data_config.pft_param_columns), 'Mismatch in PFT param feature count!'
        # Only assert Y variables if they were normalized (training mode)
        if y_scalar_data is not None:
            assert y_scalar_data.shape[1] == len(self.data_config.y_list_scalar_columns), 'Mismatch in y_scalar feature count!'
        if y_pft_1d_data is not None:
            assert y_pft_1d_data.shape[1] == len(self.data_config.y_list_columns_1d), 'Mismatch in y_pft_1d variable count!'
        if y_soil_2d is not None:
            assert y_soil_2d.shape[1] == len(self.data_config.y_list_columns_2d), 'Mismatch in y_soil_2d feature count!'

        # Store all scalers
        self.scalers = {
            'time_series': time_series_scaler,
            'static': static_scaler,
            'scalar': scalar_scaler,
            'y_scalar': y_scalar_scaler,
            'pft_1d': pft_1d_scaler,
            'y_pft_1d': y_pft_1d_scaler,
            'variables_2d_soil': variables_2d_soil_scaler,
            'y_soil_2d': y_soil_2d_scaler,
            'pft_param': pft_param_scaler,
            'water': water_scaler if 'water_scaler' in locals() else None,
            'y_water': y_water_scaler if 'y_water_scaler' in locals() else None,
            # Store individual scaler managers
            'individual_scalar': self.individual_scalers['scalar'],
            'individual_y_scalar': self.individual_scalers['y_scalar'],
            'individual_pft_1d': self.individual_scalers['pft_1d'],
            'individual_y_pft_1d': self.individual_scalers['y_pft_1d'],
            'individual_soil_2d': self.individual_scalers['soil_2d'],
            'individual_y_soil_2d': self.individual_scalers['y_soil_2d'],
        }
        
        ret = {
            'time_series_data': time_series_data,
            'static_data': static_data,
            'pft_param_data': pft_param_data,
            'scalar_data': scalar_data,
            'variables_1d_pft': pft_1d_data,
            'variables_2d_soil': variables_2d_soil,
            'water': water_tensor,
            'y_water': y_water_tensor,
            'scalers': self.scalers
        }
        
        # Only add Y variables if they were normalized (training mode) or exist (inference mode)
        if y_scalar_data is not None:
            ret['y_scalar'] = y_scalar_data
        if y_pft_1d_data is not None:
            ret['y_pft_1d'] = y_pft_1d_data
        if y_soil_2d is not None:
            ret['y_soil_2d'] = y_soil_2d
        
        # Add per-sample PFT mask derived from raw PCT_NAT_PFT_1..16 (1 where >0, else 0)
        try:
            pct_cols = [f'PCT_NAT_PFT_{i}' for i in range(1, 17)]
            if all(c in self.df.columns for c in pct_cols):
                pct = self.df[pct_cols].values.astype(np.float32)
                mask = (pct > 0.0).astype(np.float32)  # shape [N,16]
                ret['pft_presence_mask'] = torch.tensor(mask, dtype=self.preprocessing_config.data_type)
            else:
                logger.warning("Some PCT_NAT_PFT_1..16 columns are missing; pft_presence_mask not created")
        except Exception as _e:
            logger.warning(f"Failed to create pft_presence_mask: {_e}")
        # Optional dump after normalization (individual)
        if os.getenv('DUMP_ALL_PFT_SOIL', '0') == '1':
            try:
                self._dump_normalized_pft_soil(ret, stage='individual_norm')
            except Exception as _e:
                logger.warning(f"Individual-norm dump failed: {_e}")
        return ret

    def _dump_normalized_pft_soil(self, normalized_data: Dict[str, Any], stage: str) -> None:
        """Print normalized PFT1D (16 PFTs, PFT1..PFT16) and Soil2D (top 10 layers) tensors."""
        try:
            y_pft = normalized_data.get('y_pft_1d')
            if isinstance(y_pft, torch.Tensor) and y_pft.numel() > 0:
                arr = y_pft.detach().cpu().numpy()
                # Expect (samples, variables, 16)
                if arr.ndim == 3:
                    logger.info(f"[{stage}] y_pft_1d: shape={arr.shape}")
                    for i in range(arr.shape[0]):
                        flat = arr[i].reshape(arr.shape[1], arr.shape[2])
                        logger.info(f"  row{i}:")
                        for v in range(flat.shape[0]):
                            logger.info(f"    var{v}: {','.join([f'{x:.6g}' for x in flat[v]])}")
        except Exception as e:
            logger.warning(f"[{stage}] Failed dump y_pft_1d: {e}")
        try:
            x_pft = normalized_data.get('variables_1d_pft')
            if isinstance(x_pft, torch.Tensor) and x_pft.numel() > 0:
                arr = x_pft.detach().cpu().numpy()
                if arr.ndim == 3:
                    logger.info(f"[{stage}] x_pft_1d: shape={arr.shape}")
                    for i in range(arr.shape[0]):
                        flat = arr[i].reshape(arr.shape[1], arr.shape[2])
                        logger.info(f"  row{i}:")
                        for v in range(flat.shape[0]):
                            logger.info(f"    var{v}: {','.join([f'{x:.6g}' for x in flat[v]])}")
        except Exception as e:
            logger.warning(f"[{stage}] Failed dump x_pft_1d: {e}")
        try:
            y_soil = normalized_data.get('y_soil_2d')
            if isinstance(y_soil, torch.Tensor) and y_soil.numel() > 0:
                arr = y_soil.detach().cpu().numpy()
                # Expect (samples, variables, 1, 10)
                if arr.ndim == 4:
                    logger.info(f"[{stage}] y_soil_2d: shape={arr.shape}")
                    for i in range(arr.shape[0]):
                        logger.info(f"  row{i}:")
                        for v in range(arr.shape[1]):
                            vec = arr[i, v, 0, :]
                            logger.info(f"    var{v}: {','.join([f'{x:.6g}' for x in vec])}")
        except Exception as e:
            logger.warning(f"[{stage}] Failed dump y_soil_2d: {e}")
        try:
            x_soil = normalized_data.get('variables_2d_soil')
            if isinstance(x_soil, torch.Tensor) and x_soil.numel() > 0:
                arr = x_soil.detach().cpu().numpy()
                if arr.ndim == 4:
                    logger.info(f"[{stage}] x_soil_2d: shape={arr.shape}")
                    for i in range(arr.shape[0]):
                        logger.info(f"  row{i}:")
                        for v in range(arr.shape[1]):
                            vec = arr[i, v, 0, :]
                            logger.info(f"    var{v}: {','.join([f'{x:.6g}' for x in vec])}")
        except Exception as e:
            logger.warning(f"[{stage}] Failed dump x_soil_2d: {e}")

    def normalize_data_hybrid(self, use_individual_for: List[str] = None, group_soil_vars: List[str] = None) -> Dict[str, Any]:
        """
        Normalize data using a hybrid approach - individual normalization for specified types,
        group normalization for others.
        
        Args:
            use_individual_for: List of data types to use individual normalization for.
                              Options: ['scalar', 'y_scalar', 'pft_1d', 'y_pft_1d', 'soil_2d', 'y_soil_2d']
                              If None, uses group normalization for all.
            group_soil_vars: List of specific soil variables to use group normalization for,
                             even if soil_2d is in use_individual_for.
                             Example: ['sminn_vr', 'smin_no3_vr', 'smin_nh4_vr']
        
        Returns:
            Dictionary containing normalized data and appropriate scalers
        """
        if use_individual_for is None:
            use_individual_for = []
        
        logger.info(f"Normalizing data using hybrid approach. Individual normalization for: {use_individual_for}")

        # Time series (always group normalization for now)
        time_series_data, time_series_scaler = self._normalize_time_series()
        
        # Static (always group normalization for now)
        static_data, static_scaler = self._normalize_static(self.data_config.static_columns)

        # Scalar - Choose normalization method
        if 'scalar' in use_individual_for:
            scalar_data, scalar_scaler = self._normalize_scalar_individual(transform_only=False)
        else:
            scalar_data, scalar_scaler = self._normalize_scalar()

        # Y scalar - Choose normalization method
        if 'y_scalar' in use_individual_for:
            y_scalar_data, y_scalar_scaler = self._normalize_y_scalar_individual(transform_only=False)
        else:
            y_scalar_data, y_scalar_scaler = self._normalize_y_scalar()

        # 1D PFT - Choose normalization method
        if 'pft_1d' in use_individual_for:
            pft_1d_data, pft_1d_scaler = self._normalize_list_1d_individual(self.data_config.x_list_columns_1d, transform_only=False)
        else:
            pft_1d_data, pft_1d_scaler = self._normalize_list_1d(self.data_config.x_list_columns_1d)

        # Y PFT1D - Choose normalization method
        if 'y_pft_1d' in use_individual_for:
            y_pft_1d_data, y_pft_1d_scaler = self._normalize_list_1d_individual(self.data_config.y_list_columns_1d, transform_only=False)
        else:
            y_pft_1d_data, y_pft_1d_scaler = self._normalize_list_1d(self.data_config.y_list_columns_1d)

        # 2D Soil - Choose normalization method with optional selective group normalization
        if 'soil_2d' in use_individual_for:
            if group_soil_vars:
                # Apply hybrid approach for soil variables
                variables_2d_soil, variables_2d_soil_scaler = self._normalize_list_2d_selective(
                    self.data_config.x_list_columns_2d, 
                    group_vars=group_soil_vars, 
                    transform_only=False
                )
            else:
                # Standard individual normalization for all soil variables
                variables_2d_soil, variables_2d_soil_scaler = self._normalize_list_2d_individual(
                    self.data_config.x_list_columns_2d, 
                    transform_only=False
                )
        else:
            variables_2d_soil, variables_2d_soil_scaler = self._normalize_list_2d(self.data_config.x_list_columns_2d)

        # Y Soil2D - Choose normalization method with optional selective group normalization
        if 'y_soil_2d' in use_individual_for:
            if group_soil_vars:
                # Apply hybrid approach for soil variables
                y_group_vars = [f'Y_{var}' for var in group_soil_vars]
                y_soil_2d, y_soil_2d_scaler = self._normalize_list_2d_selective(
                    self.data_config.y_list_columns_2d, 
                    group_vars=y_group_vars, 
                    transform_only=False
                )
            else:
                # Standard individual normalization for all soil variables
                y_soil_2d, y_soil_2d_scaler = self._normalize_list_2d_individual(
                    self.data_config.y_list_columns_2d, 
                    transform_only=False
                )
        else:
            y_soil_2d, y_soil_2d_scaler = self._normalize_list_2d(self.data_config.y_list_columns_2d)

        # PFT param (always group normalization for now)
        pft_param_data, pft_param_scaler = self._normalize_pft_param()

        # Water (if present) - always group normalization
        water_tensor = None
        y_water_tensor = None
        if hasattr(self.data_config, 'x_list_water_columns') and self.data_config.x_list_water_columns:
            water_tensor, water_scaler = self._normalize_list_1d(self.data_config.x_list_water_columns)
        if hasattr(self.data_config, 'y_list_water_columns') and self.data_config.y_list_water_columns:
            y_water_tensor, y_water_scaler = self._normalize_list_1d(self.data_config.y_list_water_columns)

        # Assert lists are not empty
        assert len(self.data_config.x_list_columns_1d) > 0, 'x_list_columns_1d list is empty!'
        assert pft_1d_data.shape[1] == len(self.data_config.x_list_columns_1d), 'Mismatch in 1D PFT variable count!'
        assert scalar_data.shape[1] == len(self.data_config.x_list_scalar_columns), 'Mismatch in scalar feature count!'
        assert variables_2d_soil.shape[1] == len(self.data_config.x_list_columns_2d), 'Mismatch in 2D soil feature count!'
        assert pft_param_data.shape[1] == len(self.data_config.pft_param_columns), 'Mismatch in PFT param feature count!'
        assert y_scalar_data.shape[1] == len(self.data_config.y_list_scalar_columns), 'Mismatch in y_scalar feature count!'
        assert y_pft_1d_data.shape[1] == len(self.data_config.y_list_columns_1d), 'Mismatch in y_pft_1d variable count!'
        assert y_soil_2d.shape[1] == len(self.data_config.y_list_columns_2d), 'Mismatch in y_soil_2d feature count!'

        # Store all scalers
        self.scalers = {
            'time_series': time_series_scaler,
            'static': static_scaler,
            'scalar': scalar_scaler,
            'y_scalar': y_scalar_scaler,
            'pft_1d': pft_1d_scaler,
            'y_pft_1d': y_pft_1d_scaler,
            'variables_2d_soil': variables_2d_soil_scaler,
            'y_soil_2d': y_soil_2d_scaler,
            'pft_param': pft_param_scaler,
            'water': water_scaler if 'water_scaler' in locals() else None,
            'y_water': y_water_scaler if 'y_water_scaler' in locals() else None,
        }
        
        # Add individual scalers if they were used
        if 'scalar' in use_individual_for:
            self.scalers['individual_scalar'] = self.individual_scalers['scalar']
        if 'y_scalar' in use_individual_for:
            self.scalers['individual_y_scalar'] = self.individual_scalers['y_scalar']
        if 'pft_1d' in use_individual_for:
            self.scalers['individual_pft_1d'] = self.individual_scalers['pft_1d']
        if 'y_pft_1d' in use_individual_for:
            self.scalers['individual_y_pft_1d'] = self.individual_scalers['y_pft_1d']
        if 'soil_2d' in use_individual_for:
            self.scalers['individual_soil_2d'] = self.individual_scalers['soil_2d']
        if 'y_soil_2d' in use_individual_for:
            self.scalers['individual_y_soil_2d'] = self.individual_scalers['y_soil_2d']
        
        return {
            'time_series_data': time_series_data,
            'static_data': static_data,
            'pft_param_data': pft_param_data,
            'scalar_data': scalar_data,
            'variables_1d_pft': pft_1d_data,
            'variables_2d_soil': variables_2d_soil,
            'y_scalar': y_scalar_data,
            'y_pft_1d': y_pft_1d_data,
            'y_soil_2d': y_soil_2d,
            'water': water_tensor,
            'y_water': y_water_tensor,
            'scalers': self.scalers
        }
    
    def _get_static_columns(self) -> List[str]:
        """Get static columns using the fixed list from config to ensure consistency."""
        # Use the fixed static columns from config
        static_columns = []
        for col in self.data_config.static_columns:
            if col in self.df.columns:
                # Check if the column contains list data
                sample_values = self.df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Check if any value is a list
                    has_lists = any(isinstance(val, list) for val in sample_values)
                    if not has_lists:
                        static_columns.append(col)
                    else:
                        logger.warning(f"Column {col} contains list data but was not in list configurations. Skipping from static columns.")
                else:
                    # Column exists but has no data, still include it
                    static_columns.append(col)
            else:
                logger.warning(f"Static column {col} not found in dataset. This may cause shape mismatches.")
        
        logger.info(f"Found {len(static_columns)} static columns from fixed config: {static_columns}")
        return static_columns
    
    def _get_scaler(self, normalization_type: str):
        """Get a scaler instance based on normalization type."""
        if normalization_type == 'minmax':
            return MinMaxScaler()
        elif normalization_type == 'standard':
            return StandardScaler()
        elif normalization_type == 'robust':
            return RobustScaler()
        else:
            logger.warning(f"Unknown normalization type: {normalization_type}, using MinMaxScaler")
            return MinMaxScaler()
    
    def _normalize_time_series(self) -> Tuple[torch.Tensor, Any]:
        """Normalize time series data."""
        logger.info("Normalizing time series data...")
        
        # Create time series data with proper shape (samples, time_steps, features)
        time_series_list = []
        for col in self.data_config.time_series_columns:
            if col not in self.df.columns:
                logger.warning(f"Time series column {col} not found, using zeros")
                col_data = np.zeros((len(self.df), self.data_config.time_series_length), dtype=np.float32)
            else:
                # Extract time series data for this column
                col_data = np.vstack(self.df[col].values)
                if col_data.shape[1] != self.data_config.time_series_length:
                    logger.warning(f"Time series column {col} has unexpected shape {col_data.shape}, expected {len(self.df)}x{self.data_config.time_series_length}")
                if col_data.size == 0:
                    logger.error(f"Time series column {col} has empty data!")
                    col_data = np.zeros((len(self.df), self.data_config.time_series_length), dtype=np.float32)
                else:
                    col_data = np.nan_to_num(col_data, nan=0.0)
                if np.isinf(col_data).any():
                    logger.warning(f"Time series column {col} contains infinite values, filling with 0")
                    col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
            time_series_list.append(col_data)
        
        # Stack along feature dimension to get (samples, time_steps, features)
        time_series_data = np.stack(time_series_list, axis=-1)
        time_series_data = np.ascontiguousarray(time_series_data)
        
        # Check if we need to handle time series length mismatch
        actual_time_steps = time_series_data.shape[1]
        expected_time_steps = self.data_config.time_series_length
        
        if actual_time_steps != expected_time_steps:
            logger.warning(f"Time series length mismatch: actual={actual_time_steps}, expected={expected_time_steps}")
            
            if actual_time_steps > expected_time_steps:
                # Truncate to expected length (take first expected_time_steps)
                logger.info(f"Truncating time series from {actual_time_steps} to {expected_time_steps} time steps")
                time_series_data = time_series_data[:, :expected_time_steps, :]
            else:
                # Pad to expected length (repeat last time step)
                logger.info(f"Padding time series from {actual_time_steps} to {expected_time_steps} time steps")
                padding_needed = expected_time_steps - actual_time_steps
                last_time_step = time_series_data[:, -1:, :]
                padding = np.repeat(last_time_step, padding_needed, axis=1)
                time_series_data = np.concatenate([time_series_data, padding], axis=1)
        
        expected_shape = (len(self.df), self.data_config.time_series_length, len(self.data_config.time_series_columns))
        if time_series_data.shape != expected_shape:
            logger.error(f"Time series data has wrong shape after adjustment: {time_series_data.shape}, expected {expected_shape}")
            raise ValueError(f"Time series data has wrong shape after adjustment: {time_series_data.shape}, expected {expected_shape}")
        
        if time_series_data.shape[0] == 0 or time_series_data.shape[1] == 0:
            logger.error(f"Time series data has invalid shape: {time_series_data.shape}")
            raise ValueError(f"Time series data has invalid shape: {time_series_data.shape}")
        
        # Reshape for normalization: (samples * time_steps, features)
        original_shape = time_series_data.shape
        time_series_flat = time_series_data.reshape(-1, len(self.data_config.time_series_columns))
        scaler = self._get_scaler(self.preprocessing_config.time_series_normalization)
        time_series_normalized = scaler.fit_transform(time_series_flat)
        time_series_data = time_series_normalized.reshape(original_shape)
        time_series_data = np.ascontiguousarray(time_series_data)
        return torch.tensor(time_series_data, dtype=self.preprocessing_config.data_type), scaler
    
    def _normalize_static(self, static_columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize static data in the order defined by static_columns."""
        logger.info(f"Normalizing static data with columns: {static_columns}")
        # Enforce order
        for i, col in enumerate(static_columns):
            assert col in self.df.columns, f"Static column '{col}' missing in DataFrame!"
        static_data = self.df[static_columns].values
        # Clean NaN/Inf
        static_data = np.nan_to_num(static_data, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = self._get_scaler(self.preprocessing_config.static_normalization)
        static_normalized = scaler.fit_transform(static_data)
        return torch.tensor(static_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_scalar(self) -> Tuple[torch.Tensor, Any]:
        """Normalize scalar variables using group normalization."""
        scalar_columns = self.data_config.x_list_scalar_columns
        logger.info(f"Normalizing scalar data with columns: {scalar_columns}")
        
        for i, col in enumerate(scalar_columns):
            assert col in self.df.columns, f"Scalar column '{col}' missing in DataFrame!"
        
        scalar_data = self.df[scalar_columns].values
        # Clean NaN/Inf
        scalar_data = np.nan_to_num(scalar_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use group normalization
        scaler = self._get_scaler(self.preprocessing_config.target_normalization)
        scalar_normalized = scaler.fit_transform(scalar_data)
        
        return torch.tensor(scalar_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_y_scalar(self) -> Tuple[torch.Tensor, Any]:
        """Normalize y_scalar variables using group normalization."""
        y_scalar_columns = self.data_config.y_list_scalar_columns
        logger.info(f"Normalizing y_scalar data with columns: {y_scalar_columns}")
        
        for i, col in enumerate(y_scalar_columns):
            assert col in self.df.columns, f"y_scalar column '{col}' missing in DataFrame!"
        
        y_scalar_data = self.df[y_scalar_columns].values
        # Clean NaN/Inf
        y_scalar_data = np.nan_to_num(y_scalar_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use group normalization
        scaler = self._get_scaler(self.preprocessing_config.target_normalization)
        y_scalar_normalized = scaler.fit_transform(y_scalar_data)
        
        return torch.tensor(y_scalar_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_scalar_individual(self, transform_only: bool = False) -> Tuple[torch.Tensor, Any]:
        """Normalize scalar variables individually using IndividualScalerManager."""
        scalar_columns = self.data_config.x_list_scalar_columns
        logger.info(f"Normalizing scalar data with columns: {scalar_columns}")
        
        for i, col in enumerate(scalar_columns):
            assert col in self.df.columns, f"Scalar column '{col}' missing in DataFrame!"
        
        scalar_data = self.df[scalar_columns].values
        scalar_data = np.nan_to_num(scalar_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use individual normalization (fit+transform or transform-only)
        if transform_only:
            normalized_data = self.individual_scalers['scalar'].transform_scalar(scalar_data, scalar_columns)
        else:
            normalized_data = self.individual_scalers['scalar'].fit_transform_scalar(scalar_data, scalar_columns)
        
        return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['scalar']

    def _normalize_y_scalar_individual(self, transform_only: bool = False) -> Tuple[torch.Tensor, Any]:
        """Normalize y_scalar variables individually using IndividualScalerManager."""
        y_scalar_columns = self.data_config.y_list_scalar_columns
        logger.info(f"Normalizing y_scalar data with columns: {y_scalar_columns}")
        
        # For inference mode, skip if columns don't exist
        if transform_only:
            missing_cols = [col for col in y_scalar_columns if col not in self.df.columns]
            if missing_cols:
                logger.info(f"Inference mode: Y_scalar columns {missing_cols} missing in DataFrame. Skipping normalization.")
                return None, None
        
        for i, col in enumerate(y_scalar_columns):
            assert col in self.df.columns, f"y_scalar column '{col}' missing in DataFrame!"
        
        y_scalar_data = self.df[y_scalar_columns].values
        y_scalar_data = np.nan_to_num(y_scalar_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use individual normalization (fit+transform or transform-only)
        if transform_only:
            normalized_data = self.individual_scalers['y_scalar'].transform_scalar(y_scalar_data, y_scalar_columns)
        else:
            normalized_data = self.individual_scalers['y_scalar'].fit_transform_scalar(y_scalar_data, y_scalar_columns)
        
        return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['y_scalar']

    def _normalize_list_1d_individual(self, columns: List[str], transform_only: bool = False) -> Tuple[torch.Tensor, Any]:
        """Normalize 1D list data individually using IndividualScalerManager."""
        logger.info(f"Normalizing 1D list data with columns: {columns}")
        
        # For inference mode with Y variables, skip if columns don't exist
        is_y = columns == self.data_config.y_list_columns_1d
        if transform_only and is_y:
            missing_cols = [col for col in columns if col not in self.df.columns]
            if missing_cols:
                logger.info(f"Inference mode: Y_pft_1d columns {missing_cols} missing in DataFrame. Skipping normalization.")
                return None, None
        
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"1D column '{col}' missing in DataFrame!"
        
        col_data = [np.vstack(self.df[col].values) for col in columns]
        # Clean NaN/Inf in stacked data
        col_data = [np.nan_to_num(cd, nan=0.0, posinf=0.0, neginf=0.0) for cd in col_data]
        data = np.stack(col_data, axis=1)  # shape: (samples, features, length)
        
        # Handle PFT0 dropping for compatibility with model expectations
        if data.shape[2] == 17:  # 17 PFTs (including PFT0)
            data = data[:, :, 1:]  # Drop PFT0, keep PFT1-PFT16
        elif data.shape[2] != 16:
            logger.warning(f"Unexpected PFT count: {data.shape[2]}, expected 16 or 17")
        
        # For PFT1D, the data shape is (samples, features, pfts)
        # We need to transpose to (samples, pfts, features) for the scaler
        if data.shape[2] == 16:  # 16 PFTs
            data = np.transpose(data, (0, 2, 1))  # (samples, pfts, features)
            # Use PFT1-PFT16 naming to match training scaler keys (not PFT0-PFT15)
            pft_names = [f'PFT{i}' for i in range(1, 17)]
            
            if columns == self.data_config.x_list_columns_1d:
                # Input PFT1D data - use fit+transform or transform-only
                if transform_only:
                    normalized_data = self.individual_scalers['pft_1d'].transform_pft_1d(data, pft_names, columns)
                else:
                    normalized_data = self.individual_scalers['pft_1d'].fit_transform_pft_1d(data, pft_names, columns)
                # Transpose back to original shape
                normalized_data = np.transpose(normalized_data, (0, 2, 1))
                return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['pft_1d']
            else:
                # Output PFT1D data - use fit+transform or transform-only
                if transform_only:
                    normalized_data = self.individual_scalers['y_pft_1d'].transform_pft_1d(data, pft_names, columns)
                else:
                    normalized_data = self.individual_scalers['y_pft_1d'].fit_transform_pft_1d(data, pft_names, columns)
                # Transpose back to original shape
                normalized_data = np.transpose(normalized_data, (0, 2, 1))
                return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['y_pft_1d']
        else:
            # Fallback to group normalization for non-PFT data
            n_samples, n_features, n_length = data.shape
            data_reshaped = data.reshape(n_samples, -1)
            scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
            data_normalized = scaler.fit_transform(data_reshaped)
            data_normalized = data_normalized.reshape(n_samples, n_features, n_length)
            return torch.tensor(data_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_list_2d_selective(self, columns: List[str], group_vars: List[str], transform_only: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalize 2D list columns using a selective approach: group normalization for specified variables,
        individual normalization for the rest.
        
        Args:
            columns: List of columns to normalize
            group_vars: List of variable names to use group normalization for
            transform_only: If True, use existing scalers without fitting (for inference)
            
        Returns:
            Tuple of normalized tensor and dictionary of scalers
        """
        logger.info(f"Using selective normalization for 2D list columns. Group normalization for: {group_vars}")
        
        # Separate columns into group and individual normalization
        group_columns = [col for col in columns if col in group_vars]
        individual_columns = [col for col in columns if col not in group_vars]
        
        # Create empty tensor to hold all normalized data
        sample_size = len(self.df)
        num_columns = len(columns)
        tensor_shape = (sample_size, num_columns, 1, 10)  # Always (samples, variables, 1, 10) for soil2D
        normalized_tensor = torch.zeros(tensor_shape, dtype=torch.float32)
        
        # Dictionary to store all scalers
        all_scalers = {}
        
        # Process group normalization columns if any
        if group_columns:
            # Get column indices in the original list
            group_indices = [columns.index(col) for col in group_columns]
            
            # Extract data for group columns
            group_data = []
            for col in group_columns:
                if col in self.df.columns:
                    values = self.df[col].values
                    standardized_samples = []
                    
                    for val in values:
                        if isinstance(val, (list, np.ndarray)):
                            val_array = np.array(val)
                            if val_array.shape[1] == 15:  # Has 15 layers
                                # Extract first column and top 10 layers immediately
                                extracted = val_array[0:1, 0:10]  # Shape: (1, 10)
                                standardized_samples.append(extracted)
                            else:
                                # Handle other shapes by padding/truncating
                                padded = np.zeros((1, 10), dtype=np.float32)
                                if val_array.ndim == 2:
                                    h, w = val_array.shape
                                    # Take first column (or only column if h=1)
                                    col_data = val_array[0:1, :] if h > 1 else val_array
                                    # Take first 10 elements (or pad if fewer)
                                    w_take = min(w, 10)
                                    padded[0, :w_take] = col_data[0, :w_take]
                                standardized_samples.append(padded)
                        else:
                            # Handle non-array values with zeros
                            standardized_samples.append(np.zeros((1, 10), dtype=np.float32))
                    
                    # Stack all samples for this column
                    stacked = np.stack(standardized_samples, axis=0)  # Shape: (samples, 1, 10)
                    group_data.append(stacked)
            
            # Combine all group data
            if group_data:
                combined_group_data = np.concatenate(group_data, axis=1)  # Shape: (samples, n_group_vars, 10)
                combined_group_data = combined_group_data.reshape(combined_group_data.shape[0], len(group_columns), 1, 10)
                
                # Reshape for normalization
                original_shape = combined_group_data.shape
                flattened = combined_group_data.reshape(-1, 1)
                
                # Create and fit scaler
                scaler = MinMaxScaler() if self.preprocessing_config.list_2d_normalization == 'minmax' else StandardScaler()
                if transform_only and hasattr(self, 'scalers') and 'group_soil_2d' in self.scalers:
                    normalized_flat = self.scalers['group_soil_2d'].transform(flattened)
                else:
                    normalized_flat = scaler.fit_transform(flattened)
                    all_scalers['group_soil_2d'] = scaler
                
                # Reshape back to original shape
                normalized_group = normalized_flat.reshape(original_shape)
                
                # Place in the final tensor
                for i, idx in enumerate(group_indices):
                    normalized_tensor[:, idx, :, :] = torch.tensor(normalized_group[:, i, :, :], dtype=torch.float32)
        
        # Process individual normalization columns
        if individual_columns:
            # Get individual normalization for these columns
            individual_data, individual_scalers = self._normalize_list_2d_individual(individual_columns, transform_only)
            
            # Place in the final tensor
            for i, col in enumerate(individual_columns):
                idx = columns.index(col)
                normalized_tensor[:, idx, :, :] = individual_data[:, i, :, :]
            
            # Merge scalers
            all_scalers.update(individual_scalers)
        elif not group_columns:
            # If no individual columns and no group columns, something is wrong
            logger.warning("No columns to normalize in _normalize_list_2d_selective!")
        
        return normalized_tensor, all_scalers

    def _normalize_list_2d_individual(self, columns: List[str], transform_only: bool = False) -> Tuple[torch.Tensor, Any]:
        """Normalize 2D list data individually using IndividualScalerManager."""
        logger.info(f"Normalizing 2D list data with columns: {columns}")
        
        # For inference mode with Y variables, skip if columns don't exist
        is_y = columns == self.data_config.y_list_columns_2d
        if transform_only and is_y:
            missing_cols = [col for col in columns if col not in self.df.columns]
            if missing_cols:
                logger.info(f"Inference mode: Y_soil_2d columns {missing_cols} missing in DataFrame. Skipping normalization.")
                return None, None
        
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"2D column '{col}' missing in DataFrame!"
        
        # Extract first column and top 10 layers directly for consistent shapes
        col_data = []
        for col in columns:
            values = self.df[col].values
            standardized_samples = []
            
            for val in values:
                if isinstance(val, (list, np.ndarray)):
                    val_array = np.array(val)
                    val_array = np.nan_to_num(val_array, nan=0.0, posinf=0.0, neginf=0.0)
                    if val_array.shape[1] == 15:  # Has 15 layers
                        # Extract first column and top 10 layers immediately
                        extracted = val_array[0:1, 0:10]  # Shape: (1, 10)
                        standardized_samples.append(extracted)
                    else:
                        # Invalid structure, use zeros
                        standardized_samples.append(np.zeros((1, 10)))
                else:
                    # Invalid data type, use zeros
                    standardized_samples.append(np.zeros((1, 10)))
            
            col_data.append(np.stack(standardized_samples))
        
        data = np.stack(col_data, axis=1)  # shape: (samples, features, 1, 10)
        
        # Log the standardized data
        logger.info(f"Standardized Soil2D data shape: {data.shape}")
        non_zero_count_before = np.count_nonzero(data)
        logger.info(f"Before normalization - Soil2D non-zero count: {non_zero_count_before}")
        if data.shape[0] > 0:
            logger.info(f"Before normalization - Soil2D sample (first few elements): {data[0, :2, :2, :2]}")
        
        # Data is already in the correct shape: (samples, variables, 1, 10)
        # No need for additional extraction since we did it during loading
        
        # For Soil2D, we need to handle the layer dimension
        if columns == self.data_config.x_list_columns_2d:
            # Input Soil2D data - use fit+transform or transform-only
            if transform_only:
                normalized_data = self.individual_scalers['soil_2d'].transform_soil_2d(data, columns, data.shape[3])
            else:
                normalized_data = self.individual_scalers['soil_2d'].fit_transform_soil_2d(data, columns, data.shape[3])
            # Log after normalization for input soil2D
            non_zero_count_after = np.count_nonzero(normalized_data)
            logger.info(f"After normalization - Input Soil2D non-zero count: {non_zero_count_after}")
            if normalized_data.shape[0] > 0:
                logger.info(f"After normalization - Input Soil2D sample (first few elements): {normalized_data[0, :2, :2, :2]}")
            return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['soil_2d']
        else:
            # Output Soil2D data - use fit+transform or transform-only
            if transform_only:
                normalized_data = self.individual_scalers['y_soil_2d'].transform_soil_2d(data, columns, data.shape[3])
            else:
                normalized_data = self.individual_scalers['y_soil_2d'].fit_transform_soil_2d(data, columns, data.shape[3])
            # Log after normalization for output soil2D
            non_zero_count_after = np.count_nonzero(normalized_data)
            logger.info(f"After normalization - Output Soil2D non-zero count: {non_zero_count_after}")
            if normalized_data.shape[0] > 0:
                logger.info(f"After normalization - Output Soil2D sample (first few elements): {normalized_data[0, :2, :2, :2]}")
            return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['y_soil_2d']

    def _normalize_list_1d(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize 1D list data in the order defined by columns (legacy method)."""
        logger.info(f"Normalizing 1D list data with columns: {columns}")
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"1D column '{col}' missing in DataFrame!"
        col_data = [np.vstack(self.df[col].values) for col in columns]
        data = np.stack(col_data, axis=1)  # shape: (samples, features, length)
        n_samples, n_features, n_length = data.shape
        data_reshaped = data.reshape(n_samples, -1)
        scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
        data_normalized = scaler.fit_transform(data_reshaped)
        data_normalized = data_normalized.reshape(n_samples, n_features, n_length)
        return torch.tensor(data_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_list_2d(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize 2D list data in the order defined by columns (legacy method)."""
        logger.info(f"Normalizing 2D list data with columns: {columns}")
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"2D column '{col}' missing in DataFrame!"
        
        # Extract first column and top 10 layers directly for consistent shapes
        col_data = []
        for col in columns:
            values = self.df[col].values
            standardized_samples = []
            
            for val in values:
                if isinstance(val, (list, np.ndarray)):
                    val_array = np.array(val)
                    if val_array.shape[1] == 15:  # Has 15 layers
                        # Extract first column and top 10 layers immediately
                        extracted = val_array[0:1, 0:10]  # Shape: (1, 10)
                        standardized_samples.append(extracted)
                    else:
                        # Invalid structure, use zeros
                        standardized_samples.append(np.zeros((1, 10)))
                else:
                    # Invalid data type, use zeros
                    standardized_samples.append(np.zeros((1, 10)))
            
            col_data.append(np.stack(standardized_samples))
        
        data = np.stack(col_data, axis=1)  # shape: (samples, features, 1, 10)
        
        # Data is already in the correct shape: (samples, variables, 1, 10)
        # No need for additional extraction since we did it during loading
        
        n_samples, n_features, n_rows, n_cols = data.shape
        data_reshaped = data.reshape(n_samples, -1)
        scaler = self._get_scaler(self.preprocessing_config.list_2d_normalization)
        data_normalized = scaler.fit_transform(data_reshaped)
        data_normalized = data_normalized.reshape(n_samples, n_features, n_rows, n_cols)
        return torch.tensor(data_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_pft_param(self) -> Tuple[torch.Tensor, Any]:
        """Normalize and stack pft_param data as [batch, 44, 17] in config order."""
        pft_param_columns = self.data_config.pft_param_columns

        num_params = len(pft_param_columns)
        num_pfts = 17  # Always use 17 PFTs
        logger.info(f"Normalizing pft_param data with columns: {pft_param_columns}")
        # Enforce order and presence
        for col in pft_param_columns:
            assert col in self.df.columns, f"PFT param column '{col}' missing in DataFrame!"
        # Stack in config order
        param_matrix = []
        for idx, row in self.df.iterrows():
            row_vectors = []
            for col in pft_param_columns:
                val = row[col]
                if isinstance(val, (list, np.ndarray)) and len(val) == num_pfts:
                    arr = np.array(val, dtype=np.float32)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    row_vectors.append(arr)
                else:
                    row_vectors.append(np.zeros(num_pfts, dtype=np.float32))
            row_matrix = np.stack(row_vectors, axis=0)  # [44, 17]
            param_matrix.append(row_matrix)
        param_matrix = np.stack(param_matrix, axis=0)  # [batch, 44, 17]
        assert param_matrix.shape[1:] == (num_params, num_pfts), f"pft_param_data shape {param_matrix.shape} does not match [batch, 44, 17]"
        # Flatten for normalization
        flat_param_matrix = param_matrix.reshape(param_matrix.shape[0], -1)
        scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
        flat_param_matrix_norm = scaler.fit_transform(flat_param_matrix)
        param_matrix_norm = flat_param_matrix_norm.reshape(param_matrix.shape)
        pft_param_data = torch.tensor(param_matrix_norm, dtype=self.preprocessing_config.data_type)
        return pft_param_data, scaler

    def save_scalers(self, directory: str):
        """Save all scalers to disk."""
        scaler_dir = Path(directory) / "scalers"
        scaler_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual scalers
        for name, scaler_manager in self.individual_scalers.items():
            if hasattr(scaler_manager, 'save_scalers'):
                scaler_manager.save_scalers(scaler_dir / name)
                logger.info(f"Saved individual scalers for {name}")
        
        # Save group scalers
        for name, scaler in self.scalers.items():
            if scaler is not None and not name.startswith('individual_'):
                scaler_file = scaler_dir / f"{name}_scaler.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"Saved group scaler: {scaler_file}")
        
        # Save comprehensive metadata
        metadata = {
            'normalization_type': 'individual_variable',
            'individual_scalers': {name: scaler_manager.get_scaler_info() for name, scaler_manager in self.individual_scalers.items()},
            'group_scalers': {name: type(scaler).__name__ if scaler is not None else None for name, scaler in self.scalers.items() if not name.startswith('individual_')}
        }
        
        metadata_file = scaler_dir / "scaler_metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved scaler metadata: {metadata_file}")

    def load_scalers(self, directory: str):
        """Load all scalers from disk."""
        scaler_dir = Path(directory) / "scalers"
        
        if not scaler_dir.exists():
            logger.warning(f"Scaler directory {scaler_dir} does not exist")
            return
        
        # Load individual scalers
        for name, scaler_manager in self.individual_scalers.items():
            if (scaler_dir / name).exists():
                scaler_manager.load_scalers(scaler_dir / name)
                logger.info(f"Loaded individual scalers for {name}")
        
        # Load group scalers
        for name, scaler in self.scalers.items():
            if not name.startswith('individual_'):
                scaler_file = scaler_dir / f"{name}_scaler.pkl"
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.scalers[name] = pickle.load(f)
                    logger.info(f"Loaded group scaler: {scaler_file}")

    def get_original_data_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get original data ranges for each variable type."""
        ranges = {}
        
        # Scalar variables
        if self.data_config.x_list_scalar_columns:
            ranges['scalar'] = {}
            for col in self.data_config.x_list_scalar_columns:
                if col in self.df.columns:
                    data = self.df[col].values
                    ranges['scalar'][col] = (float(np.min(data)), float(np.max(data)))
        
        # Y scalar variables
        if self.data_config.y_list_scalar_columns:
            ranges['y_scalar'] = {}
            for col in self.data_config.y_list_scalar_columns:
                if col in self.df.columns:
                    data = self.df[col].values
                    ranges['y_scalar'][col] = (float(np.min(data)), float(np.max(data)))
        
        # PFT1D variables
        if self.data_config.x_list_columns_1d:
            ranges['pft_1d'] = {}
            for col in self.data_config.x_list_columns_1d:
                if col in self.df.columns:
                    data = np.vstack(self.df[col].values)
                    ranges['pft_1d'][col] = (float(np.min(data)), float(np.max(data)))
        
        # Y PFT1D variables
        if self.data_config.y_list_columns_1d:
            ranges['y_pft_1d'] = {}
            for col in self.data_config.y_list_columns_1d:
                if col in self.df.columns:
                    data = np.vstack(self.df[col].values)
                    ranges['y_pft_1d'][col] = (float(np.min(data)), float(np.max(data)))
        
        # Soil2D variables
        if self.data_config.x_list_columns_2d:
            ranges['soil_2d'] = {}
            for col in self.data_config.x_list_columns_2d:
                if col in self.df.columns:
                    data = np.stack(self.df[col].values)
                    ranges['soil_2d'][col] = (float(np.min(data)), float(np.max(data)))
        
        # Y Soil2D variables
        if self.data_config.y_list_columns_2d:
            ranges['y_soil_2d'] = {}
            for col in self.data_config.y_list_columns_2d:
                if col in self.df.columns:
                    data = np.stack(self.df[col].values)
                    ranges['y_soil_2d'][col] = (float(np.min(data)), float(np.max(data)))
        
        return ranges

    def inverse_transform_predictions(self, predictions: Dict[str, torch.Tensor], variable_names: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Apply inverse transformation to predictions using individual scalers."""
        denormalized_predictions = {}
        
        # Scalar predictions
        if 'scalar' in predictions and 'scalar' in variable_names:
            denormalized_predictions['scalar'] = torch.tensor(
                self.individual_scalers['scalar'].inverse_transform_scalar(
                    predictions['scalar'].numpy(), 
                    variable_names['scalar']
                )
            )
        
        # Y scalar predictions
        if 'y_scalar' in predictions and 'y_scalar' in variable_names:
            denormalized_predictions['y_scalar'] = torch.tensor(
                self.individual_scalers['y_scalar'].inverse_transform_scalar(
                    predictions['y_scalar'].numpy(), 
                    variable_names['y_scalar']
                )
            )
        
        # PFT1D predictions
        if 'pft_1d' in predictions and 'pft_1d' in variable_names:
            pft_names = [f'PFT{i}' for i in range(predictions['pft_1d'].shape[1])]
            denormalized_predictions['pft_1d'] = torch.tensor(
                self.individual_scalers['pft_1d'].inverse_transform_pft_1d(
                    predictions['pft_1d'].numpy(), 
                    pft_names, 
                    variable_names['pft_1d']
                )
            )
        
        # Y PFT1D predictions
        if 'y_pft_1d' in predictions and 'y_pft_1d' in variable_names:
            pft_names = [f'PFT{i}' for i in range(predictions['y_pft_1d'].shape[1])]
            denormalized_predictions['y_pft_1d'] = torch.tensor(
                self.individual_scalers['y_pft_1d'].inverse_transform_pft_1d(
                    predictions['y_pft_1d'].numpy(), 
                    pft_names, 
                    variable_names['y_pft_1d']
                )
            )
        
        # Soil2D predictions
        if 'soil_2d' in predictions and 'soil_2d' in variable_names:
            num_layers = predictions['soil_2d'].shape[3]
            denormalized_predictions['soil_2d'] = torch.tensor(
                self.individual_scalers['soil_2d'].inverse_transform_soil_2d(
                    predictions['soil_2d'].numpy(), 
                    variable_names['soil_2d'], 
                    num_layers
                )
            )
        
        # Y Soil2D predictions
        if 'y_soil_2d' in predictions and 'y_soil_2d' in variable_names:
            num_layers = predictions['y_soil_2d'].shape[3]
            denormalized_predictions['y_soil_2d'] = torch.tensor(
                self.individual_scalers['y_soil_2d'].inverse_transform_soil_2d(
                    predictions['y_soil_2d'].numpy(), 
                    variable_names['y_soil_2d'], 
                    num_layers
                )
            )
        
        return denormalized_predictions

    def split_data(self, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split normalized data into train and test sets.
        """
        logger.info("Splitting data into train/test sets...")

        train_data = {}
        test_data = {}
        
        total_samples = len(self.df)
        train_size = int(self.data_config.train_split * total_samples)
        test_size = total_samples - train_size
        
        logger.info(f"Data splitting details:")
        logger.info(f"  - Total samples: {total_samples}")
        logger.info(f"  - Train split ratio: {self.data_config.train_split}")
        logger.info(f"  - Train size: {train_size}")
        logger.info(f"  - Test size: {test_size}")

        # Expose split indices for downstream use (e.g., location validation)
        # Matches the contiguous slicing used below
        try:
            self.train_indices = np.arange(0, train_size, dtype=int)
            self.test_indices = np.arange(train_size, total_samples, dtype=int)
        except Exception:
            # Fallback without crashing if numpy not available for some reason
            self.train_indices = list(range(0, train_size))
            self.test_indices = list(range(train_size, total_samples))
        
        if test_size == 0:
            logger.error("Test size is 0! This will cause evaluation issues.")
            logger.error("Consider reducing train_split ratio or increasing dataset size.")
        
        # Split time series data
        train_time_series = normalized_data['time_series_data'][:train_size, :, :]
        test_time_series = normalized_data['time_series_data'][train_size:, :, :]
        train_data['time_series'] = train_time_series
        test_data['time_series'] = test_time_series
        
        # Split static data
        train_static = normalized_data['static_data'][:train_size]
        test_static = normalized_data['static_data'][train_size:]
        train_data['static'] = train_static
        test_data['static'] = test_static
        
        # Split pft_param data
        train_pft_param = normalized_data['pft_param_data'][:train_size]
        test_pft_param = normalized_data['pft_param_data'][train_size:]
        train_data['pft_param'] = train_pft_param
        test_data['pft_param'] = test_pft_param

        # Split scalar data (input)
        train_list_scalar = normalized_data['scalar_data'][:train_size]
        test_list_scalar = normalized_data['scalar_data'][train_size:]
        train_data['scalar'] = train_list_scalar
        test_data['scalar'] = test_list_scalar 

        # Split y_scalar (target) - skip if not present (inference mode)
        if 'y_scalar' in normalized_data and normalized_data['y_scalar'] is not None:
            y_scalar = normalized_data['y_scalar']
            train_data['y_scalar'] = y_scalar[:train_size]
            test_data['y_scalar'] = y_scalar[train_size:]

        # Split variables_1d_pft (input)
        variables_1d_pft = normalized_data['variables_1d_pft']
        train_data['variables_1d_pft'] = variables_1d_pft[:train_size]
        test_data['variables_1d_pft'] = variables_1d_pft[train_size:]
        
        # Split y_pft_1d (target) - skip if not present (inference mode)
        if 'y_pft_1d' in normalized_data and normalized_data['y_pft_1d'] is not None:
            y_pft_1d = normalized_data['y_pft_1d']
            train_data['y_pft_1d'] = y_pft_1d[:train_size]
            test_data['y_pft_1d'] = y_pft_1d[train_size:]

        # Split y_soil_2d (target) - skip if not present (inference mode)
        if 'y_soil_2d' in normalized_data and normalized_data['y_soil_2d'] is not None:
            y_soil_2d = normalized_data['y_soil_2d']
            train_data['y_soil_2d'] = y_soil_2d[:train_size]
            test_data['y_soil_2d'] = y_soil_2d[train_size:]

        # Split variables_2d_soil (input)
        variables_2d_soil = normalized_data['variables_2d_soil']
        train_data['variables_2d_soil'] = variables_2d_soil[:train_size]
        test_data['variables_2d_soil'] = variables_2d_soil[train_size:]
        
        # Split water data if present
        if 'water' in normalized_data and normalized_data['water'] is not None:
            train_data['water'] = normalized_data['water'][:train_size]
            test_data['water'] = normalized_data['water'][train_size:]
        if 'y_water' in normalized_data and normalized_data['y_water'] is not None:
            train_data['y_water'] = normalized_data['y_water'][:train_size]
            test_data['y_water'] = normalized_data['y_water'][train_size:]

        # Split PFT presence mask if present
        if 'pft_presence_mask' in normalized_data:
            ppm = normalized_data['pft_presence_mask']
            try:
                train_data['pft_presence_mask'] = ppm[:train_size]
                test_data['pft_presence_mask'] = ppm[train_size:]
            except Exception:
                logger.warning("pft_presence_mask present but could not be split; skipping")
        
        logger.info(f"Split completed:")
        logger.info(f"  - Train time_series shape: {train_time_series.shape}")
        logger.info(f"  - Test time_series shape: {test_time_series.shape}")
        logger.info(f"  - Train static shape: {train_static.shape}")
        logger.info(f"  - Test static shape: {test_static.shape}")
        
        # Only keep final keys in output
        final_keys = [
            'time_series', 'static', 'pft_param', 'scalar',
            'variables_1d_pft', 'variables_2d_soil',
            'y_scalar', 'y_pft_1d', 'y_soil_2d'
        ]   

        # we need to make sure water is optional 
        if 'water' in train_data:
            final_keys.append('water')
            final_keys.append('y_water')

        # Optionally include presence mask
        if 'pft_presence_mask' in train_data:
            final_keys.append('pft_presence_mask')
        train_data = {k: v for k, v in train_data.items() if k in final_keys}
        test_data = {k: v for k, v in test_data.items() if k in final_keys}
        
        return {
            'train': train_data,
            'test': test_data,
            'train_size': train_size,
            'test_size': test_size
        }

    def _normalize_list_1d(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize 1D list data in the order defined by columns."""
        logger.info(f"Normalizing 1D list data with columns: {columns}")
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"1D column '{col}' missing in DataFrame!"
        col_data = [np.vstack(self.df[col].values) for col in columns]
        data = np.stack(col_data, axis=1)  # shape: (samples, features, length)
        n_samples, n_features, n_length = data.shape
        data_reshaped = data.reshape(n_samples, -1)
        scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
        data_normalized = scaler.fit_transform(data_reshaped)
        data_normalized = data_normalized.reshape(n_samples, n_features, n_length)
        return torch.tensor(data_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_list_2d(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize 2D list data in the order defined by columns (group path).

        Extract FIRST GROUP (column) -> top 10 layers for consistency with inspector/individual.
        """
        logger.info(f"Normalizing 2D list data with columns: {columns}")
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"2D column '{col}' missing in DataFrame!"
        
        # Extract first column and top 10 layers directly for consistent shapes
        col_data = []
        for col in columns:
            values = self.df[col].values
            standardized_samples = []
            
            for val in values:
                try:
                    if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (list, tuple, np.ndarray)):
                        arr = np.array(val[0], dtype=float).reshape(1, -1)
                    else:
                        arr = np.array(val, dtype=object)
                        if getattr(arr, 'ndim', 1) == 2 and arr.shape[0] >= 1:
                            arr = np.array(arr[0, :], dtype=float).reshape(1, -1)
                        elif getattr(arr, 'ndim', 1) == 1 and len(arr) >= 1 and not isinstance(arr[0], (list, tuple, np.ndarray)):
                            arr = np.array(arr, dtype=float).reshape(1, -1)
                        else:
                            arr = None
                    if arr is None:
                        standardized_samples.append(np.zeros((1, 10)))
                    else:
                        # take top 10 layers
                        out = np.zeros((1, 10), dtype=float)
                        take = min(10, arr.shape[1])
                        if take > 0:
                            out[:, :take] = arr[:, :take]
                        standardized_samples.append(out)
                except Exception:
                    standardized_samples.append(np.zeros((1, 10)))
            
            col_data.append(np.stack(standardized_samples))
        
        data = np.stack(col_data, axis=1)  # shape: (samples, features, 1, 10)
        
        # Data is already in the correct shape: (samples, variables, 1, 10)
        # No need for additional extraction since we did it during loading
        
        n_samples, n_features, n_rows, n_cols = data.shape
        data_reshaped = data.reshape(n_samples, -1)
        scaler = self._get_scaler(self.preprocessing_config.list_2d_normalization)
        data_normalized = scaler.fit_transform(data_reshaped)
        data_normalized = data_normalized.reshape(n_samples, n_features, n_rows, n_cols)
        return torch.tensor(data_normalized, dtype=self.preprocessing_config.data_type), scaler

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data for configuration and logging."""
        data_info = {
            'time_series_columns': self.data_config.time_series_columns,
            'static_columns': self.data_config.static_columns,
            'pft_param_columns': self.data_config.pft_param_columns,
            'x_list_scalar_columns': self.data_config.x_list_scalar_columns,
            'y_list_scalar_columns': self.data_config.y_list_scalar_columns,
            'variables_1d_pft': self.data_config.x_list_columns_1d,
            'y_list_columns_1d': self.data_config.y_list_columns_1d,
            'x_list_columns_2d': self.data_config.x_list_columns_2d,
            'y_list_columns_2d': self.data_config.y_list_columns_2d,
            'num_samples': len(self.df) if hasattr(self, 'df') else 0,
            'data_shape': self.df.shape if hasattr(self, 'df') else None
        }
        return data_info
