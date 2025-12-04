"""
CNP Combined Model Architecture

This module provides a specialized neural network model for CNP (Carbon-Nitrogen-Phosphorus)
cycle prediction based on the CNP_IO_list1.txt structure.

Architecture:
- LSTM for 6 time-series variables (20 years)
- FC for surface properties (geographic, soil texture, P forms, PFT coverage)
- FC for 44 PFT characteristics parameters
- FC for water variables (optional)
- FC for scalar variables
- FC for 16 non-scalar variables
- CNN for 87 2D variables
- Transformer encoder for feature fusion
- Multi-task perceptrons for separate predictions
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging

from config.training_config import ModelConfig

logger = logging.getLogger(__name__)


class CNPCombinedModel(nn.Module):
    """
    CNP Combined Model for climate data prediction.
    
    This model implements the specific architecture described in CNP_IO_list1.txt:
    - LSTM for time series (6 variables, 20 years)
    - FC for surface properties (19 variables)
    - FC for PFT parameters (44 variables)
    - FC for water variables (6 variables, optional)
    - FC for scalar variables (5 variables)
    - FC for 1D variables (16 variables)
    - CNN for 2D variables (87 variables)
    - Transformer encoder for feature fusion
    - Multi-task perceptrons for separate predictions
    """
    
    def __init__(self, model_config: ModelConfig, data_info: Dict[str, Any], 
                 include_water: bool = True, include_scalar: bool = True, use_learnable_loss_weights: bool = False):
        """
        Initialize the CNP combined model.
        """
        super(CNPCombinedModel, self).__init__()
        
        self.model_config = model_config
        self.data_info = data_info
        self.include_water = include_water
        self.include_scalar = include_scalar
        self.use_learnable_loss_weights = use_learnable_loss_weights
        self.token_dim = self.model_config.token_dim  # <-- Fix: set token_dim before feature fusion
        # Centralized dropout probability (allows disabling for strict determinism)
        self.dropout_p = getattr(self.model_config, 'dropout_p', 0.1)
        
        # Calculate input dimensions
        self._calculate_input_dimensions()
        
        # Build model components
        self._build_lstm()
        self._build_surface_encoder()
        self._build_pft_1d_encoder()
        self._build_water_encoder()
        self._build_scalar_encoder()
        self._build_soil2d_encoder()
        self._build_pft_param_encoder()
        
        # Track active encoders and their output sizes
        self._track_active_encoders()
        
        self._build_feature_fusion()
        self._build_output_heads()
        
        # Learnable log_sigma parameters for loss weighting (optional)
        if self.use_learnable_loss_weights:
            if self.include_scalar:
                self.log_sigma_scalar = nn.Parameter(torch.zeros(1))
            else:
                self.log_sigma_scalar = None
            self.log_sigma_soil_2d = nn.Parameter(torch.zeros(1))
            if self.include_water:
                self.log_sigma_water = nn.Parameter(torch.zeros(1))
            else:
                self.log_sigma_water = None
            self.log_sigma_pft_1d = nn.Parameter(torch.zeros(1))
        else:
            self.log_sigma_scalar = None
            self.log_sigma_soil_2d = None
            self.log_sigma_water = None
            self.log_sigma_pft_1d = None
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"CNP Model initialized with {self._count_parameters()} parameters")
        logger.info(f"Water variables included: {include_water}")


    def _calculate_input_dimensions(self):
        """Calculate input dimensions based on data info."""
        # Time series input size (6 variables)
        self.lstm_input_size = len(self.data_info['time_series_columns'])
        
        # Surface properties input size (now static + scalar)
        self.surface_input_size = len(self.data_info['static_columns'])
        
        # PFT parameters input size (44 variables)
        self.pft_param_input_size = len(self.data_info.get('pft_param_columns', []))

        # 1D PFT state variables input size (14 variables)
        self.pft_1d_input_size = len(self.data_info.get('variables_1d_pft', []))
        self.vector_length = getattr(self.model_config, 'vector_length', 16)  # fallback if not set
        self.actual_1d_size = len(self.data_info.get('variables_1d_pft', [])) * self.vector_length
        # print(f"[DEBUG] len(data_info['variables_1d_pft']): {len(self.data_info.get('variables_1d_pft', []))}")
        # print(f"[DEBUG] vector_length: {self.vector_length}")
        # print(f"[DEBUG] actual_1d_size: {self.actual_1d_size}")
        
        # Water variables input size (6 variables, optional)
        if self.include_water:
            self.water_input_size = len(self.data_info.get('x_list_water_columns', []))
        else:
            self.water_input_size = 0
        
        # Scalar variables input size (4 variables, optional)
        if self.include_scalar:
            self.scalar_input_size = len(self.data_info.get('x_list_scalar_columns', []))
        else:
            self.scalar_input_size = 0
        # print(f"[DEBUG] scalar_input_size at model init: {self.scalar_input_size}")
        
        # 2D input size
        self.actual_2d_channels = len(self.data_info.get('x_list_columns_2d', []))

        logger.info(f"Input dimensions - LSTM: {self.lstm_input_size}, Surface: {self.surface_input_size}, "
                   f"PFT_param: {self.pft_param_input_size}, Water: {self.water_input_size}, "
                   f"Scalar: {self.scalar_input_size}, 1D PFT: {self.pft_1d_input_size}, "
                   f"2D Soil: {self.actual_2d_channels}")
    
    def _build_lstm(self):
        """Build LSTM component for time series processing (6 variables, 20 years)."""
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.model_config.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_p
        )
    
    def _build_surface_encoder(self):
        """Build encoder for surface properties (geographic, soil texture, P forms, PFT coverage)."""
        if self.surface_input_size > 0:
            self.fc_surface = nn.Sequential(
                nn.Linear(self.surface_input_size, self.model_config.static_fc_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.model_config.static_fc_size, self.model_config.static_fc_size // 2)
            )
        else:
            self.fc_surface = None
            logger.warning("No surface properties found.")
    
    def _build_pft_1d_encoder(self):
        input_dim = self.pft_1d_input_size * 16  # 14*16=224
        output_dim = 128  # Feature size for PFT 1D encoder
        if self.pft_1d_input_size > 0:
            self.fc_pft_1d = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(256, output_dim)
            )
            self.pft_1d_output_dim = output_dim
        else:
            self.fc_pft_1d = None
            self.pft_1d_output_dim = 0
            logger.warning("No PFT 1D variables found.")

    def _get_pft_channels(self):
        # Count number of PFT 1D variables
        return len(self.data_info['variables_1d_pft'])
    
    def _build_water_encoder(self):
        """Build encoder for water variables (6 variables, optional)."""
        if self.include_water and self.water_input_size > 0:
            self.fc_water = nn.Sequential(
                nn.Linear(self.water_input_size, 32),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(32, 16)
            )
        else:
            self.fc_water = None
    
    def _build_scalar_encoder(self):
        """Build encoder for scalar variables (4 variables, optional)."""
        if self.include_scalar and self.scalar_input_size > 0:
            self.fc_scalar = nn.Sequential(
                nn.Linear(self.scalar_input_size, 32),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(32, 16)
            )
        else:
            self.fc_scalar = None
            logger.warning("No scalar variables found.")
    
    
    def _build_soil2d_encoder(self):
        """Build both 1D and 2D encoders for soil variables and select at runtime.

        - 1D encoder runs along the layers axis when height (columns) == 1
        - 2D encoder runs over (rows x cols) when height > 1
        """
        soil2d_channels = self._get_soil2d_channels()
        self.cnn_soil2d_1d = None
        self.cnn_soil2d_2d = None
        if soil2d_channels > 0:
            h, w = self.model_config.matrix_rows, self.model_config.matrix_cols
            # 1D branch (always available for safety)
            layers_1d = []
            in_channels_1d = soil2d_channels
            current_len = w
            for i, out_channels in enumerate(self.model_config.conv_channels):
                layers_1d.extend([
                    nn.Conv1d(
                        in_channels_1d, out_channels,
                        kernel_size=self.model_config.conv_kernel_size,
                        padding=self.model_config.conv_padding
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_p)
                ])
                if i < len(self.model_config.conv_channels) - 1 and current_len >= 2:
                    layers_1d.append(nn.MaxPool1d(2))
                    current_len = max(1, current_len // 2)
                in_channels_1d = out_channels
            conv1d_output_size = self._calculate_conv_output_size_soil1d(initial_len=w)
            layers_1d.extend([
                nn.Flatten(),
                nn.Linear(conv1d_output_size, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(128, 128)
            ])
            self.cnn_soil2d_1d = nn.Sequential(*layers_1d)

            # 2D branch (also available if needed)
            layers_2d = []
            in_channels_2d = soil2d_channels
            hh, ww = h, w
            for i, out_channels in enumerate(self.model_config.conv_channels):
                layers_2d.extend([
                    nn.Conv2d(
                        in_channels_2d, out_channels, 
                        kernel_size=self.model_config.conv_kernel_size,
                        padding=self.model_config.conv_padding
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(self.dropout_p)
                ])
                if i < len(self.model_config.conv_channels) - 1 and (hh >= 2 and ww >= 2):
                    layers_2d.append(nn.MaxPool2d(2))
                    hh = hh // 2
                    ww = ww // 2
                in_channels_2d = out_channels
            conv2d_output_size = self._calculate_conv_output_size_soil2d()
            layers_2d.extend([
                nn.Flatten(),
                nn.Linear(conv2d_output_size, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(128, 128)
            ])
            self.cnn_soil2d_2d = nn.Sequential(*layers_2d)
        else:
            logger.warning("No soil 2D variables found.")

    def _build_pft_param_encoder(self):
        pft_param_size = self.model_config.pft_param_size
        num_pfts = self.model_config.num_pfts
        if not hasattr(self.model_config, 'use_cnn_for_pft_param') or not self.model_config.use_cnn_for_pft_param:
            # Use FC for mini/simple model
            self.fc_pft_param = nn.Sequential(
                nn.Linear(num_pfts, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            self.cnn_pft_param = None
            logger.info(f"Using FC encoder for PFT parameters with {num_pfts} PFTs")
        else:
            # Use 2D CNN for CNPCombinedModel
            self.cnn_pft_param = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),  # Output: [batch, 64]
            )
            self.fc_pft_param = None
            logger.info(f"Using 2D CNN encoder for PFT parameters with {pft_param_size} parameters and {num_pfts} PFTs")

    def _get_soil2d_channels(self):
        # Count number of soil 2D variables
        soil2d_vars = [v for v in self.data_info['x_list_columns_2d']]
        return len(soil2d_vars)

    def _calculate_conv_output_size_soil2d(self) -> int:
        h, w = self.model_config.matrix_rows, self.model_config.matrix_cols
        for i, _ in enumerate(self.model_config.conv_channels):
            if i < len(self.model_config.conv_channels) - 1 and (h >= 2 and w >= 2):
                h = h // 2
                w = w // 2
        return self.model_config.conv_channels[-1] * h * w

    def _calculate_conv_output_size_soil1d(self, initial_len: int) -> int:
        """Compute flattened size after Conv1d(+optional pooling) stack for soil-1D encoder."""
        length = initial_len
        for i, _ in enumerate(self.model_config.conv_channels):
            if i < len(self.model_config.conv_channels) - 1 and length >= 2:
                length = max(1, length // 2)
        return self.model_config.conv_channels[-1] * length
    
    def _track_active_encoders(self):
        """Track which encoders are active and their output sizes."""
        self.active_encoders = []
        self.active_encoder_output_sizes = []

        if self.lstm is not None:
            self.active_encoders.append('lstm')
            self.active_encoder_output_sizes.append(self.model_config.lstm_hidden_size)
        if self.fc_surface is not None:
            self.active_encoders.append('surface')
            self.active_encoder_output_sizes.append(self.model_config.static_fc_size // 2)
        if self.fc_pft_param is not None:
            self.active_encoders.append('pft_param')
            self.active_encoder_output_sizes.append(64)
        elif self.cnn_pft_param is not None:
            self.active_encoders.append('pft_param')
            self.active_encoder_output_sizes.append(64) # Changed from fc_hidden_size // 2 to 64
        if self.fc_scalar is not None:
            self.active_encoders.append('scalar')
            self.active_encoder_output_sizes.append(16)
        if self.fc_water is not None:
            self.active_encoders.append('water')
            self.active_encoder_output_sizes.append(16)
        if self.fc_pft_1d is not None:
            self.active_encoders.append('pft_1d')
            self.active_encoder_output_sizes.append(self.pft_1d_output_dim)
        # Soil2D encoder contributes features if either branch exists
        if self.cnn_soil2d_1d is not None or self.cnn_soil2d_2d is not None:
            self.active_encoders.append('soil2d')
            self.active_encoder_output_sizes.append(128)

        logger.info(f"Active encoders: {self.active_encoders}")
        logger.info(f"Active encoder output sizes: {self.active_encoder_output_sizes}")
        logger.info(f"Total concatenated feature size: {sum(self.active_encoder_output_sizes)}")
    
    def _build_feature_fusion(self):
        """Build feature fusion layers (projection + transformer)."""
        # Calculate concatenated feature size
        concatenated_feature_size = sum(self.active_encoder_output_sizes)
        self.concatenated_feature_size = concatenated_feature_size
        # Debug: Print out active encoder sizes
        logger.info(f"Active encoder output sizes: {self.active_encoder_output_sizes}")
        logger.info(f"Total concatenated feature size: {concatenated_feature_size}")
        # Ensure output size is a multiple of token_dim
        num_tokens = max(1, (concatenated_feature_size + self.token_dim - 1) // self.token_dim)
        self.model_config.num_tokens = num_tokens
        output_size = num_tokens * self.token_dim
        logger.info(f"Feature projection: input size {concatenated_feature_size}, output size {output_size}")
        self.feature_projection = nn.Linear(
            concatenated_feature_size, output_size
        )
        self.feature_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.token_dim, 
                nhead=self.model_config.transformer_heads, 
                dim_feedforward=self.token_dim * 4,
                dropout=self.dropout_p,
                batch_first=True
            ),
            num_layers=self.model_config.transformer_layers
        )

    def _build_output_heads(self):
        """Build multi-task output heads for different prediction types."""
        # Output heads now expect input size self.token_dim (after pooling)
        # Water output head (6 variables)
        if self.include_water:
            self.water_head = nn.Sequential(
                nn.Linear(self.token_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, 6)  # 6 water variables
            )
        else:
            self.water_head = None
        # Scalar output head (6 variables, optional)
        if self.include_scalar and self.model_config.scalar_output_size > 0:
            self.scalar_head = nn.Sequential(
                nn.Linear(self.token_dim, 64),
                nn.BatchNorm1d(64),  # Add BatchNorm
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, self.model_config.scalar_output_size)  # 6 scalar variables
            )
        else:
            self.scalar_head = None
        # 2D output head (dynamic number of 2D soil variables)
        n_2d_vars = len(self.data_info.get('y_list_columns_2d', []))
        self.matrix_head = nn.Sequential(
            nn.Linear(self.token_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, n_2d_vars * self.model_config.matrix_rows * self.model_config.matrix_cols)
        )
        # 1D PFT output head (14 variables x 16 PFTs)
        self.pft_1d_head = nn.Sequential(
            nn.Linear(self.token_dim, 128),
            nn.GELU(),  # Allows negative values while providing non-linearity
            nn.Dropout(0.0),
            nn.Linear(128, self.pft_1d_input_size * self.model_config.vector_length)  # 14 x 16
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _extract_features(self, time_series_data: torch.Tensor, static_data: torch.Tensor, scalar_data: torch.Tensor,
                         variables_1d_pft: torch.Tensor, variables_2d_soil: torch.Tensor, pft_param_data: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from different input types. All inputs are required."""
        features = []
        
        # Extract features in the same order as active_encoders
        for encoder_name in self.active_encoders:
            if encoder_name == 'lstm':
                lstm_out, (hidden, cell) = self.lstm(time_series_data)
                lstm_features = hidden[-1]  # Use last hidden state
                features.append(lstm_features)
                logger.info(f"Feature group: lstm, shape: {lstm_features.shape}")
            elif encoder_name == 'surface':
                if self.fc_surface is not None:
                    surface_features = self.fc_surface(static_data)  # static_data now includes scalar
                    features.append(surface_features)
                    logger.info(f"Feature group: surface, shape: {surface_features.shape}")
            elif encoder_name == 'pft_param':
                if pft_param_data is not None and self.cnn_pft_param is not None:
                    # pft_param_data: [batch, num_pfts, pft_param_size]
                    # CNN expects (batch, channels, sequence_length) = (batch, pft_param_size, num_pfts)
                    x = pft_param_data.transpose(1, 2)  # (batch, pft_param_size, num_pfts)
                    pft_param_features = self.cnn_pft_param(x)
                    features.append(pft_param_features)
            elif encoder_name == 'scalar':
                if self.fc_scalar is not None:
                    scalar_features = self.fc_scalar(scalar_data)
                    features.append(scalar_features)
                    logger.info(f"Feature group: scalar, shape: {scalar_features.shape}")
            elif encoder_name == 'pft_1d':
                if self.fc_pft_1d is not None:
                    # variables_1d_pft: [batch, 14, 16]
                    batch_size = variables_1d_pft.size(0)
                    pft_1d_flat = variables_1d_pft.view(batch_size, -1)  # [batch, 224]
                    pft_1d_features = self.fc_pft_1d(pft_1d_flat)        # [batch, 128]
                    features.append(pft_1d_features)
                    logger.info(f"Feature group: pft_1d, shape: {pft_1d_features.shape}")
            elif encoder_name == 'water':
                if self.fc_water is not None:
                    water_data = torch.stack([variables_1d_pft[:, i] for i, col in enumerate(self.data_info['x_list_columns_1d']) 
                                            if 'H2O' in col], dim=1)
                    water_features = self.fc_water(water_data)
                    features.append(water_features)
                    logger.info(f"Feature group: water, shape: {water_features.shape}")
            elif encoder_name == 'soil2d':
                # Split 2D data: extract soil variables only
                batch_size = variables_2d_soil.size(0)
                soil2d_indices = []
                for i, col in enumerate(self.data_info['x_list_columns_2d']):
                    soil2d_indices.append(i)
                logger.info(f"Total 2D variables: {len(self.data_info['x_list_columns_2d'])}")
                logger.info(f"Number of soil variables: {len(soil2d_indices)}")
                soil2d_data = variables_2d_soil[:, soil2d_indices, :, :]  # [batch, channels, height, width]
                if soil2d_data.size(2) <= 1 and self.cnn_soil2d_1d is not None:
                    soil1d_data = soil2d_data.squeeze(2)  # [batch, channels, width]
                    logger.info(f"Soil1D data shape for CNN1d: {soil1d_data.shape}")
                    conv_features = self.cnn_soil2d_1d(soil1d_data)
                elif soil2d_data.size(2) > 1 and self.cnn_soil2d_2d is not None:
                    logger.info(f"Soil2D data shape for CNN2d: {soil2d_data.shape}")
                    conv_features = self.cnn_soil2d_2d(soil2d_data)
                else:
                    # Fallback: flatten and use a simple linear projection
                    logger.warning("No suitable soil2D encoder found for current shape; using fallback flatten+linear")
                    flat = soil2d_data.view(soil2d_data.size(0), -1)
                    conv_features = nn.Sequential(nn.Linear(flat.size(1), 128), nn.ReLU())(flat)
                features.append(conv_features)
                logger.info(f"Feature group: soil2d, shape: {conv_features.shape}")

        
        return features
    
    def forward(self, time_series_data: torch.Tensor, static_data: torch.Tensor, 
                pft_param_data: torch.Tensor, scalar: torch.Tensor, 
                variables_1d_pft: torch.Tensor, variables_2d_soil: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CNP model.
        Returns a dictionary with keys: 'scalar', 'pft_1d', 'soil_2d'.
        """
        # Debug: Check NaNs in all model inputs
        # print("NaNs in time_series_data:", torch.isnan(time_series_data).sum().item())
        # print("NaNs in static_data:", torch.isnan(static_data).sum().item())
        # print("NaNs in pft_param_data:", torch.isnan(pft_param_data).sum().item())
        # print("NaNs in scalar:", torch.isnan(scalar).sum().item())
        # print("NaNs in variables_1d_pft:", torch.isnan(variables_1d_pft).sum().item())
        # print("NaNs in variables_2d_soil:", torch.isnan(variables_2d_soil).sum().item())

        features = []
        # LSTM for time_series
        if self.lstm is not None:
            lstm_out, (hidden, cell) = self.lstm(time_series_data)
            lstm_features = hidden[-1]
            # print("NaNs in lstm_features:", torch.isnan(lstm_features).sum().item(), "shape:", lstm_features.shape)
            features.append(lstm_features)
        # Static encoder
        if self.fc_surface is not None:
            static_features = self.fc_surface(static_data)
            # print("NaNs in static_features:", torch.isnan(static_features).sum().item(), "shape:", static_features.shape)
            features.append(static_features)
        # PFT param encoder
        if self.fc_pft_param is not None:
            # mini/simple model: pft_param_data shape [batch, num_pfts, 1] or [batch, 1, num_pfts]
            if pft_param_data.shape[-1] == self.model_config.num_pfts:
                x = pft_param_data.squeeze(-2)  # [batch, num_pfts]
            else:
                x = pft_param_data.squeeze(-1)  # [batch, num_pfts]
            pft_param_features = self.fc_pft_param(x)
            features.append(pft_param_features)
        elif self.cnn_pft_param is not None:
            # CNPCombinedModel: pft_param_data shape [batch, num_pfts, pft_param_size] or [batch, pft_param_size, num_pfts]
            if pft_param_data.shape[1] == self.model_config.num_pfts:
                x = pft_param_data.permute(0, 2, 1)  # [batch, pft_param_size, num_pfts]
            else:
                x = pft_param_data
            x = x.unsqueeze(1)  # [batch, 1, pft_param_size, num_pfts]
            pft_param_features = self.cnn_pft_param(x)
            features.append(pft_param_features)
        # Scalar encoder
        if self.include_scalar and self.fc_scalar is not None:
            scalar_features = self.fc_scalar(scalar)
            # print("NaNs in scalar_features:", torch.isnan(scalar_features).sum().item(), "shape:", scalar_features.shape)
            features.append(scalar_features)
        # 1D PFT encoder
        if self.fc_pft_1d is not None:
            batch_size = variables_1d_pft.size(0)
            pft_1d_flat = variables_1d_pft.view(batch_size, -1)
            pft_1d_features = self.fc_pft_1d(pft_1d_flat)
            # print("NaNs in pft_1d_features:", torch.isnan(pft_1d_features).sum().item(), "shape:", pft_1d_features.shape)
            features.append(pft_1d_features)
        # Soil encoder (select 1D vs 2D at runtime)
        if self.cnn_soil2d_1d is not None or self.cnn_soil2d_2d is not None:
            soil2d_data = variables_2d_soil  # [batch, channels, height, width]
            if soil2d_data.size(2) <= 1 and self.cnn_soil2d_1d is not None:
                conv_features = self.cnn_soil2d_1d(soil2d_data.squeeze(2))  # [batch, channels, width]
            elif soil2d_data.size(2) > 1 and self.cnn_soil2d_2d is not None:
                conv_features = self.cnn_soil2d_2d(soil2d_data)
            else:
                flat = soil2d_data.view(soil2d_data.size(0), -1)
                conv_features = nn.Sequential(nn.Linear(flat.size(1), 128), nn.ReLU())(flat)
            features.append(conv_features)
        # Feature fusion
        concatenated_features = torch.cat(features, dim=1)
        batch_size = concatenated_features.size(0)
        projected_features = self.feature_projection(concatenated_features)
        projected_features = projected_features.view(batch_size, self.model_config.num_tokens, self.token_dim)
        fused_features = self.feature_fusion(projected_features)
        fused_features = torch.mean(fused_features, dim=1)
        # Debug: Check for NaNs and stats in fused_features
        # print("NaNs in fused_features:", torch.isnan(fused_features).sum().item())
        # print("Max/Min/Mean fused_features:", fused_features.max().item(), fused_features.min().item(), fused_features.mean().item())
        # Output heads
        outputs = {}
        if self.include_scalar and self.scalar_head is not None:
            scalar_pred = self.scalar_head(fused_features)
            # Apply non-negativity constraint to all outputs (all are pools)
            outputs['scalar'] = torch.relu(scalar_pred)

        # Process PFT 1D outputs to apply specific constraints per variable
        pft_1d_raw_output = self.pft_1d_head(fused_features)
        pft_1d_varnames = self.data_info.get('variables_1d_pft', [])
        n_vars = len(pft_1d_varnames)
        n_pfts = getattr(self, 'vector_length', 16) # Use getattr for safety

        if pft_1d_raw_output.dim() == 2 and pft_1d_raw_output.shape[1] == n_vars * n_pfts:
            pft_1d_reshaped = pft_1d_raw_output.view(-1, n_vars, n_pfts)
            processed_slices = []
            
            for i, var_name in enumerate(pft_1d_varnames):
                if var_name == 'xsmrpool':
                    # During training allow free values; enforce non-positivity only in eval
                    if self.training:
                        processed_slices.append(pft_1d_reshaped[:, i, :].unsqueeze(1))
                    else:
                        processed_slices.append(torch.clamp(pft_1d_reshaped[:, i, :], max=0.0).unsqueeze(1))
                else:
                    # Apply ReLU for other variables (non-negative)
                    processed_slices.append(torch.relu(pft_1d_reshaped[:, i, :]).unsqueeze(1))
            
            # Concatenate all processed slices back along the variable dimension
            pft_1d_pred_final = torch.cat(processed_slices, dim=1)
            # Reshape back to the original flat output shape
            outputs['pft_1d'] = pft_1d_pred_final.view(-1, n_vars * n_pfts)
        else:
            # Fallback if the shape is not as expected, apply ReLU to all as a general constraint
            outputs['pft_1d'] = torch.relu(pft_1d_raw_output)
            
        # Use Softplus to avoid dead ReLU on small positive targets
        outputs['soil_2d'] = torch.nn.functional.softplus(self.matrix_head(fused_features))
        return outputs
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for different output types."""
        if self.use_learnable_loss_weights:
            weights = {}
            if self.include_scalar and self.log_sigma_scalar is not None:
                weights['scalar'] = (1 / (2 * torch.exp(self.log_sigma_scalar) ** 2)).item()
            if self.log_sigma_matrix is not None:
                weights['matrix'] = (1 / (2 * torch.exp(self.log_sigma_matrix) ** 2)).item()
            if self.include_water and self.log_sigma_water is not None:
                weights['water'] = (1 / (2 * torch.exp(self.log_sigma_water) ** 2)).item()
            if self.log_sigma_pft_1d is not None:
                weights['pft_1d'] = (1 / (2 * torch.exp(self.log_sigma_pft_1d) ** 2)).item()
            return weights
        else:
            weights = {
                'matrix': 1.0
            }
            if self.include_scalar:
                weights['scalar'] = 1.0
            if self.include_water:
                weights['water'] = 1.0
            # Optionally add pft_1d if used in loss
            weights['pft_1d'] = 1.0
            return weights
    
    def predict(self, time_series_data: torch.Tensor, static_data: torch.Tensor,
                pft_param_data: torch.Tensor, scalar_data: torch.Tensor,
                variables_1d_pft: torch.Tensor, variables_2d_soil: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate predictions without computing gradients.
        
        Args:
            time_series_data: Time series data
            static_data: Static surface data
            pft_param_data: PFT parameter data
            scalar_data: Scalar input data
            variables_1d_pft: 1D PFT data
            variables_2d_soil: 2D soil data
        Returns:
            Dictionary containing predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(time_series_data, static_data, pft_param_data, scalar_data, variables_1d_pft, variables_2d_soil) 