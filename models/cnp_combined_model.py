import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
import logging
from config.training_config import ModelConfig

logger = logging.getLogger(__name__)

# --- 辅助函数 ---
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """生成 1D Sin-Cos 位置编码"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return torch.from_numpy(emb).float()

# --- Stream 1: 动态变量编码器 ---
class ForcingTemporalEncoder(nn.Module):
    def __init__(self, num_vars, embed_dim, total_time_steps, patch_size, lat_lon_dim=2):
        super().__init__()
        assert total_time_steps % patch_size == 0
        self.num_patches = total_time_steps // patch_size
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        
        # 1. 独立时间分块 (Grouped Conv1d)
        self.patch_embed = nn.Conv1d(
            in_channels=num_vars,
            out_channels=num_vars * embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            groups=num_vars
        )
        
        # 2. Variable ID & Time Pos
        self.var_embed = nn.Parameter(torch.zeros(1, num_vars, 1, embed_dim))
        self.time_pos = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))
        
        # 3. LatLon Injection
        self.lat_lon_mlp = nn.Sequential(
            nn.Linear(lat_lon_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        # SinCos 初始化
        t_pos = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.arange(self.num_patches, dtype=np.float32))
        self.time_pos.data.copy_(t_pos.unsqueeze(0).unsqueeze(0))
        
        v_pos = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.arange(self.num_vars, dtype=np.float32))
        self.var_embed.data.copy_(v_pos.unsqueeze(0).unsqueeze(2))
        
        # Init MLP
        for m in self.lat_lon_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, lat_lon):
        B, T, V = x.shape
        x = x.transpose(1, 2) # [B, V, T]
        
        # Patching -> [B, V*D, N]
        x = self.patch_embed(x)
        
        # Reshape -> [B, V, N, D]
        x = x.view(B, V, self.embed_dim, self.num_patches).permute(0, 1, 3, 2)
        
        # Add Embeddings (Broadcasting)
        x = x + self.var_embed
        x = x + self.time_pos
        
        # Add Global Context
        geo = self.lat_lon_mlp(lat_lon).view(B, 1, 1, self.embed_dim)
        x = x + geo
        
        # Flatten -> [B, V*N, D]
        return x.reshape(B, -1, self.embed_dim)

# --- Stream 2: 静态变量编码器 ---
class StaticVariableEncoder(nn.Module):
    def __init__(self, input_group_indices, total_input_dim, embed_dim, group_ids):
        super().__init__()
        self.embed_dim = embed_dim
        # input_group_indices: List[List[int]]. 每个元素是一个列表，包含该 Token 对应的输入变量索引。
        # 例如: [[0,1,2], [3], [4]...] 表示第一个Token由变量0,1,2组成，后续是单变量。
        
        self.input_group_indices = input_group_indices
        self.num_tokens = len(input_group_indices)
        
        # 1. 分离单变量和多变量
        self.single_var_indices = [] # List of ints (input indices)
        self.multi_var_configs = []  # List of (output_token_index, input_indices)
        
        # 为了保持输出 Token 的顺序与 group_ids 一致，我们需要记录映射关系
        # 但为了计算效率，我们将单变量批量处理。
        # 策略：先计算所有 Token，最后按顺序拼回去？或者直接拼接？
        # 简单起见，我们将单变量和多变量分开处理，然后拼接。
        # 注意：group_ids 必须与这里生成的 Token 顺序对应。
        # 在外部调用者那里，我们约定：先放多变量组(如果被置顶)，或者按 input_group_indices 的顺序。
        
        # 实际上，为了效率，我们应该把所有 Single Vars 收集起来一次性处理。
        # 我们记录 Single Vars 在 input_group_indices 中的位置，以便最后恢复顺序（如果需要）。
        # 这里简化：我们假设输入 group_ids 已经按照 [Multi_Groups..., Single_Groups...] 或者任何我们处理后的顺序排列。
        # 为了通用性，我们记录每个 Token 是怎么生成的。
        
        self.token_generators = nn.ModuleList()
        self.token_types = [] # 'single' or 'multi'
        self.token_indices = [] # input indices for each token
        
        # 优化：收集所有单变量索引
        self.single_indices_flat = []
        self.single_token_positions = [] # 记录这些单变量对应最终输出的第几个 Token
        
        for i, indices in enumerate(input_group_indices):
            dim = len(indices)
            if dim == 1:
                self.token_types.append('single')
                self.single_indices_flat.append(indices[0])
                self.single_token_positions.append(i)
                self.token_generators.append(None) # Placeholder
            else:
                # 多变量或向量类型：使用 Linear 层映射到 embed_dim
                self.token_types.append('multi')
                self.token_generators.append(nn.Linear(dim, embed_dim))
                self.token_indices.append(indices) # Keep list
                
        # 构建批量单变量处理器
        self.num_single = len(self.single_indices_flat)
        if self.num_single > 0:
            self.single_proj = nn.Conv1d(
                in_channels=self.num_single,
                out_channels=self.num_single * embed_dim,
                kernel_size=1,
                groups=self.num_single
            )
            # Init single proj
            w = self.single_proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # 注册缓冲区
        if self.num_single > 0:
            self.register_buffer('single_indices_tensor', torch.tensor(self.single_indices_flat, dtype=torch.long))
        
        # 2. Embeddings
        self.var_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.group_embed = nn.Embedding(len(set(group_ids)) + 1, embed_dim)
        self.register_buffer('group_ids_tensor', torch.tensor(group_ids, dtype=torch.long))
        
        self._init_weights()

    def _init_weights(self):
        # SinCos Pos Emb
        v_pos = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.arange(self.num_tokens, dtype=np.float32))
        self.var_embed.data.copy_(v_pos.unsqueeze(0))
        nn.init.normal_(self.group_embed.weight, std=0.02)
        
        # Multi-var Linear layers init
        for m in self.token_generators:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, Total_Static_Vars] or [B, Total_Static_Vars, 1]
        if x.dim() == 3:
            x = x.squeeze(-1)
        B = x.shape[0]
        
        # 容器用于存放每个位置的 Token [B, 1, D]
        # 为了处理顺序，我们可以创建一个列表 list of [B, 1, D]
        tokens = [None] * self.num_tokens
        
        # 1. Process Multi-vars
        multi_idx = 0
        for i, type_ in enumerate(self.token_types):
            if type_ == 'multi':
                indices = self.token_indices[multi_idx]
                # x[:, indices] -> [B, Dim]
                # Gather requires tensor index
                # 优化: 将 indices 转为 tensor 放在 loop 外面如果性能成问题，但这里通常只有几个 multi group
                idx_tensor = torch.tensor(indices, device=x.device)
                inp = x.index_select(1, idx_tensor) 
                out = self.token_generators[i](inp) # [B, D]
                tokens[i] = out.unsqueeze(1) # [B, 1, D]
                multi_idx += 1
        
        # 2. Process Single-vars (Batch)
        if self.num_single > 0:
            # Gather all single inputs
            # x: [B, Total]
            inp_single = x.index_select(1, self.single_indices_tensor) # [B, N_single]
            inp_single = inp_single.unsqueeze(-1) # [B, N_single, 1]
            
            out_single = self.single_proj(inp_single) # [B, N_single*D, 1]
            out_single = out_single.view(B, self.num_single, self.embed_dim) # [B, N_single, D]
            
            # Distribute back to tokens list
            for j, pos in enumerate(self.single_token_positions):
                tokens[pos] = out_single[:, j, :].unsqueeze(1)
                
        # 3. Concat
        final_tokens = torch.cat(tokens, dim=1) # [B, N_tokens, D]
        
        # 4. Add Embeddings
        final_tokens = final_tokens + self.var_embed
        g_emb = self.group_embed(self.group_ids_tensor)
        final_tokens = final_tokens + g_emb.unsqueeze(0)
        
        return final_tokens

# --- Main Model ---
class CNPCombinedModel(nn.Module):
    def __init__(self, model_config: ModelConfig, data_info: Dict[str, Any], 
                 include_water: bool = True, use_learnable_loss_weights: bool = False):
        super(CNPCombinedModel, self).__init__()
        
        self.model_config = model_config
        self.data_info = data_info
        self.include_water = include_water
        self.use_learnable_loss_weights = use_learnable_loss_weights
        
        # --- 核心修改 1: 统一维度 ---
        # 强制所有流使用相同的 Embed Dim，确保可以拼接
        self.embed_dim = getattr(self.model_config, 'embed_dim', 256)
        self.token_dim = self.embed_dim 
        self.dropout_p = getattr(self.model_config, 'dropout_p', 0.1)
        
        # 1. 计算维度
        self._calculate_input_dimensions()
        
        # 2. 构建 Stream 1: 动态 (Forcing)
        # 修正: 使用 self.embed_dim 而不是 lstm_hidden_size
        self._build_temporal_encoder()
        
        # 3. 构建 Stream 2: 静态 (Static)
        self.input_group_indices, self.group_ids_list = self._configure_static_structure()
        
        self.static_encoder = StaticVariableEncoder(
            input_group_indices=self.input_group_indices,
            total_input_dim=self.total_static_input_dim,
            embed_dim=self.embed_dim,
            group_ids=self.group_ids_list
        )
        
        # 4. Backbone (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=getattr(self.model_config, 'transformer_heads', 4),
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout_p,
            batch_first=True,
            norm_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=getattr(self.model_config, 'transformer_layers', 6)
        )
        
        # 5. Multihead/Multimode Configuration
        self.use_multihead_attention = getattr(self.model_config, 'use_multihead_attention', False)
        self.multihead_num_heads = getattr(self.model_config, 'multihead_num_heads', 4)
        self.use_mode_specific_heads = getattr(self.model_config, 'use_mode_specific_heads', False)
        
        # 6. Output Heads
        self._build_output_heads()
        
        # Loss weights setup (保持原样)
        self._setup_loss_weights()
        self._initialize_weights()

    def _calculate_input_dimensions(self):
        # 保持原逻辑，获取各变量维度
        self.lstm_input_size = len(self.data_info['time_series_columns'])
        self.time_series_length = self.data_info.get('time_series_length', 240)
        self.surface_input_size = len(self.data_info['static_columns'])
        self.pft_param_input_size = len(self.data_info.get('pft_param_columns', []))
        # PFT Param 数据形状是 [batch, num_params, num_pfts]，需要计算实际 flatten 后的大小
        num_pfts = getattr(self.model_config, 'num_pfts', 17)  # 默认 17 个 PFT
        self.actual_pft_param_size = self.pft_param_input_size * num_pfts
        self.pft_1d_input_size = len(self.data_info.get('variables_1d_pft', []))
        self.vector_length = getattr(self.model_config, 'vector_length', 16)
        self.actual_1d_size = self.pft_1d_input_size * self.vector_length
        self.water_input_size = len(self.data_info.get('x_list_water_columns', [])) if self.include_water else 0
        self.scalar_input_size = len(self.data_info.get('x_list_scalar_columns', []))
        self.actual_2d_channels = len(self.data_info.get('x_list_columns_2d', []))
        
        logger.info(f"Stream 1 Dim: {self.lstm_input_size} vars over {self.time_series_length} steps")
        logger.info(f"PFT Param: {self.pft_param_input_size} params × {num_pfts} PFTs = {self.actual_pft_param_size} tokens")

    def _build_temporal_encoder(self):
        # 修正: 传入 self.embed_dim
        patch_size = getattr(self.model_config, 'patch_size', 60) # 默认5年
        if self.time_series_length % patch_size != 0: 
            logger.warning(f"Configured patch_size {patch_size} does not divide time_series_length {self.time_series_length}. Fallback to 12.")
            patch_size = 12 # fallback
        
        self.temporal_encoder = ForcingTemporalEncoder(
            num_vars=self.lstm_input_size,
            embed_dim=self.embed_dim,
            total_time_steps=self.time_series_length,
            patch_size=patch_size,
            lat_lon_dim=2 
        )

    def _configure_static_structure(self):
        # 构建 input_group_indices 和 group_ids
        # 逻辑：将 Surface 中的特定变量（PFT, Clay, Sand）分组，其他保持单变量
        input_group_indices = []
        group_ids = []
        
        current_offset = 0
        
        # 1. Surface (Group 0)
        surface_cols = self.data_info.get('static_columns', [])
        
        # Identify groups
        special_prefixes = ['PCT_NAT_PFT', 'PCT_CLAY', 'PCT_SAND']
        grouped_indices = {p: [] for p in special_prefixes}
        single_indices = []
        
        for i, col in enumerate(surface_cols):
            matched = False
            for prefix in special_prefixes:
                if col.startswith(prefix):
                    grouped_indices[prefix].append(i)
                    matched = True
                    break
            if not matched:
                single_indices.append(i)
        
        # Emit Groups First
        for prefix in special_prefixes:
            indices = grouped_indices[prefix]
            if indices:
                abs_indices = [idx + current_offset for idx in indices]
                input_group_indices.append(abs_indices)
                group_ids.append(0)
        
        # Emit Singles
        for idx in single_indices:
            input_group_indices.append([idx + current_offset])
            group_ids.append(0)
            
        current_offset += len(surface_cols)
        
        # 2. PFT Param (Group 1) - 向量整体输入
        # PFT Param 数据形状是 [batch, num_params, num_pfts]
        # 每个参数作为一个向量整体输入 Linear(num_pfts, embed_dim)
        num_pfts = getattr(self.model_config, 'num_pfts', 17)
        for i in range(self.pft_param_input_size):
            # 每个 token 对应一个参数的所有 PFT 值（向量）
            start_idx = current_offset + i * num_pfts
            end_idx = current_offset + (i + 1) * num_pfts
            input_group_indices.append(list(range(start_idx, end_idx)))  # 向量索引范围
            group_ids.append(1)
        current_offset += self.actual_pft_param_size
        
        # 3. Scalar (Group 2)
        for i in range(self.scalar_input_size):
            input_group_indices.append([current_offset + i])
            group_ids.append(2)
        current_offset += self.scalar_input_size
        
        # 4. Water (Group 3)
        if self.include_water:
            for i in range(self.water_input_size):
                input_group_indices.append([current_offset + i])
                group_ids.append(3)
            current_offset += self.water_input_size
            
        # 5. PFT 1D (Group 4) - 向量整体输入
        # PFT 1D 数据形状是 [batch, num_vars, vector_length]
        # 每个变量作为一个向量整体输入 Linear(vector_length, embed_dim)
        for i in range(self.pft_1d_input_size):
            # 每个 token 对应一个变量的所有 PFT 值（向量）
            start_idx = current_offset + i * self.vector_length
            end_idx = current_offset + (i + 1) * self.vector_length
            input_group_indices.append(list(range(start_idx, end_idx)))  # 向量索引范围
            group_ids.append(4)
        current_offset += self.actual_1d_size
        
        # 6. Soil 2D (Group 5) - 向量整体输入
        # 计算每个变量实际包含的元素总数 (Rows * Cols)
        elements_per_var = self.model_config.matrix_rows * self.model_config.matrix_cols
        
        for i in range(self.actual_2d_channels):
            # 计算起始位置：必须跳过前面所有变量的完整长度
            start_idx = current_offset + i * elements_per_var
            
            # 结束位置：我们要把该变量的所有数据(所有行和列)打包成一个Token
            end_idx = start_idx + elements_per_var
            
            # 添加索引列表
            input_group_indices.append(list(range(start_idx, end_idx)))
            group_ids.append(5)
            
        # 更新总偏移量
        current_offset += self.actual_2d_channels * elements_per_var
        
        self.total_static_input_dim = current_offset
        return input_group_indices, group_ids

    def _build_output_heads(self):
        # Multihead Attention layers for mode-specific feature extraction
        if self.use_multihead_attention:
            # Multi-head attention for each output mode
            self.scalar_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.multihead_num_heads,
                dropout=self.dropout_p,
                batch_first=True
            )
            self.soil2d_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.multihead_num_heads,
                dropout=self.dropout_p,
                batch_first=True
            )
            self.pft1d_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.multihead_num_heads,
                dropout=self.dropout_p,
                batch_first=True
            )
            
            # Query vectors for each mode (learnable)
            self.scalar_query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.soil2d_query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.pft1d_query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            
            # Mode-specific feature projection
            self.scalar_feat_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.soil2d_feat_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.pft1d_feat_proj = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.scalar_attention = None
            self.soil2d_attention = None
            self.pft1d_attention = None
        
        # Mode-specific feature extractors (optional)
        if self.use_mode_specific_heads:
            # Separate feature extractors for each mode
            self.scalar_feature_extractor = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p)
            )
            self.soil2d_feature_extractor = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p)
            )
            self.pft1d_feature_extractor = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p)
            )
        else:
            self.scalar_feature_extractor = None
            self.soil2d_feature_extractor = None
            self.pft1d_feature_extractor = None
        
        # Output heads - 输入维度改为 self.embed_dim (因为 backbone 输出也是这个维度)
        self.scalar_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(self.dropout_p),
            nn.Linear(64, self.model_config.scalar_output_size)
        )
        # ... (其他 Head 结构类似，只需把输入 dim 改为 self.embed_dim)
        n_2d_vars = len(self.data_info.get('y_list_columns_2d', []))
        self.matrix_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(), nn.Dropout(self.dropout_p),
            nn.Linear(128, n_2d_vars * self.model_config.matrix_rows * self.model_config.matrix_cols)
        )
        self.pft_1d_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.GELU(), nn.Dropout(0.0),
            nn.Linear(128, self.pft_1d_input_size * self.model_config.vector_length)
        )
        
        # Water head
        if self.include_water:
             self.water_head = nn.Sequential(
                nn.Linear(self.embed_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, 6)
            )
        else:
            self.water_head = None


    def _setup_loss_weights(self):
        # 简化的 loss weights 初始化
        if self.use_learnable_loss_weights:
            self.log_sigma_scalar = nn.Parameter(torch.zeros(1))
            self.log_sigma_soil_2d = nn.Parameter(torch.zeros(1))
            self.log_sigma_pft_1d = nn.Parameter(torch.zeros(1))
            self.log_sigma_water = nn.Parameter(torch.zeros(1)) if self.include_water else None
        else:
            self.log_sigma_scalar = None
            self.log_sigma_soil_2d = None
            self.log_sigma_pft_1d = None
            self.log_sigma_water = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        # Initialize multihead attention query vectors
        if self.use_multihead_attention:
            nn.init.normal_(self.scalar_query, std=0.02)
            nn.init.normal_(self.soil2d_query, std=0.02)
            nn.init.normal_(self.pft1d_query, std=0.02)
            
            # Initialize feature projection layers
            if hasattr(self, 'scalar_feat_proj'):
                nn.init.xavier_uniform_(self.scalar_feat_proj.weight)
                if self.scalar_feat_proj.bias is not None:
                    nn.init.zeros_(self.scalar_feat_proj.bias)
            if hasattr(self, 'soil2d_feat_proj'):
                nn.init.xavier_uniform_(self.soil2d_feat_proj.weight)
                if self.soil2d_feat_proj.bias is not None:
                    nn.init.zeros_(self.soil2d_feat_proj.bias)
            if hasattr(self, 'pft1d_feat_proj'):
                nn.init.xavier_uniform_(self.pft1d_feat_proj.weight)
                if self.pft1d_feat_proj.bias is not None:
                    nn.init.zeros_(self.pft1d_feat_proj.bias)

    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for different output types."""
        if self.use_learnable_loss_weights:
            weights = {}
            if self.log_sigma_scalar is not None:
                weights['scalar'] = (1 / (2 * torch.exp(self.log_sigma_scalar) ** 2)).item()
            if self.log_sigma_soil_2d is not None:
                weights['soil_2d'] = (1 / (2 * torch.exp(self.log_sigma_soil_2d) ** 2)).item()
            if self.include_water and self.log_sigma_water is not None:
                weights['water'] = (1 / (2 * torch.exp(self.log_sigma_water) ** 2)).item()
            if self.log_sigma_pft_1d is not None:
                weights['pft_1d'] = (1 / (2 * torch.exp(self.log_sigma_pft_1d) ** 2)).item()
            return weights
        else:
            weights = {
                'scalar': 1.0,
                'soil_2d': 1.0,
                'pft_1d': 1.0
            }
            if self.include_water:
                weights['water'] = 1.0
            return weights

    def predict(self, time_series_data: torch.Tensor, static_data: torch.Tensor,
                pft_param_data: torch.Tensor, scalar_data: torch.Tensor,
                variables_1d_pft: torch.Tensor, variables_2d_soil: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate predictions without computing gradients.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(time_series_data, static_data, pft_param_data, scalar_data, variables_1d_pft, variables_2d_soil)

    def forward(self, time_series_data, static_data, pft_param_data, scalar, variables_1d_pft, variables_2d_soil):
        # 1. Stream 1 Encode
        lat_lon = static_data[:, :2]
        dyn_tokens = self.temporal_encoder(time_series_data, lat_lon)

        # 2. Stream 2 Encode (Data Preparation)
        batch_size = static_data.shape[0]
        
        # Flatten components
        flat_surface = static_data.reshape(batch_size, -1)  # Ensure 2D [Batch, Num_Vars]
        flat_pft_param = pft_param_data.reshape(batch_size, -1)
        flat_scalar = scalar
        flat_pft_1d = variables_1d_pft.reshape(batch_size, -1)
        flat_soil_2d = variables_2d_soil.reshape(batch_size, -1)
        
        static_components = [flat_surface, flat_pft_param, flat_scalar]
        
        # --- 核心修正 2: 正确处理 Water ---
        if self.include_water:
            # 假设 water 数据包含在 variables_1d_pft 中，或者是独立的输入。
            # 这里为了代码健壮性，我们暂时假设 water 需要被手动提取或已经是输入的一部分。
            # 这是一个占位符逻辑，你需要根据实际数据加载器调整：
            # 如果 water 是第4个参数传入的，请确保它被加上：
            # flat_water = ...
            # static_components.append(flat_water)
            pass 
        
        static_components.append(flat_pft_1d)
        static_components.append(flat_soil_2d)
        
        static_input_vector = torch.cat(static_components, dim=1)
        sta_tokens = self.static_encoder(static_input_vector)

        # 3. Direct Concatenation
        # [B, N_dyn + N_sta, D]
        all_tokens = torch.cat([dyn_tokens, sta_tokens], dim=1)

        # 4. Backbone
        features = self.backbone(all_tokens)

        # 5. Multihead/Multimode Feature Extraction
        if self.use_multihead_attention:
            # Extract mode-specific features using multi-head attention
            batch_size = features.shape[0]
            
            # Scalar mode: attention over all tokens with scalar-specific query
            scalar_query = self.scalar_query.expand(batch_size, -1, -1)
            scalar_attn_out, _ = self.scalar_attention(scalar_query, features, features)
            scalar_feat = scalar_attn_out.squeeze(1)  # [B, D]
            scalar_feat = self.scalar_feat_proj(scalar_feat)
            
            # Soil 2D mode: attention over all tokens with soil2d-specific query
            soil2d_query = self.soil2d_query.expand(batch_size, -1, -1)
            soil2d_attn_out, _ = self.soil2d_attention(soil2d_query, features, features)
            soil2d_feat = soil2d_attn_out.squeeze(1)  # [B, D]
            soil2d_feat = self.soil2d_feat_proj(soil2d_feat)
            
            # PFT 1D mode: attention over all tokens with pft1d-specific query
            pft1d_query = self.pft1d_query.expand(batch_size, -1, -1)
            pft1d_attn_out, _ = self.pft1d_attention(pft1d_query, features, features)
            pft1d_feat = pft1d_attn_out.squeeze(1)  # [B, D]
            pft1d_feat = self.pft1d_feat_proj(pft1d_feat)
        else:
            # Standard global pooling
            global_feat = features.mean(dim=1)
            scalar_feat = global_feat
            soil2d_feat = global_feat
            pft1d_feat = global_feat
        
        # Apply mode-specific feature extractors if enabled
        if self.use_mode_specific_heads:
            scalar_feat = self.scalar_feature_extractor(scalar_feat)
            soil2d_feat = self.soil2d_feature_extractor(soil2d_feat)
            pft1d_feat = self.pft1d_feature_extractor(pft1d_feat)

        # 6. Heads
        outputs = {}
        outputs['scalar'] = torch.relu(self.scalar_head(scalar_feat))
        outputs['soil_2d'] = torch.nn.functional.softplus(self.matrix_head(soil2d_feat))
        
        pft_out = self.pft_1d_head(pft1d_feat)
        # Process PFT output
        pft_1d_varnames = self.data_info.get('variables_1d_pft', [])
        n_vars = len(pft_1d_varnames)
        n_pfts = getattr(self, 'vector_length', 16)

        def _apply_pft1d_activation(var_name: str, x: torch.Tensor) -> torch.Tensor:
            activation = getattr(self.model_config, 'pft1d_activation', 'abs')
            overrides = getattr(self.model_config, 'pft1d_activation_overrides', {}) or {}
            key = var_name
            if key in overrides:
                activation = overrides[key]
            elif f'Y_{key}' in overrides:
                activation = overrides[f'Y_{key}']
            activation = str(activation).lower()
            if activation == 'relu':
                return torch.relu(x)
            if activation == 'softplus':
                return torch.nn.functional.softplus(x)
            if activation == 'linear':
                return x
            # default abs
            return torch.abs(x)

        if pft_out.dim() == 2 and pft_out.shape[1] == n_vars * n_pfts:
            pft_reshaped = pft_out.view(-1, n_vars, n_pfts)
            processed_slices = []
            for i, var_name in enumerate(pft_1d_varnames):
                if var_name == 'xsmrpool':
                    if self.training:
                        processed_slices.append(pft_reshaped[:, i, :].unsqueeze(1))
                    else:
                        processed_slices.append(torch.clamp(pft_reshaped[:, i, :], max=0.0).unsqueeze(1))
                else:
                    processed_slices.append(_apply_pft1d_activation(var_name, pft_reshaped[:, i, :]).unsqueeze(1))
            pft_final = torch.cat(processed_slices, dim=1)
            outputs['pft_1d'] = pft_final.view(-1, n_vars * n_pfts)
        else:
            outputs['pft_1d'] = _apply_pft1d_activation('default', pft_out)

        return outputs
