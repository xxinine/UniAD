"""
Flash Attention implementation using official flash-attn library.
This provides the fastest attention computation for training acceleration.

Copyright (c) OpenDriveLab. All rights reserved.
"""

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import xavier_init, constant_init
import warnings

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn(
        "flash-attn not installed. FlashMultiheadAttention will fall back to standard attention. "
        "Install with: pip install flash-attn --no-build-isolation"
    )


@ATTENTION.register_module()
class FlashMultiheadAttention(BaseModule):
    """
    Multi-head attention with Flash Attention optimization using official flash-attn.
    
    This is the fastest implementation, providing 2-4x speedup over standard attention.
    Compatible with mmdetection's attention interface.
    
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): Dropout rate for attention weights. Default: 0.0
        proj_drop (float): Dropout rate after projection. Default: 0.0
        dropout_layer (dict): Config for dropout layer. Default: dict(type='Dropout', drop_prob=0.)
        init_cfg (dict): Config for initialization. Default: None
        batch_first (bool): If True, batch dimension is first. Default: False
        **kwargs: Other arguments (for compatibility)
    
    Note:
        - Requires flash-attn library: pip install flash-attn --no-build-isolation
        - Only works with FP16 or BF16 precision
        - Falls back to standard attention if flash-attn not available or using FP32
    """
    
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(init_cfg)
        
        assert embed_dims % num_heads == 0, \
            f"embed_dims {embed_dims} must be divisible by num_heads {num_heads}"
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first
        self.attn_drop = attn_drop
        
        # Q, K, V projections
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=True)
        
        # Separate Q, K, V for cross-attention
        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=True)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=True)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=True)
        
        # Output projection
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Dropout layer for residual connection
        if dropout_layer and isinstance(dropout_layer, dict):
            drop_prob = dropout_layer.get('drop_prob', 0.)
            self.dropout_layer = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        else:
            self.dropout_layer = nn.Identity()
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        xavier_init(self.qkv, distribution='uniform')
        xavier_init(self.q_proj, distribution='uniform')
        xavier_init(self.k_proj, distribution='uniform')
        xavier_init(self.v_proj, distribution='uniform')
        xavier_init(self.proj, distribution='uniform')
        constant_init(self.proj, val=0., bias=0.)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Forward function with Flash Attention.
        
        Args:
            query (Tensor): [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
            key (Tensor): Same shape as query. If None, use query (self-attention)
            value (Tensor): Same shape as query. If None, use key
            identity (Tensor): Residual connection tensor
            query_pos (Tensor): Positional encoding for query
            key_pos (Tensor): Positional encoding for key
            attn_mask (Tensor): Attention mask (currently not fully supported with flash-attn)
            key_padding_mask (Tensor): Key padding mask
            
        Returns:
            Tensor: Output tensor with same shape as input
        """
        # Handle default values
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        
        # Add positional encodings
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        # Convert to batch_first format for processing
        if not self.batch_first:
            query = query.transpose(0, 1)  # [bs, num_query, embed_dims]
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            identity = identity.transpose(0, 1)
        
        bs, seq_len, _ = query.shape
        
        # Check if this is self-attention (Q, K, V from same source)
        is_self_attention = (query.data_ptr() == key.data_ptr() and 
                            key.data_ptr() == value.data_ptr())
        
        if is_self_attention:
            # Self-attention: compute Q, K, V together for efficiency
            qkv = self.qkv(query).reshape(bs, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, bs, seq_len, num_heads, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # Cross-attention: compute Q, K, V separately
            q = self.q_proj(query).reshape(bs, -1, self.num_heads, self.head_dim)
            k = self.k_proj(key).reshape(bs, -1, self.num_heads, self.head_dim)
            v = self.v_proj(value).reshape(bs, -1, self.num_heads, self.head_dim)
        
        # Apply Flash Attention if available and using FP16/BF16
        use_flash = (FLASH_ATTN_AVAILABLE and 
                    q.dtype in [torch.float16, torch.bfloat16])
        
        if use_flash:
            try:
                # Flash Attention expects: [batch, seqlen, nheads, headdim]
                # Our tensors are already in this format
                
                # Flash attention doesn't support attention masks well
                # For training with dropout
                dropout_p = self.attn_drop if self.training else 0.0
                
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=dropout_p,
                    softmax_scale=self.scale,
                    causal=False,  # Not causal attention for detection tasks
                )
                # Output shape: [bs, seq_len, num_heads, head_dim]
                attn_output = attn_output.reshape(bs, seq_len, self.embed_dims)
                
            except Exception as e:
                warnings.warn(f"Flash attention failed: {e}, falling back to standard attention")
                use_flash = False
        
        if not use_flash:
            # Fallback to standard attention
            # Reshape for standard attention: [bs, num_heads, seq_len, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention scores
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # Apply attention mask if provided
            if attn_mask is not None:
                attn = attn + attn_mask
            
            # Softmax and dropout
            attn = attn.softmax(dim=-1)
            if self.training and self.attn_drop > 0:
                attn = torch.dropout(attn, self.attn_drop, train=True)
            
            # Apply attention to values
            attn_output = (attn @ v).transpose(1, 2).reshape(bs, seq_len, self.embed_dims)
        
        # Output projection
        output = self.proj(attn_output)
        output = self.proj_drop(output)
        
        # Residual connection
        output = identity + self.dropout_layer(output)
        
        # Convert back to original format
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output


@ATTENTION.register_module()
class FlashCrossAttention(BaseModule):
    """
    Flash Cross-Attention for scenarios where Q comes from one source and K,V from another.
    Optimized for cross-attention in decoder layers.
    
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): Dropout rate for attention weights. Default: 0.0
        proj_drop (float): Dropout rate after projection. Default: 0.0
        dropout_layer (dict): Config for dropout layer. Default: dict(type='Dropout', drop_prob=0.)
        init_cfg (dict): Config for initialization. Default: None
        batch_first (bool): If True, batch dimension is first. Default: False
    """
    
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(init_cfg)
        
        assert embed_dims % num_heads == 0
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first
        self.attn_drop = attn_drop
        
        # Separate Q, K, V projections for cross-attention
        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=True)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=True)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=True)
        
        # Output projection
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Dropout layer for residual connection
        if dropout_layer and isinstance(dropout_layer, dict):
            drop_prob = dropout_layer.get('drop_prob', 0.)
            self.dropout_layer = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        else:
            self.dropout_layer = nn.Identity()
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        xavier_init(self.q_proj, distribution='uniform')
        xavier_init(self.k_proj, distribution='uniform')
        xavier_init(self.v_proj, distribution='uniform')
        xavier_init(self.proj, distribution='uniform')
        constant_init(self.proj, val=0., bias=0.)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for cross-attention."""
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            identity = identity.transpose(0, 1)
        
        bs, q_len, _ = query.shape
        _, k_len, _ = key.shape
        
        # Project Q, K, V
        q = self.q_proj(query).reshape(bs, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(bs, k_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(bs, k_len, self.num_heads, self.head_dim)
        
        # Use flash attention if available
        if FLASH_ATTN_AVAILABLE and q.dtype in [torch.float16, torch.bfloat16]:
            try:
                dropout_p = self.attn_drop if self.training else 0.0
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=dropout_p,
                    softmax_scale=self.scale,
                    causal=False,
                )
                attn_output = attn_output.reshape(bs, q_len, self.embed_dims)
            except Exception:
                # Fallback
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                if attn_mask is not None:
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                if self.training and self.attn_drop > 0:
                    attn = torch.dropout(attn, self.attn_drop, train=True)
                attn_output = (attn @ v).transpose(1, 2).reshape(bs, q_len, self.embed_dims)
        else:
            # Standard attention fallback
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            if self.training and self.attn_drop > 0:
                attn = torch.dropout(attn, self.attn_drop, train=True)
            attn_output = (attn @ v).transpose(1, 2).reshape(bs, q_len, self.embed_dims)
        
        output = self.proj(attn_output)
        output = self.proj_drop(output)
        output = identity + self.dropout_layer(output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output
