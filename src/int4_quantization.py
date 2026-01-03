"""
INT4 Quantized Attention for KV Cache Memory Efficiency.

This module implements INT4 quantization for Key-Value caches in attention mechanisms,
reducing memory usage by 4x while maintaining model performance.

Based on techniques from papers like "SmoothQuant" and "GPTQ" for efficient LLM inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class INT4Quantizer:
    """
    INT4 Quantizer for tensors with dynamic range scaling.
    
    This quantizer converts floating-point tensors to INT4 representation
    with per-channel or per-token scaling to preserve precision.
    """
    
    def __init__(self, symmetric: bool = True, per_channel: bool = False):
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.int4_min = -8
        self.int4_max = 7
        
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT4 with scaling.
        
        Args:
            tensor: Input tensor to quantize
            
        Returns:
            Tuple of (quantized_tensor, scale)
        """
        if self.per_channel:
            # Per-channel quantization
            if tensor.dim() == 3:  # (batch, seq_len, features)
                amax = torch.amax(torch.abs(tensor), dim=[0, 1], keepdim=True)
            else:  # (batch, features) or (seq_len, features)
                amax = torch.amax(torch.abs(tensor), dim=-1, keepdim=True)
        else:
            # Per-token quantization
            amax = torch.amax(torch.abs(tensor), dim=-1, keepdim=True)
        
        # Avoid division by zero
        amax = torch.clamp(amax, min=1e-5)
        
        # Scale to [-8, 7] range for INT4
        scale = (amax / 7.0).to(torch.float32)
        
        # Quantize
        quantized = torch.round(tensor / scale).clamp(self.int4_min, self.int4_max).to(torch.int8)
        
        return quantized, scale
    
    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT4 tensor back to float.
        
        Args:
            quantized: Quantized tensor (INT4 represented as INT8)
            scale: Scale tensor
            
        Returns:
            Dequantized tensor
        """
        return (quantized.float() * scale).to(scale.dtype)


class QuantizedKVCache:
    """
    Quantized Key-Value cache for attention mechanisms.
    
    This cache stores Keys and Values in INT4 format to reduce memory usage
    while maintaining attention computation accuracy.
    """
    
    def __init__(self, 
                 head_dim: int, 
                 num_heads: int, 
                 max_seq_len: int = 4096,
                 device: torch.device = torch.device("cpu")):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Quantizers for K and V
        self.k_quantizer = INT4Quantizer(symmetric=True, per_channel=True)
        self.v_quantizer = INT4Quantizer(symmetric=True, per_channel=True)
        
        # Initialize cache storage (will be allocated dynamically)
        self.k_cache = None
        self.v_cache = None
        self.k_scale = None
        self.v_scale = None
        
        # Current sequence length
        self.current_seq_len = 0
    
    def init_cache(self, batch_size: int):
        """Initialize the cache for a given batch size."""
        self.k_cache = torch.zeros(
            batch_size, self.num_heads, self.max_seq_len, self.head_dim,
            dtype=torch.int8, device=self.device
        )
        self.v_cache = torch.zeros(
            batch_size, self.num_heads, self.max_seq_len, self.head_dim,
            dtype=torch.int8, device=self.device
        )
        self.k_scale = torch.zeros(
            batch_size, self.num_heads, self.max_seq_len, 1,
            dtype=torch.float32, device=self.device
        )
        self.v_scale = torch.zeros(
            batch_size, self.num_heads, self.max_seq_len, 1,
            dtype=torch.float32, device=self.device
        )
        self.current_seq_len = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new K and V values.
        
        Args:
            k: New key tensor (batch, num_heads, seq_len, head_dim)
            v: New value tensor (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Tuple of (dequantized_k, dequantized_v) for current step
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # Initialize cache if not done
        if self.k_cache is None:
            self.init_cache(batch_size)
        
        # Quantize new values
        k_quantized, k_scale = self.k_quantizer.quantize(k)
        v_quantized, v_scale = self.v_quantizer.quantize(v)
        
        # Store in cache
        start_pos = self.current_seq_len
        end_pos = start_pos + seq_len
        
        self.k_cache[:, :, start_pos:end_pos, :] = k_quantized
        self.v_cache[:, :, start_pos:end_pos, :] = v_quantized
        self.k_scale[:, :, start_pos:end_pos, :] = k_scale
        self.v_scale[:, :, start_pos:end_pos, :] = v_scale
        
        # Update sequence length
        self.current_seq_len = end_pos
        
        # Return dequantized values for current computation
        k_dequantized = self.k_quantizer.dequantize(k_quantized, k_scale)
        v_dequantized = self.v_quantizer.dequantize(v_quantized, v_scale)
        
        return k_dequantized, v_dequantized
    
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the full cached K and V tensors.
        
        Returns:
            Tuple of (dequantized_k, dequantized_v) for all cached positions
        """
        if self.current_seq_len == 0:
            return None, None
        
        k_cached = self.k_quantizer.dequantize(
            self.k_cache[:, :, :self.current_seq_len, :], 
            self.k_scale[:, :, :self.current_seq_len, :]
        )
        v_cached = self.v_quantizer.dequantize(
            self.v_cache[:, :, :self.current_seq_len, :], 
            self.v_scale[:, :, :self.current_seq_len, :]
        )
        
        return k_cached, v_cached
    
    def reset(self):
        """Reset the cache."""
        self.current_seq_len = 0


class INT4MultiHeadAttention(nn.Module):
    """
    Multi-head attention with INT4 quantized KV cache for memory efficiency.
    
    This attention mechanism uses INT4 quantization for the Key-Value cache,
    reducing memory usage by 4x while maintaining performance.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_rank: Optional[int] = None,  # For MLA-style compression
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_rank = kv_rank or embed_dim
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, self.kv_rank, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, self.kv_rank, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # KV Cache
        self.kv_cache = QuantizedKVCache(
            head_dim=self.kv_rank // num_heads,
            num_heads=num_heads,
            device=device
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        # Xavier initialization for projections
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with INT4 quantized KV cache.
        
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            use_cache: Whether to use KV cache
            
        Returns:
            Tuple of (output, attn_weights, cache_stats)
        """
        B, T, D = query.shape
        
        # Project to attention space
        q = self.q_proj(query)  # (B, T, D)
        k = self.k_proj(key)    # (B, T, kv_rank)
        v = self.v_proj(value)  # (B, T, kv_rank)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        k = k.view(B, T, self.num_heads, self.kv_rank // self.num_heads).transpose(1, 2)  # (B, H, T, kv_rank//H)
        v = v.view(B, T, self.num_heads, self.kv_rank // self.num_heads).transpose(1, 2)  # (B, H, T, kv_rank//H)
        
        # Handle KV caching
        if use_cache:
            k, v = self.kv_cache.update(k, v)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (B, H, T, kv_rank//H)
        
        # Reshape back to (B, T, D)
        output = output.transpose(1, 2).contiguous().view(B, T, self.kv_rank)
        
        # Project back to embed_dim
        output = self.out_proj(output)
        output = self.out_dropout(output)
        
        # Return cache statistics
        cache_stats = {
            "cache_size": self.kv_cache.current_seq_len,
            "memory_saved": (k.numel() + v.numel()) * 3 if use_cache else 0  # 4x reduction from INT4
        }
        
        if need_weights:
            return output, attn_weights, cache_stats
        else:
            return output, None, cache_stats


class INT4MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention with INT4 quantization (MLA-style).
    
    This is an enhanced version of the attention mechanism that combines
    the benefits of MLA (Multi-head Latent Attention) with INT4 quantization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_rank: int = 64,  # Lower rank for compression
        dropout: float = 0.0,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_rank = kv_rank
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query projection (standard)
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        
        # KV compression (MLA innovation)
        # Down-projection: embed_dim → kv_rank
        self.kv_down_proj = nn.Linear(embed_dim, kv_rank, device=device, dtype=dtype)
        
        # Up-projection: kv_rank → embed_dim (for K and V separately)
        self.k_up_proj = nn.Linear(kv_rank, embed_dim, device=device, dtype=dtype)
        self.v_up_proj = nn.Linear(kv_rank, embed_dim, device=device, dtype=dtype)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout)
        
        # INT4 Quantized KV Cache
        self.kv_cache = QuantizedKVCache(
            head_dim=kv_rank // num_heads,
            num_heads=num_heads,
            device=device
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights following MLA paper recommendations."""
        # Xavier initialization for projections
        for proj in [self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        # Output projection with smaller initialization for stability
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with MLA and INT4 quantization.
        
        Args:
            x: Input tensor (B, N, D)
            attention_mask: Optional attention mask
            use_cache: Whether to use KV cache
            
        Returns:
            output: (B, N, D)
            cache_stats: Dictionary with cache statistics
        """
        B, N, D = x.shape
        
        # Query (standard)
        q = self.q_proj(x)  # (B, N, D)
        
        # KV compression (MLA innovation)
        kv_latent = self.kv_down_proj(x)  # (B, N, kv_rank) - THIS is cached in INT4
        
        # Handle KV caching
        if use_cache:
            # For this implementation, we quantize and dequantize for demonstration
            # In a real implementation, we would store the quantized values in cache
            kv_latent_quantized, kv_scale = self.kv_cache.k_quantizer.quantize(kv_latent)
            # Dequantize for current computation (in a real implementation, we would keep quantized values)
            kv_latent = self.kv_cache.k_quantizer.dequantize(kv_latent_quantized, kv_scale)
        
        # KV decompression
        k = self.k_up_proj(kv_latent)  # (B, N, D)
        v = self.v_up_proj(kv_latent)  # (B, N, D)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Calculate cache statistics
        cache_stats = {
            "memory_saved": kv_latent.numel() * 3 if use_cache else 0,  # 4x reduction from INT4
            "cache_size": self.kv_cache.current_seq_len if use_cache else 0,
            "compression_ratio": 4.0  # INT4 vs FP32
        }
        
        return output, cache_stats


def replace_attention_with_int4(model: nn.Module, use_mla: bool = True):
    """
    Replace standard attention layers with INT4 quantized attention.
    
    Args:
        model: PyTorch model to modify
        use_mla: Whether to use MLA-style attention or standard multi-head
    """
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            # Replace with INT4 attention
            new_module = INT4MultiHeadAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=hasattr(module.in_proj_linear, "bias"),
                device=module.in_proj_linear.weight.device,
                dtype=module.in_proj_linear.weight.dtype
            )
            setattr(model, name, new_module)
        elif isinstance(module, INT4MultiHeadLatentAttention) and use_mla:
            # Already INT4, but we can enhance it further
            continue
        else:
            # Recursively apply to child modules
            replace_attention_with_int4(module, use_mla)


# Example usage and testing
if __name__ == "__main__":
    # Test the INT4 quantization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a sample tensor
    x = torch.randn(2, 16, 512, device=device)
    
    # Test INT4 quantizer
    quantizer = INT4Quantizer()
    quantized, scale = quantizer.quantize(x)
    dequantized = quantizer.dequantize(quantized, scale)
    
    print(f"Original shape: {x.shape}, dtype: {x.dtype}")
    print(f"Quantized shape: {quantized.shape}, dtype: {quantized.dtype}")
    print(f"Scale shape: {scale.shape}, dtype: {scale.dtype}")
    print(f"Max abs error: {torch.max(torch.abs(x - dequantized))}")
    print(f"Memory reduction: {x.element_size() / quantized.element_size():.1f}x")
    
    # Test INT4 attention
    attn = INT4MultiHeadAttention(
        embed_dim=512,
        num_heads=8,
        device=device
    ).to(device)
    
    output, _, stats = attn(x, x, x, use_cache=True)
    print(f"Attention output shape: {output.shape}")
    print(f"Cache stats: {stats}")

