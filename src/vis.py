from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleAdaptor(nn.Module):
    def __init__(self, input_channels: int = 64, hidden_dim: int = 256, output_dim: int = 256, shuffle_factor: int = 2):
        super().__init__()
        self.shuffle_factor = shuffle_factor
        shuffled_channels = input_channels * (shuffle_factor ** 2)
        self.mlp = nn.Sequential(
            nn.Linear(shuffled_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        r = self.shuffle_factor
        H_pad, W_pad = (r - H % r) % r, (r - W % r) % r
        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad))
            H, W = H + H_pad, W + W_pad
        x = x.view(B, C, H // r, r, W // r, r).permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * r * r, H // r, W // r)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, (H // r) * (W // r), C * r * r)
        return self.norm(self.mlp(x))


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, kv_rank: int = 64, dropout: float = 0.0, use_int4: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_int4 = use_int4

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_down_proj = nn.Linear(embed_dim, kv_rank)
        self.k_up_proj = nn.Linear(kv_rank, embed_dim)
        self.v_up_proj = nn.Linear(kv_rank, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if use_int4:
            from .int import INT4Quantizer
            self.kv_quantizer = INT4Quantizer(symmetric=True, per_channel=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        q = self.q_proj(x)
        kv_latent = self.kv_down_proj(x)

        if self.use_int4 and use_cache and hasattr(self, 'kv_quantizer'):
            quantized, scale = self.kv_quantizer.quantize(kv_latent)
            kv_latent = self.kv_quantizer.dequantize(quantized, scale)

        k, v = self.k_up_proj(kv_latent), self.v_up_proj(kv_latent)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out), kv_latent if use_cache else None


class VisionBackbone(nn.Module):
    def __init__(self, input_channels: int = 4, embed_dim: int = 256, num_heads: int = 4, kv_rank: int = 64, num_mla_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, 4, 2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.adaptor = PixelShuffleAdaptor(64, embed_dim, embed_dim, 2)
        self.mla = nn.ModuleList([
            MultiHeadLatentAttention(embed_dim, num_heads, kv_rank, 0.1)
            for _ in range(num_mla_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self._init_conv()

    def _init_conv(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adaptor(self.conv(x))
        for layer in self.mla:
            x = x + layer(x)[0]
        return self.proj(self.norm(x.mean(dim=1)))


class VisionPPONetwork(nn.Module):
    def __init__(self, n_actions: int, input_channels: int = 4, embed_dim: int = 256, num_heads: int = 4, kv_rank: int = 64, num_mla_layers: int = 2):
        super().__init__()
        self.backbone = VisionBackbone(input_channels, embed_dim, num_heads, kv_rank, num_mla_layers)
        self.policy = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, n_actions))
        self.value = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, 1))
        self._init_heads()

    def _init_heads(self):
        for head in [self.policy, self.value]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.policy(features), self.value(features)


class ExplorationNetwork(VisionPPONetwork):
    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(n_actions, input_channels, embed_dim=192, num_heads=3, kv_rank=48, num_mla_layers=1)


class BattleNetwork(VisionPPONetwork):
    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(n_actions, input_channels, embed_dim=320, num_heads=5, kv_rank=80, num_mla_layers=3)


class MenuNetwork(VisionPPONetwork):
    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(n_actions, input_channels, embed_dim=128, num_heads=2, kv_rank=32, num_mla_layers=1)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}
