"""
Vision Encoder per Pokemon AI Agent.

Architettura ispirata a DeepSeek-VL2 (arXiv:2412.10302), adattata per
reinforcement learning su Game Boy.

COMPONENTI:
1. PixelShuffleAdaptor: Compressione 2×2 dei token visivi (riduce 4x i token)
2. MultiHeadLatentAttention (MLA): Comprime KV cache in vettori latenti
3. VisionPPO: Rete PPO con attention-based vision encoder

ADATTAMENTI PER POKEMON:
- Input 144×160 (Game Boy) invece di 384×384 (SigLIP)
- Backbone CNN invece di ViT (più efficiente per immagini piccole)
- MLA con rank ridotto (64-128) per efficienza su hardware consumer

RIFERIMENTI:
- DeepSeek-VL2 paper (arXiv:2412.10302)
- Section 2.2: Vision-Language Adaptor (pixel shuffle 2×2)
- Section 2.3: Multi-head Latent Attention (KV compression)
"""
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PixelShuffleAdaptor(nn.Module):
    """
    Vision-Language Adaptor con Pixel Shuffle dal paper DeepSeek-VL2.

    FUNZIONAMENTO (Section 2.2 del paper):
    1. Riceve feature map da CNN backbone (H×W×C)
    2. Applica pixel shuffle 2×2: raggruppa 4 pixel adiacenti in 1
    3. Proietta nel nuovo spazio embedding con MLP a 2 layer

    EFFETTO:
    - Riduce token visivi da H×W a (H/2)×(W/2) = 4× meno token
    - Mantiene informazione spaziale (non è un semplice pooling)
    - Comprime senza perdere dettagli locali

    ESEMPIO (dal paper):
    - Input: 27×27 = 729 token per tile
    - Output: 14×14 = 196 token (riduzione 3.7×)
    - Qui: 15×17 → 7×8 = 56 token (era 255)

    Args:
        input_channels: Canali in input dalla CNN
        hidden_dim: Dimensione layer nascosto MLP
        output_dim: Dimensione embedding output
        shuffle_factor: Fattore di shuffle (default 2 = 2×2)
    """

    def __init__(
        self,
        input_channels: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 256,
        shuffle_factor: int = 2
    ):
        super().__init__()
        self.shuffle_factor = shuffle_factor

        # Dopo pixel shuffle: canali aumentano di shuffle_factor^2
        shuffled_channels = input_channels * (shuffle_factor ** 2)

        # MLP a 2 layer come nel paper
        self.mlp = nn.Sequential(
            nn.Linear(shuffled_channels, hidden_dim),
            nn.GELU(),  # GELU come nel paper (non ReLU)
            nn.Linear(hidden_dim, output_dim)
        )

        # Layer norm per stabilità
        self.norm = nn.LayerNorm(output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inizializzazione Xavier per MLP."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def pixel_shuffle_down(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pixel shuffle inverso (space-to-depth).

        Raggruppa pixel 2×2 adiacenti in un singolo token con 4× canali.
        Opposto di nn.PixelShuffle che fa depth-to-space.

        Args:
            x: Tensor (B, C, H, W)

        Returns:
            Tensor (B, C*4, H/2, W/2)
        """
        B, C, H, W = x.shape
        r = self.shuffle_factor

        # Assicura che H e W siano divisibili per r
        H_pad = (r - H % r) % r
        W_pad = (r - W % r) % r
        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad))
            H, W = H + H_pad, W + W_pad

        # Reshape: (B, C, H, W) → (B, C, H/r, r, W/r, r)
        x = x.view(B, C, H // r, r, W // r, r)

        # Permute: (B, C, H/r, r, W/r, r) → (B, C, r, r, H/r, W/r)
        x = x.permute(0, 1, 3, 5, 2, 4)

        # Reshape: (B, C*r*r, H/r, W/r)
        x = x.contiguous().view(B, C * r * r, H // r, W // r)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del PixelShuffleAdaptor.

        Args:
            x: Feature map CNN (B, C, H, W)

        Returns:
            Token compressi (B, N, D) dove N = (H/2)*(W/2), D = output_dim
        """
        # 1. Pixel shuffle inverso: (B, C, H, W) → (B, C*4, H/2, W/2)
        x = self.pixel_shuffle_down(x)

        # 2. Flatten spaziale: (B, C*4, H', W') → (B, H'*W', C*4)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', C)
        x = x.view(B, H * W, C)  # (B, N, C)

        # 3. MLP projection: (B, N, C) → (B, N, output_dim)
        x = self.mlp(x)

        # 4. Layer norm
        x = self.norm(x)

        return x


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) dal paper DeepSeek-VL2 with INT4 quantization.

    IDEA CHIAVE (Section 2.3 del paper):
    Invece di cachare K e V di dimensione (heads × head_dim),
    comprime in vettori latenti di dimensione rank << heads × head_dim.

    VANTAGGI:
    1. KV cache ridotta di ~10× (rank=64 vs 512 standard)
    2. Inference più veloce (meno memoria da leggere)
    3. Throughput maggiore (più batch in memoria)
    4. INT4 quantization riduce ulteriormente la memoria del 4×

    FORMULA (dal paper):
    - Latent: c = W_down @ x  (compressione)
    - K = W_k_up @ c          (decompressione per K)
    - V = W_v_up @ c          (decompressione per V)

    CONFIGURAZIONE PAPER (Table 2):
    - DeepSeek-VL2-Small: rank=512, heads=16
    - Qui usiamo: rank=64-128, heads=4-8 (per efficienza)

    Args:
        embed_dim: Dimensione embedding input
        num_heads: Numero di attention heads
        kv_rank: Rank della compressione KV (più basso = più efficiente)
        dropout: Dropout rate
        use_int4: Whether to use INT4 quantization for KV cache
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        kv_rank: int = 64,
        dropout: float = 0.0,
        use_int4: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_rank = kv_rank
        self.scale = self.head_dim ** -0.5
        self.use_int4 = use_int4

        assert embed_dim % num_heads == 0, "embed_dim deve essere divisibile per num_heads"

        # Query projection (standard)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # KV compression (innovazione MLA)
        # Down-projection: embed_dim → kv_rank
        self.kv_down_proj = nn.Linear(embed_dim, kv_rank)

        # Up-projection: kv_rank → embed_dim (per K e V separatamente)
        self.k_up_proj = nn.Linear(kv_rank, embed_dim)
        self.v_up_proj = nn.Linear(kv_rank, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize INT4 quantization if enabled
        if use_int4:
            from .int4_quantization import INT4Quantizer
            self.kv_quantizer = INT4Quantizer(symmetric=True, per_channel=True)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inizializzazione come nel paper (Xavier + small output)."""
        for module in [self.q_proj, self.kv_down_proj, self.k_up_proj,
                       self.v_up_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Output projection con scala ridotta per stabilità
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass MLA with optional INT4 quantization.

        Args:
            x: Input tensor (B, N, D)
            attention_mask: Maschera opzionale (B, N) o (B, 1, N, N)
            return_kv_cache: Se True, ritorna anche KV latent per caching
            use_cache: Se True, usa la cache INT4 quantizzata

        Returns:
            output: (B, N, D)
            kv_latent: (B, N, rank) se return_kv_cache=True, altrimenti None
        """
        B, N, D = x.shape

        # Query (standard)
        q = self.q_proj(x)  # (B, N, D)

        # KV compression (innovazione MLA)
        kv_latent = self.kv_down_proj(x)  # (B, N, rank) - QUESTO è cachato

        # Apply INT4 quantization if enabled
        if self.use_int4 and use_cache:
            # Usa il quantizer interno se disponibile
            if hasattr(self, 'kv_quantizer'):
                kv_latent_quantized, scale = self.kv_quantizer.quantize(kv_latent)
                # Per il calcolo, dequantizza
                kv_latent = self.kv_quantizer.dequantize(kv_latent_quantized, scale)
            else:
                # Fallback: crea temporaneamente un quantizer
                from .int4_quantization import INT4Quantizer
                quantizer = INT4Quantizer(symmetric=True, per_channel=True)
                kv_latent_quantized, scale = quantizer.quantize(kv_latent)
                # Per il calcolo, dequantizza
                kv_latent = quantizer.dequantize(kv_latent_quantized, scale)

        # KV decompression
        k = self.k_up_proj(kv_latent)  # (B, N, D)
        v = self.v_up_proj(kv_latent)  # (B, N, D)

        # Reshape per multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, d)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)

        # Output projection
        output = self.out_proj(attn_output)

        if return_kv_cache:
            return output, kv_latent
        return output, None


class VisionBackbone(nn.Module):
    """
    Backbone CNN + MLA per vision encoder.

    ARCHITETTURA:
    1. CNN per estrazione features
    2. PixelShuffleAdaptor per compressione token
    3. MLA per attenzione efficiente

    VANTAGGI:
    - Attenzione: comprende relazioni spaziali
    - Compressione token: più efficiente
    - GELU + BatchNorm: training stabile
    """

    def __init__(
        self,
        input_channels: int = 4,
        cnn_channels: int = 64,
        embed_dim: int = 256,
        num_heads: int = 4,
        kv_rank: int = 64,
        num_mla_layers: int = 2,
        use_int4: bool = True
    ):
        super().__init__()

        # CNN backbone (simile a prima ma con GELU)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, cnn_channels, kernel_size=3, stride=1)

        # Batch norm per stabilità
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(cnn_channels)

        # PixelShuffleAdaptor (innovazione DeepSeek-VL2)
        self.adaptor = PixelShuffleAdaptor(
            input_channels=cnn_channels,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            shuffle_factor=2
        )

        # Stack di MLA layers (innovazione DeepSeek-VL2)
        self.mla_layers = nn.ModuleList([
            MultiHeadLatentAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kv_rank=kv_rank,
                use_int4=use_int4
            )
            for _ in range(num_mla_layers)
        ])

        # Feed-forward dopo MLA (come in transformer)
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_mla_layers)
        ])

        # Layer norms
        self.mla_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_mla_layers)
        ])
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_mla_layers)
        ])

        self.embed_dim = embed_dim

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inizializzazione ortogonale per CNN, Xavier per il resto."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del backbone.

        Args:
            x: Input frame stack (B, C, H, W) tipicamente (B, 4, 144, 160)

        Returns:
            Features (B, embed_dim) pronte per policy/value heads
        """
        # CNN con GELU
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))

        # PixelShuffle: (B, 64, H', W') → (B, N, embed_dim)
        x = self.adaptor(x)

        # MLA layers con residual connections
        for mla, ff, mla_norm, ff_norm in zip(
            self.mla_layers, self.ff_layers,
            self.mla_norms, self.ff_norms
        ):
            # MLA con residual
            residual = x
            x = mla_norm(x)
            x, _ = mla(x)  # Usa il metodo originale senza cache per ora
            x = residual + x

            # FF con residual
            residual = x
            x = ff_norm(x)
            x = residual + ff(x)

        # Global average pooling sui token: (B, N, D) → (B, D)
        x = x.mean(dim=1)

        return x


class VisionPPO(nn.Module):
    """
    Rete PPO con vision encoder attention-based.

    Combina:
    - VisionBackbone (CNN + PixelShuffle + MLA)
    - Policy head (Actor)
    - Value head (Critic)

    VANTAGGI:
    1. Attenzione: Comprende relazioni spaziali tra elementi dello schermo
    2. Compressione token: Meno parametri, inference più veloce
    3. KV cache compresso: Più efficiente in memoria
    4. GELU + LayerNorm: Training più stabile

    Args:
        n_actions: Numero di azioni possibili (9 per Game Boy)
        input_channels: Canali input (4 per frame stack)
        embed_dim: Dimensione embedding (256 default)
        num_heads: Attention heads (4 default)
        kv_rank: Rank KV compression (64 default)
        num_mla_layers: Numero layer MLA (2 default)
    """

    def __init__(
        self,
        n_actions: int,
        input_channels: int = 4,
        embed_dim: int = 256,
        num_heads: int = 4,
        kv_rank: int = 64,
        num_mla_layers: int = 2,
        use_int4: bool = True
    ):
        super().__init__()

        self.backbone = VisionBackbone(
            input_channels=input_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            kv_rank=kv_rank,
            num_mla_layers=num_mla_layers,
            use_int4=use_int4
        )

        # Policy head (Actor)
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, n_actions)
        )

        # Value head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self._initialize_heads()

    def _initialize_heads(self) -> None:
        """Inizializzazione heads con scala ridotta."""
        for head in [self.policy_head, self.value_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input (B, C, H, W)

        Returns:
            policy_logits: (B, n_actions)
            value: (B, 1)
        """
        features = self.backbone(x)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value


# ============== VARIANTI SPECIALIZZATE ==============

class ExplorationPPO(VisionPPO):
    """
    Variante leggera per esplorazione overworld.

    Configurazione ridotta per decisioni rapide:
    - embed_dim=192
    - num_heads=3
    - kv_rank=48
    - num_mla_layers=1
    """

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(
            n_actions=n_actions,
            input_channels=input_channels,
            embed_dim=192,
            num_heads=3,
            kv_rank=48,
            num_mla_layers=1,
            use_int4=True
        )


class BattlePPO(VisionPPO):
    """
    Variante più profonda per battaglie.

    Configurazione aumentata per ragionamento strategico:
    - embed_dim=320
    - num_heads=5
    - kv_rank=80
    - num_mla_layers=3
    """

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(
            n_actions=n_actions,
            input_channels=input_channels,
            embed_dim=320,
            num_heads=5,
            kv_rank=80,
            num_mla_layers=3,
            use_int4=True
        )


class MenuPPO(VisionPPO):
    """
    Variante minimale per navigazione menu.

    Configurazione minima per UI semplici:
    - embed_dim=128
    - num_heads=2
    - kv_rank=32
    - num_mla_layers=1
    """

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(
            n_actions=n_actions,
            input_channels=input_channels,
            embed_dim=128,
            num_heads=2,
            kv_rank=32,
            num_mla_layers=1,
            use_int4=True
        )


# Alias per retrocompatibilità
DeepSeekVL2ExplorationPPO = ExplorationPPO
DeepSeekVL2BattlePPO = BattlePPO
DeepSeekVL2MenuPPO = MenuPPO
DeepSeekVL2PPO = VisionPPO


# ============== UTILITY FUNCTIONS ==============

def count_parameters(model: nn.Module) -> dict:
    """
    Conta parametri del modello.

    Returns:
        Dict con total, trainable, e per-component
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    components = {}
    if hasattr(model, 'backbone'):
        components['backbone'] = sum(
            p.numel() for p in model.backbone.parameters()
        )
        if hasattr(model.backbone, 'adaptor'):
            components['adaptor'] = sum(
                p.numel() for p in model.backbone.adaptor.parameters()
            )
        if hasattr(model.backbone, 'mla_layers'):
            components['mla'] = sum(
                p.numel() for layer in model.backbone.mla_layers
                for p in layer.parameters()
            )
    if hasattr(model, 'policy_head'):
        components['policy_head'] = sum(
            p.numel() for p in model.policy_head.parameters()
        )
    if hasattr(model, 'value_head'):
        components['value_head'] = sum(
            p.numel() for p in model.value_head.parameters()
        )

    return {
        'total': total,
        'trainable': trainable,
        'components': components
    }


