from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExpertNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MoERouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_output_dim: int = 64,
        hidden_dim: int = 512,
        top_k: int = 2,
        capacity_factor: float = 1.25
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )

        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=input_dim,
                output_dim=expert_output_dim,
                hidden_dim=hidden_dim
            )
            for _ in range(num_experts)
        ])

        self.output_proj = nn.Linear(expert_output_dim, expert_output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = x.size(0)

        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros(batch_size, self.experts[0].network[-1].out_features,
                            device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).float()
            expert_weights = (top_k_weights * expert_mask).sum(dim=-1, keepdim=True)
            expert_out = expert(x)
            weighted_expert_out = expert_out * expert_weights
            output = output + weighted_expert_out

        output = self.output_proj(output)

        routing_info = {
            "gate_weights": gate_weights,
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
            "expert_usage": gate_weights.mean(dim=0)
        }

        return output, routing_info


class GameStateMoERouter(nn.Module):
    def __init__(
        self,
        vision_features_dim: int = 1024,
        state_embedding_dim: int = 256,
        num_experts: int = 6,
        hidden_dim: int = 512,
        top_k: int = 2
    ):
        super().__init__()

        self.vision_features_dim = vision_features_dim
        self.state_embedding_dim = state_embedding_dim

        self.vision_processor = nn.Sequential(
            nn.Linear(vision_features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.memory_processor = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.moe_router = MoERouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            expert_output_dim=state_embedding_dim,
            hidden_dim=hidden_dim,
            top_k=top_k
        )

        self.state_classifier = nn.Sequential(
            nn.Linear(state_embedding_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 5)
        )

        self.state_extractors = nn.ModuleDict({
            "exploring": nn.Linear(state_embedding_dim, state_embedding_dim),
            "battle": nn.Linear(state_embedding_dim, state_embedding_dim),
            "menu": nn.Linear(state_embedding_dim, state_embedding_dim),
            "dialogue": nn.Linear(state_embedding_dim, state_embedding_dim),
            "other": nn.Linear(state_embedding_dim, state_embedding_dim)
        })

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        vision_features: torch.Tensor,
        memory_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = vision_features.size(0)

        vision_processed = self.vision_processor(vision_features)

        if memory_features is not None:
            memory_processed = self.memory_processor(memory_features)
        else:
            memory_processed = torch.zeros(batch_size, vision_processed.size(-1),
                                         device=vision_features.device,
                                         dtype=vision_features.dtype)

        combined_features = torch.cat([vision_processed, memory_processed], dim=-1)
        combined_features = self.feature_combiner(combined_features)

        state_embedding, routing_info = self.moe_router(combined_features)

        state_logits = self.state_classifier(state_embedding)
        state_probs = F.softmax(state_logits, dim=-1)

        state_pred = torch.argmax(state_probs, dim=-1)

        state_names = ["exploring", "battle", "menu", "dialogue", "other"]
        state_features = {}
        for i, state_name in enumerate(state_names):
            mask = (state_pred == i).float().unsqueeze(-1)
            extractor = self.state_extractors[state_name]
            state_features[state_name] = extractor(state_embedding) * mask

        result = {
            "state_embedding": state_embedding,
            "state_logits": state_logits,
            "state_probs": state_probs,
            "predicted_state": state_pred,
            "state_features": state_features,
            "routing_info": routing_info
        }

        return result

    def get_state_name(self, state_idx: int) -> str:
        state_names = ["exploring", "battle", "menu", "dialogue", "other"]
        return state_names[state_idx] if 0 <= state_idx < len(state_names) else "unknown"

    def get_state_confidence(self, state_probs: torch.Tensor) -> torch.Tensor:
        return torch.max(state_probs, dim=-1)[0]


class ExpertActivationTracker:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.activation_counts = torch.zeros(num_experts)
        self.total_calls = 0

    def update(self, routing_info: Dict[str, torch.Tensor]):
        expert_usage = routing_info["expert_usage"]
        self.activation_counts += expert_usage.cpu()
        self.total_calls += 1

    def get_activation_stats(self) -> Dict[str, float]:
        if self.total_calls == 0:
            return {}

        avg_usage = self.activation_counts / self.total_calls
        return {
            f"expert_{i}_usage": avg_usage[i].item()
            for i in range(self.num_experts)
        }

    def reset(self):
        self.activation_counts.zero_()
        self.total_calls = 0


class MoEVisionBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        embed_dim: int = 256,
        num_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self._cnn_out_size = 128 * 14 * 16

        self.cnn_proj = nn.Sequential(
            nn.Linear(self._cnn_out_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        self.moe = MoELayer(
            input_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            output_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k
        )

        self.post_moe_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_experts = num_experts

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = x.shape[0]

        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))

        h = h.view(B, -1)
        h = self.cnn_proj(h)

        h, aux_losses = self.moe(h, training)
        h = self.post_moe_norm(h)

        return h, aux_losses


class MoELayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k
        )

        self.experts = nn.ModuleList([
            ExpertFFN(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])

        self.shared_expert = ExpertFFN(input_dim, hidden_dim // 2, output_dim, dropout)
        self.use_shared_expert = True

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, D = x.shape

        weights, indices, aux_losses = self.router(x, training)

        output = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            expert_weight = weights[:, k:k+1]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask] * expert_output

        if self.use_shared_expert:
            shared_out = self.shared_expert(x)
            output = output + 0.5 * shared_out

        return output, aux_losses


class TopKRouter(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 0.1,
        capacity_factor: float = 1.25
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(input_dim, num_experts, bias=False)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))

        nn.init.normal_(self.router.weight, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, D = x.shape

        router_logits = self.router(x) + self.expert_bias

        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        aux_losses = self._compute_aux_losses(router_logits, router_probs)

        return top_k_weights, top_k_indices, aux_losses

    def _compute_aux_losses(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        tokens_per_expert = router_probs.mean(dim=0)
        uniform = torch.ones_like(tokens_per_expert) / self.num_experts
        load_balance_loss = ((tokens_per_expert - uniform) ** 2).sum() * self.num_experts
        router_z_loss = (router_logits ** 2).mean()

        return {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss * 0.001
        }


class ExpertFFN(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.0
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEPPO(nn.Module):
    def __init__(
        self,
        n_actions: int = 9,
        input_channels: int = 4,
        embed_dim: int = 256,
        num_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()

        self.backbone = MoEVisionBackbone(
            input_channels=input_channels,
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k
        )

        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, n_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.n_actions = n_actions
        self.num_experts = num_experts

        for head in [self.policy_head, self.value_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        features, aux_losses = self.backbone(x, training)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value, aux_losses

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        action_mask: Optional[List[float]] = None
    ) -> Tuple[int, float, float]:
        from torch.distributions import Categorical

        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)

            policy_logits, value, _ = self.forward(x, training=False)

            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, device=x.device)
                policy_logits = policy_logits + torch.log(mask_tensor + 1e-8)

            dist = Categorical(logits=policy_logits)

            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()


def create_default_game_state_moe(
    vision_features_dim: int = 1024,
    state_embedding_dim: int = 256
) -> GameStateMoERouter:
    return GameStateMoERouter(
        vision_features_dim=vision_features_dim,
        state_embedding_dim=state_embedding_dim,
        num_experts=6,
        hidden_dim=512,
        top_k=2
    )


def count_moe_parameters(model: MoEPPO) -> Dict[str, int]:
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    policy_params = sum(p.numel() for p in model.policy_head.parameters())
    value_params = sum(p.numel() for p in model.value_head.parameters())

    router_params = sum(p.numel() for p in model.backbone.moe.router.parameters())
    experts_params = sum(
        sum(p.numel() for p in expert.parameters())
        for expert in model.backbone.moe.experts
    )
    shared_expert_params = sum(
        p.numel() for p in model.backbone.moe.shared_expert.parameters()
    )

    return {
        "backbone_cnn": backbone_params - router_params - experts_params - shared_expert_params,
        "router": router_params,
        "experts": experts_params,
        "shared_expert": shared_expert_params,
        "policy_head": policy_params,
        "value_head": value_params,
        "total": sum(p.numel() for p in model.parameters())
    }
