"""Mixture of Experts (MoE) Router for Pokemon AI Agent.

This module implements an end-to-end state detection system that replaces
the traditional CV-based state detector with a learned neural network approach.

The MoE Router uses multiple expert networks specialized for different game states
and a gating network to route inputs to the appropriate experts.

Based on the paper: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
"""
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ExpertNetwork(nn.Module):
    """
    Individual expert network specialized for a particular game state or task.
    """
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
    """
    Mixture of Experts Router that learns to route inputs to appropriate experts.
    
    The router uses a gating network to determine which experts to activate
    for each input, enabling efficient computation through sparsity.
    """
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
        
        # Gating network to determine which experts to use
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=input_dim,
                output_dim=expert_output_dim,
                hidden_dim=hidden_dim
            )
            for _ in range(num_experts)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(expert_output_dim, expert_output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the MoE router.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, routing_info)
            - output: Final output tensor
            - routing_info: Dictionary with routing statistics
        """
        batch_size = x.size(0)
        
        # Compute gate logits
        gate_logits = self.gate(x)  # (batch_size, num_experts)
        
        # Apply softmax to get routing weights
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch_size, num_experts)
        
        # Select top-k experts for each input
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        # Normalize the top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros(batch_size, self.experts[0].network[-1].out_features, 
                            device=x.device, dtype=x.dtype)
        
        # Process each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Check which inputs should use this expert
            expert_mask = (top_k_indices == i).float()  # (batch_size, top_k)
            expert_weights = (top_k_weights * expert_mask).sum(dim=-1, keepdim=True)  # (batch_size, 1)
            
            # Compute expert output
            expert_out = expert(x)  # (batch_size, expert_output_dim)
            
            # Weight the expert output
            weighted_expert_out = expert_out * expert_weights  # (batch_size, expert_output_dim)
            output = output + weighted_expert_out
            
            expert_outputs.append(expert_out)
        
        # Apply final projection
        output = self.output_proj(output)
        
        # Return routing information
        routing_info = {
            gate_weights: gate_weights,
            top_k_indices: top_k_indices,
            top_k_weights: top_k_weights,
            expert_usage: gate_weights.mean(dim=0)  # Average usage per expert
        }
        
        return output, routing_info


class GameStateMoERouter(nn.Module):
    """
    Specialized MoE Router for Pokemon game state detection.
    
    This router has experts specialized for different game states:
    - Exploration expert (overworld navigation)
    - Battle expert (combat scenarios)
    - Menu expert (UI navigation)
    - Dialogue expert (text interactions)
    - Specialized experts for other game states
    """
    def __init__(
        self,
        vision_features_dim: int = 1024,  # Output from vision encoder
        state_embedding_dim: int = 256,
        num_experts: int = 6,
        hidden_dim: int = 512,
        top_k: int = 2
    ):
        super().__init__()
        
        self.vision_features_dim = vision_features_dim
        self.state_embedding_dim = state_embedding_dim
        
        # Vision feature processing
        self.vision_processor = nn.Sequential(
            nn.Linear(vision_features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Memory feature processing (from game memory)
        self.memory_processor = nn.Sequential(
            nn.Linear(128, hidden_dim),  # Assuming 128-dim memory features
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Combined feature processing
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # MoE Router
        self.moe_router = MoERouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            expert_output_dim=state_embedding_dim,
            hidden_dim=hidden_dim,
            top_k=top_k
        )
        
        # State classification head
        self.state_classifier = nn.Sequential(
            nn.Linear(state_embedding_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 5)  # 5 game states: exploring, battle, menu, dialogue, other
        )
        
        # State-specific feature extractors
        self.state_extractors = nn.ModuleDict({
            exploring: nn.Linear(state_embedding_dim, state_embedding_dim),
            battle: nn.Linear(state_embedding_dim, state_embedding_dim),
            menu: nn.Linear(state_embedding_dim, state_embedding_dim),
            dialogue: nn.Linear(state_embedding_dim, state_embedding_dim),
            other: nn.Linear(state_embedding_dim, state_embedding_dim)
        })
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability."""
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
        """
        Forward pass for game state detection.
        
        Args:
            vision_features: Features from vision encoder (B, vision_features_dim)
            memory_features: Features from game memory (B, 128) - optional
            
        Returns:
            Dictionary with state detection results
        """
        batch_size = vision_features.size(0)
        
        # Process vision features
        vision_processed = self.vision_processor(vision_features)
        
        # Process memory features if provided
        if memory_features is not None:
            memory_processed = self.memory_processor(memory_features)
        else:
            # Create dummy memory features if not provided
            memory_processed = torch.zeros(batch_size, vision_processed.size(-1), 
                                         device=vision_features.device, 
                                         dtype=vision_features.dtype)
        
        # Combine vision and memory features
        combined_features = torch.cat([vision_processed, memory_processed], dim=-1)
        combined_features = self.feature_combiner(combined_features)
        
        # Pass through MoE router
        state_embedding, routing_info = self.moe_router(combined_features)
        
        # Classify game state
        state_logits = self.state_classifier(state_embedding)
        state_probs = F.softmax(state_logits, dim=-1)
        
        # Get predicted state
        state_pred = torch.argmax(state_probs, dim=-1)
        
        # Extract state-specific features
        state_names = [exploring, battle, menu, dialogue, other]
        state_features = {}
        for i, state_name in enumerate(state_names):
            mask = (state_pred == i).float().unsqueeze(-1)
            extractor = self.state_extractors[state_name]
            state_features[state_name] = extractor(state_embedding) * mask
        
        # Combine all results
        result = {
            state_embedding: state_embedding,
            state_logits: state_logits,
            state_probs: state_probs,
            predicted_state: state_pred,
            state_features: state_features,
            routing_info: routing_info
        }
        
        return result

    def get_state_name(self, state_idx: int) -> str:
        """Convert state index to name."""
        state_names = [exploring, battle, menu, dialogue, other]
        return state_names[state_idx] if 0 <= state_idx < len(state_names) else unknown

    def get_state_confidence(self, state_probs: torch.Tensor) -> torch.Tensor:
        """Get confidence of state prediction."""
        return torch.max(state_probs, dim=-1)[0]


class ExpertActivationTracker:
    """
    Utility class to track expert activation patterns for debugging and optimization.
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.activation_counts = torch.zeros(num_experts)
        self.total_calls = 0

    def update(self, routing_info: Dict[str, torch.Tensor]):
        """Update activation counts based on routing info."""
        expert_usage = routing_info[expert_usage]
        self.activation_counts += expert_usage.cpu()
        self.total_calls += 1

    def get_activation_stats(self) -> Dict[str, float]:
        """Get expert activation statistics."""
        if self.total_calls == 0:
            return {}
        
        avg_usage = self.activation_counts / self.total_calls
        return {
            fexpert_{i}_usage: avg_usage[i].item()
            for i in range(self.num_experts)
        }

    def reset(self):
        """Reset activation counts."""
        self.activation_counts.zero_()
        self.total_calls = 0


def create_default_game_state_moe(
    vision_features_dim: int = 1024,
    state_embedding_dim: int = 256
) -> GameStateMoERouter:
    """
    Create a default game state MoE router with recommended parameters.
    """
    return GameStateMoERouter(
        vision_features_dim=vision_features_dim,
        state_embedding_dim=state_embedding_dim,
        num_experts=6,
        hidden_dim=512,
        top_k=2
    )
