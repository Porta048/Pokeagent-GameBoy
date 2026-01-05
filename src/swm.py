"""
Simple World Model for Pokemon Agent - Weekend Implementation
Based on DreamerV3 but simplified for quick integration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SimpleWorldModel(nn.Module):
    """
    Simplified world model for quick weekend implementation.
    Predicts: next latent state, reward, done.
    """
    def __init__(
        self,
        input_channels: int = 4,
        latent_dim: int = 192,  # Match ExplorationPPO
        action_dim: int = 9,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Vision encoder (simplified)
        # Input: 144x160
        # Conv1 (k=8, s=4): (144-8)/4+1=35, (160-8)/4+1=39 -> 35x39
        # Conv2 (k=4, s=2): (35-4)/2+1=16, (39-4)/2+1=18 -> 16x18
        # Conv3 (k=3, s=1): (16-3)/1+1=14, (18-3)/1+1=16 -> 14x16
        # Flatten: 64 * 14 * 16 = 14336

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 14 * 16, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.vision_encoder = nn.Sequential(
            self.conv_layers,
            self.flatten,
            self.fc
        )

        # Action embedding
        self.action_embed = nn.Embedding(action_dim, 32)

        # Dynamics model: predicts next latent from current latent + action
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Reward predictor with symlog transformation to handle high reward values
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Done predictor
        self.done_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Reward normalization statistics
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_std', torch.ones(1))
        self.register_buffer('reward_count', torch.zeros(1))
        self.register_buffer('reward_sum', torch.zeros(1))
        self.register_buffer('reward_sum_sq', torch.zeros(1))

        # Small epsilon to prevent division by zero
        self.eps = 1e-8

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode game frame to latent state."""
        return self.vision_encoder(obs)

    def predict_next(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state given current latent and action."""
        # Embed action
        action_emb = self.action_embed(action)

        # Concatenate and predict next state
        x = torch.cat([latent, action_emb], dim=-1)
        next_latent = self.dynamics(x)

        return next_latent

    def update_reward_stats(self, rewards: torch.Tensor):
        """Update running statistics for reward normalization."""
        batch_size = rewards.size(0)

        # Update sum and sum of squares
        self.reward_sum += rewards.sum()
        self.reward_sum_sq += (rewards ** 2).sum()
        self.reward_count += batch_size

        # Update mean and std
        self.reward_mean = self.reward_sum / self.reward_count
        variance = (self.reward_sum_sq / self.reward_count) - (self.reward_mean ** 2)
        self.reward_std = torch.sqrt(torch.clamp(variance, min=self.eps))

    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize reward using running statistics."""
        return (reward - self.reward_mean) / (self.reward_std + self.eps)

    def symlog_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply symmetric logarithm transformation to handle high values."""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def inverse_symlog_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse symmetric logarithm transformation."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def decode_reward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        raw_reward = self.reward_head(latent)
        # Apply inverse symlog to get back to original scale
        return self.inverse_symlog_transform(raw_reward)

    def decode_done(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict done probability from latent state."""
        return self.done_head(latent)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: obs + action -> next_latent, predicted reward, predicted done.

        Args:
            obs: Current observation (B, C, H, W)
            action: Action taken (B,)

        Returns:
            next_latent: Predicted next latent state (B, latent_dim)
            pred_reward: Predicted reward (B, 1)
            pred_done: Predicted done logits (B, 1)
        """
        # Encode current observation
        latent = self.encode_observation(obs)

        # Predict next state
        next_latent = self.predict_next(latent, action)

        # Predict reward and done from next state
        pred_reward = self.decode_reward(next_latent)
        pred_done = self.decode_done(next_latent)

        return next_latent, pred_reward, pred_done


def compute_world_model_loss(
    model: SimpleWorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute world model training loss.

    Args:
        model: SimpleWorldModel instance
        states: Current states (B, C, H, W)
        actions: Actions taken (B,)
        next_states: Next states (B, C, H, W)
        rewards: Actual rewards (B,)
        dones: Actual dones (B,)

    Returns:
        Dictionary of losses
    """
    # Update reward statistics before computing loss
    model.update_reward_stats(rewards)

    # Encode current and next states
    current_latent = model.encode_observation(states)
    actual_next_latent = model.encode_observation(next_states)

    # Predict next latent
    predicted_next_latent = model.predict_next(current_latent, actions)

    # Transform rewards for training
    normalized_rewards = model.normalize_reward(rewards)
    symlog_rewards = model.symlog_transform(normalized_rewards)

    # Predict reward (this will be in symlog space)
    pred_reward = model.reward_head(predicted_next_latent)  # Raw output before inverse symlog

    # Predict done
    pred_done = model.decode_done(predicted_next_latent)

    # Latent prediction loss (MSE between predicted and actual next latent)
    latent_loss = F.mse_loss(predicted_next_latent, actual_next_latent.detach())

    # Reward prediction loss (in symlog space)
    reward_loss = F.mse_loss(pred_reward.squeeze(), symlog_rewards)

    # Done prediction loss (binary cross-entropy)
    done_loss = F.binary_cross_entropy_with_logits(pred_done.squeeze(), dones)

    # Total loss (weighted sum)
    total_loss = latent_loss + reward_loss + done_loss

    return {
        'latent_loss': latent_loss,
        'reward_loss': reward_loss,
        'done_loss': done_loss,
        'total_loss': total_loss
    }
