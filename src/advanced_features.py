"""
Advanced Features Integration Module for Pokemon AI Agent V4.0.

Questo modulo integra le tecnologie avanzate ispirate alla ricerca cinese 2025-2026:

1. World Model (DreamerV3) - Imagination-based learning
2. MoE Router (DeepSeek-V3) - Learned routing invece di CV-based
3. INT4 KV Cache Quantization - 4x memory reduction

UTILIZZO:
    from src.advanced_features import AdvancedPokemonAgent

    agent = AdvancedPokemonAgent(
        n_actions=9,
        device=torch.device("cuda"),
        use_world_model=True,
        use_moe=True,
        use_int4=True
    )

    # Training loop
    action, log_prob, value = agent.get_action(frame_stack)
    agent.store_transition(frame, action, reward, done, next_frame)
    metrics = agent.train_step()

BENCHMARKS (stimati):
    | Feature          | Memory  | Speed  | Sample Eff. |
    |------------------|---------|--------|-------------|
    | Baseline V3.0    | 13 MB   | 1.0x   | 1.0x        |
    | + INT4 KV Cache  | 4 MB    | 1.1x   | 1.0x        |
    | + MoE Router     | 8 MB    | 0.9x   | 1.2x        |
    | + World Model    | 25 MB   | 0.7x   | 3.0x        |
    | Full V4.0        | 30 MB   | 0.8x   | 3.5x        |
"""
from typing import Dict, List, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class AdvancedPokemonAgent(nn.Module):
    """
    Agente Pokemon avanzato con tutte le feature V4.0.

    Combina:
    - World Model per imagination-based learning
    - MoE Router per routing end-to-end
    - INT4 KV Cache per efficienza memoria
    """

    def __init__(
        self,
        n_actions: int = 9,
        input_channels: int = 4,
        embed_dim: int = 256,
        device: torch.device = None,
        use_world_model: bool = True,
        use_moe: bool = True,
        use_int4: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        imagination_horizon: int = 15,
        world_model_lr: float = 1e-4,
        policy_lr: float = 3e-4
    ):
        super().__init__()

        self.device = device or torch.device('cpu')
        self.n_actions = n_actions
        self.use_world_model = use_world_model
        self.use_moe = use_moe
        self.use_int4 = use_int4
        self.imagination_horizon = imagination_horizon

        # Initialize components based on configuration
        self._init_policy_network(
            n_actions, input_channels, embed_dim,
            num_experts, top_k, use_moe, use_int4
        )

        if use_world_model:
            self._init_world_model(input_channels, embed_dim, n_actions)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=policy_lr
        )

        if use_world_model:
            self.world_model_optimizer = torch.optim.AdamW(
                self.world_model.parameters(),
                lr=world_model_lr,
                weight_decay=1e-5
            )

        # Experience buffer for world model training
        self.experience_buffer = {
            'frames': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'next_frames': []
        }
        self.buffer_size = 100000
        self.buffer_idx = 0

        # Training stats
        self.total_steps = 0

        self.to(self.device)
        self._log_architecture()

    def _init_policy_network(
        self,
        n_actions: int,
        input_channels: int,
        embed_dim: int,
        num_experts: int,
        top_k: int,
        use_moe: bool,
        use_int4: bool
    ):
        """Initialize the policy network."""
        if use_moe:
            from .moe_router import MoEPPO
            self.policy_network = MoEPPO(
                n_actions=n_actions,
                input_channels=input_channels,
                embed_dim=embed_dim,
                num_experts=num_experts,
                top_k=top_k
            )
        else:
            from .vision_encoder import VisionPPO
            self.policy_network = VisionPPO(
                n_actions=n_actions,
                input_channels=input_channels,
                embed_dim=embed_dim,
                use_int4=use_int4
            )

    def _init_world_model(
        self,
        input_channels: int,
        embed_dim: int,
        action_dim: int
    ):
        """Initialize the world model."""
        from .world_model import DreamerV3WorldModel
        self.world_model = DreamerV3WorldModel(
            input_channels=input_channels,
            embed_dim=embed_dim,
            action_dim=action_dim
        )

    def _log_architecture(self):
        """Log architecture information."""
        total_params = sum(p.numel() for p in self.parameters())

        logger.info("=" * 60)
        logger.info("POKEMON AI AGENT V4.0 - Advanced Features")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Features enabled:")
        logger.info(f"  - World Model (DreamerV3): {self.use_world_model}")
        logger.info(f"  - MoE Router: {self.use_moe}")
        logger.info(f"  - INT4 KV Cache: {self.use_int4}")

        if self.use_world_model:
            wm_params = sum(p.numel() for p in self.world_model.parameters())
            logger.info(f"  - World Model params: {wm_params:,}")

        policy_params = sum(p.numel() for p in self.policy_network.parameters())
        logger.info(f"  - Policy Network params: {policy_params:,}")
        logger.info("=" * 60)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        action_mask: Optional[List[float]] = None
    ) -> Tuple[int, float, float]:
        """
        Get action from policy network.

        Args:
            state: Frame stack (C, H, W) or (B, C, H, W)
            deterministic: Use argmax instead of sampling
            action_mask: Optional soft action mask

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
        """
        self.policy_network.eval()

        with torch.no_grad():
            if state.dim() == 3:
                state = state.unsqueeze(0)

            state = state.to(self.device)

            # Forward pass (handle MoE and non-MoE)
            if self.use_moe:
                policy_logits, value, _ = self.policy_network(state, training=False)
            else:
                policy_logits, value = self.policy_network(state)

            # Apply action mask
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, device=self.device)
                policy_logits = policy_logits + torch.log(mask_tensor + 1e-8)

            dist = Categorical(logits=policy_logits)

            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def store_transition(
        self,
        frame: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        next_frame: torch.Tensor
    ):
        """Store transition for world model training."""
        if not self.use_world_model:
            return

        if len(self.experience_buffer['frames']) < self.buffer_size:
            self.experience_buffer['frames'].append(frame.cpu())
            self.experience_buffer['actions'].append(action)
            self.experience_buffer['rewards'].append(reward)
            self.experience_buffer['dones'].append(done)
            self.experience_buffer['next_frames'].append(next_frame.cpu())
        else:
            idx = self.buffer_idx % self.buffer_size
            self.experience_buffer['frames'][idx] = frame.cpu()
            self.experience_buffer['actions'][idx] = action
            self.experience_buffer['rewards'][idx] = reward
            self.experience_buffer['dones'][idx] = done
            self.experience_buffer['next_frames'][idx] = next_frame.cpu()

        self.buffer_idx += 1

    def train_world_model_step(self, batch_size: int = 32) -> Optional[Dict[str, float]]:
        """
        Train world model on a batch of experiences.

        Returns:
            Dictionary with loss metrics, or None if buffer too small
        """
        if not self.use_world_model:
            return None

        if len(self.experience_buffer['frames']) < batch_size:
            return None

        # Sample batch
        indices = torch.randint(
            0, len(self.experience_buffer['frames']), (batch_size,)
        )

        frames = torch.stack([
            self.experience_buffer['frames'][i] for i in indices
        ]).to(self.device)

        actions = torch.tensor([
            self.experience_buffer['actions'][i] for i in indices
        ]).to(self.device)

        rewards = torch.tensor([
            self.experience_buffer['rewards'][i] for i in indices
        ], dtype=torch.float32).to(self.device)

        dones = torch.tensor([
            self.experience_buffer['dones'][i] for i in indices
        ], dtype=torch.float32).to(self.device)

        next_frames = torch.stack([
            self.experience_buffer['next_frames'][i] for i in indices
        ]).to(self.device)

        # Forward + loss
        self.world_model.train()

        # Reshape for sequence dimension (batch, 1, ...)
        frames = frames.unsqueeze(1)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        losses = self.world_model.compute_loss(
            frames, actions, rewards, dones, next_frames
        )

        # Backward
        self.world_model_optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_model_optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train_policy_with_imagination(
        self,
        initial_states: torch.Tensor,
        horizon: int = None
    ) -> Dict[str, float]:
        """
        Train policy using imagined trajectories from world model.

        This is the key innovation from DreamerV3 - training on imagined
        data rather than just real experiences.

        Args:
            initial_states: Starting states for imagination (B, C, H, W)
            horizon: Number of imagination steps (default: self.imagination_horizon)

        Returns:
            Training metrics
        """
        if not self.use_world_model:
            return {}

        horizon = horizon or self.imagination_horizon
        batch_size = initial_states.shape[0]

        self.world_model.eval()
        self.policy_network.train()

        # Encode initial states
        with torch.no_grad():
            z = self.world_model.encode(initial_states)

        # Initialize world model state
        posterior, prior, h = self.world_model.initial_state(batch_size)

        # Imagination rollout
        imagined_states = []
        imagined_actions = []
        imagined_rewards = []
        imagined_values = []
        imagined_log_probs = []

        z_current = z
        post_current = posterior

        for t in range(horizon):
            # Get action from policy
            if self.use_moe:
                policy_logits, value, _ = self.policy_network(
                    self.world_model.decode(z_current), training=True
                )
            else:
                decoded = self.world_model.decode(z_current)
                policy_logits, value = self.policy_network(decoded)

            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Store for training
            imagined_states.append(z_current)
            imagined_actions.append(action)
            imagined_values.append(value.squeeze())
            imagined_log_probs.append(log_prob)

            # Predict next state using world model
            with torch.no_grad():
                post_current, _, h = self.world_model.update_state(
                    post_current, action, h
                )
                reward = self.world_model.predict_reward(post_current)
                z_current = self.world_model._to_dense(post_current)

            imagined_rewards.append(reward)

        # Compute advantages using lambda-returns
        returns = self._compute_lambda_returns(
            imagined_rewards, imagined_values, gamma=0.99, lambda_=0.95
        )

        # PPO loss on imagined trajectories
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for t in range(horizon):
            if self.use_moe:
                policy_logits, value, aux_losses = self.policy_network(
                    self.world_model.decode(imagined_states[t]), training=True
                )
            else:
                decoded = self.world_model.decode(imagined_states[t])
                policy_logits, value = self.policy_network(decoded)
                aux_losses = {}

            dist = Categorical(logits=policy_logits)
            new_log_prob = dist.log_prob(imagined_actions[t])
            entropy = dist.entropy().mean()

            # Advantage
            advantage = returns[t] - imagined_values[t].detach()

            # Policy loss (no clipping in imagination - full gradient)
            policy_loss = -(new_log_prob * advantage.detach()).mean()

            # Value loss
            value_loss = F.mse_loss(value.squeeze(), returns[t].detach())

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy

        # Average over horizon
        total_policy_loss /= horizon
        total_value_loss /= horizon
        total_entropy /= horizon

        # Total loss
        loss = total_policy_loss + 0.5 * total_value_loss - 0.01 * total_entropy

        # Add MoE auxiliary losses if using MoE
        if self.use_moe and aux_losses:
            loss += 0.01 * sum(aux_losses.values())

        # Backward
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()

        return {
            'imagination_policy_loss': total_policy_loss.item(),
            'imagination_value_loss': total_value_loss.item(),
            'imagination_entropy': total_entropy.item(),
            'imagination_loss': loss.item()
        }

    def _compute_lambda_returns(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> List[torch.Tensor]:
        """Compute lambda-returns for advantage estimation."""
        horizon = len(rewards)
        returns = [None] * horizon

        # Bootstrap from last value
        next_return = values[-1].detach()

        for t in reversed(range(horizon)):
            td_target = rewards[t] + gamma * next_return
            returns[t] = (1 - lambda_) * td_target + lambda_ * (
                rewards[t] + gamma * (returns[t + 1] if t < horizon - 1 else next_return)
            )
            next_return = values[t].detach()

        return returns

    def train_step(
        self,
        batch_data: Dict[str, torch.Tensor],
        entropy_coeff: float = 0.01
    ) -> Dict[str, float]:
        """
        Complete training step including world model and policy.

        Args:
            batch_data: Dictionary with states, actions, advantages, returns, etc.
            entropy_coeff: Entropy coefficient for exploration

        Returns:
            Dictionary with all training metrics
        """
        metrics = {}

        # 1. Train world model on real experiences
        if self.use_world_model:
            wm_metrics = self.train_world_model_step(batch_size=32)
            if wm_metrics:
                metrics.update({f'wm_{k}': v for k, v in wm_metrics.items()})

        # 2. Train policy on real data (standard PPO)
        policy_metrics = self._train_policy_ppo(batch_data, entropy_coeff)
        metrics.update(policy_metrics)

        # 3. Train policy on imagined data (if world model trained enough)
        if self.use_world_model and len(self.experience_buffer['frames']) > 1000:
            # Sample initial states for imagination
            indices = torch.randint(
                0, len(self.experience_buffer['frames']), (16,)
            )
            initial_states = torch.stack([
                self.experience_buffer['frames'][i] for i in indices
            ]).to(self.device)

            imagination_metrics = self.train_policy_with_imagination(initial_states)
            metrics.update(imagination_metrics)

        self.total_steps += 1

        return metrics

    def _train_policy_ppo(
        self,
        batch_data: Dict,
        entropy_coeff: float
    ) -> Dict[str, float]:
        """Standard PPO training on real data."""
        self.policy_network.train()

        states = batch_data['states'].to(self.device)
        actions = batch_data['actions'].to(self.device)
        old_log_probs = batch_data['old_log_probs'].to(self.device)
        advantages = batch_data['advantages'].to(self.device)
        returns = batch_data['returns'].to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        if self.use_moe:
            policy_logits, values, aux_losses = self.policy_network(states, training=True)
        else:
            policy_logits, values = self.policy_network(states)
            aux_losses = {}

        dist = Categorical(logits=policy_logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

        # Add MoE losses
        if aux_losses:
            loss += 0.01 * sum(aux_losses.values())

        # Backward
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def save(self, path: str):
        """Save all components."""
        checkpoint = {
            'policy_network': self.policy_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': {
                'use_world_model': self.use_world_model,
                'use_moe': self.use_moe,
                'use_int4': self.use_int4
            }
        }

        if self.use_world_model:
            checkpoint['world_model'] = self.world_model.state_dict()
            checkpoint['world_model_optimizer'] = self.world_model_optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load all components."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)

        if self.use_world_model and 'world_model' in checkpoint:
            self.world_model.load_state_dict(checkpoint['world_model'])
            self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])

        logger.info(f"Loaded checkpoint from {path}")


def benchmark_advanced_features(device: torch.device = None):
    """
    Benchmark the advanced features to compare performance.

    Returns metrics for:
    - Memory usage
    - Inference speed
    - Training speed
    """
    import time

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test input
    batch_size = 32
    test_input = torch.randn(batch_size, 4, 144, 160, device=device)

    results = {}

    # Baseline (no advanced features)
    agent_baseline = AdvancedPokemonAgent(
        device=device,
        use_world_model=False,
        use_moe=False,
        use_int4=False
    )

    # Full V4.0
    agent_full = AdvancedPokemonAgent(
        device=device,
        use_world_model=True,
        use_moe=True,
        use_int4=True
    )

    # Benchmark inference
    for name, agent in [('baseline', agent_baseline), ('full_v4', agent_full)]:
        # Warm up
        for _ in range(10):
            agent.get_action(test_input[0])

        # Measure
        start = time.time()
        for _ in range(100):
            agent.get_action(test_input[0])
        elapsed = time.time() - start

        results[f'{name}_inference_ms'] = (elapsed / 100) * 1000

        # Memory
        params = sum(p.numel() for p in agent.parameters())
        results[f'{name}_params'] = params
        results[f'{name}_memory_mb'] = params * 4 / (1024 * 1024)  # FP32

    logger.info("Benchmark Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.2f}")

    return results
