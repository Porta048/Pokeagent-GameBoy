from typing import Dict, Tuple, Optional, Any, List
import logging
import torch
import torch.nn as nn
import numpy as np
from .hyp import HYPERPARAMETERS
from .vis import (
    ExplorationPPO,
    BattlePPO,
    MenuPPO,
)
logger = logging.getLogger(__name__)
__all__ = ['ExplorationPPO', 'BattlePPO', 'MenuPPO', 'PPONetworkGroup']
class PPONetworkGroup:
    def __init__(
        self,
        n_actions: int,
        device: torch.device,
        input_channels: int = 4,
        **kwargs  
    ) -> None:
        self.device = device
        self.n_actions = n_actions
        self.exploration_network = ExplorationPPO(n_actions, input_channels).to(device)
        self.battle_network = BattlePPO(n_actions, input_channels).to(device)
        self.menu_network = MenuPPO(n_actions, input_channels).to(device)
        self.exploration_optimizer = torch.optim.Adam(
            self.exploration_network.parameters(),
            lr=HYPERPARAMETERS['PPO_LR']
        )
        self.battle_optimizer = torch.optim.Adam(
            self.battle_network.parameters(),
            lr=HYPERPARAMETERS['PPO_LR']
        )
        self.menu_optimizer = torch.optim.Adam(
            self.menu_network.parameters(),
            lr=HYPERPARAMETERS['PPO_LR']
        )
        self._log_architecture_info()
    def _log_architecture_info(self) -> None:
        total_params = sum(
            sum(p.numel() for p in net.parameters())
            for net in [self.exploration_network, self.battle_network, self.menu_network]
        )
        logger.info("[ARCH] Architettura: DEEPSEEK-VL2")
        logger.info(f"[ARCH] Parametri totali: {total_params:,}")
        logger.info("[ARCH] Componenti:")
        logger.info("  - PixelShuffleAdaptor: Compressione 2×2 token visivi")
        logger.info("  - Multi-head Latent Attention: KV cache compresso")
    def get_architecture_info(self) -> Dict:
        info = {
            'architecture': 'deepseek_vl2',
            'networks': {}
        }
        for name, net in [
            ('exploration', self.exploration_network),
            ('battle', self.battle_network),
            ('menu', self.menu_network)
        ]:
            params = sum(p.numel() for p in net.parameters())
            info['networks'][name] = {
                'parameters': params,
                'class': net.__class__.__name__
            }
        info['total_parameters'] = sum(
            n['parameters'] for n in info['networks'].values()
        )
        return info
    def select_network(self, game_state: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
        if game_state == "battle":
            return self.battle_network, self.battle_optimizer
        elif game_state == "menu":
            return self.menu_network, self.menu_optimizer
        else:
            return self.exploration_network, self.exploration_optimizer
    def choose_action(
        self,
        state: torch.Tensor,
        game_state: str,
        deterministic: bool = False,
        action_mask: list = None,
        llm_bias: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, float, float]:
        """
        Choose action using policy network with optional LLM bias.

        Args:
            state: Current game state tensor
            game_state: 'exploring', 'battle', or 'menu'
            deterministic: If True, use argmax instead of sampling
            action_mask: Optional mask for valid actions
            llm_bias: Optional LLM response with 'suggested_action' and 'confidence'

        Returns:
            (action_index, log_probability, value_estimate)
        """
        from torch.distributions import Categorical
        network, _ = self.select_network(game_state)
        network.eval()
        with torch.no_grad():
            state_batch = state.unsqueeze(0) if state.dim() == 3 else state
            policy_logits, value = network(state_batch)

            # Apply action mask first
            if action_mask is not None:
                from .act import ContextAwareActionFilter
                policy_logits = ContextAwareActionFilter.apply_mask_to_logits(
                    policy_logits, action_mask
                )

            # Apply LLM bias as soft influence on action probabilities
            if llm_bias is not None and llm_bias.get('suggested_action') is not None:
                policy_logits = self._apply_llm_bias(policy_logits, llm_bias)

            dist = Categorical(logits=policy_logits)
            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def _apply_llm_bias(
        self,
        policy_logits: torch.Tensor,
        llm_bias: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply LLM suggestion as soft bias to policy logits.

        The LLM acts as a "strategic advisor" - it boosts the logit
        of suggested actions but doesn't override the policy network.

        Args:
            policy_logits: Raw logits from policy network [batch, n_actions]
            llm_bias: Dict with 'suggested_action' (str) and 'confidence' (float)

        Returns:
            Adjusted policy logits
        """
        suggested_action = llm_bias.get('suggested_action')
        confidence = llm_bias.get('confidence', 0.5)

        # Map action name to index
        action_to_idx = {
            None: 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4,
            'a': 5, 'b': 6, 'start': 7, 'select': 8
        }

        action_idx = action_to_idx.get(suggested_action)
        if action_idx is None:
            return policy_logits

        # Calculate boost based on confidence
        # Higher confidence = stronger boost (0.5 to 2.0 logit addition)
        logit_boost = confidence * 2.0

        # Apply boost to suggested action
        boosted_logits = policy_logits.clone()
        boosted_logits[:, action_idx] += logit_boost

        return boosted_logits

    def train_grpo(
        self,
        batch_data: Dict,
        game_state: str,
        entropy_coeff: float = None
    ) -> Dict[str, float]:
        """
        GRPO (Group Relative Policy Optimization) training.

        Based on DeepSeek-R1 (January 2025). Key difference from PPO:
        - Advantages are pre-normalized by group in trajectory buffer
        - No additional global normalization during training
        - Improves stability for multi-modal reward distributions

        Args:
            batch_data: Dict with 'states', 'actions', 'old_log_probs',
                       'advantages' (pre-normalized), 'returns', 'game_states'
            game_state: Current game state for network selection
            entropy_coeff: Entropy bonus coefficient

        Returns:
            metrics: Dict with 'policy_loss', 'value_loss', 'entropy'
        """
        from torch.distributions import Categorical
        import torch.nn.functional as F

        network, optimizer = self.select_network(game_state)
        network.train()

        if entropy_coeff is None:
            entropy_coeff = HYPERPARAMETERS['PPO_ENTROPY_COEFF']

        # Prepare batch (GPU transfer)
        states = torch.stack(batch_data['states']).to(self.device)
        actions = torch.tensor(batch_data['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(batch_data['old_log_probs'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch_data['advantages'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(batch_data['returns'], dtype=torch.float32).to(self.device)

        # GRPO: Advantages pre-normalized by group in trajectory buffer

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        dataset_size = len(states)
        minibatch_size = HYPERPARAMETERS['PPO_MINIBATCH_SIZE']

        # Multi-epoch training
        for epoch in range(HYPERPARAMETERS['PPO_EPOCHS']):
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                mb_indices = indices[start:end]

                # Minibatch extraction
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass
                policy_logits, values = network(mb_states)
                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # GRPO Policy Loss (same clipping as PPO)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - HYPERPARAMETERS['PPO_CLIP_EPSILON'],
                    1.0 + HYPERPARAMETERS['PPO_CLIP_EPSILON']
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)

                # Combined Loss
                loss = (
                    policy_loss +
                    HYPERPARAMETERS['PPO_VALUE_COEFF'] * value_loss -
                    entropy_coeff * entropy
                )

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    network.parameters(),
                    HYPERPARAMETERS['PPO_MAX_GRAD_NORM']
                )
                optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
