from typing import Dict, List, Any, Tuple, Optional
import torch
from .hyp import HYPERPARAMETERS
class TrajectoryBuffer:
    def __init__(self, capacity: int = HYPERPARAMETERS['PPO_TRAJECTORY_LENGTH']):
        self.capacity = capacity
        self.reset()
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.game_states = []
    def add(self, state, action, reward, value, log_prob, done, game_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.game_states.append(game_state)
    def __len__(self):
        return len(self.states)
    def is_full(self):
        return len(self) >= self.capacity
    def calculate_gae_advantages(self, next_value):
        """Legacy PPO advantage calculation (global normalization)."""
        advantages = []
        gae = 0
        gamma = HYPERPARAMETERS['PPO_GAE_GAMMA']
        lam = HYPERPARAMETERS['PPO_GAE_LAMBDA']
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def calculate_grpo_advantages(
        self,
        next_value: float,
        group_by: str = 'game_state'
    ) -> Tuple[List[float], List[float], Dict[str, Dict[str, float]]]:
        """
        Calculate GAE advantages with group-relative normalization (GRPO).

        Based on DeepSeek-R1 Group Relative Policy Optimization (January 2025).
        Normalizes advantages within groups (game states) instead of globally,
        improving training stability for multi-modal reward distributions.

        Args:
            next_value: Bootstrap value for last state
            group_by: Grouping strategy
                - 'game_state': Group by game state (exploring/battle/menu/dialogue/other)
                - 'none': No grouping (fallback to standard PPO)

        Returns:
            advantages: List of normalized advantages (same length as trajectory)
            returns: List of returns (advantages + values)
            group_stats: Dict mapping group name to {'mean', 'std', 'count'}
        """
        # Step 1: Calculate raw GAE advantages (same as standard PPO)
        raw_advantages = []
        gae = 0
        gamma = HYPERPARAMETERS['PPO_GAE_GAMMA']
        lam = HYPERPARAMETERS['PPO_GAE_LAMBDA']
        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            raw_advantages.insert(0, gae)

        # Step 2: Group-relative normalization
        if group_by == 'game_state' and HYPERPARAMETERS.get('GRPO_ENABLED', True):
            # Group experiences by game state
            groups: Dict[str, Dict[str, List]] = {}
            for i, state in enumerate(self.game_states):
                if state not in groups:
                    groups[state] = {'indices': [], 'advantages': []}
                groups[state]['indices'].append(i)
                groups[state]['advantages'].append(raw_advantages[i])

            # Normalize within each group
            normalized_advantages = raw_advantages.copy()
            group_stats = {}
            min_group_size = HYPERPARAMETERS.get('GRPO_MIN_GROUP_SIZE', 2)

            for state, data in groups.items():
                if len(data['advantages']) < min_group_size:
                    # Skip normalization for small groups, keep raw advantages
                    group_stats[state] = {
                        'mean': 0.0,
                        'std': 1.0,
                        'count': len(data['advantages']),
                        'normalized': False
                    }
                    continue

                # Convert to tensor for efficient computation
                group_adv = torch.tensor(data['advantages'], dtype=torch.float32)
                group_mean = group_adv.mean()
                group_std = group_adv.std()

                # Normalize: (x - mean) / (std + epsilon)
                normalized_group = (group_adv - group_mean) / (group_std + 1e-8)

                # Update advantages in place
                for idx, norm_val in zip(data['indices'], normalized_group.tolist()):
                    normalized_advantages[idx] = norm_val

                # Store statistics for logging
                group_stats[state] = {
                    'mean': group_mean.item(),
                    'std': group_std.item(),
                    'count': len(data['advantages']),
                    'normalized': True
                }

            advantages = normalized_advantages
        else:
            # Fallback: global normalization (standard PPO behavior)
            if len(raw_advantages) > 1:
                adv_tensor = torch.tensor(raw_advantages, dtype=torch.float32)
                adv_mean = adv_tensor.mean()
                adv_std = adv_tensor.std()
                advantages = ((adv_tensor - adv_mean) / (adv_std + 1e-8)).tolist()
                group_stats = {
                    'global': {
                        'mean': adv_mean.item(),
                        'std': adv_std.item(),
                        'count': len(raw_advantages),
                        'normalized': True
                    }
                }
            else:
                advantages = raw_advantages
                group_stats = {}

        # Calculate returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns, group_stats
    def get_batch(self, advantages, returns):
        return {
            'states': self.states,
            'actions': self.actions,
            'old_log_probs': self.log_probs,
            'advantages': advantages,
            'returns': returns,
            'game_states': self.game_states
        }