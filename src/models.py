from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hyperparameters import HYPERPARAMETERS
from utils import EXPLORATION_CONV_DIMENSIONS, MENU_CONV_DIMENSIONS


class BaseActorCriticNetwork(nn.Module):
    """Base Actor-Critic network with shared backbone."""
    def __init__(self, n_actions, input_channels=4):
        super(BaseActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_height, conv_width = EXPLORATION_CONV_DIMENSIONS
        linear_input_size = conv_height * conv_width * 64

        self.fc_shared = nn.Linear(linear_input_size, 512)

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


class ExplorationPPO(BaseActorCriticNetwork):
    """PPO Actor-Critic for exploration."""
    def __init__(self, n_actions, input_channels=4):
        super(ExplorationPPO, self).__init__(n_actions, input_channels)


class BattlePPO(BaseActorCriticNetwork):
    """PPO Actor-Critic for battle (more capacity)."""
    def __init__(self, n_actions, input_channels=4):
        super(BattlePPO, self).__init__(n_actions, input_channels)
        conv_height, conv_width = EXPLORATION_CONV_DIMENSIONS
        linear_input_size = conv_height * conv_width * 64

        self.fc_shared = nn.Linear(linear_input_size, 512)
        self.fc_shared2 = nn.Linear(512, 256)

        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        x = F.relu(self.fc_shared2(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


class MenuPPO(nn.Module):
    """PPO Actor-Critic for menu (smaller)."""
    def __init__(self, n_actions, input_channels=4):
        super(MenuPPO, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        conv_height, conv_width = MENU_CONV_DIMENSIONS
        linear_input_size = conv_height * conv_width * 32

        self.fc_shared = nn.Linear(linear_input_size, 128)

        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


from errors import PokemonAIError, ModelSaveError


class PPONetworkGroup:
    """Manager for multiple PPO networks for game states."""
    def __init__(self, n_actions: int, device: 'torch.device', input_channels: int = 4) -> None:
        self.device = device
        self.n_actions = n_actions

        self.exploration_network = ExplorationPPO(n_actions, input_channels).to(device)
        self.battle_network = BattlePPO(n_actions, input_channels).to(device)
        self.menu_network = MenuPPO(n_actions, input_channels).to(device)

        import torch.optim as optim
        self.exploration_optimizer = optim.Adam(self.exploration_network.parameters(), lr=HYPERPARAMETERS['PPO_LR'])
        self.battle_optimizer = optim.Adam(self.battle_network.parameters(), lr=HYPERPARAMETERS['PPO_LR'])
        self.menu_optimizer = optim.Adam(self.menu_network.parameters(), lr=HYPERPARAMETERS['PPO_LR'])

    def select_network(self, game_state: str):
        """Select network and optimizer for game state."""
        if game_state == "battle":
            return self.battle_network, self.battle_optimizer
        elif game_state == "menu":
            return self.menu_network, self.menu_optimizer
        else:
            return self.exploration_network, self.exploration_optimizer

    def choose_action(self, state: torch.Tensor, game_state: str, deterministic: bool = False,
                      action_mask: 'List[float]' = None):
        """Choose action with stochastic policy and optional action masking."""
        from torch.distributions import Categorical
        from typing import List
        
        network, _ = self.select_network(game_state)
        network.eval()

        with torch.no_grad():
            state_batch = state.unsqueeze(0) if state.dim() == 3 else state
            policy_logits, value = network(state_batch)

            # Apply action mask if provided (context-aware filtering)
            if action_mask is not None:
                from action_filter import ContextAwareActionFilter
                policy_logits = ContextAwareActionFilter.apply_mask_to_logits(
                    policy_logits, action_mask
                )

            dist = Categorical(logits=policy_logits)

            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def train_ppo(self, batch_data: dict, game_state: str, entropy_coeff: float = None) -> dict:
        """PPO Training with clipped surrogate objective and adaptive entropy."""
        import torch.optim as optim
        import torch.nn.functional as F
        from torch.distributions import Categorical
        
        network, optimizer = self.select_network(game_state)
        network.train()

        # Use entropy coefficient provided or default
        if entropy_coeff is None:
            entropy_coeff = HYPERPARAMETERS['PPO_ENTROPY_COEFF']

        states = torch.stack(batch_data['states']).to(self.device)
        actions = torch.tensor(batch_data['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(batch_data['old_log_probs'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch_data['advantages'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(batch_data['returns'], dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        dataset_size = len(states)
        minibatch_size = HYPERPARAMETERS['PPO_MINIBATCH_SIZE']

        for epoch in range(HYPERPARAMETERS['PPO_EPOCHS']):
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                policy_logits, values = network(mb_states)
                dist = Categorical(logits=policy_logits)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - HYPERPARAMETERS['PPO_CLIP_EPSILON'],
                                   1.0 + HYPERPARAMETERS['PPO_CLIP_EPSILON']) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(), mb_returns)

                loss = (policy_loss +
                       HYPERPARAMETERS['PPO_VALUE_COEFF'] * value_loss -
                       entropy_coeff * entropy)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), HYPERPARAMETERS['PPO_MAX_GRAD_NORM'])
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }