"""
Architetture Reti Neurali PPO per Pokemon Rosso/Blu.

DESIGN: Tre reti specializzate per diversi contesti di gioco
- ExplorationPPO: Esplorazione overworld (leggera, veloce)
- BattlePPO: Combattimenti (profonda, strategica)
- MenuPPO: Navigazione menu (minimalista, rapidissima)

ARCHITETTURA ACTOR-CRITIC:
- Actor (Policy): Decide quale azione fare (output = probabilità per ogni azione)
- Critic (Value): Stima quanto è "buono" lo stato corrente (output = valore atteso)

PERCHÉ 3 RETI SEPARATE?
Ogni contesto ha priorità diverse:
- Esplorazione: Serve velocità, decisioni rapide su movimento
- Battaglia: Serve profondità, ragionamento strategico su mosse
- Menu: Serve minimalismo, navigazione efficiente interfaccia

INPUT: Frame stack 4x grayscale (4, 144, 160)
OUTPUT: Logit azioni (9) + valore stato (1)
"""
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .hyperparameters import HYPERPARAMETERS
from .utils import EXPLORATION_CONV_DIMENSIONS, MENU_CONV_DIMENSIONS
from .errors import PokemonAIError


class BaseActorCriticNetwork(nn.Module):
    """
    Rete Actor-Critic BASE per PPO.

    COMPONENTI:
    1. Backbone CNN: Estrae features visive da frame Game Boy
       - Conv1: 4→32 canali, kernel 8x8, stride 4 (riduzione aggressive)
       - Conv2: 32→64 canali, kernel 4x4, stride 2
       - Conv3: 64→64 canali, kernel 3x3, stride 1 (raffinamento)

    2. Fully Connected: 512 neuroni per rappresentazione ad alto livello

    3. Policy Head: FC 512→9 (logit per ogni azione)

    4. Value Head: FC 512→1 (stima valore stato)

    INIZIALIZZAZIONE: Orthogonal con gain=sqrt(2)
    - Migliora stabilità training PPO
    - Previene vanishing/exploding gradients
    """

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_height, conv_width = EXPLORATION_CONV_DIMENSIONS
        linear_input_size = conv_height * conv_width * 64

        self.fc_shared = nn.Linear(linear_input_size, 512)
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inizializzazione ortogonale per stabilità del training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


class ExplorationPPO(BaseActorCriticNetwork):
    """Rete PPO specializzata per esplorazione del mondo di gioco."""

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(n_actions, input_channels)


class BattlePPO(BaseActorCriticNetwork):
    """
    Rete PPO specializzata per combattimenti.
    Maggiore capacità (layer aggiuntivo) per gestire tattiche complesse.
    """

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__(n_actions, input_channels)
        conv_height, conv_width = EXPLORATION_CONV_DIMENSIONS
        linear_input_size = conv_height * conv_width * 64

        self.fc_shared = nn.Linear(linear_input_size, 512)
        self.fc_shared2 = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """
    Rete PPO leggera specializzata per navigazione menu.
    Architettura ridotta per elaborazione veloce delle interfacce UI.
    """

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        conv_height, conv_width = MENU_CONV_DIMENSIONS
        linear_input_size = conv_height * conv_width * 32

        self.fc_shared = nn.Linear(linear_input_size, 128)
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


class PPONetworkGroup:
    """
    Gestore delle 3 reti PPO specializzate per diversi stati di gioco.
    Seleziona automaticamente la rete appropriata basandosi sul contesto.
    """

    def __init__(self, n_actions: int, device: torch.device, input_channels: int = 4) -> None:
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

    def select_network(self, game_state: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """Seleziona rete e optimizer in base allo stato di gioco."""
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
        action_mask: List[float] = None
    ) -> Tuple[int, float, float]:
        """
        Sceglie azione usando policy stocastica con mascheramento opzionale.

        Returns:
            action: Indice dell'azione scelta
            log_prob: Log-probabilità dell'azione
            value: Valore stimato dello stato
        """
        from torch.distributions import Categorical

        network, _ = self.select_network(game_state)
        network.eval()

        with torch.no_grad():
            state_batch = state.unsqueeze(0) if state.dim() == 3 else state
            policy_logits, value = network(state_batch)

            if action_mask is not None:
                from .action_filter import ContextAwareActionFilter
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

    def train_ppo(
        self,
        batch_data: Dict,
        game_state: str,
        entropy_coeff: float = None
    ) -> Dict[str, float]:
        """
        Training PPO con clipped surrogate objective e entropia adattiva.

        Args:
            batch_data: Dizionario con states, actions, advantages, returns
            game_state: Stato di gioco per selezione rete
            entropy_coeff: Coefficiente entropia (opzionale, usa default se None)

        Returns:
            Metriche di training (policy_loss, value_loss, entropy)
        """
        from torch.distributions import Categorical

        network, optimizer = self.select_network(game_state)
        network.train()

        if entropy_coeff is None:
            entropy_coeff = HYPERPARAMETERS['PPO_ENTROPY_COEFF']

        states = torch.stack(batch_data['states']).to(self.device)
        actions = torch.tensor(batch_data['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(batch_data['old_log_probs'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch_data['advantages'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(batch_data['returns'], dtype=torch.float32).to(self.device)

        # Normalizzazione advantages per stabilità numerica
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
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

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - HYPERPARAMETERS['PPO_CLIP_EPSILON'],
                    1.0 + HYPERPARAMETERS['PPO_CLIP_EPSILON']
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(), mb_returns)

                loss = (
                    policy_loss +
                    HYPERPARAMETERS['PPO_VALUE_COEFF'] * value_loss -
                    entropy_coeff * entropy
                )

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
