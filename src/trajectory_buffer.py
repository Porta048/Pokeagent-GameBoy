from typing import Dict, List, Any
from .hyperparameters import HYPERPARAMETERS


class TrajectoryBuffer:
    """Buffer per traiettorie PPO con calcolo GAE."""
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
        """Calcola vantaggi con GAE-Lambda."""
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

    def get_batch(self, advantages, returns):
        """Ritorna dati per training."""
        return {
            'states': self.states,
            'actions': self.actions,
            'old_log_probs': self.log_probs,
            'advantages': advantages,
            'returns': returns,
            'game_states': self.game_states
        }