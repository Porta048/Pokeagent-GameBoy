import time
from collections import deque
from .hyperparameters import HYPERPARAMETERS
class AdaptiveEntropyScheduler:
    def __init__(self):
        self.start_entropy = HYPERPARAMETERS['PPO_ENTROPY_START']
        self.end_entropy = HYPERPARAMETERS['PPO_ENTROPY_END']
        self.decay_frames = HYPERPARAMETERS['PPO_ENTROPY_DECAY_FRAMES']
    def get_entropy(self, current_frame: int) -> float:
        progress = min(1.0, current_frame / self.decay_frames)
        entropy = self.start_entropy + (self.end_entropy - self.start_entropy) * progress
        return entropy
class AntiLoopMemoryBuffer:
    def __init__(self):
        self.buffer_size = HYPERPARAMETERS['ANTI_LOOP_BUFFER_SIZE']
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_history = deque(maxlen=20)  
        self.position_history = deque(maxlen=50)  
    def add_state(self, pos_x: int, pos_y: int, id_map: int, action: int):
        state_key = (id_map, pos_x, pos_y)
        self.state_buffer.append(state_key)
        self.action_history.append(action)
        self.position_history.append(state_key)
    def detect_position_loop(self) -> bool:
        if len(self.state_buffer) < 20:
            return False
        recent_states = list(self.state_buffer)[-20:]
        unique_states = set(recent_states)
        return len(unique_states) <= 2
    def detect_action_loop(self) -> bool:
        if len(self.action_history) < HYPERPARAMETERS['ACTION_REPEAT_MAX']:
            return False
        recent_actions = list(self.action_history)[-HYPERPARAMETERS['ACTION_REPEAT_MAX']:]
        return len(set(recent_actions)) == 1
    def detect_oscillation(self) -> bool:
        if len(self.position_history) < 16:  
            return False
        recent_pos = list(self.position_history)[-16:]  
        unique_pos = set(recent_pos)
        if len(unique_pos) == 2:
            alternating_count = sum(
                1 for i in range(len(recent_pos)-1)
                if recent_pos[i] != recent_pos[i+1]
            )
            return alternating_count >= 14  
        return False
    def calculate_loop_penalty(self) -> float:
        penalty = 0.0
        if self.detect_position_loop():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY']
        if self.detect_action_loop():
            penalty += HYPERPARAMETERS['ACTION_REPEAT_PENALTY']
        if self.detect_oscillation():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY'] * 0.5
        return penalty
    def get_exploration_bonus(self) -> float:
        if len(self.state_buffer) < 20:
            return 0.0
        recent_states = list(self.state_buffer)[-20:]
        unique_ratio = len(set(recent_states)) / len(recent_states)
        if unique_ratio > 0.6:  
            return 1.5  
        elif unique_ratio > 0.4:  
            return 0.8  
        return 0.0