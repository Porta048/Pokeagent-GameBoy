import time
from collections import deque
from .hyp import HYPERPARAMETERS
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
        # Track menu opening behavior (inspired by ByteDance UI-TARS reasoning)
        self.menu_open_history = deque(maxlen=30)
        self.last_menu_open_time = 0  
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

    def track_menu_action(self, action_index: int, start_button_index: int, frame_count: int):
        """
        Track menu opening actions (inspired by ByteDance VAPO value-based reasoning).
        Penalizes non-productive menu usage.
        """
        if action_index == start_button_index:
            self.menu_open_history.append(frame_count)
            self.last_menu_open_time = frame_count

    def detect_menu_spam(self, current_frame: int) -> bool:
        """
        Detect if agent is opening menu too frequently without productive actions.
        Uses temporal reasoning similar to UI-TARS.
        """
        if len(self.menu_open_history) < 3:
            return False

        # Check if menu was opened 3+ times in last 100 frames
        recent_opens = [f for f in self.menu_open_history if current_frame - f < 100]
        if len(recent_opens) >= 3:
            return True

        # Check if menu opened very recently (less than 20 frames ago)
        if current_frame - self.last_menu_open_time < 20:
            return True

        return False

    def get_menu_spam_penalty(self, current_frame: int) -> float:
        """
        Calculate penalty for menu spam behavior.
        Implements value-based action filtering from VAPO framework.
        """
        if self.detect_menu_spam(current_frame):
            # Stronger penalty for frequent menu opening
            return -1.5
        return 0.0