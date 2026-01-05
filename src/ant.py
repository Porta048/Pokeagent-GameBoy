from collections import deque, defaultdict
from .hyp import HYPERPARAMETERS


class AdaptiveEntropyScheduler:
    def __init__(self):
        self.start = HYPERPARAMETERS['PPO_ENTROPY_START']
        self.end = HYPERPARAMETERS['PPO_ENTROPY_END']
        self.decay = HYPERPARAMETERS['PPO_ENTROPY_DECAY_FRAMES']
        self.stuck_counter = 0
        self.last_progress = 0

    def get_entropy(self, frame: int, stuck: bool = False) -> float:
        progress = min(1.0, frame / self.decay)
        base = self.start + (self.end - self.start) * progress
        if stuck:
            self.stuck_counter += 1
            return min(base * (1 + 0.1 * min(self.stuck_counter, 10)), self.start)
        self.stuck_counter = max(0, self.stuck_counter - 1)
        return base


class AntiLoopMemoryBuffer:
    def __init__(self):
        self.buffer_size = HYPERPARAMETERS['ANTI_LOOP_BUFFER_SIZE']
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_history = deque(maxlen=30)
        self.position_history = deque(maxlen=self.buffer_size)
        self.menu_history = deque(maxlen=20)
        self.last_menu_frame = 0
        self.state_visits = defaultdict(int)
        self.recent_unique_states = set()

    def add_state(self, x: int, y: int, map_id: int, action: int):
        state = (map_id, x, y)
        self.state_buffer.append(state)
        self.action_history.append(action)
        self.position_history.append(state)
        self.state_visits[state] += 1
        if len(self.recent_unique_states) > 200:
            self.recent_unique_states.clear()
        self.recent_unique_states.add(state)

    def detect_position_loop(self) -> bool:
        if len(self.state_buffer) < 12:
            return False
        recent = list(self.state_buffer)[-12:]
        return len(set(recent)) <= HYPERPARAMETERS['ANTI_LOOP_THRESHOLD']

    def detect_action_loop(self) -> bool:
        if len(self.action_history) < HYPERPARAMETERS['ACTION_REPEAT_MAX']:
            return False
        recent = list(self.action_history)[-HYPERPARAMETERS['ACTION_REPEAT_MAX']:]
        return len(set(recent)) == 1

    def detect_oscillation(self) -> bool:
        if len(self.position_history) < 10:
            return False
        recent = list(self.position_history)[-10:]
        unique = set(recent)
        if len(unique) != 2:
            return False
        alternations = sum(1 for i in range(len(recent)-1) if recent[i] != recent[i+1])
        return alternations >= 8

    def detect_small_area_loop(self) -> bool:
        if len(self.position_history) < 30:
            return False
        recent = list(self.position_history)[-30:]
        unique = set(recent)
        return len(unique) <= 5

    def calculate_loop_penalty(self) -> float:
        penalty = 0.0
        if self.detect_position_loop():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY']
        if self.detect_action_loop():
            penalty += HYPERPARAMETERS['ACTION_REPEAT_PENALTY']
        if self.detect_oscillation():
            penalty += HYPERPARAMETERS.get('OSCILLATION_PENALTY', -4.0)
        if self.detect_small_area_loop():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY'] * 0.7
        return penalty

    def get_exploration_bonus(self, x: int, y: int, map_id: int) -> float:
        state = (map_id, x, y)
        visits = self.state_visits[state]
        if visits == 1:
            return HYPERPARAMETERS.get('EXPLORATION_BONUS', 0.5)
        elif visits <= 3:
            return HYPERPARAMETERS.get('EXPLORATION_BONUS', 0.5) * 0.3
        return 0.0

    def get_curiosity_reward(self) -> float:
        if len(self.state_buffer) < 20:
            return 0.0
        recent = list(self.state_buffer)[-20:]
        unique_ratio = len(set(recent)) / len(recent)
        if unique_ratio > 0.8:
            return HYPERPARAMETERS.get('CURIOSITY_COEFF', 0.1) * 2.0
        elif unique_ratio > 0.5:
            return HYPERPARAMETERS.get('CURIOSITY_COEFF', 0.1)
        return 0.0

    def track_menu_action(self, action: int, start_idx: int, frame: int):
        if action == start_idx:
            self.menu_history.append(frame)
            self.last_menu_frame = frame

    def detect_menu_spam(self, frame: int) -> bool:
        if len(self.menu_history) < 3:
            return False
        recent = [f for f in self.menu_history if frame - f < 80]
        return len(recent) >= 3 or (frame - self.last_menu_frame < 15)

    def get_menu_spam_penalty(self, frame: int) -> float:
        return -2.0 if self.detect_menu_spam(frame) else 0.0

    def is_stuck(self) -> bool:
        return self.detect_position_loop() or self.detect_small_area_loop() or self.detect_oscillation()
