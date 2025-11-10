import time
from collections import deque
from hyperparameters import HYPERPARAMETERS


class AdaptiveEntropyScheduler:
    """
    Scheduler for dynamic entropy coefficient.
    Gradually reduces random exploration as the agent learns.

    Formula: entropy = start + (end - start) * min(1.0, frames / decay_frames)
    """
    def __init__(self):
        self.start_entropy = HYPERPARAMETERS['PPO_ENTROPY_START']
        self.end_entropy = HYPERPARAMETERS['PPO_ENTROPY_END']
        self.decay_frames = HYPERPARAMETERS['PPO_ENTROPY_DECAY_FRAMES']

    def get_entropy(self, current_frame: int) -> float:
        """Calculate current entropy coefficient based on frames."""
        progress = min(1.0, current_frame / self.decay_frames)
        entropy = self.start_entropy + (self.end_entropy - self.start_entropy) * progress
        return entropy


class AntiLoopMemoryBuffer:
    """
    Buffer for detection and prevention of behavioral loops.
    Tracks recent states and penalizes repetitive patterns.

    Pattern Strategy: different detection algorithms for different types of loops.
    """
    def __init__(self):
        self.buffer_size = HYPERPARAMETERS['ANTI_LOOP_BUFFER_SIZE']
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_history = deque(maxlen=20)  # Last 20 actions
        self.position_history = deque(maxlen=50)  # Last 50 positions

    def add_state(self, pos_x: int, pos_y: int, id_map: int, action: int):
        """Add current state to buffer."""
        state_key = (id_map, pos_x, pos_y)
        self.state_buffer.append(state_key)
        self.action_history.append(action)
        self.position_history.append(state_key)

    def detect_position_loop(self) -> bool:
        """
        Detects position loop (e.g. back-and-forth repeatedly).
        Returns True if agent is in a loop.
        """
        if len(self.state_buffer) < 20:
            return False

        # Count occurrences of last 20 states (was 10)
        recent_states = list(self.state_buffer)[-20:]
        unique_states = set(recent_states)

        # If visiting less than 2 unique positions in last 20 steps = loop (was <3 in 10)
        # Much more permissive
        return len(unique_states) <= 2

    def detect_action_loop(self) -> bool:
        """
        Detects action loops (e.g. pressing A repeatedly without progress).
        Returns True if agent repeats the same action too much.
        """
        if len(self.action_history) < HYPERPARAMETERS['ACTION_REPEAT_MAX']:
            return False

        recent_actions = list(self.action_history)[-HYPERPARAMETERS['ACTION_REPEAT_MAX']:]
        # If last N actions are identical = loop
        return len(set(recent_actions)) == 1

    def detect_oscillation(self) -> bool:
        """
        Detects oscillation (e.g. up-down-up-down).
        Returns True if oscillatory pattern detected.
        """
        if len(self.position_history) < 16:  # Increased from 8 to 16
            return False

        # Check if positions alternate between 2 values
        recent_pos = list(self.position_history)[-16:]  # Increased from 8 to 16
        unique_pos = set(recent_pos)

        if len(unique_pos) == 2:
            # Check perfect alternation for at least 12 steps (was 8)
            alternating_count = sum(
                1 for i in range(len(recent_pos)-1)
                if recent_pos[i] != recent_pos[i+1]
            )
            # Only if alternates for more than 90% of cases (very restrictive)
            return alternating_count >= 14  # 14 out of 15 transitions

        return False

    def calculate_loop_penalty(self) -> float:
        """
        Calculate penalty based on detected loops.
        Returns negative penalty if loop detected, 0 otherwise.
        """
        penalty = 0.0

        if self.detect_position_loop():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY']

        if self.detect_action_loop():
            penalty += HYPERPARAMETERS['ACTION_REPEAT_PENALTY']

        if self.detect_oscillation():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY'] * 0.5

        return penalty

    def get_exploration_bonus(self) -> float:
        """
        Bonus for exploratory behavior (opposite of loop).
        Returns positive bonus if agent actively explores.
        """
        if len(self.state_buffer) < 20:
            return 0.0

        # Count unique states in last 20 steps
        recent_states = list(self.state_buffer)[-20:]
        unique_ratio = len(set(recent_states)) / len(recent_states)

        # Bonus proportional to diversity of visited states (easier to achieve)
        if unique_ratio > 0.6:  # >60% unique states = excellent exploration (was 0.7)
            return 1.5  # Reduced from 2.0
        elif unique_ratio > 0.4:  # >40% unique states = good exploration (was 0.5)
            return 0.8  # Reduced from 1.0

        return 0.0