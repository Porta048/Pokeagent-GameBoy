"""Tests for anti-loop system."""
import pytest

from src.anti_loop import AdaptiveEntropyScheduler, AntiLoopMemoryBuffer


class TestAdaptiveEntropyScheduler:
    """Test suite for AdaptiveEntropyScheduler."""

    def test_entropy_at_start(self):
        """Test entropy value at training start."""
        scheduler = AdaptiveEntropyScheduler()
        entropy = scheduler.get_entropy(0)

        # At frame 0, should be at start value
        assert entropy == scheduler.start_entropy

    def test_entropy_at_end(self):
        """Test entropy value after decay period."""
        scheduler = AdaptiveEntropyScheduler()
        # After decay frames, should be at end value
        entropy = scheduler.get_entropy(scheduler.decay_frames)

        assert entropy == pytest.approx(scheduler.end_entropy)

    def test_entropy_decreases_over_time(self):
        """Test that entropy decreases monotonically."""
        scheduler = AdaptiveEntropyScheduler()

        entropy_start = scheduler.get_entropy(0)
        entropy_mid = scheduler.get_entropy(scheduler.decay_frames // 2)
        entropy_end = scheduler.get_entropy(scheduler.decay_frames)

        assert entropy_start > entropy_mid > entropy_end

    def test_entropy_stays_at_minimum_after_decay(self):
        """Test that entropy stays at minimum after decay period."""
        scheduler = AdaptiveEntropyScheduler()

        entropy_at_decay = scheduler.get_entropy(scheduler.decay_frames)
        entropy_after = scheduler.get_entropy(scheduler.decay_frames * 2)

        assert entropy_at_decay == pytest.approx(scheduler.end_entropy)
        assert entropy_after == pytest.approx(scheduler.end_entropy)


class TestAntiLoopMemoryBuffer:
    """Test suite for AntiLoopMemoryBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create an AntiLoopMemoryBuffer instance."""
        return AntiLoopMemoryBuffer()

    def test_initialization(self, buffer):
        """Test that buffer initializes empty."""
        assert len(buffer.state_buffer) == 0
        assert len(buffer.action_history) == 0
        assert len(buffer.position_history) == 0

    def test_add_state(self, buffer):
        """Test adding states to buffer."""
        buffer.add_state(pos_x=10, pos_y=20, id_map=1, action=0)

        assert len(buffer.state_buffer) == 1
        assert len(buffer.action_history) == 1
        assert len(buffer.position_history) == 1

    def test_position_loop_not_detected_with_few_states(self, buffer):
        """Test that loop is not detected with insufficient data."""
        for i in range(10):
            buffer.add_state(pos_x=i, pos_y=i, id_map=1, action=0)

        assert not buffer.detect_position_loop()

    def test_position_loop_detected_when_stuck(self, buffer):
        """Test that loop is detected when agent is stuck."""
        # Add 20 states at same position
        for _ in range(20):
            buffer.add_state(pos_x=10, pos_y=20, id_map=1, action=0)

        assert buffer.detect_position_loop()

    def test_position_loop_not_detected_with_exploration(self, buffer):
        """Test that loop is not detected during active exploration."""
        # Add 20 states at different positions
        for i in range(20):
            buffer.add_state(pos_x=i, pos_y=i, id_map=1, action=0)

        assert not buffer.detect_position_loop()

    def test_action_loop_not_detected_with_varied_actions(self, buffer):
        """Test that action loop is not detected with varied actions."""
        actions = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        for i, action in enumerate(actions):
            buffer.add_state(pos_x=i, pos_y=i, id_map=1, action=action)

        assert not buffer.detect_action_loop()

    def test_action_loop_detected_with_repeated_action(self, buffer):
        """Test that action loop is detected when spamming one action."""
        # Spam action 0 for 15 times (more than ACTION_REPEAT_MAX)
        for i in range(15):
            buffer.add_state(pos_x=i, pos_y=i, id_map=1, action=0)

        assert buffer.detect_action_loop()

    def test_oscillation_not_detected_initially(self, buffer):
        """Test that oscillation is not detected with few states."""
        for i in range(10):
            buffer.add_state(pos_x=i % 2, pos_y=0, id_map=1, action=0)

        assert not buffer.detect_oscillation()

    def test_oscillation_detected_with_back_and_forth(self, buffer):
        """Test that oscillation is detected with perfect alternation."""
        # Perfect alternation between two positions for 16 steps
        for i in range(16):
            buffer.add_state(pos_x=i % 2, pos_y=0, id_map=1, action=0)

        assert buffer.detect_oscillation()

    def test_loop_penalty_zero_without_loops(self, buffer):
        """Test that no penalty is applied without loops."""
        # Add varied states
        for i in range(20):
            buffer.add_state(pos_x=i, pos_y=i, id_map=1, action=i % 5)

        penalty = buffer.calculate_loop_penalty()

        assert penalty == 0.0

    def test_loop_penalty_negative_with_loops(self, buffer):
        """Test that penalty is negative when loops detected."""
        # Create stuck condition
        for _ in range(20):
            buffer.add_state(pos_x=10, pos_y=20, id_map=1, action=0)

        penalty = buffer.calculate_loop_penalty()

        assert penalty < 0

    def test_exploration_bonus_with_high_diversity(self, buffer):
        """Test that exploration bonus is given for diverse states."""
        # Add 20 unique states
        for i in range(20):
            buffer.add_state(pos_x=i, pos_y=i, id_map=1, action=i % 5)

        bonus = buffer.get_exploration_bonus()

        assert bonus > 0

    def test_exploration_bonus_zero_with_repetition(self, buffer):
        """Test that no bonus is given for repetitive behavior."""
        # Add 20 states at same position
        for _ in range(20):
            buffer.add_state(pos_x=10, pos_y=20, id_map=1, action=0)

        bonus = buffer.get_exploration_bonus()

        assert bonus == 0.0
