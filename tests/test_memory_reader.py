"""Tests for game memory reading and reward calculation."""
import pytest
from unittest.mock import Mock, MagicMock

from src.mem import GameMemoryReader


class TestGameMemoryReader:
    """Test suite for GameMemoryReader."""

    @pytest.fixture
    def mock_pyboy(self):
        """Create a mock PyBoy instance."""
        pyboy = Mock()
        pyboy.get_memory_value = Mock(return_value=0)
        return pyboy

    @pytest.fixture
    def reader(self, mock_pyboy):
        """Create a GameMemoryReader instance with mock PyBoy."""
        return GameMemoryReader(mock_pyboy)

    def test_initialization(self, reader):
        """Test that GameMemoryReader initializes correctly."""
        assert reader.previous_state == {}
        assert len(reader.visited_coordinates) == 0
        assert len(reader.previous_event_flags) == 0

    def test_badge_reward_calculation(self, reader):
        """Test that badge acquisition gives large reward."""
        # Setup initial state with 0 badges
        reader.previous_state = {'badges': 0}

        # Current state with 1 badge
        current_state = {'badges': 1}

        reward = reader._calculate_badge_rewards(current_state)

        assert reward == 2000  # Badge reward is 2000

    def test_no_badge_change_gives_zero_reward(self, reader):
        """Test that no badge change gives no reward."""
        reader.previous_state = {'badges': 1}
        current_state = {'badges': 1}

        reward = reader._calculate_badge_rewards(current_state)

        assert reward == 0

    def test_pokemon_caught_reward(self, reader):
        """Test that catching Pokemon gives reward."""
        reader.previous_state = {'pokedex_caught': 0}
        current_state = {'pokedex_caught': 1}

        reward = reader._calculate_pokemon_rewards(current_state)

        assert reward == 300

    def test_pokemon_seen_reward(self, reader):
        """Test that seeing new Pokemon gives smaller reward."""
        reader.previous_state = {'pokedex_seen': 0}
        current_state = {'pokedex_seen': 1}

        reward = reader._calculate_pokemon_rewards(current_state)

        assert reward == 30

    def test_exploration_map_change_reward(self, reader):
        """Test that changing maps gives exploration reward."""
        reader.previous_state = {'map_id': 0, 'pos_x': 0, 'pos_y': 0}
        current_state = {'map_id': 1, 'pos_x': 0, 'pos_y': 0}

        reward = reader._calculate_exploration_rewards(current_state)

        assert reward >= 80  # Map change gives at least 80

    def test_money_gain_reward(self, reader):
        """Test that gaining money gives reward."""
        reader.previous_state = {'player_money': 0}
        current_state = {'player_money': 1000}

        reward = reader._calculate_money_rewards(current_state)

        assert reward > 0
        assert reward <= 20  # Capped at 20

    def test_money_loss_penalty(self, reader):
        """Test that losing significant money gives penalty."""
        reader.previous_state = {'player_money': 1000}
        current_state = {'player_money': 0}

        reward = reader._calculate_money_rewards(current_state)

        assert reward == -20  # Penalty for large money loss

    def test_battle_victory_reward(self, reader):
        """Test that winning a battle gives reward."""
        reader.previous_state = {
            'in_battle': True,
            'hp_team': [100, 50, 0, 0, 0, 0]
        }
        current_state = {
            'in_battle': False,
            'hp_team': [80, 40, 0, 0, 0, 0]  # Team still has HP
        }

        reward = reader._calculate_battle_rewards(current_state)

        assert reward == 120

    def test_battle_loss_penalty(self, reader):
        """Test that losing a battle gives penalty."""
        reader.previous_state = {
            'in_battle': True,
            'hp_team': [100, 50, 0, 0, 0, 0]
        }
        current_state = {
            'in_battle': False,
            'hp_team': [0, 0, 0, 0, 0, 0]  # All fainted
        }

        reward = reader._calculate_battle_rewards(current_state)

        assert reward == -200

    def test_navigation_reward_new_coordinate(self, reader):
        """Test that visiting new coordinate gives reward."""
        state = {'map_id': 1, 'pos_x': 10, 'pos_y': 15}

        reward = reader._calculate_navigation_rewards(state)

        assert reward == 5.0

    def test_navigation_no_reward_revisit(self, reader):
        """Test that revisiting coordinate gives no reward."""
        state = {'map_id': 1, 'pos_x': 10, 'pos_y': 15}

        # First visit
        reader._calculate_navigation_rewards(state)

        # Second visit
        reward = reader._calculate_navigation_rewards(state)

        assert reward == 0.0  # No reward for revisit

    def test_healing_reward(self, reader):
        """Test that healing Pokemon gives reward."""
        reader.previous_state = {
            'hp_team': [50, 0, 0, 0, 0, 0],
            'hp_max_team': [100, 0, 0, 0, 0, 0]
        }
        current_state = {
            'hp_team': [100, 0, 0, 0, 0, 0],
            'hp_max_team': [100, 0, 0, 0, 0, 0]
        }

        reward = reader._calculate_healing_rewards(current_state)

        assert reward > 0  # Healing gives positive reward
        assert reward <= 5.0  # Max 5 per full heal
