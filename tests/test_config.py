"""Tests for configuration management."""
import pytest
import torch
from pathlib import Path

from src.config import Config


class TestConfig:
    """Test suite for Config class."""

    def test_default_initialization(self):
        """Test that Config initializes with valid defaults."""
        config = Config()
        assert config.ROM_PATH == r"C:\Users\chatg\Documents\GitHub\Pokemon Red.gb"
        assert config.FRAME_STACK_SIZE == 4
        assert config.SAVE_FREQUENCY > 0
        assert isinstance(config.DEVICE, torch.device)

    def test_validation_emulation_speed_negative(self):
        """Test that negative emulation speed raises ValueError."""
        with pytest.raises(ValueError, match="EMULATION_SPEED deve essere >= 0"):
            Config(EMULATION_SPEED=-1)

    def test_validation_frame_stack_size_zero(self):
        """Test that zero frame stack size raises ValueError."""
        with pytest.raises(ValueError, match="FRAME_STACK_SIZE deve essere >= 1"):
            Config(FRAME_STACK_SIZE=0)

    def test_validation_menu_threshold_out_of_range(self):
        """Test that menu_threshold outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="MENU_THRESHOLD deve essere in"):
            Config(MENU_THRESHOLD=1.5)

        with pytest.raises(ValueError, match="MENU_THRESHOLD deve essere in"):
            Config(MENU_THRESHOLD=-0.1)

    def test_get_save_dir_creates_directory(self, tmp_path):
        """Test that get_save_dir creates the directory."""
        config = Config(SAVE_DIR_PREFIX=str(tmp_path / "test_saves"))
        rom_path = "pokemon_red.gb"

        save_dir = config.get_save_dir(rom_path)

        assert save_dir.exists()
        assert save_dir.is_dir()
        assert save_dir.name == "test_saves_pokemon_red"

    def test_get_model_path_returns_correct_path(self):
        """Test that get_model_path returns correct file path."""
        config = Config()
        rom_path = "pokemon_red.gb"

        model_path = config.get_model_path(rom_path)

        assert model_path.name == "model_ppo.pth"
        assert "pokemon_red" in str(model_path)

    def test_actions_list_is_valid(self):
        """Test that actions list contains valid button names."""
        config = Config()

        assert None in config.ACTIONS  # No-op
        assert 'a' in config.ACTIONS
        assert 'b' in config.ACTIONS
        assert 'up' in config.ACTIONS
        assert 'down' in config.ACTIONS
        assert 'left' in config.ACTIONS
        assert 'right' in config.ACTIONS
        assert 'start' in config.ACTIONS
        assert 'select' in config.ACTIONS

    def test_frameskip_map_contains_all_states(self):
        """Test that frameskip_map has entries for all game states."""
        config = Config()

        required_states = ['dialogue', 'battle', 'menu', 'exploring', 'base']
        for state in required_states:
            assert state in config.FRAMESKIP_MAP
            assert config.FRAMESKIP_MAP[state] > 0
