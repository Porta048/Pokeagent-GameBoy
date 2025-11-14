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
        assert config.rom_path == "pokemon_red.gb"
        assert config.frame_stack_size == 4
        assert config.save_frequency > 0
        assert isinstance(config.device, torch.device)

    def test_validation_emulation_speed_negative(self):
        """Test that negative emulation speed raises ValueError."""
        with pytest.raises(ValueError, match="emulation_speed must be >= 0"):
            Config(emulation_speed=-1)

    def test_validation_frame_stack_size_zero(self):
        """Test that zero frame stack size raises ValueError."""
        with pytest.raises(ValueError, match="frame_stack_size must be >= 1"):
            Config(frame_stack_size=0)

    def test_validation_menu_threshold_out_of_range(self):
        """Test that menu_threshold outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="menu_threshold must be in"):
            Config(menu_threshold=1.5)

        with pytest.raises(ValueError, match="menu_threshold must be in"):
            Config(menu_threshold=-0.1)

    def test_get_save_dir_creates_directory(self, tmp_path):
        """Test that get_save_dir creates the directory."""
        config = Config(save_dir_prefix=str(tmp_path / "test_saves"))
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

        assert None in config.actions  # No-op
        assert 'a' in config.actions
        assert 'b' in config.actions
        assert 'up' in config.actions
        assert 'down' in config.actions
        assert 'left' in config.actions
        assert 'right' in config.actions
        assert 'start' in config.actions
        assert 'select' in config.actions

    def test_frameskip_map_contains_all_states(self):
        """Test that frameskip_map has entries for all game states."""
        config = Config()

        required_states = ['dialogue', 'battle', 'menu', 'exploring', 'base']
        for state in required_states:
            assert state in config.frameskip_map
            assert config.frameskip_map[state] > 0
