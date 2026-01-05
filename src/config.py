import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
@dataclass
class Config:
    ROM_PATH: str = r"C:\Users\chatg\Documents\GitHub\Pokemon Red.gb"
    HEADLESS: bool = False
    EMULATION_SPEED: int = 0  
    RENDER_ENABLED: bool = True
    RENDER_EVERY_N_FRAMES: int = 2
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    SAVE_DIR_PREFIX: str = "pokemon_ai_saves"
    MODEL_FILENAME: str = "model_ppo.pth"
    STATS_FILENAME: str = "stats_ppo.json"
    GAME_STATE_FILENAME: str = "game_state.state"
    FRAME_STACK_SIZE: int = 4
    SAVE_FREQUENCY: int = 10000
    PERFORMANCE_LOG_INTERVAL: int = 1000
    ACTIONS: List[Optional[str]] = field(default_factory=lambda: [
        None,      
        'up',      
        'down',    
        'left',    
        'right',   
        'a',       
        'b',       
        'start',   
        'select'   
    ])
    FRAMESKIP_MAP: Dict[str, int] = field(default_factory=lambda: {
        "dialogue": 6,
        "battle": 12,
        "menu": 8,
        "exploring": 10,
        "base": 8
    })
    HP_THRESHOLD: int = 500
    MENU_THRESHOLD: float = 0.15
    DIALOGUE_THRESHOLD: int = 30
    ANTI_LOOP_ENABLED: bool = True  # Enable anti-loop with menu spam detection
    LOG_FILE: str = "pokemon_ai.log"
    LOG_LEVEL: str = "INFO"

    # LLM Integration (Ollama + ministral-3b)
    LLM_ENABLED: bool = True
    LLM_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "ministral-3b:latest"
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT: float = 5.0
    LLM_MIN_INTERVAL_MS: int = 500
    LLM_USE_VISION: bool = True
    LLM_USE_FOR_EXPLORATION: bool = True
    LLM_USE_FOR_BATTLE: bool = True
    LLM_USE_FOR_MENU: bool = False
    def __post_init__(self) -> None:
        self._validate_paths()
        self._validate_ranges()
        self._validate_device()
    def _validate_paths(self) -> None:
        if self.SAVE_DIR_PREFIX:
            Path(self.SAVE_DIR_PREFIX).mkdir(parents=True, exist_ok=True)
    def _validate_ranges(self) -> None:
        if self.EMULATION_SPEED < 0:
            raise ValueError(f"EMULATION_SPEED deve essere >= 0, ottenuto {self.EMULATION_SPEED}")
        if self.FRAME_STACK_SIZE < 1:
            raise ValueError(f"FRAME_STACK_SIZE deve essere >= 1, ottenuto {self.FRAME_STACK_SIZE}")
        if self.RENDER_EVERY_N_FRAMES < 1:
            raise ValueError(f"RENDER_EVERY_N_FRAMES deve essere >= 1, ottenuto {self.RENDER_EVERY_N_FRAMES}")
        if self.SAVE_FREQUENCY < 1:
            raise ValueError(f"SAVE_FREQUENCY deve essere >= 1, ottenuto {self.SAVE_FREQUENCY}")
        if self.PERFORMANCE_LOG_INTERVAL < 1:
            raise ValueError(f"PERFORMANCE_LOG_INTERVAL deve essere >= 1, ottenuto {self.PERFORMANCE_LOG_INTERVAL}")
        if not 0.0 <= self.MENU_THRESHOLD <= 1.0:
            raise ValueError(f"MENU_THRESHOLD deve essere in [0.0, 1.0], ottenuto {self.MENU_THRESHOLD}")
    def _validate_device(self) -> None:
        if not isinstance(self.DEVICE, torch.device):
            raise TypeError(f"DEVICE deve essere torch.device, ottenuto {type(self.DEVICE)}")
    def get_save_dir(self, rom_path: str) -> Path:
        rom_name = Path(rom_path).stem
        save_dir = Path(f"{self.SAVE_DIR_PREFIX}_{rom_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
    def get_model_path(self, rom_path: str) -> Path:
        return self.get_save_dir(rom_path) / self.MODEL_FILENAME
    def get_stats_path(self, rom_path: str) -> Path:
        return self.get_save_dir(rom_path) / self.STATS_FILENAME
    def get_game_state_path(self, rom_path: str) -> Path:
        return self.get_save_dir(rom_path) / self.GAME_STATE_FILENAME
    @classmethod
    def from_cli_args(cls, args) -> 'Config':
        config = cls()
        if hasattr(args, 'rom_path') and args.rom_path:
            config.ROM_PATH = args.rom_path
        if hasattr(args, 'headless') and args.headless is not None:
            config.HEADLESS = args.headless
        if hasattr(args, 'speed') and args.speed is not None:
            config.EMULATION_SPEED = args.speed
        if hasattr(args, 'log_level') and args.log_level:
            config.LOG_LEVEL = args.log_level.upper()
        return config
config = Config()
