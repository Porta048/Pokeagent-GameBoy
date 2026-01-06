from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch

@dataclass
class Config:
    ROM_PATH: str = "roms/Pokemon Red.gb"
    HEADLESS: bool = False
    EMULATION_SPEED: int = 1
    RENDER_ENABLED: bool = True
    RENDER_EVERY_N_FRAMES: int = 2
    DEVICE: torch.device = field(default_factory=lambda: torch.device("mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")))
    SAVE_DIR_PREFIX: str = "pokemon_ai_saves"
    MODEL_FILENAME: str = "model_ppo.pth"
    STATS_FILENAME: str = "stats_ppo.json"
    GAME_STATE_FILENAME: str = "game_state.state"
    KNOWLEDGE_BASE_FILE: str = "data/knowledge_base.json"
    FRAME_STACK_SIZE: int = 4
    SAVE_FREQUENCY: int = 10000
    PERFORMANCE_LOG_INTERVAL: int = 10000
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
        "battle": 10,
        "menu": 6,
        "exploring": 8,
        "base": 6
    })
    HP_THRESHOLD: int = 500
    MENU_THRESHOLD: float = 0.15
    DIALOGUE_THRESHOLD: int = 30
    ANTI_LOOP_ENABLED: bool = True
    LOG_FILE: str = "pokemon_ai.log"
    LOG_LEVEL: str = "INFO"
    ENABLE_PATHFINDER: bool = False
    CRITIQUE_ENABLED: bool = True
    LONG_TERM_MEMORY_PATH: str = "chroma_db"
    MEMORY_SUMMARIZE_EVERY_N_STEPS: int = 50
    PLANNER_PROMPT_TEMPLATE: str = (
        "Sei un assistente che pianifica obiettivi a breve termine per Pokémon.\n"
        "Stato di gioco: {game_state}\n"
        "Obiettivo a lungo termine: {long_term_goal}\n"
        "Eventi recenti:\n{recent_history}\n"
        "Fatti rilevanti:\n{relevant_facts}\n"
        "Proponi un singolo prossimo obiettivo chiaro e fattibile."
    )

    # LLM Integration
    LLM_ENABLED: bool = True
    LLM_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:0.5b"
    LLM_TEMPERATURE: float = 0.3
    LLM_TIMEOUT: float = 60.0
    LLM_MIN_INTERVAL_MS: int = 50
    LLM_MAX_CALLS_PER_MINUTE: int = 600
    LLM_CACHE_TTL_SECONDS: int = 20
    LLM_USE_VISION: bool = False
    LLM_USE_TOKEN_BUCKET: bool = True
    LLM_USE_FAST_CACHE_KEY: bool = True
    LLM_USE_FOR_EXPLORATION: bool = True
    LLM_USE_FOR_BATTLE: bool = True
    LLM_USE_FOR_MENU: bool = True
    LLM_FALLBACK_ON_ERROR: bool = True
    LLM_RETRY_ATTEMPTS: int = 2
    LLM_CONSECUTIVE_FAILURE_THRESHOLD: int = 3
    LLM_FAILURE_COOLDOWN_SECONDS: int = 60
    
    RAM_OFFSETS: Dict[str, int] = field(default_factory=lambda: {
        "player_x": 0xD362,
        "player_y": 0xD361,
        "current_map": 0xD35E,
        "in_battle_flag": 0xD057,
        "menu_flag": 0xD4E0,
        "dialog_state": 0xD730,
        "money": 0xD347,
        "badges": 0xD356,
        "pokedex_owned": 0xD2F7,
        "pokedex_seen": 0xD30A,
        "party_count": 0xD163,
        "party_species": 0xD164,
        "party_levels": 0xD18C,
        "party_hp_current": 0xD16B,
        "party_hp_max": 0xD18D,
        "opponent_species": 0xCFE5,
        "opponent_level": 0xD8C4,
        "opponent_hp_current": 0xCFE6,
        "opponent_hp_max": 0xD8B5
    })
    
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

config = Config()
