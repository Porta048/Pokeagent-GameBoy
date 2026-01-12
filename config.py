from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_device() -> Any:
    try:
        import torch
    except Exception:
        return "cpu"
    return torch.device(
        "mps"
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


@dataclass
class Config:
    ROM_PATH: str = "roms/Pokemon Red.gb"
    HEADLESS: bool = False
    EMULATION_SPEED: int = 0
    RENDER_ENABLED: bool = True
    RENDER_EVERY_N_FRAMES: int = 2
    DEVICE: Any = field(default_factory=_default_device)
    SAVE_DIR_PREFIX: str = "pokemon_ai_saves"
    MODEL_FILENAME: str = "model_bc.pth"
    RECORDINGS_DIR: str = "recordings"
    KNOWLEDGE_BASE_FILE: str = "data/knowledge_base.json"
    LOG_FILE: str = "pokemon_ai.log"
    LOG_LEVEL: str = "INFO"

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

    BC_LEARNING_RATE: float = 1e-4
    BC_BATCH_SIZE: int = 64
    BC_EPOCHS: int = 50
    BC_VAL_SPLIT: float = 0.1
    INFERENCE_TEMPERATURE: float = 0.5

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
        self._validate_device()

    def _validate_paths(self) -> None:
        if self.SAVE_DIR_PREFIX:
            Path(self.SAVE_DIR_PREFIX).mkdir(parents=True, exist_ok=True)
        Path(self.RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)

    def _validate_device(self) -> None:
        try:
            import torch
        except Exception:
            return
        if not isinstance(self.DEVICE, torch.device):
            raise TypeError(f"DEVICE must be torch.device, got {type(self.DEVICE)}")

    def get_save_dir(self, rom_path: str) -> Path:
        rom_name = Path(rom_path).stem
        save_dir = Path(f"{self.SAVE_DIR_PREFIX}_{rom_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def get_model_path(self, rom_path: str) -> Path:
        return self.get_save_dir(rom_path) / self.MODEL_FILENAME


config = Config()
