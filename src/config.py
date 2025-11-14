"""Gestione configurazione per Pokemon AI Agent."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch


@dataclass
class Config:
    """Configurazione per training Pokemon AI con validazione."""

    # Impostazioni ROM
    rom_path: str = "pokemon_red.gb"

    # Impostazioni emulatore
    headless: bool = False
    emulation_speed: int = 0  # 0 = illimitata, 1 = velocità normale, 2 = 2x, ecc.
    render_enabled: bool = True
    render_every_n_frames: int = 2

    # Impostazioni agente AI
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir_prefix: str = "pokemon_ai_saves"
    model_filename: str = "model_ppo.pth"
    stats_filename: str = "stats_ppo.json"
    game_state_filename: str = "game_state.state"

    # Impostazioni rete neurale
    frame_stack_size: int = 4

    # Impostazioni training
    save_frequency: int = 10000
    performance_log_interval: int = 1000

    # Azioni disponibili per l'agente
    actions: List[Optional[str]] = field(default_factory=lambda: [
        None,      # No-op
        'up',      # Freccia su
        'down',    # Freccia giù
        'left',    # Freccia sinistra
        'right',   # Freccia destra
        'a',       # Pulsante A
        'b',       # Pulsante B
        'start',   # Pulsante Start
        'select'   # Pulsante Select
    ])

    # Mappa frameskip adattivo
    frameskip_map: Dict[str, int] = field(default_factory=lambda: {
        "dialogue": 6,
        "battle": 12,
        "menu": 8,
        "exploring": 10,
        "base": 8
    })

    # Soglie per rilevamento stato gioco
    hp_threshold: int = 500
    menu_threshold: float = 0.15
    dialogue_threshold: int = 30

    # Impostazioni anti-loop
    anti_loop_enabled: bool = False

    # Impostazioni logging
    log_file: str = "pokemon_ai.log"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Valida configurazione dopo inizializzazione."""
        self._validate_paths()
        self._validate_ranges()
        self._validate_device()

    def _validate_paths(self) -> None:
        """Valida percorsi file e crea directory se necessario."""
        if self.save_dir_prefix:
            Path(self.save_dir_prefix).mkdir(parents=True, exist_ok=True)

    def _validate_ranges(self) -> None:
        """Valida range numerici."""
        if self.emulation_speed < 0:
            raise ValueError(f"emulation_speed deve essere >= 0, ottenuto {self.emulation_speed}")

        if self.frame_stack_size < 1:
            raise ValueError(f"frame_stack_size deve essere >= 1, ottenuto {self.frame_stack_size}")

        if self.render_every_n_frames < 1:
            raise ValueError(f"render_every_n_frames deve essere >= 1, ottenuto {self.render_every_n_frames}")

        if self.save_frequency < 1:
            raise ValueError(f"save_frequency deve essere >= 1, ottenuto {self.save_frequency}")

        if self.performance_log_interval < 1:
            raise ValueError(f"performance_log_interval deve essere >= 1, ottenuto {self.performance_log_interval}")

        if not 0.0 <= self.menu_threshold <= 1.0:
            raise ValueError(f"menu_threshold deve essere in [0.0, 1.0], ottenuto {self.menu_threshold}")

    def _validate_device(self) -> None:
        """Valida e logga selezione device."""
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device deve essere torch.device, ottenuto {type(self.device)}")

    def get_save_dir(self, rom_path: str) -> Path:
        """Ottieni directory salvataggio per ROM specifica."""
        rom_name = Path(rom_path).stem
        save_dir = Path(f"{self.save_dir_prefix}_{rom_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def get_model_path(self, rom_path: str) -> Path:
        """Ottieni percorso checkpoint modello per ROM specifica."""
        return self.get_save_dir(rom_path) / self.model_filename

    def get_stats_path(self, rom_path: str) -> Path:
        """Ottieni percorso file statistiche per ROM specifica."""
        return self.get_save_dir(rom_path) / self.stats_filename

    def get_game_state_path(self, rom_path: str) -> Path:
        """Ottieni percorso file stato gioco per ROM specifica."""
        return self.get_save_dir(rom_path) / self.game_state_filename

    @classmethod
    def from_cli_args(cls, args) -> 'Config':
        """Crea Config da argomenti linea di comando."""
        config = cls()

        if hasattr(args, 'rom_path') and args.rom_path:
            config.rom_path = args.rom_path
        if hasattr(args, 'headless') and args.headless is not None:
            config.headless = args.headless
        if hasattr(args, 'speed') and args.speed is not None:
            config.emulation_speed = args.speed
        if hasattr(args, 'log_level') and args.log_level:
            config.log_level = args.log_level.upper()

        return config


# Istanza configurazione globale (per retrocompatibilità)
config = Config()
