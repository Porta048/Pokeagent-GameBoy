import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import config as CFG
from agent.emulator import EmulatorHarness

logger = logging.getLogger("pokeagent.environment")


class PokemonEnv:
    def __init__(self, rom_path: Optional[str] = None, render: bool = True):
        self.rom_path = rom_path or CFG.ROM_PATH
        self.render = render
        self.emulator: Optional[EmulatorHarness] = None
        self._step_count = 0

    def _process_state(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state = self.emulator.get_current_state()
        obs = self._screen_to_obs(state["visual_np"])
        ram = state.get("ram", {})
        parsed = state.get("parsed", {})
        info = {
            "badges": parsed.get("badges", 0),
            "money": parsed.get("money", 0),
            "party_count": parsed.get("party_count", 0),
            "location": parsed.get("location", ""),
            "in_battle": parsed.get("in_battle", False),
            "current_map": ram.get("current_map", 0),
            "player_x": ram.get("player_x", 0),
            "player_y": ram.get("player_y", 0),
            "step_count": self._step_count,
        }
        return obs, info

    @staticmethod
    def _screen_to_obs(screen: np.ndarray) -> np.ndarray:
        from PIL import Image
        if len(screen.shape) == 3:
            gray = (0.299 * screen[:, :, 0] + 0.587 * screen[:, :, 1] + 0.114 * screen[:, :, 2]).astype(np.uint8)
        else:
            gray = screen
        img = Image.fromarray(gray)
        img = img.resize((84, 84), Image.BILINEAR)
        return np.array(img, dtype=np.uint8).reshape(84, 84, 1)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.emulator is not None:
            self.emulator.close()
        self.emulator = EmulatorHarness(self.rom_path)
        self._step_count = 0
        return self._process_state()

    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step_count += 1
        action_name = CFG.ACTIONS[action]
        if action_name is not None:
            self.emulator.press_button(action_name.upper())
        return self._process_state()

    def close(self):
        if self.emulator is not None:
            self.emulator.close()
            self.emulator = None
