import io
from pathlib import Path

import gymnasium as gym
import numpy as np
from pyboy import PyBoy

from agent.game_state import GameState, read_state

ACTIONS = ("down", "left", "right", "up", "a", "b")
PRESS_FRAMES = 8
ACTION_FRAMES = 24
FRAME_STACK = 3
SCREEN_SHAPE = (72, 80)
# Finestra di tile visibili attorno al giocatore (9 righe x 10 colonne, giocatore in (4, 4))
VIEW_ROWS = 9
VIEW_COLS = 10
PLAYER_ROW = 4
PLAYER_COL = 4
NUM_BADGES = 8
# hp, livelli, dimensione squadra, in battaglia + una flag per medaglia
NUM_STATS = 4 + NUM_BADGES

REWARD_TILE = 0.02
REWARD_LEVEL = 1.0
REWARD_BADGE = 10.0
REWARD_EVENT = 1.0
REWARD_HEAL = 0.5
# Oltre questa soglia i livelli valgono meno, per scoraggiare il grinding
LEVEL_SOFT_CAP = 15


def heal_amount(prev: GameState, cur: GameState) -> float:
    # Cura (es. Centro Pokemon) solo a squadra invariata, senza level up e non dopo un KO totale
    if sum(prev.party_hp) == 0 or len(prev.party_hp) != len(cur.party_hp) or sum(cur.party_levels) != sum(prev.party_levels):
        return 0.0
    gain = sum(cur.party_hp) - sum(prev.party_hp)
    return gain / max(1, sum(cur.party_max_hp)) if gain > 0 else 0.0


def score(state: GameState, tiles: int, events: int, heal_total: float) -> float:
    levels = sum(state.party_levels)
    level_score = min(levels, LEVEL_SOFT_CAP) + max(0, levels - LEVEL_SOFT_CAP) / 4
    return (
        REWARD_TILE * tiles
        + REWARD_LEVEL * level_score
        + REWARD_BADGE * state.badges
        + REWARD_EVENT * events
        + REWARD_HEAL * heal_total
    )


class PokemonRedEnv(gym.Env):
    def __init__(self, rom_path: str, max_steps: int = 20480, headless: bool = True) -> None:
        self.rom_path = rom_path
        self.max_steps = max_steps
        self.pyboy = PyBoy(rom_path, window="null" if headless else "SDL2", sound_emulated=False)
        self.pyboy.set_emulation_speed(0 if headless else 1)
        self.init_state = Path(rom_path + ".state").read_bytes()

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "screen": gym.spaces.Box(0, 255, (FRAME_STACK, *SCREEN_SHAPE), np.uint8),
            "visited": gym.spaces.Box(0.0, 1.0, (VIEW_ROWS * VIEW_COLS,), np.float32),
            "stats": gym.spaces.Box(0.0, 1.0, (NUM_STATS,), np.float32),
        })

        self.frames = np.zeros((FRAME_STACK, *SCREEN_SHAPE), np.uint8)
        self.visited: set[tuple[int, int, int]] = set()
        self.steps = 0
        self.score = 0.0
        self.base_events = 0
        self.max_events = 0
        self.heal_total = 0.0
        self.state = read_state(self.pyboy)

    def reset(self, *, seed: int | None = None, options: dict[str, int] | None = None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        super().reset(seed=seed)
        self.pyboy.load_state(io.BytesIO(self.init_state))
        self.pyboy.tick(1, True)
        self.state = read_state(self.pyboy)
        self.frames[:] = self._screen()
        self.visited = {(self.state.map_id, self.state.x, self.state.y)}
        self.steps = 0
        self.base_events = self.state.event_flags
        self.max_events = 0
        self.heal_total = 0.0
        self.score = score(self.state, len(self.visited), self.max_events, self.heal_total)
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, float]]:
        self.pyboy.button(ACTIONS[action], PRESS_FRAMES)
        self.pyboy.tick(ACTION_FRAMES - 1, False)
        # tick restituisce False se la finestra e' stata chiusa
        alive = self.pyboy.tick(1, True)
        prev = self.state
        self.state = read_state(self.pyboy)
        self.steps += 1

        self.frames = np.roll(self.frames, 1, axis=0)
        self.frames[0] = self._screen()
        self.visited.add((self.state.map_id, self.state.x, self.state.y))
        self.max_events = max(self.max_events, self.state.event_flags - self.base_events)
        self.heal_total += heal_amount(prev, self.state)

        new_score = score(self.state, len(self.visited), self.max_events, self.heal_total)
        reward = new_score - self.score
        self.score = new_score
        truncated = self.steps >= self.max_steps or not alive
        return self._obs(), reward, False, truncated, self._info()

    def _screen(self) -> np.ndarray:
        return self.pyboy.screen.ndarray[::2, ::2, 0]

    def _obs(self) -> dict[str, np.ndarray]:
        s = self.state
        visited = np.array([
            (s.map_id, s.x + c - PLAYER_COL, s.y + r - PLAYER_ROW) in self.visited
            for r in range(VIEW_ROWS) for c in range(VIEW_COLS)
        ], np.float32)
        hp = sum(s.party_hp) / max(1, sum(s.party_max_hp))
        badges = [(s.badges > i) * 1.0 for i in range(NUM_BADGES)]
        stats = np.array([hp, min(sum(s.party_levels) / 100, 1.0), len(s.party_hp) / 6, min(s.in_battle, 1), *badges], np.float32)
        return {"screen": self.frames.copy(), "visited": visited, "stats": stats}

    def _info(self) -> dict[str, float]:
        s = self.state
        return {
            "score": self.score,
            "tiles": len(self.visited),
            "levels": sum(s.party_levels),
            "badges": s.badges,
            "events": self.max_events,
            "dex": s.dex_owned,
            "map_id": s.map_id,
        }

    def close(self) -> None:
        self.pyboy.stop(save=False)
