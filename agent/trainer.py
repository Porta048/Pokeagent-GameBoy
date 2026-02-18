import logging
import time
import threading
import signal
import atexit
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from config import config as CFG
from agent.environment import PokemonEnv
from agent.model import BCAgent, GameplayDataset, GameplayRecorder

logger = logging.getLogger("pokeagent.trainer")

_active_recorder = None

def _save_on_exit(signum=None, frame=None):
    global _active_recorder
    if _active_recorder is not None:
        logger.info("Saving recordings before exit...")
        _active_recorder.save()
        _active_recorder = None
    if signum is not None:
        exit(0)


class BCTrainer:
    def __init__(self, rom_path: Optional[str] = None, load_checkpoint: bool = True):
        self.rom_path = rom_path or CFG.ROM_PATH
        self.agent = BCAgent(n_actions=len(CFG.ACTIONS), device=CFG.DEVICE)
        self.model_path = CFG.get_model_path(self.rom_path)
        if load_checkpoint and self.model_path.exists():
            self.agent.load(str(self.model_path))
        logger.info("BCTrainer initialized: device=%s", CFG.DEVICE)

    def train(self, data_dir: str, epochs: int = 50, batch_size: int = 64, val_split: float = 0.1):
        dataset = GameplayDataset(data_dir)
        if len(dataset) == 0:
            logger.error("No training data found in %s", data_dir)
            return
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        logger.info("Training: %d samples, Validation: %d samples", train_size, val_size)
        best_val_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0
            self.agent.policy.train()
            for obs_batch, action_batch in train_loader:
                stats = self.agent.train_step(obs_batch, action_batch)
                epoch_loss += stats["loss"]
                epoch_acc += stats["accuracy"]
                n_batches += 1
            avg_loss = epoch_loss / max(1, n_batches)
            avg_acc = epoch_acc / max(1, n_batches)
            val_stats = self.agent.validate(val_loader)
            logger.info(
                "Epoch %d/%d: loss=%.4f, acc=%.2f%%, val_loss=%.4f, val_acc=%.2f%%",
                epoch + 1, epochs, avg_loss, avg_acc * 100, val_stats["val_loss"], val_stats["val_accuracy"] * 100
            )
            if val_stats["val_loss"] < best_val_loss:
                best_val_loss = val_stats["val_loss"]
                self.agent.save(str(self.model_path))
                logger.info("New best model saved (val_loss=%.4f)", best_val_loss)
        logger.info("Training complete. Best val_loss: %.4f", best_val_loss)

    def close(self):
        pass


class HumanRecorder:
    def __init__(self, rom_path: Optional[str] = None):
        self.rom_path = rom_path or CFG.ROM_PATH
        self.recorder = GameplayRecorder(CFG.RECORDINGS_DIR)
        self.emulator = None
        self.running = True
        logger.info("HumanRecorder initialized")

    def record_session(self):
        global _active_recorder
        from pyboy import PyBoy
        from pyboy.utils import WindowEvent
        from PIL import Image
        from pathlib import Path

        _active_recorder = self.recorder
        signal.signal(signal.SIGINT, _save_on_exit)
        signal.signal(signal.SIGTERM, _save_on_exit)
        atexit.register(_save_on_exit)

        self.emulator = PyBoy(self.rom_path, window="SDL2")
        self.emulator.set_emulation_speed(1)

        state_path = Path(self.rom_path + ".state")
        if state_path.exists():
            with open(state_path, "rb") as f:
                self.emulator.load_state(f)
            logger.info("Save state loaded from %s", state_path)

        logger.info("=" * 60)
        logger.info("RECORDING MODE - All keypresses are recorded!")
        logger.info("=" * 60)
        logger.info("Controls:")
        logger.info("  Arrow Keys = Move (UP/DOWN/LEFT/RIGHT)")
        logger.info("  Z = A button (confirm/interact)")
        logger.info("  X = B button (cancel/run)")
        logger.info("  Enter = START (menu/pause)")
        logger.info("  Backspace = SELECT")
        logger.info("=" * 60)
        logger.info("Play the game! Recording EVERY keypress...")
        logger.info("Close the PyBoy window to save and exit")
        logger.info("=" * 60)

        key_to_action = {
            WindowEvent.PRESS_ARROW_UP: 1,
            WindowEvent.PRESS_ARROW_DOWN: 2,
            WindowEvent.PRESS_ARROW_LEFT: 3,
            WindowEvent.PRESS_ARROW_RIGHT: 4,
            WindowEvent.PRESS_BUTTON_A: 5,
            WindowEvent.PRESS_BUTTON_B: 6,
            WindowEvent.PRESS_BUTTON_START: 7,
            WindowEvent.PRESS_BUTTON_SELECT: 8,
        }

        action_names = ['none', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']
        total_recorded = 0

        while self.running:
            try:
                still_running = self.emulator.tick(render=True)
                if not still_running:
                    break
            except Exception:
                break

            for event in self.emulator.events:
                if event in key_to_action:
                    action = key_to_action[event]

                    screenshot = self.emulator.screen.image
                    gray = screenshot.convert('L')
                    gray = gray.resize((84, 84), Image.BILINEAR)
                    obs = np.array(gray, dtype=np.uint8).reshape(84, 84, 1)

                    self.recorder.record(obs, action)
                    total_recorded += 1

                    current_map = self.emulator.memory[0xD35E]
                    current_x = self.emulator.memory[0xD362]
                    current_y = self.emulator.memory[0xD361]
                    in_battle = self.emulator.memory[0xD057] > 0

                    if total_recorded % 50 == 0:
                        battle_str = " [BATTLE]" if in_battle else ""
                        logger.info(
                            "Recorded: %d | Map: %d | Pos: (%d, %d) | %s%s",
                            total_recorded, current_map, current_x, current_y,
                            action_names[action], battle_str
                        )

        self.recorder.save()
        _active_recorder = None
        stats = self.recorder.get_stats()
        logger.info("=" * 60)
        logger.info("RECORDING COMPLETE!")
        logger.info("Total samples saved: %d", stats["total_recorded"])
        logger.info("Files created: %d", stats["files_saved"] + (1 if stats["current_buffer"] > 0 else 0))
        logger.info("=" * 60)

        try:
            self.emulator.stop()
        except Exception:
            pass

    def close(self):
        global _active_recorder
        self.running = False
        if self.recorder:
            self.recorder.save()
            _active_recorder = None
        if self.emulator:
            try:
                self.emulator.stop()
            except Exception:
                pass


class InferenceRunner:
    def __init__(self, rom_path: Optional[str] = None):
        self.rom_path = rom_path or CFG.ROM_PATH
        self.env = PokemonEnv(rom_path=self.rom_path, render=True)
        self.agent = BCAgent(n_actions=len(CFG.ACTIONS), device=CFG.DEVICE)
        model_path = CFG.get_model_path(self.rom_path)
        if model_path.exists():
            self.agent.load(str(model_path))
        else:
            logger.warning("No model found at %s", model_path)
        logger.info("InferenceRunner initialized")

    def play(self, num_steps: int = 10000, temperature: float = 0.5):
        obs, info = self.env.reset()
        logger.info("Playing for %d steps (temperature=%.2f)", num_steps, temperature)
        for step in range(num_steps):
            action = self.agent.select_action(obs, temperature)
            obs, info = self.env.step(action)
            if step % 1000 == 0:
                logger.info("Step %d: badges=%d, location=%s", step, info.get("badges", 0), info.get("location", ""))
        self.env.close()

    def close(self):
        self.env.close()
