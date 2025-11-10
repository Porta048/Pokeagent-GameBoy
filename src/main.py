import os
import sys
import time
import threading
import json
import hashlib
import logging
from typing import Union, List, Dict, Optional, Tuple, Any
from collections import deque
from functools import lru_cache
import numpy as np
import torch

try:
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not found. Install with: pip install torch")

try:
    from pyboy import PyBoy
    import keyboard
    from PIL import Image
    import cv2
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"WARNING: Missing dependencies: {e}")

from config import config
from memory_reader import GameMemoryReader
from state_detector import GameStateDetector
from utils import AsyncSaver, FrameStack, ImageCache, EXPLORATION_CONV_DIMENSIONS, MENU_CONV_DIMENSIONS
from hyperparameters import HYPERPARAMETERS
from errors import PokemonAIError, ROMLoadError, MemoryReadError, CheckpointLoadError, GameEnvironmentError
from anti_loop import AdaptiveEntropyScheduler, AntiLoopMemoryBuffer
from action_filter import ContextAwareActionFilter
from trajectory_buffer import TrajectoryBuffer
from models import PPONetworkGroup
from screen_regions import SCREEN_REGIONS


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pokemon_ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PokemonAIAgent:
    """Pokemon AI Agent with PPO."""
    def __init__(self, rom_path: str, headless: bool = False) -> None:
        if not TORCH_AVAILABLE or not DEPS_AVAILABLE:
            raise PokemonAIError("Missing dependencies. Install: torch, pyboy, keyboard, PIL, cv2")

        self.device = config.DEVICE
        print(f"Using device: {self.device}")

        # Support for parallel training
        self.use_shared_buffer = False  # Set to True by parallel_trainer
        self.shared_buffer = None
        self.rank = 0  # Worker rank (0 = visible, >0 = headless)

        # PyBoy with optimized window
        window_type = "headless" if headless else "SDL2"
        try:
            self.pyboy = PyBoy(rom_path, window=window_type)
        except Exception as e:
            raise ROMLoadError(f"Failed to load ROM file '{rom_path}': {str(e)}") from e

        # Set emulation speed (0=unlimited, 1=normal, 2=2x, etc.)
        emulation_speed = config.EMULATION_SPEED
        self.pyboy.set_emulation_speed(emulation_speed)
        speed_desc = "unlimited" if emulation_speed == 0 else f"{emulation_speed}x"
        print(f"[INFO] Emulation speed: {speed_desc}")

        # PyBoy 2.6.0+ uses string-based API for buttons
        self.actions = config.ACTIONS

        rom_name = os.path.splitext(os.path.basename(rom_path))[0]
        self.save_dir = f"{config.SAVE_DIR_PREFIX}_{rom_name}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_path = os.path.join(self.save_dir, config.MODEL_FILENAME)
        self.stats_path = os.path.join(self.save_dir, config.STATS_FILENAME)
        self.game_state_path = os.path.join(self.save_dir, config.GAME_STATE_FILENAME)

        input_channels = config.FRAME_STACK_SIZE
        self.network_group = PPONetworkGroup(len(self.actions), self.device, input_channels)

        self.trajectory_buffer = TrajectoryBuffer()
        self.frame_stack = FrameStack(config.FRAME_STACK_SIZE)
        self.image_cache = ImageCache()
        self.memory_reader = GameMemoryReader(self.pyboy)
        self.state_detector = GameStateDetector()
        self.async_saver = AsyncSaver()

        # Anti-Confusion System Components
        self.entropy_scheduler = AdaptiveEntropyScheduler()
        self.anti_loop_buffer = AntiLoopMemoryBuffer()
        self.action_filter = ContextAwareActionFilter()

        self.current_game_state = "exploring"
        self.last_screen_array = None

        self.stats = self._load_stats()
        self.episode_count = 0
        self.frame_count = 0
        self.total_reward = 0
        self.reward_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=100)

        self._load_checkpoint()

    def _load_stats(self):
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load stats file: {str(e)}, starting with default stats")
                return {'episodes': 0, 'total_frames': 0, 'best_reward': float('-inf')}
        return {'episodes': 0, 'total_frames': 0, 'best_reward': float('-inf')}

    def _save_stats(self):
        try:
            final_state = self.memory_reader.get_current_state()
            if final_state:
                self.stats['final_state'] = final_state

            self.stats.update({
                'episodes': self.episode_count,
                'total_frames': self.stats.get('total_frames', 0) + self.frame_count,
                'best_reward': max(self.stats.get('best_reward', float('-inf')), self.total_reward)
            })

            if len(self.reward_history) > 0:
                self.stats['avg_reward_last_1000'] = float(np.mean(list(self.reward_history)))

            with open(self.stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save stats file: {str(e)}")

    def _load_checkpoint(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.network_group.exploration_network.load_state_dict(checkpoint['explorer_state'])
                self.network_group.battle_network.load_state_dict(checkpoint['battle_state'])
                self.network_group.menu_network.load_state_dict(checkpoint['menu_state'])
                self.episode_count = checkpoint.get('episode', 0)
                self.frame_count = checkpoint.get('frame', 0)
                print(f"[LOAD] Checkpoint loaded: episode {self.episode_count}, frame {self.frame_count}")

                # Load game state if exists
                if os.path.exists(self.game_state_path):
                    try:
                        with open(self.game_state_path, 'rb') as f:
                            self.pyboy.load_state(f)
                        print(f"[LOAD] Game state loaded from {self.game_state_path}")
                    except Exception as e:
                        logger.warning(f"Unable to load game state: {e}")
                        print("[INFO] Starting from beginning of game")
            except Exception as e:
                import logging
                logging.error(f"Error loading checkpoint: {e}")
                raise PokemonAIError(f"Failed to load checkpoint: {e}") from e

    def _save_checkpoint(self):
        checkpoint = {
            'explorer_state': self.network_group.exploration_network.state_dict(),
            'battle_state': self.network_group.battle_network.state_dict(),
            'menu_state': self.network_group.menu_network.state_dict(),
            'episode': self.episode_count,
            'frame': self.frame_count
        }
        self.async_saver.save_async(lambda d: torch.save(d, self.model_path), checkpoint)

        # Save game state
        try:
            with open(self.game_state_path, 'wb') as f:
                self.pyboy.save_state(f)
        except Exception as e:
            logger.warning(f"Unable to save game state: {e}")

    def _get_screen_tensor(self):
        """Get preprocessed current frame."""
        try:
            screen = self.pyboy.screen.image  # PyBoy 2.6.0+ API
            gray = np.array(screen.convert('L'))
            self.last_screen_array = gray.copy()

            cached = self.image_cache.get(gray)
            if cached is not None:
                normalized = cached
            else:
                normalized = gray.astype(np.float32) / 255.0
                self.image_cache.save(gray, normalized)

            tensor = torch.from_numpy(normalized).unsqueeze(0)
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error getting screen tensor: {str(e)}")
            # Return a blank tensor as fallback
            blank_frame = torch.zeros((1, 144, 160), dtype=torch.float32, device=self.device)
            return blank_frame

    def _detect_game_state(self, screen_array):
        if self.state_detector.detect_battle(screen_array):
            self.current_game_state = "battle"
        elif self.state_detector.detect_dialogue(screen_array):
            self.current_game_state = "dialogue"
        elif self.state_detector.detect_menu(screen_array):
            self.current_game_state = "menu"
        else:
            self.current_game_state = "exploring"

        return self.current_game_state

    def _calculate_reward(self):
        """
        Calculate reward based ONLY on game progression (memory).
        No reward for generic movement - prevents premature convergence.
        """
        reward = 0

        try:
            memory_state = self.memory_reader.get_current_state()
            memory_reward = self.memory_reader.calculate_event_rewards(memory_state)
            reward += memory_reward

            # Anti-Loop System: DISABLED if flag is False
            if config.ANTI_LOOP_ENABLED:
                # Anti-Loop System: penalty for repetitive behaviors
                loop_penalty = self.anti_loop_buffer.calculate_loop_penalty()
                if loop_penalty < 0:
                    reward += loop_penalty
                    # Log when loop detected (only occasionally to avoid spam)
                    if self.frame_count % 5000 == 0:  # Every 5000 frames instead of 100
                        print(f"[ANTI-LOOP] Loop detected! Penalty: {loop_penalty:.2f}")

                # Exploration bonus: reward for active exploration
                exploration_bonus = self.anti_loop_buffer.get_exploration_bonus()
                if exploration_bonus > 0:
                    reward += exploration_bonus

            self.reward_history.append(reward)
            self.total_reward += reward
            return reward
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            # Return a small penalty when reward calculation fails to encourage stability
            return -1.0

    def start_training(self) -> None:
        """Main PPO training loop with smooth rendering."""
        paused = False
        last_save_frame = 0
        perf_start_time = time.time()
        perf_frame_count = 0

        initial_frame = self._get_screen_tensor()
        self.frame_stack.reset(initial_frame)

        render_counter = 0

        try:
            while True:
                if keyboard.is_pressed('escape'):
                    break
                if keyboard.is_pressed('space'):
                    paused = not paused
                    time.sleep(0.3)

                # Speed controls: + and - to increase/decrease
                if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                    current_speed = config.EMULATION_SPEED
                    config.EMULATION_SPEED = min(current_speed + 1, 10)
                    self.pyboy.set_emulation_speed(config.EMULATION_SPEED)
                    print(f"[INFO] Speed: {config.EMULATION_SPEED}x")
                    time.sleep(0.2)
                if keyboard.is_pressed('-') or keyboard.is_pressed('_'):
                    current_speed = config.EMULATION_SPEED
                    config.EMULATION_SPEED = max(current_speed - 1, 0)
                    self.pyboy.set_emulation_speed(config.EMULATION_SPEED)
                    speed_desc = "unlimited" if config.EMULATION_SPEED == 0 else f"{config.EMULATION_SPEED}x"
                    print(f"[INFO] Speed: {speed_desc}")
                    time.sleep(0.2)

                if paused:
                    self.pyboy.tick()
                    continue

                # Capture current state
                single_frame = self._get_screen_tensor()
                self.frame_stack.add(single_frame)
                stacked_state = self.frame_stack.get_stack()

                # Context-Aware Action Masking
                action_mask = self.action_filter.get_action_mask(self.current_game_state)

                # AI decides action to execute (with action masking)
                action, log_prob, value = self.network_group.choose_action(
                    stacked_state, self.current_game_state, action_mask=action_mask
                )

                # Execute action
                button = self.actions[action]
                if button is not None:
                    self.pyboy.button_press(button)

                # Adaptive frameskip based on game state
                frameskip = config.FRAMESKIP_MAP.get(self.current_game_state, config.FRAMESKIP_MAP["base"])

                # Smooth rendering without lag
                if config.RENDER_ENABLED:
                    # Render every N frames for smoothness
                    render_freq = config.RENDER_EVERY_N_FRAMES
                    for i in range(frameskip):
                        should_render = (i % render_freq == 0)
                        self.pyboy.tick(1, render=should_render)
                        render_counter += 1
                else:
                    # Headless mode: maximum speed
                    self.pyboy.tick(count=frameskip, render=False)

                # Release button
                if button is not None:
                    self.pyboy.button_release(button)

                # Capture next state
                next_single_frame = self._get_screen_tensor()

                # Update Anti-Loop Buffer only if enabled
                if config.ANTI_LOOP_ENABLED:
                    mem_state = self.memory_reader.get_current_state()
                    self.anti_loop_buffer.add_state(
                        mem_state.get('pos_x', 0),
                        mem_state.get('pos_y', 0),
                        mem_state.get('map_id', 0),
                        action
                    )

                reward = self._calculate_reward()

                # Performance monitoring
                perf_frame_count += 1
                if perf_frame_count % config.PERFORMANCE_LOG_INTERVAL == 0:
                    elapsed = time.time() - perf_start_time
                    fps = perf_frame_count / elapsed
                    avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0

                    # Calculate current entropy
                    current_entropy = self.entropy_scheduler.get_entropy(self.frame_count)

                    # Detailed game state
                    mem_state = self.memory_reader.get_current_state()
                    print(f"[PERF] {fps:.1f} FPS | Frame: {self.frame_count} | "
                          f"State: {self.current_game_state} | Avg Reward: {avg_reward:.2f}")
                    print(f"[GAME] Badges: {mem_state.get('badges', 0)} | "
                          f"Pokedex: {mem_state.get('pokedex_caught', 0)}/{mem_state.get('pokedex_seen', 0)} | "
                          f"Map: {mem_state.get('map_id', 0)} | "
                          f"Pos: ({mem_state.get('pos_x', 0)},{mem_state.get('pos_y', 0)})")
                    print(f"[ADAPTIVE] Entropy: {current_entropy:.4f} | "
                          f"Exploration: {'High' if current_entropy > 0.03 else 'Medium' if current_entropy > 0.01 else 'Low'}")

                    perf_start_time = time.time()
                    perf_frame_count = 0

                # Add to trajectory buffer
                self.trajectory_buffer.add(
                    stacked_state.cpu(), action, reward, value,
                    log_prob, False, self.current_game_state
                )

                # Training when buffer is full
                if self.trajectory_buffer.is_full():
                    self.frame_stack.add(next_single_frame)
                    next_stacked_state = self.frame_stack.get_stack()
                    _, _, next_value = self.network_group.choose_action(
                        next_stacked_state, self.current_game_state, deterministic=True
                    )

                    advantages, returns = self.trajectory_buffer.calculate_gae_advantages(next_value)

                    # Calculate adaptive entropy coefficient based on total frames
                    current_entropy_coeff = self.entropy_scheduler.get_entropy(self.frame_count)

                    # Train per game state
                    game_states_in_buffer = set(self.trajectory_buffer.game_states)
                    for gs in game_states_in_buffer:
                        indices = [i for i, g in enumerate(self.trajectory_buffer.game_states) if g == gs]
                        if len(indices) < HYPERPARAMETERS['PPO_MINIBATCH_SIZE']:
                            continue

                        batch_data = {
                            'states': [self.trajectory_buffer.states[i] for i in indices],
                            'actions': [self.trajectory_buffer.actions[i] for i in indices],
                            'old_log_probs': [self.trajectory_buffer.log_probs[i] for i in indices],
                            'advantages': [advantages[i] for i in indices],
                            'returns': [returns[i] for i in indices],
                            'game_states': [gs] * len(indices)
                        }

                        # Training with adaptive entropy coefficient
                        metrics = self.network_group.train_ppo(batch_data, gs, entropy_coeff=current_entropy_coeff)
                        self.loss_history.append(metrics['policy_loss'])

                        if self.frame_count % 100 == 0:
                            print(f"[TRAIN] Frame {self.frame_count} [{gs}] - "
                                  f"Policy Loss: {metrics['policy_loss']:.4f}, "
                                  f"Value Loss: {metrics['value_loss']:.4f}, "
                                  f"Entropy: {metrics['entropy']:.4f}")

                    self.trajectory_buffer.reset()

                # Checkpoint saving
                if self.frame_count - last_save_frame >= config.SAVE_FREQUENCY:
                    self._save_checkpoint()
                    self._save_stats()
                    last_save_frame = self.frame_count
                    print(f"[SAVE] Checkpoint saved at frame {self.frame_count}")

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user")
        finally:
            self._save_checkpoint()
            self._save_stats()
            self.pyboy.stop()
            print(f"[INFO] Training completed. Frames: {self.frame_count}, Total Reward: {self.total_reward:.2f}")


def main() -> None:
    rom_path = config.ROM_PATH
    if not os.path.exists(rom_path) or not (rom_path.lower().endswith('.gbc') or rom_path.lower().endswith('.gb')) or os.path.getsize(rom_path) == 0:
        print(f"Error: ROM file not found or invalid at path: {rom_path}")
        return

    headless = config.HEADLESS

    try:
        agent = PokemonAIAgent(rom_path, headless=headless)
        agent.start_training()
    except PokemonAIError as e:
        logger.error(f"Pokemon AI Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()