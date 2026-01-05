import os
import sys
import time
import json
import logging
from typing import Dict, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn

try:
    from pyboy import PyBoy
    import keyboard
    from PIL import Image
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"Missing dependencies: {e}")

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = "src"

from .cfg import config
from .mem import GameMemoryReader
from .moe import GameStateMoERouter
from .uti import AsyncSaver, FrameStack, ImageCache
from .hyp import HYPERPARAMETERS
from .err import PokemonAIError, ROMLoadError
from .ant import AdaptiveEntropyScheduler, AntiLoopMemoryBuffer
from .act import ContextAwareActionFilter
from .trj import TrajectoryBuffer
from .mod import PPONetworkGroup
from .swm import SimpleWorldModel, compute_world_model_loss
from .llm import OllamaLLMClient, LLMConfig
from .hrs import HierarchicalRewardCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pokemon_ai.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class PokemonAIAgent:
    def __init__(self, rom_path: str, headless: bool = False):
        if not DEPS_AVAILABLE:
            raise PokemonAIError("Missing dependencies: torch, pyboy, keyboard, PIL")

        self.device = config.DEVICE
        print(f"Device: {self.device}")

        try:
            self.pyboy = PyBoy(rom_path, window="headless" if headless else "SDL2")
        except Exception as e:
            raise ROMLoadError(f"Failed to load ROM '{rom_path}': {e}") from e

        self.pyboy.set_emulation_speed(config.EMULATION_SPEED)
        self.actions = config.ACTIONS

        rom_name = os.path.splitext(os.path.basename(rom_path))[0]
        self.save_dir = f"{config.SAVE_DIR_PREFIX}_{rom_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_path = os.path.join(self.save_dir, config.MODEL_FILENAME)
        self.stats_path = os.path.join(self.save_dir, config.STATS_FILENAME)
        self.game_state_path = os.path.join(self.save_dir, config.GAME_STATE_FILENAME)

        self.network_group = PPONetworkGroup(len(self.actions), self.device, config.FRAME_STACK_SIZE)
        self.world_model = SimpleWorldModel(
            config.FRAME_STACK_SIZE, HYPERPARAMETERS['WORLD_MODEL_LATENT_DIM'],
            len(self.actions), 256
        ).to(self.device)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=HYPERPARAMETERS['WORLD_MODEL_LR'])
        self.world_model_enabled = False
        self.imagination_horizon = HYPERPARAMETERS['WORLD_MODEL_IMAGINATION_HORIZON']

        self.trajectory_buffer = TrajectoryBuffer()
        self.frame_stack = FrameStack(config.FRAME_STACK_SIZE)
        self.image_cache = ImageCache()
        self.memory_reader = GameMemoryReader(self.pyboy)
        self.moe_router = GameStateMoERouter(vision_features_dim=192, state_embedding_dim=256).to(self.device)
        self.async_saver = AsyncSaver()
        self.entropy_scheduler = AdaptiveEntropyScheduler()
        self.anti_loop_buffer = AntiLoopMemoryBuffer()
        self.action_filter = ContextAwareActionFilter()
        self.current_game_state = "exploring"
        self.last_screen_array = None
        self.action_history = []
        self.llm_decisions = 0
        self.rl_fallbacks = 0

        llm_config = LLMConfig(
            enabled=config.LLM_ENABLED, host=config.LLM_HOST, model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE, timeout=config.LLM_TIMEOUT,
            min_interval_ms=config.LLM_MIN_INTERVAL_MS, max_calls_per_minute=config.LLM_MAX_CALLS_PER_MINUTE,
            cache_ttl_seconds=config.LLM_CACHE_TTL_SECONDS, use_vision=config.LLM_USE_VISION,
            use_for_exploration=config.LLM_USE_FOR_EXPLORATION, use_for_battle=config.LLM_USE_FOR_BATTLE,
            use_for_menu=config.LLM_USE_FOR_MENU
        )
        self.llm_client = OllamaLLMClient(llm_config)
        print(f"[LLM] {'Ready' if self.llm_client.available else 'Disabled'}")

        self.hierarchical_reward_calculator = HierarchicalRewardCalculator(
            llm_client=self.llm_client if self.llm_client.available else None
        )

        self.stats = self._load_stats()
        self.episode_count = 0
        self.frame_count = 0
        self.total_reward = 0
        self.reward_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=100)
        self._load_checkpoint()

    def _extract_memory_features(self, mem: Dict) -> torch.Tensor:
        features = torch.zeros(128, dtype=torch.float32, device=self.device)
        features[0] = mem.get('pos_x', 0) / 255.0
        features[1] = mem.get('pos_y', 0) / 255.0
        features[2] = mem.get('map_id', 0) / 255.0
        features[3] = mem.get('badges', 0) / 8.0
        features[4] = mem.get('pokedex_caught', 0) / 151.0
        features[5] = mem.get('pokedex_seen', 0) / 151.0
        features[6] = mem.get('party_count', 0) / 6.0
        features[7] = min(mem.get('money', 0) / 100000.0, 1.0)
        features[8] = 1.0 if mem.get('in_battle') else 0.0
        return features.unsqueeze(0)

    def _load_stats(self):
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {'episodes': 0, 'total_frames': 0, 'best_reward': float('-inf')}

    def _save_stats(self):
        try:
            state = self.memory_reader.get_current_state()
            if state:
                self.stats['final_state'] = {k: list(v) if isinstance(v, set) else v for k, v in state.items()}
            self.stats.update({
                'episodes': self.episode_count,
                'total_frames': self.stats.get('total_frames', 0) + self.frame_count,
                'best_reward': max(self.stats.get('best_reward', float('-inf')), self.total_reward)
            })
            if self.reward_history:
                self.stats['avg_reward'] = float(np.mean(list(self.reward_history)))
            with open(self.stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Save stats failed: {e}")

    def _load_checkpoint(self):
        if not os.path.exists(self.model_path):
            return
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.network_group.exploration_network.load_state_dict(ckpt['explorer_state'])
            self.network_group.battle_network.load_state_dict(ckpt['battle_state'])
            self.network_group.menu_network.load_state_dict(ckpt['menu_state'])
            if 'moe_router_state' in ckpt:
                self.moe_router.load_state_dict(ckpt['moe_router_state'])
            if 'world_model_state' in ckpt:
                try:
                    self.world_model.load_state_dict(ckpt['world_model_state'])
                    self.world_model_optimizer.load_state_dict(ckpt['world_model_optimizer_state'])
                except RuntimeError:
                    logger.warning("World model architecture changed, training from scratch")
            self.episode_count = ckpt.get('episode', 0)
            self.frame_count = ckpt.get('frame', 0)
            print(f"[LOAD] Episode {self.episode_count}, Frame {self.frame_count}")
            if os.path.exists(self.game_state_path):
                try:
                    with open(self.game_state_path, 'rb') as f:
                        self.pyboy.load_state(f)
                except Exception:
                    pass
        except Exception as e:
            raise PokemonAIError(f"Checkpoint load failed: {e}") from e

    def _save_checkpoint(self):
        ckpt = {
            'explorer_state': self.network_group.exploration_network.state_dict(),
            'battle_state': self.network_group.battle_network.state_dict(),
            'menu_state': self.network_group.menu_network.state_dict(),
            'moe_router_state': self.moe_router.state_dict(),
            'world_model_state': self.world_model.state_dict(),
            'world_model_optimizer_state': self.world_model_optimizer.state_dict(),
            'episode': self.episode_count, 'frame': self.frame_count
        }
        self.async_saver.save_async(lambda d: torch.save(d, self.model_path), ckpt)
        try:
            with open(self.game_state_path, 'wb') as f:
                self.pyboy.save_state(f)
        except Exception:
            pass

    def _get_screen_tensor(self):
        try:
            gray = np.array(self.pyboy.screen.image.convert('L'))
            self.last_screen_array = gray.copy()
            cached = self.image_cache.get(gray)
            normalized = cached if cached is not None else gray.astype(np.float32) / 255.0
            if cached is None:
                self.image_cache.save(gray, normalized)
            return torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        except Exception:
            return torch.zeros((1, 144, 160), dtype=torch.float32, device=self.device)

    def _calculate_reward(self):
        try:
            mem = self.memory_reader.get_current_state()
            prev = self.memory_reader.previous_state

            loop_penalty = 0.0
            curiosity_bonus = 0.0
            exploration_bonus = 0.0

            if config.ANTI_LOOP_ENABLED:
                loop_penalty = self.anti_loop_buffer.calculate_loop_penalty()
                curiosity_bonus = self.anti_loop_buffer.get_curiosity_reward()
                exploration_bonus = self.anti_loop_buffer.get_exploration_bonus(
                    mem.get('pos_x', 0), mem.get('pos_y', 0), mem.get('map_id', 0)
                )

            reward, details = self.hierarchical_reward_calculator.calculate_total_reward(
                mem, prev, loop_penalty, self.last_screen_array
            )

            reward += curiosity_bonus + exploration_bonus

            if curiosity_bonus > 0:
                details['curiosity'] = round(curiosity_bonus, 4)
            if exploration_bonus > 0:
                details['exploration'] = round(exploration_bonus, 4)

            if details and self.frame_count % 100 == 0:
                print(f"[REWARD] {details} = {reward:.4f}")

            self.reward_history.append(reward)
            self.total_reward += reward
            return reward
        except Exception as e:
            logger.error(f"Reward calc failed: {e}")
            return -1.0

    def _train_world_model(self, buf) -> Dict[str, float]:
        self.world_model.train()
        total_latent, total_reward_loss, total_done, n = 0.0, 0.0, 0.0, 0
        max_idx = len(buf.states) - self.imagination_horizon
        if max_idx < 1:
            return {'latent_loss': 0, 'reward_loss': 0, 'done_loss': 0}

        for i in range(max_idx):
            states, next_states, actions, rewards, dones = [], [], [], [], []
            for j in range(i, min(i + self.imagination_horizon, len(buf.states) - 1)):
                states.append(buf.states[j])
                next_states.append(buf.states[j + 1])
                actions.append(buf.actions[j])
                rewards.append(buf.rewards[j])
                dones.append(buf.dones[j])
            if not states:
                continue

            losses = compute_world_model_loss(
                self.world_model,
                torch.stack(states).to(self.device),
                torch.tensor(actions, dtype=torch.long, device=self.device),
                torch.stack(next_states).to(self.device),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.float32, device=self.device)
            )
            self.world_model_optimizer.zero_grad()
            losses['total_loss'].backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_model_optimizer.step()
            total_latent += losses['latent_loss'].item()
            total_reward_loss += losses['reward_loss'].item()
            total_done += losses['done_loss'].item()
            n += 1

        return {
            'latent_loss': total_latent / max(n, 1),
            'reward_loss': total_reward_loss / max(n, 1),
            'done_loss': total_done / max(n, 1)
        }

    def start_training(self):
        paused, last_save = False, 0
        perf_start, perf_frames = time.time(), 0
        self.frame_stack.reset(self._get_screen_tensor())

        try:
            while True:
                if keyboard.is_pressed('escape'):
                    break
                if keyboard.is_pressed('space'):
                    paused = not paused
                    time.sleep(0.3)
                if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                    config.EMULATION_SPEED = min(config.EMULATION_SPEED + 1, 10)
                    self.pyboy.set_emulation_speed(config.EMULATION_SPEED)
                    time.sleep(0.2)
                if keyboard.is_pressed('-'):
                    config.EMULATION_SPEED = max(config.EMULATION_SPEED - 1, 0)
                    self.pyboy.set_emulation_speed(config.EMULATION_SPEED)
                    time.sleep(0.2)
                if paused:
                    self.pyboy.tick()
                    continue

                frame = self._get_screen_tensor()
                self.frame_stack.add(frame)
                stacked = self.frame_stack.get_stack()
                mem = self.memory_reader.get_current_state()

                self.current_game_state = "battle" if mem.get('in_battle') else "exploring"

                # LLM is PRIMARY decision maker
                llm_action = self.llm_client.get_action(
                    self.current_game_state, mem, self.last_screen_array, self.action_history
                )

                if llm_action is not None:
                    action = llm_action
                    log_prob = 0.0
                    _, _, value = self.network_group.choose_action(stacked, self.current_game_state, deterministic=True)
                    value = value
                    self.llm_decisions += 1
                else:
                    # RL network as FALLBACK
                    mask = self.action_filter.get_action_mask(self.current_game_state)
                    action, log_prob, value = self.network_group.choose_action(stacked, self.current_game_state, action_mask=mask)
                    self.rl_fallbacks += 1

                self.action_history.append(self.actions[action])
                if len(self.action_history) > 20:
                    self.action_history.pop(0)

                button = self.actions[action]
                if button:
                    self.pyboy.button_press(button)
                frameskip = config.FRAMESKIP_MAP.get(self.current_game_state, 6)
                self.pyboy.tick(count=frameskip, render=config.RENDER_ENABLED)
                if button:
                    self.pyboy.button_release(button)

                if config.ANTI_LOOP_ENABLED:
                    m = self.memory_reader.get_current_state()
                    self.anti_loop_buffer.add_state(m.get('pos_x', 0), m.get('pos_y', 0), m.get('map_id', 0), action)
                    self.anti_loop_buffer.track_menu_action(action, self.actions.index('start'), self.frame_count)

                reward = self._calculate_reward()
                perf_frames += 1

                if perf_frames % config.PERFORMANCE_LOG_INTERVAL == 0:
                    fps = perf_frames / (time.time() - perf_start)
                    avg = np.mean(list(self.reward_history)) if self.reward_history else 0
                    m = self.memory_reader.get_current_state()
                    total_decisions = self.llm_decisions + self.rl_fallbacks
                    llm_pct = (self.llm_decisions / max(total_decisions, 1)) * 100
                    print(f"[PERF] {fps:.1f}fps Frame:{self.frame_count} Reward:{avg:.2f}")
                    print(f"[GAME] Badges:{m.get('badges',0)} Pokemon:{m.get('pokedex_caught',0)} Map:{m.get('map_id',0)} Pos:({m.get('pos_x',0)},{m.get('pos_y',0)})")
                    print(f"[DECISION] LLM:{self.llm_decisions} ({llm_pct:.0f}%) | RL fallback:{self.rl_fallbacks}")
                    perf_start, perf_frames = time.time(), 0

                self.trajectory_buffer.add(stacked.cpu(), action, reward, value, log_prob, False, self.current_game_state)

                if self.trajectory_buffer.is_full():
                    next_frame = self._get_screen_tensor()
                    self.frame_stack.add(next_frame)
                    _, _, next_val = self.network_group.choose_action(self.frame_stack.get_stack(), self.current_game_state, deterministic=True)
                    adv, ret, _ = self.trajectory_buffer.calculate_grpo_advantages(next_val, group_by=HYPERPARAMETERS.get('GRPO_GROUP_BY', 'game_state'))
                    is_stuck = self.anti_loop_buffer.is_stuck() if config.ANTI_LOOP_ENABLED else False
                    entropy_coeff = self.entropy_scheduler.get_entropy(self.frame_count, stuck=is_stuck)

                    for gs in set(self.trajectory_buffer.game_states):
                        idx = [i for i, g in enumerate(self.trajectory_buffer.game_states) if g == gs]
                        if len(idx) < HYPERPARAMETERS['PPO_MINIBATCH_SIZE']:
                            continue
                        batch = {
                            'states': [self.trajectory_buffer.states[i] for i in idx],
                            'actions': [self.trajectory_buffer.actions[i] for i in idx],
                            'old_log_probs': [self.trajectory_buffer.log_probs[i] for i in idx],
                            'advantages': [adv[i] for i in idx],
                            'returns': [ret[i] for i in idx],
                            'game_states': [gs] * len(idx)
                        }
                        metrics = self.network_group.train_grpo(batch, gs, entropy_coeff=entropy_coeff)
                        self.loss_history.append(metrics['policy_loss'])
                        print(f"[GRPO] {gs} Policy:{metrics['policy_loss']:.4f} Value:{metrics['value_loss']:.4f}")

                    if self.frame_count > HYPERPARAMETERS['WORLD_MODEL_START_FRAME']:
                        self.world_model_enabled = True
                        wm = self._train_world_model(self.trajectory_buffer)
                        logger.info(f"[WM] Latent:{wm['latent_loss']:.4f} Reward:{wm['reward_loss']:.4f}")

                    self.trajectory_buffer.reset()

                if self.frame_count - last_save >= config.SAVE_FREQUENCY:
                    self._save_checkpoint()
                    self._save_stats()
                    last_save = self.frame_count
                    print(f"[SAVE] Frame {self.frame_count}")

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self._save_checkpoint()
            self._save_stats()
            self.pyboy.stop()
            print(f"Done. Frames:{self.frame_count} Reward:{self.total_reward:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pokemon AI Agent")
    parser.add_argument('--rom-path', type=str, default=config.ROM_PATH)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--speed', type=int, default=config.EMULATION_SPEED)
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=config.LOG_LEVEL)
    args = parser.parse_args()

    if args.speed != config.EMULATION_SPEED:
        config.EMULATION_SPEED = args.speed
    if args.log_level != config.LOG_LEVEL:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    rom = args.rom_path
    if not os.path.exists(rom):
        print(f"ROM not found: {rom}")
        return
    if not rom.lower().endswith(('.gb', '.gbc')):
        print(f"Invalid ROM extension: {rom}")
        return

    print(f"ROM: {rom} | {'Headless' if args.headless else 'SDL2'} | Speed: {args.speed or 'unlimited'}")
    try:
        PokemonAIAgent(rom, headless=args.headless or config.HEADLESS).start_training()
    except (PokemonAIError, Exception) as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
