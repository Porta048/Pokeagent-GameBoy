"""
Main Entry Point per Pokemon AI Agent.

ARCHITETTURA GENERALE:
1. PokemonAIAgent: Classe principale che gestisce tutto
2. PyBoy: Emulatore Game Boy che esegue la ROM
3. PPO Networks: 3 reti neurali specializzate (esplorazione, battaglia, menu)
4. Memory Reader: Legge RAM del Game Boy per stato gioco e ricompense
5. State Detector: Rileva contesto (battaglia/menu/dialogo/esplorazione)
6. Trajectory Buffer: Raccoglie esperienze per training PPO

LOOP PRINCIPALE:
1. Cattura frame corrente dallo schermo
2. State Detector determina contesto (battle/menu/dialogue/exploring)
3. Action Filter maschera azioni inutili per il contesto
4. Rete neurale decide azione da eseguire
5. Esegue azione sull'emulatore (con frameskip adattivo)
6. Memory Reader calcola ricompensa dalla RAM
7. Salva esperienza nel trajectory buffer
8. Quando buffer pieno → training PPO
9. Ogni 10k frame → salva checkpoint

OTTIMIZZAZIONI PERFORMANCE:
- Rendering selettivo (ogni N frame)
- Frameskip adattivo (6-12 frame) basato su contesto
- Image caching per frame identici
- Modalità headless per FPS massimi
- Async saving per checkpoint non bloccanti
"""
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

from .config import config
from .memory_reader import GameMemoryReader
from .state_detector import GameStateDetector
from .utils import AsyncSaver, FrameStack, ImageCache, EXPLORATION_CONV_DIMENSIONS, MENU_CONV_DIMENSIONS
from .hyperparameters import HYPERPARAMETERS
from .errors import PokemonAIError, ROMLoadError, MemoryReadError, CheckpointLoadError, GameEnvironmentError
from .anti_loop import AdaptiveEntropyScheduler, AntiLoopMemoryBuffer
from .action_filter import ContextAwareActionFilter
from .trajectory_buffer import TrajectoryBuffer
from .models import PPONetworkGroup
from .screen_regions import SCREEN_REGIONS


# Configurazione logging
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
    """
    Agente AI principale per Pokemon Rosso/Blu con algoritmo PPO.

    RESPONSABILITÀ:
    1. Inizializza emulatore Game Boy (PyBoy)
    2. Gestisce 3 reti neurali PPO specializzate
    3. Coordina loop training (cattura frame, decide azioni, calcola reward)
    4. Salva/carica checkpoint per resume training
    5. Monitora performance e progressione gioco

    COMPONENTI PRINCIPALI:
    - pyboy: Emulatore Game Boy
    - network_group: 3 reti PPO (exploration, battle, menu)
    - trajectory_buffer: Buffer esperienze per training PPO
    - memory_reader: Lettore RAM Game Boy per stato/reward
    - state_detector: Rilevatore contesto (CV-based)
    - anti_loop_buffer: Sistema anti-loop per penalizzare comportamenti ripetitivi
    - action_filter: Filtro contestuale azioni

    TRAINING:
    - On-policy PPO con GAE (Generalized Advantage Estimation)
    - Trajectory length: 512 step
    - Minibatch size: 32
    - Epoche per update: 3
    - Entropy scheduling: 0.1 → 0.01 (decay 200k frame)
    """
    def __init__(self, rom_path: str, headless: bool = False) -> None:
        if not TORCH_AVAILABLE or not DEPS_AVAILABLE:
            raise PokemonAIError("Missing dependencies. Install: torch, pyboy, keyboard, PIL, cv2")

        self.device = config.DEVICE
        print(f"Using device: {self.device}")

        # Supporto per training parallelo
        self.use_shared_buffer = False  # Impostato a True da parallel_trainer
        self.shared_buffer = None
        self.rank = 0  # Rank del worker (0 = visibile, >0 = headless)

        # PyBoy con finestra ottimizzata
        window_type = "headless" if headless else "SDL2"
        try:
            self.pyboy = PyBoy(rom_path, window=window_type)
        except Exception as e:
            raise ROMLoadError(f"Failed to load ROM file '{rom_path}': {str(e)}") from e

        # Imposta velocità emulazione (0=illimitata, 1=normale, 2=2x, ecc.)
        emulation_speed = config.EMULATION_SPEED
        self.pyboy.set_emulation_speed(emulation_speed)
        speed_desc = "unlimited" if emulation_speed == 0 else f"{emulation_speed}x"
        print(f"[INFO] Emulation speed: {speed_desc}")

        # PyBoy 2.6.0+ usa API basata su stringhe per i pulsanti
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

        # Componenti del sistema anti-confusione
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
                # Convert sets to lists for JSON serialization
                serializable_state = {}
                for key, value in final_state.items():
                    if isinstance(value, set):
                        serializable_state[key] = list(value)
                    else:
                        serializable_state[key] = value
                self.stats['final_state'] = serializable_state

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

                # Carica stato gioco se esiste
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

        # Salva stato gioco
        try:
            with open(self.game_state_path, 'wb') as f:
                self.pyboy.save_state(f)
        except Exception as e:
            logger.warning(f"Unable to save game state: {e}")

    def _get_screen_tensor(self):
        """Ottiene il frame corrente preprocessato."""
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
            # Ritorna un tensore vuoto come fallback
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
        Calcola reward basato SOLO su progressione di gioco (memoria).
        Nessun reward per movimento generico - previene convergenza prematura.
        """
        reward = 0

        try:
            memory_state = self.memory_reader.get_current_state()
            memory_reward = self.memory_reader.calculate_event_rewards(memory_state)
            reward += memory_reward

            # Sistema Anti-Loop: DISABILITATO se flag è False
            if config.ANTI_LOOP_ENABLED:
                # Sistema Anti-Loop: penalità per comportamenti ripetitivi
                loop_penalty = self.anti_loop_buffer.calculate_loop_penalty()
                if loop_penalty < 0:
                    reward += loop_penalty
                    # Log quando loop rilevato (solo occasionalmente per evitare spam)
                    if self.frame_count % 5000 == 0:  # Ogni 5000 frame invece di 100
                        print(f"[ANTI-LOOP] Loop detected! Penalty: {loop_penalty:.2f}")

                # Bonus esplorazione: reward per esplorazione attiva
                exploration_bonus = self.anti_loop_buffer.get_exploration_bonus()
                if exploration_bonus > 0:
                    reward += exploration_bonus

            self.reward_history.append(reward)
            self.total_reward += reward
            return reward
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            # Ritorna una piccola penalità quando il calcolo del reward fallisce per incoraggiare stabilità
            return -1.0

    def start_training(self) -> None:
        """
        Loop principale di training PPO.

        STRUTTURA LOOP:
        1. INPUT: Cattura frame + stato gioco
        2. DECISIONE: Rete neurale sceglie azione (con action mask)
        3. ESECUZIONE: Applica azione su emulatore (con frameskip adattivo)
        4. REWARD: Calcola ricompensa da RAM Game Boy
        5. STORAGE: Salva esperienza in trajectory buffer
        6. TRAINING: Quando buffer pieno, esegui update PPO
        7. CHECKPOINT: Ogni 10k frame, salva modello

        CONTROLLI KEYBOARD:
        - ESC: Esci e salva
        - SPACE: Pausa/Riprendi
        - +/=: Aumenta velocità emulazione
        - -/_: Diminuisci velocità emulazione

        OTTIMIZZAZIONI:
        - Rendering adattivo (ogni N frame) per non rallentare training
        - Frameskip contestuale (battaglia 12, dialogo 6, esplorazione 10)
        - Performance monitoring ogni 1000 frame
        """
        paused = False
        last_save_frame = 0
        perf_start_time = time.time()
        perf_frame_count = 0

        # Inizializza frame stack con primo frame
        initial_frame = self._get_screen_tensor()
        self.frame_stack.reset(initial_frame)

        render_counter = 0

        try:
            while True:
                # === GESTIONE INPUT UTENTE ===
                if keyboard.is_pressed('escape'):
                    break
                if keyboard.is_pressed('space'):
                    paused = not paused
                    time.sleep(0.3)

                # Controlli velocità: + e - per aumentare/diminuire
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

                # Cattura stato corrente
                single_frame = self._get_screen_tensor()
                self.frame_stack.add(single_frame)
                stacked_state = self.frame_stack.get_stack()

                # Mascheramento azioni context-aware
                action_mask = self.action_filter.get_action_mask(self.current_game_state)

                # L'AI decide l'azione da eseguire (con mascheramento azioni)
                action, log_prob, value = self.network_group.choose_action(
                    stacked_state, self.current_game_state, action_mask=action_mask
                )

                # Esegue azione
                button = self.actions[action]
                if button is not None:
                    self.pyboy.button_press(button)

                # Frameskip adattivo basato su stato gioco
                frameskip = config.FRAMESKIP_MAP.get(self.current_game_state, config.FRAMESKIP_MAP["base"])

                # Rendering fluido senza lag
                if config.RENDER_ENABLED:
                    # Renderizza ogni N frame per fluidità
                    render_freq = config.RENDER_EVERY_N_FRAMES
                    for i in range(frameskip):
                        should_render = (i % render_freq == 0)
                        self.pyboy.tick(1, render=should_render)
                        render_counter += 1
                else:
                    # Modalità headless: velocità massima
                    self.pyboy.tick(count=frameskip, render=False)

                # Rilascia pulsante
                if button is not None:
                    self.pyboy.button_release(button)

                # Cattura stato successivo
                next_single_frame = self._get_screen_tensor()

                # Aggiorna buffer anti-loop solo se abilitato
                if config.ANTI_LOOP_ENABLED:
                    mem_state = self.memory_reader.get_current_state()
                    self.anti_loop_buffer.add_state(
                        mem_state.get('pos_x', 0),
                        mem_state.get('pos_y', 0),
                        mem_state.get('map_id', 0),
                        action
                    )

                reward = self._calculate_reward()

                # Monitoraggio performance
                perf_frame_count += 1
                if perf_frame_count % config.PERFORMANCE_LOG_INTERVAL == 0:
                    elapsed = time.time() - perf_start_time
                    fps = perf_frame_count / elapsed
                    avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0

                    # Calcola entropia corrente
                    current_entropy = self.entropy_scheduler.get_entropy(self.frame_count)

                    # Stato dettagliato di gioco
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

                # Aggiungi a trajectory buffer
                self.trajectory_buffer.add(
                    stacked_state.cpu(), action, reward, value,
                    log_prob, False, self.current_game_state
                )

                # Training quando buffer è pieno
                if self.trajectory_buffer.is_full():
                    self.frame_stack.add(next_single_frame)
                    next_stacked_state = self.frame_stack.get_stack()
                    _, _, next_value = self.network_group.choose_action(
                        next_stacked_state, self.current_game_state, deterministic=True
                    )

                    advantages, returns = self.trajectory_buffer.calculate_gae_advantages(next_value)

                    # Calcola coefficiente entropia adattivo basato su frame totali
                    current_entropy_coeff = self.entropy_scheduler.get_entropy(self.frame_count)

                    # Training per stato gioco
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

                        # Training con coefficiente entropia adattivo
                        metrics = self.network_group.train_ppo(batch_data, gs, entropy_coeff=current_entropy_coeff)
                        self.loss_history.append(metrics['policy_loss'])

                        if self.frame_count % 100 == 0:
                            print(f"[TRAIN] Frame {self.frame_count} [{gs}] - "
                                  f"Policy Loss: {metrics['policy_loss']:.4f}, "
                                  f"Value Loss: {metrics['value_loss']:.4f}, "
                                  f"Entropy: {metrics['entropy']:.4f}")

                    self.trajectory_buffer.reset()

                # Salvataggio checkpoint
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Pokemon AI Agent - Reinforcement Learning con PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  python -m src --rom-path pokemon_red.gb
  python -m src --rom-path "C:/Games/Pokemon Red.gb" --headless
  python -m src --rom-path roms/pokemon.gb --speed 2 --log-level DEBUG
        """
    )

    parser.add_argument(
        '--rom-path',
        type=str,
        default=config.ROM_PATH,
        help=f'Percorso al file ROM Pokemon (.gb o .gbc). Default: {config.ROM_PATH}'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Esegui in modalità headless (senza finestra, più veloce)'
    )
    parser.add_argument(
        '--speed',
        type=int,
        default=config.EMULATION_SPEED,
        help='Velocità emulazione (0=illimitata, 1=normale, 2=2x, ecc.). Default: 0'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=config.LOG_LEVEL,
        help='Livello di logging. Default: INFO'
    )

    args = parser.parse_args()

    # Aggiorna config con argomenti CLI
    rom_path = args.rom_path
    headless = args.headless if args.headless else config.HEADLESS

    if args.speed != config.EMULATION_SPEED:
        config.EMULATION_SPEED = args.speed

    if args.log_level != config.LOG_LEVEL:
        config.LOG_LEVEL = args.log_level
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Valida ROM
    if not os.path.exists(rom_path):
        print(f"[ERROR] File ROM non trovato: {rom_path}")
        print(f"\nSpecifica il percorso corretto con --rom-path:")
        print(f"  python -m src --rom-path percorso/al/tuo/file.gb")
        return

    if not (rom_path.lower().endswith('.gbc') or rom_path.lower().endswith('.gb')):
        print(f"[ERROR] Il file deve essere .gb o .gbc: {rom_path}")
        return

    if os.path.getsize(rom_path) == 0:
        print(f"[ERROR] Il file ROM è vuoto: {rom_path}")
        return

    print(f"[INFO] ROM: {rom_path}")
    print(f"[INFO] Modalità: {'headless' if headless else 'finestra SDL2'}")
    print(f"[INFO] Velocità: {'illimitata' if args.speed == 0 else f'{args.speed}x'}")
    print(f"[INFO] Device: {config.DEVICE}")
    print()

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