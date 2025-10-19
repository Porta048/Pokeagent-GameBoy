import os, sys, time, threading, json, hashlib
from typing import Union, List, Dict, Optional, Tuple, Any
from collections import deque
from functools import lru_cache
import numpy as np

try:
    import torch
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

REGIONI_SCHERMO = {
    'BARRA_HP': (slice(100, 120), slice(90, 150)),
    'AREA_MENU': (slice(110, 140), slice(0, 80)),
    'CASELLA_DIALOGO': (slice(100, 140), slice(10, 150)),
    'REGIONE_CENTRALE': (slice(60, 100), slice(70, 90))
}

IPERPARAMETRI = {
    'SOGLIA_HP': 500,
    'SOGLIA_MENU': 0.15,
    'SOGLIA_DIALOGO': 30,
    'SOGLIA_MOVIMENTO': 0.02,
    'SOGLIA_BLOCCATO': 50,
    'INTERVALLO_CONTROLLO_MEMORIA': 30,
    'FREQUENZA_SALVATAGGIO': 10000,
    'DIMENSIONE_CACHE': 1000,
    # PPO Hyperparameters
    'PPO_CLIP_EPSILON': 0.2,
    'PPO_VALUE_COEFF': 0.5,
    'PPO_ENTROPY_COEFF': 0.01,  # Dinamico - gestito da AdaptiveEntropyScheduler
    'PPO_ENTROPY_START': 0.1,    # Aumentato da 0.05 a 0.1 (MOLTA più esplorazione)
    'PPO_ENTROPY_END': 0.01,     # Aumentato da 0.005 a 0.01 (mantiene esplorazione)
    'PPO_ENTROPY_DECAY_FRAMES': 1000000,  # Aumentato da 500k a 1M (decay più lento)
    'PPO_GAE_LAMBDA': 0.95,
    'PPO_GAE_GAMMA': 0.99,
    'PPO_EPOCHS': 3,
    'PPO_MINIBATCH_SIZE': 32,
    'PPO_TRAJECTORY_LENGTH': 512,
    'PPO_LR': 3e-4,
    'PPO_MAX_GRAD_NORM': 0.5,
    'FRAME_STACK': 4,
    # Anti-Confusion System
    'ANTI_LOOP_ENABLED': False,          # DISABILITATO - causava blocchi dell'agente
    'ANTI_LOOP_BUFFER_SIZE': 100,        # Traccia ultimi 100 stati
    'ANTI_LOOP_THRESHOLD': 8,            # Penalità se >8 stati simili
    'ANTI_LOOP_PENALTY': -2.0,           # Penalità ridotta (era -5.0)
    'ACTION_REPEAT_MAX': 10,             # Aumentato a 10 (era 5) - più permissivo
    'ACTION_REPEAT_PENALTY': -1.0,       # Penalità ridotta (era -2.0)
    # Emulator Optimizations
    'FRAMESKIP_BASE': 8,              # Aumentato per velocità (era 4)
    'FRAMESKIP_DIALOGUE': 6,          # Aumentato (era 3)
    'FRAMESKIP_BATTLE': 12,           # Aumentato (era 8)
    'FRAMESKIP_EXPLORING': 10,        # Aumentato (era 6)
    'FRAMESKIP_MENU': 8,              # Aumentato (era 4)
    'TURN_BASED_MODE': False,         # True = ClaudePlayer style
    'RENDER_ENABLED': True,           # True per visualizzazione completa
    'RENDER_EVERY_N_FRAMES': 2,       # Renderizza ogni 2 frame (era 1) per velocità
    'PERFORMANCE_LOG_INTERVAL': 1000, # Log performance every N frames
    # Velocita Emulazione
    'EMULATION_SPEED': 0              # 0=illimitata (era 1) per massima velocità
}

class ErroreAIPokemon(Exception): pass

class AdaptiveEntropyScheduler:
    """
    Scheduler per entropy coefficient dinamico.
    Riduce progressivamente l'esplorazione casuale man mano che l'agente impara.

    Formula: entropy = start + (end - start) * min(1.0, frames / decay_frames)
    """
    def __init__(self):
        self.start_entropy = IPERPARAMETRI['PPO_ENTROPY_START']
        self.end_entropy = IPERPARAMETRI['PPO_ENTROPY_END']
        self.decay_frames = IPERPARAMETRI['PPO_ENTROPY_DECAY_FRAMES']

    def get_entropy(self, current_frame: int) -> float:
        """Calcola entropy coefficient corrente basato sui frame."""
        progress = min(1.0, current_frame / self.decay_frames)
        entropy = self.start_entropy + (self.end_entropy - self.start_entropy) * progress
        return entropy

class AntiLoopMemoryBuffer:
    """
    Buffer per detection e prevenzione di loop comportamentali.
    Traccia stati recenti e penalizza pattern ripetitivi.

    Pattern Strategy: diversi detection algorithm per diversi tipi di loop.
    """
    def __init__(self):
        self.buffer_size = IPERPARAMETRI['ANTI_LOOP_BUFFER_SIZE']
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_history = deque(maxlen=20)  # Ultimi 20 azioni
        self.position_history = deque(maxlen=50)  # Ultimi 50 posizioni

    def add_state(self, pos_x: int, pos_y: int, id_mappa: int, action: int):
        """Aggiungi stato corrente al buffer."""
        state_key = (id_mappa, pos_x, pos_y)
        self.state_buffer.append(state_key)
        self.action_history.append(action)
        self.position_history.append(state_key)

    def detect_position_loop(self) -> bool:
        """
        Detecta loop di posizione (es. avanti-indietro ripetuto).
        Returns True se l'agente è in un loop.
        """
        if len(self.state_buffer) < 20:
            return False

        # Conta occorrenze degli ultimi 20 stati (era 10)
        recent_states = list(self.state_buffer)[-20:]
        unique_states = set(recent_states)

        # Se visita meno di 2 posizioni uniche negli ultimi 20 step = loop (era <3 in 10)
        # Molto più permissivo
        return len(unique_states) <= 2

    def detect_action_loop(self) -> bool:
        """
        Detecta loop di azioni (es. premere A ripetutamente senza progresso).
        Returns True se l'agente ripete la stessa azione troppo.
        """
        if len(self.action_history) < IPERPARAMETRI['ACTION_REPEAT_MAX']:
            return False

        recent_actions = list(self.action_history)[-IPERPARAMETRI['ACTION_REPEAT_MAX']:]
        # Se le ultime N azioni sono identiche = loop
        return len(set(recent_actions)) == 1

    def detect_oscillation(self) -> bool:
        """
        Detecta oscillazione (es. su-giù-su-giù).
        Returns True se pattern oscillatorio detectato.
        """
        if len(self.position_history) < 16:  # Aumentato da 8 a 16
            return False

        # Controlla se posizioni alternano tra 2 valori
        recent_pos = list(self.position_history)[-16:]  # Aumentato da 8 a 16
        unique_pos = set(recent_pos)

        if len(unique_pos) == 2:
            # Verifica alternanza perfetta per almeno 12 step (era 8)
            alternating_count = sum(
                1 for i in range(len(recent_pos)-1)
                if recent_pos[i] != recent_pos[i+1]
            )
            # Solo se alterna per più del 90% dei casi (molto restrittivo)
            return alternating_count >= 14  # 14 su 15 transizioni

        return False

    def calculate_loop_penalty(self) -> float:
        """
        Calcola penalità basata sui loop detectati.
        Returns penalità negativa se loop detectato, 0 altrimenti.
        """
        penalty = 0.0

        if self.detect_position_loop():
            penalty += IPERPARAMETRI['ANTI_LOOP_PENALTY']

        if self.detect_action_loop():
            penalty += IPERPARAMETRI['ACTION_REPEAT_PENALTY']

        if self.detect_oscillation():
            penalty += IPERPARAMETRI['ANTI_LOOP_PENALTY'] * 0.5

        return penalty

    def get_exploration_bonus(self) -> float:
        """
        Bonus per comportamento esplorativo (opposto di loop).
        Returns bonus positivo se l'agente esplora attivamente.
        """
        if len(self.state_buffer) < 20:
            return 0.0

        # Conta stati unici negli ultimi 20 step
        recent_states = list(self.state_buffer)[-20:]
        unique_ratio = len(set(recent_states)) / len(recent_states)

        # Bonus proporzionale alla diversità di stati visitati (più facile da ottenere)
        if unique_ratio > 0.6:  # >60% stati unici = ottima esplorazione (era 0.7)
            return 1.5  # Ridotto da 2.0
        elif unique_ratio > 0.4:  # >40% stati unici = buona esplorazione (era 0.5)
            return 0.8  # Ridotto da 1.0

        return 0.0

class ContextAwareActionFilter:
    """
    Filtro intelligente per azioni basato sul contesto di gioco.
    Riduce azioni irrilevanti in base allo stato corrente.

    Pattern Strategy: diverse strategie di filtering per ogni game state.
    """
    # Action indices (matching self.actions in AgentePokemonAI)
    ACTIONS = {
        'NOOP': 0, 'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4,
        'A': 5, 'B': 6, 'START': 7, 'SELECT': 8
    }

    @staticmethod
    def get_action_mask(game_state: str) -> List[float]:
        """
        Restituisce maschera di probabilità per azioni valide nel contesto.
        1.0 = azione utile, 0.7 = azione meno utile (molto più permissivo).

        Questo non blocca azioni, ma riduce LEGGERMENTE la loro probabilità di selezione.
        """
        mask = [1.0] * 9  # Default: tutte le azioni hanno peso normale

        if game_state == "battle":
            # In battaglia: leggera preferenza ad A, frecce, B
            mask[ContextAwareActionFilter.ACTIONS['NOOP']] = 0.7  # Era 0.3, molto più permissivo
            mask[ContextAwareActionFilter.ACTIONS['START']] = 0.7
            mask[ContextAwareActionFilter.ACTIONS['SELECT']] = 0.7

        elif game_state == "menu":
            # In menu: leggera preferenza ad A, B, frecce
            mask[ContextAwareActionFilter.ACTIONS['NOOP']] = 0.8  # Era 0.5
            mask[ContextAwareActionFilter.ACTIONS['START']] = 0.9  # Era 0.7
            mask[ContextAwareActionFilter.ACTIONS['SELECT']] = 0.7  # Era 0.3

        elif game_state == "dialogue":
            # In dialogo: leggera preferenza ad A e B
            mask[ContextAwareActionFilter.ACTIONS['UP']] = 0.7  # Era 0.2
            mask[ContextAwareActionFilter.ACTIONS['DOWN']] = 0.7
            mask[ContextAwareActionFilter.ACTIONS['LEFT']] = 0.7
            mask[ContextAwareActionFilter.ACTIONS['RIGHT']] = 0.7
            mask[ContextAwareActionFilter.ACTIONS['START']] = 0.7  # Era 0.3
            mask[ContextAwareActionFilter.ACTIONS['SELECT']] = 0.7  # Era 0.2

        elif game_state == "exploring":
            # In esplorazione: tutte le azioni quasi uguali
            mask[ContextAwareActionFilter.ACTIONS['NOOP']] = 0.8  # Era 0.4

        return mask

    @staticmethod
    def apply_mask_to_logits(logits: 'torch.Tensor', mask: List[float]) -> 'torch.Tensor':
        """
        Applica maschera ai logits della policy.
        Riduce logits per azioni meno utili nel contesto.
        """
        if TORCH_AVAILABLE:
            mask_tensor = torch.tensor(mask, device=logits.device, dtype=logits.dtype)
            # Log-space mask: logits * mask (equivalente a prob^mask in linear space)
            return logits + torch.log(mask_tensor + 1e-8)
        return logits

class BufferTraiettorie:
    """Buffer per trajectories PPO con calcolo GAE."""
    def __init__(self, capacity: int = IPERPARAMETRI['PPO_TRAJECTORY_LENGTH']):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.game_states = []

    def aggiungi(self, state, action, reward, value, log_prob, done, game_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.game_states.append(game_state)

    def __len__(self):
        return len(self.states)

    def is_full(self):
        return len(self) >= self.capacity

    def calcola_vantaggi_gae(self, next_value):
        """Calcola advantages con GAE-Lambda."""
        advantages = []
        gae = 0

        gamma = IPERPARAMETRI['PPO_GAE_GAMMA']
        lam = IPERPARAMETRI['PPO_GAE_LAMBDA']

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def ottieni_batch(self, advantages, returns):
        """Restituisce dati per training."""
        return {
            'states': self.states,
            'actions': self.actions,
            'old_log_probs': self.log_probs,
            'advantages': advantages,
            'returns': returns,
            'game_states': self.game_states
        }

class FrameStack:
    """Gestisce frame stacking 4x per input temporale."""
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, initial_frame):
        """Inizializza con frame iniziale ripetuto."""
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(initial_frame)

    def aggiungi(self, frame):
        """Aggiungi nuovo frame allo stack."""
        self.frames.append(frame)

    def ottieni_stack(self):
        """Ottieni stack corrente come array."""
        if TORCH_AVAILABLE:
            return torch.cat(list(self.frames), dim=0)
        return np.concatenate(list(self.frames), axis=0)

class CacheImmagini:
    """Cache LRU per immagini preprocessate."""
    def __init__(self, dimensione_max: int = IPERPARAMETRI['DIMENSIONE_CACHE']):
        self.cache = {}
        self.coda_accesso = deque()
        self.dimensione_max = dimensione_max
        self.contatori_accesso = {}
        self.soglia_pulizia = int(dimensione_max * 0.8)

    def _ottieni_hash_immagine(self, img: np.ndarray) -> str:
        return hashlib.md5(img.tobytes()).hexdigest()

    def ottieni(self, img: np.ndarray) -> Optional[np.ndarray]:
        h = self._ottieni_hash_immagine(img)
        if h in self.cache:
            try:
                self.coda_accesso.remove(h)
                self.coda_accesso.append(h)
            except ValueError: pass
            self.contatori_accesso[h] = self.contatori_accesso.get(h, 0) + 1
            return self.cache[h]
        return None

    def salva(self, img: np.ndarray, img_proc: np.ndarray):
        h = self._ottieni_hash_immagine(img)
        if h in self.cache: return

        if len(self.cache) >= self.soglia_pulizia: self._pulisci()

        if len(self.cache) >= self.dimensione_max:
            h_old = self.coda_accesso.popleft()
            self.cache.pop(h_old, None)
            self.contatori_accesso.pop(h_old, None)

        self.cache[h] = img_proc
        self.coda_accesso.append(h)
        self.contatori_accesso[h] = 1

    def _pulisci(self):
        if len(self.contatori_accesso) > 10:
            sorted_items = sorted(self.contatori_accesso.items(), key=lambda x: x[1])
            n_remove = int(len(sorted_items) * 0.2)
            for h, _ in sorted_items[:n_remove]:
                if h in self.cache:
                    del self.cache[h]
                    self.coda_accesso.remove(h)
                    del self.contatori_accesso[h]

class SalvatoreAsincrono:
    def __init__(self):
        self.thread = None
        self.coda = deque()
        self.attivo = False
        self.lock = threading.Lock()

    def salva_asincrono(self, func, dati):
        with self.lock:
            self.coda.append((func, dati))
            if not self.attivo:
                self.thread = threading.Thread(target=self._processa_coda)
                self.thread.daemon = True
                self.thread.start()
                self.attivo = True

    def _processa_coda(self):
        while True:
            with self.lock:
                if not self.coda:
                    self.attivo = False
                    break
                func, dati = self.coda.popleft()
            try:
                func(dati)
            except Exception as e:
                print(f"Errore salvataggio asincrono: {e}")

class LettoreMemoriaGioco:
    """Legge memoria del gioco per reward ed eventi con sistema multi-componente sofisticato."""
    INDIRIZZI_MEMORIA = {
        'SOLDI': (0xD347, 0xD349),
        'MEDAGLIE': 0xD356,
        'POKEDEX_POSSEDUTI': 0xD2F7,
        'POKEDEX_VISTI': 0xD30A,
        'LIVELLI_SQUADRA': [(0xD18C + i*44, 0xD18C + i*44 + 1) for i in range(6)],
        'HP_SQUADRA': [(0xD16D + i*44, 0xD16E + i*44) for i in range(6)],
        'HP_MAX_SQUADRA': [(0xD18D + i*44, 0xD18E + i*44) for i in range(6)],
        'ID_MAPPA': 0xD35E,
        'POS_X': 0xD361,
        'POS_Y': 0xD362,
        'EVENT_FLAGS': (0xD747, 0xD886),  # Event flags range (320 bytes)
        'TRAINER_FLAGS': (0xD5A0, 0xD5F7)  # Trainer defeat flags (88 bytes)
    }

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.stato_precedente = {}

        # Navigation tracking
        self.coordinate_visitate = set()
        self.ultima_posizione = None

        # Event tracking
        self.event_flags_precedenti = set()
        self.trainer_flags_precedenti = set()

        # Anti-grinding tracking
        self.wild_battle_count = 0
        self.ultimo_livello_medio = 0
        self.level_up_recenti = deque(maxlen=10)  # Track recent level-ups

    def ottieni_stato_corrente(self) -> Dict[str, Any]:
        try:
            mem = self.pyboy.get_memory_value

            soldi_bcd = [mem(addr) for addr in range(self.INDIRIZZI_MEMORIA['SOLDI'][0], self.INDIRIZZI_MEMORIA['SOLDI'][1] + 1)]
            soldi = sum(((b >> 4) * 10 + (b & 0xF)) * (100 ** i) for i, b in enumerate(reversed(soldi_bcd)))

            medaglie = bin(mem(self.INDIRIZZI_MEMORIA['MEDAGLIE'])).count('1')

            pokedex_posseduti = sum(bin(mem(addr)).count('1') for addr in range(self.INDIRIZZI_MEMORIA['POKEDEX_POSSEDUTI'], self.INDIRIZZI_MEMORIA['POKEDEX_POSSEDUTI'] + 19))
            pokedex_visti = sum(bin(mem(addr)).count('1') for addr in range(self.INDIRIZZI_MEMORIA['POKEDEX_VISTI'], self.INDIRIZZI_MEMORIA['POKEDEX_VISTI'] + 19))

            livelli_squadra = [mem(addr[0]) for addr in self.INDIRIZZI_MEMORIA['LIVELLI_SQUADRA']]
            hp_squadra = [mem(addr[0]) * 256 + mem(addr[1]) for addr in self.INDIRIZZI_MEMORIA['HP_SQUADRA']]
            hp_max_squadra = [mem(addr[0]) * 256 + mem(addr[1]) for addr in self.INDIRIZZI_MEMORIA['HP_MAX_SQUADRA']]

            in_battaglia = any(hp > 0 for hp in hp_squadra[:3])

            # Leggi event flags
            event_flags = set()
            for addr in range(self.INDIRIZZI_MEMORIA['EVENT_FLAGS'][0], self.INDIRIZZI_MEMORIA['EVENT_FLAGS'][1] + 1):
                byte_val = mem(addr)
                for bit in range(8):
                    if byte_val & (1 << bit):
                        event_flags.add((addr - self.INDIRIZZI_MEMORIA['EVENT_FLAGS'][0]) * 8 + bit)

            # Leggi trainer flags
            trainer_flags = set()
            for addr in range(self.INDIRIZZI_MEMORIA['TRAINER_FLAGS'][0], self.INDIRIZZI_MEMORIA['TRAINER_FLAGS'][1] + 1):
                byte_val = mem(addr)
                for bit in range(8):
                    if byte_val & (1 << bit):
                        trainer_flags.add((addr - self.INDIRIZZI_MEMORIA['TRAINER_FLAGS'][0]) * 8 + bit)

            return {
                'soldi_giocatore': soldi,
                'medaglie': medaglie,
                'pokedex_posseduti': pokedex_posseduti,
                'pokedex_visti': pokedex_visti,
                'livelli_squadra': livelli_squadra,
                'hp_squadra': hp_squadra,
                'hp_max_squadra': hp_max_squadra,
                'in_battaglia': in_battaglia,
                'id_mappa': mem(self.INDIRIZZI_MEMORIA['ID_MAPPA']),
                'pos_x': mem(self.INDIRIZZI_MEMORIA['POS_X']),
                'pos_y': mem(self.INDIRIZZI_MEMORIA['POS_Y']),
                'event_flags': event_flags,
                'trainer_flags': trainer_flags
            }
        except:
            return self.stato_precedente.copy() if self.stato_precedente else {}

    def calcola_ricompense_eventi(self, stato_corrente: Dict[str, Any]) -> float:
        if not self.stato_precedente or not stato_corrente:
            self.stato_precedente = stato_corrente.copy()
            return 0.0

        ricompensa_totale = 0.0
        rewards_dettaglio = {}

        # Sistema reward multi-componente sofisticato
        r = self._calcola_ricompense_medaglie(stato_corrente)
        if r != 0: rewards_dettaglio['medaglie'] = r
        ricompensa_totale += r

        r = self._calcola_ricompense_pokemon(stato_corrente)
        if r != 0: rewards_dettaglio['pokemon'] = r
        ricompensa_totale += r

        r = self._calcola_ricompense_livelli_bilanciato(stato_corrente)
        if r != 0: rewards_dettaglio['livelli'] = r
        ricompensa_totale += r

        r = self._calcola_ricompense_soldi(stato_corrente)
        if r != 0: rewards_dettaglio['soldi'] = r
        ricompensa_totale += r

        r = self._calcola_ricompense_esplorazione(stato_corrente)
        if r != 0: rewards_dettaglio['esplorazione'] = r
        ricompensa_totale += r

        r = self._calcola_ricompense_battaglia(stato_corrente)
        if r != 0: rewards_dettaglio['battaglia'] = r
        ricompensa_totale += r

        # Intrinsic Curiosity Module (ICM)
        r = self.calcola_ricompensa_curiosity(stato_corrente)
        if r != 0: rewards_dettaglio['curiosity'] = r
        ricompensa_totale += r

        # Advanced rewards
        r = self._calcola_ricompense_event_flags(stato_corrente)
        if r != 0: rewards_dettaglio['events'] = r
        ricompensa_totale += r

        r = self._calcola_ricompense_navigation(stato_corrente)
        if r != 0: rewards_dettaglio['navigation'] = r
        ricompensa_totale += r

        r = self._calcola_ricompense_healing(stato_corrente)
        if r != 0: rewards_dettaglio['healing'] = r
        ricompensa_totale += r

        # Log reward significativi
        if rewards_dettaglio:
            print(f"[REWARD] {rewards_dettaglio} = {ricompensa_totale:.2f}")

        self.stato_precedente = stato_corrente.copy()
        return ricompensa_totale

    def _calcola_ricompense_medaglie(self, s: Dict[str, Any]) -> float:
        # Medaglie = progressione principale del gioco
        return 2000 if s.get('medaglie', 0) > self.stato_precedente.get('medaglie', 0) else 0

    def _calcola_ricompense_pokemon(self, s: Dict[str, Any]) -> float:
        r = 0
        if s.get('pokedex_posseduti', 0) > self.stato_precedente.get('pokedex_posseduti', 0):
            # Catturare Pokemon = obiettivo importante
            r += 150 * (s.get('pokedex_posseduti', 0) - self.stato_precedente.get('pokedex_posseduti', 0))
        if s.get('pokedex_visti', 0) > self.stato_precedente.get('pokedex_visti', 0):
            # Vedere nuovi Pokemon = esplorazione utile
            r += 20 * (s.get('pokedex_visti', 0) - self.stato_precedente.get('pokedex_visti', 0))
        return r

    def _calcola_ricompense_livelli_bilanciato(self, s: Dict[str, Any]) -> float:
        """
        Balanced level reward: Incentiva level-up iniziali, scoraggia grinding eccessivo.
        Formula: reward = base_reward * diminishing_factor
        """
        lv_curr = s.get('livelli_squadra', [0] * 6)
        lv_prev = self.stato_precedente.get('livelli_squadra', [0] * 6)

        reward = 0.0

        for i in range(min(len(lv_curr), len(lv_prev))):
            if lv_curr[i] > lv_prev[i]:
                level_up_amount = lv_curr[i] - lv_prev[i]

                # Base reward diminuisce con il livello (anti-grinding)
                if lv_curr[i] <= 15:
                    base_reward = 50  # Early game: full reward
                elif lv_curr[i] <= 30:
                    base_reward = 30  # Mid game: reduced
                elif lv_curr[i] <= 45:
                    base_reward = 15  # Late game: minimal
                else:
                    base_reward = 5   # End game: very small

                # Penalizza grinding su wild Pokemon (se troppi level-up recenti)
                self.level_up_recenti.append(time.time())
                level_ups_last_minute = sum(1 for t in self.level_up_recenti if time.time() - t < 60)

                if level_ups_last_minute > 5:
                    # Troppi level-up in poco tempo = grinding
                    grinding_penalty = 0.3
                else:
                    grinding_penalty = 1.0

                reward += base_reward * level_up_amount * grinding_penalty

        return reward

    def _calcola_ricompense_soldi(self, s: Dict[str, Any]) -> float:
        diff = s.get('soldi_giocatore', 0) - self.stato_precedente.get('soldi_giocatore', 0)
        return min(diff / 100, 20) if diff > 0 else -20 if diff < -100 else 0

    def _calcola_ricompense_esplorazione(self, s: Dict[str, Any]) -> float:
        # Reward maggiore per cambiare mappa (importante per progressione)
        r = 80 if s.get('id_mappa', 0) != self.stato_precedente.get('id_mappa', 0) else 0
        # Reward per movimento significativo (incoraggia esplorazione attiva)
        diff_pos = abs(s.get('pos_x', 0) - self.stato_precedente.get('pos_x', 0)) + abs(s.get('pos_y', 0) - self.stato_precedente.get('pos_y', 0))
        return r + (8 if diff_pos > 5 else 0)

    def _calcola_ricompense_battaglia(self, s: Dict[str, Any]) -> float:
        prev_battle = self.stato_precedente.get('in_battaglia', False)
        curr_battle = s.get('in_battaglia', False)
        if prev_battle and not curr_battle:
            return 50 if any(hp > 0 for hp in s.get('hp_squadra', [])) else -100
        return 2 if not prev_battle and curr_battle else 0

    def _calcola_ricompense_event_flags(self, s: Dict[str, Any]) -> float:
        """
        Event Flags Reward: Premia eventi completati (trainer battles, quest progress, gym badges).
        Traccia event flags e trainer flags per incentivare progressione nella storia.
        """
        reward = 0.0

        # Event flags (quest progress, items, story events)
        event_flags_curr = s.get('event_flags', set())
        new_events = event_flags_curr - self.event_flags_precedenti

        if new_events:
            # Eventi della storia = progressione importante
            reward += 5.0 * len(new_events)
            self.event_flags_precedenti = event_flags_curr.copy()

        # Trainer flags (trainer battles vinti)
        trainer_flags_curr = s.get('trainer_flags', set())
        new_trainers = trainer_flags_curr - self.trainer_flags_precedenti

        if new_trainers:
            # Trainer battles valgono molto (non grinding, progressione obbligatoria)
            reward += 100.0 * len(new_trainers)
            self.trainer_flags_precedenti = trainer_flags_curr.copy()

        return reward

    def _calcola_ricompense_navigation(self, s: Dict[str, Any]) -> float:
        """
        Navigation Reward: Premia esplorazione di nuove coordinate.
        Incentiva l'esplorazione sistematica senza grinding nella stessa area.
        """
        pos_x = s.get('pos_x', 0)
        pos_y = s.get('pos_y', 0)
        id_mappa = s.get('id_mappa', 0)

        # Crea chiave unica per posizione (mappa, x, y)
        coord_key = (id_mappa, pos_x, pos_y)

        # Ricompensa solo se nuova coordinata
        if coord_key not in self.coordinate_visitate:
            self.coordinate_visitate.add(coord_key)
            return 2.0  # Aumentato per incentivare esplorazione sistematica

        return 0.0

    def _calcola_ricompense_healing(self, s: Dict[str, Any]) -> float:
        """
        Healing Reward: Proporzionale agli HP recuperati.
        Premia l'uso di Pokemon Centers e pozioni per mantenere la squadra in salute.
        """
        hp_curr = s.get('hp_squadra', [0] * 6)
        hp_max_curr = s.get('hp_max_squadra', [0] * 6)
        hp_prev = self.stato_precedente.get('hp_squadra', [0] * 6)

        reward = 0.0

        for i in range(min(len(hp_curr), len(hp_prev), len(hp_max_curr))):
            if hp_max_curr[i] > 0:  # Pokemon esiste
                hp_recovered = hp_curr[i] - hp_prev[i]

                # Ricompensa solo se HP aumentati (healing)
                if hp_recovered > 0:
                    # Reward proporzionale a % HP recuperati
                    recovery_percent = hp_recovered / hp_max_curr[i]
                    reward += recovery_percent * 5.0  # Max 5 reward per full heal

        return reward

    def calcola_ricompensa_curiosity(self, state_corrente: Dict[str, Any]) -> float:
        """
        Intrinsic Curiosity Module (ICM) reward.
        Ricompensa per esplorare stati nuovi del gioco.
        """
        reward = 0.0

        # Curiosity per nuove mappe visitate
        current_map = state_corrente.get('id_mappa', 0)
        if not hasattr(self, 'maps_visited'):
            self.maps_visited = set()

        if current_map not in self.maps_visited:
            self.maps_visited.add(current_map)
            reward += 100.0  # Grande bonus per nuova mappa (aumentato da 50 a 100)

        # Curiosity per nuovi Pokemon visti/catturati
        pokedex_visti = state_corrente.get('pokedex_visti', 0)
        if pokedex_visti > self.stato_precedente.get('pokedex_visti', 0):
            reward += 50.0 * (pokedex_visti - self.stato_precedente.get('pokedex_visti', 0))

        return reward

class RilevatorStatoGioco:
    @lru_cache(maxsize=50)
    def _calc_feat(self, h: str, b: bytes) -> Tuple[float, float]:
        img = np.frombuffer(b, dtype=np.uint8).reshape((144, 160))
        hp_var = np.var(img[REGIONI_SCHERMO['BARRA_HP']])
        menu_edges = cv2.Canny(img[REGIONI_SCHERMO['AREA_MENU']].astype(np.uint8), 50, 150)
        return hp_var, np.sum(menu_edges > 0) / menu_edges.size

    def rileva_battaglia(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160): return False
        try:
            h = hashlib.md5(scr.tobytes()).hexdigest()
            hp_var, edge_dens = self._calc_feat(h, scr.tobytes())
            return hp_var > IPERPARAMETRI['SOGLIA_HP'] and edge_dens > 0.1
        except: return False

    def rileva_menu(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160): return False
        try:
            edges = cv2.Canny(scr.astype(np.uint8), 50, 150)
            edge_dens = np.sum(edges > 0) / edges.size
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            return edge_dens > IPERPARAMETRI['SOGLIA_MENU'] and (len(lines) if lines is not None else 0) > 5
        except: return False

    def rileva_dialogo(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160): return False
        try:
            dlg = scr[REGIONI_SCHERMO['CASELLA_DIALOGO']]
            edges = cv2.Canny(dlg.astype(np.uint8), 50, 150)
            return np.std(dlg) > IPERPARAMETRI['SOGLIA_DIALOGO'] and np.sum(edges[0, :]) > 20 and np.sum(edges[-1, :]) > 20
        except: return False

def calcola_dimensioni_output_conv(altezza_input, larghezza_input, layer_conv):
    h, w = altezza_input, larghezza_input
    for dimensione_kernel, stride in layer_conv:
        h = (h - dimensione_kernel) // stride + 1
        w = (w - dimensione_kernel) // stride + 1
    return h, w

DIMENSIONI_CONV_ESPLORAZIONE = calcola_dimensioni_output_conv(144, 160, [(8, 4), (4, 2), (3, 1)])
DIMENSIONI_CONV_MENU = calcola_dimensioni_output_conv(144, 160, [(4, 2), (3, 2)])

if TORCH_AVAILABLE:
    class ReteActorCriticBase(nn.Module):
        """Base Actor-Critic network con shared backbone."""
        def __init__(self, n_azioni, input_channels=4):
            super(ReteActorCriticBase, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

            altezza_conv, larghezza_conv = DIMENSIONI_CONV_ESPLORAZIONE
            dimensione_input_lineare = altezza_conv * larghezza_conv * 64

            self.fc_shared = nn.Linear(dimensione_input_lineare, 512)

            self.policy_head = nn.Linear(512, n_azioni)
            self.value_head = nn.Linear(512, 1)

            self._initialize_weights()

        def _initialize_weights(self):
            """Orthogonal initialization per stabilità."""
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc_shared(x))

            policy_logits = self.policy_head(x)
            value = self.value_head(x)

            return policy_logits, value

    class ReteEsplorazionePPO(ReteActorCriticBase):
        """PPO Actor-Critic per esplorazione."""
        pass

    class ReteBattagliaPPO(ReteActorCriticBase):
        """PPO Actor-Critic per battaglia (più capacità)."""
        def __init__(self, n_azioni, input_channels=4):
            super(ReteBattagliaPPO, self).__init__(n_azioni, input_channels)
            altezza_conv, larghezza_conv = DIMENSIONI_CONV_ESPLORAZIONE
            dimensione_input_lineare = altezza_conv * larghezza_conv * 64

            self.fc_shared = nn.Linear(dimensione_input_lineare, 512)
            self.fc_shared2 = nn.Linear(512, 256)

            self.policy_head = nn.Linear(256, n_azioni)
            self.value_head = nn.Linear(256, 1)

            self._initialize_weights()

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc_shared(x))
            x = F.relu(self.fc_shared2(x))

            policy_logits = self.policy_head(x)
            value = self.value_head(x)

            return policy_logits, value

    class ReteMenuPPO(nn.Module):
        """PPO Actor-Critic per menu (più piccolo)."""
        def __init__(self, n_azioni, input_channels=4):
            super(ReteMenuPPO, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

            altezza_conv, larghezza_conv = DIMENSIONI_CONV_MENU
            dimensione_input_lineare = altezza_conv * larghezza_conv * 32

            self.fc_shared = nn.Linear(dimensione_input_lineare, 128)

            self.policy_head = nn.Linear(128, n_azioni)
            self.value_head = nn.Linear(128, 1)

            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc_shared(x))

            policy_logits = self.policy_head(x)
            value = self.value_head(x)

            return policy_logits, value

    class GruppoRetiPPO:
        """Gestore reti PPO multiple per stati di gioco."""
        def __init__(self, n_azioni: int, device: Any, input_channels: int = 4) -> None:
            self.device = device
            self.n_azioni = n_azioni

            self.rete_esplorazione = ReteEsplorazionePPO(n_azioni, input_channels).to(device)
            self.rete_battaglia = ReteBattagliaPPO(n_azioni, input_channels).to(device)
            self.rete_menu = ReteMenuPPO(n_azioni, input_channels).to(device)

            self.ottimizzatore_esplorazione = optim.Adam(self.rete_esplorazione.parameters(), lr=IPERPARAMETRI['PPO_LR'])
            self.ottimizzatore_battaglia = optim.Adam(self.rete_battaglia.parameters(), lr=IPERPARAMETRI['PPO_LR'])
            self.ottimizzatore_menu = optim.Adam(self.rete_menu.parameters(), lr=IPERPARAMETRI['PPO_LR'])

        def seleziona_rete(self, game_state: str):
            """Seleziona rete e ottimizzatore per game state."""
            if game_state == "battle":
                return self.rete_battaglia, self.ottimizzatore_battaglia
            elif game_state == "menu":
                return self.rete_menu, self.ottimizzatore_menu
            else:
                return self.rete_esplorazione, self.ottimizzatore_esplorazione

        def scegli_azione(self, state: torch.Tensor, game_state: str, deterministic: bool = False,
                          action_mask: Optional[List[float]] = None):
            """Scelta azione con policy stocastica e optional action masking."""
            rete, _ = self.seleziona_rete(game_state)
            rete.eval()

            with torch.no_grad():
                state_batch = state.unsqueeze(0) if state.dim() == 3 else state
                policy_logits, value = rete(state_batch)

                # Applica action mask se fornito (context-aware filtering)
                if action_mask is not None:
                    policy_logits = ContextAwareActionFilter.apply_mask_to_logits(
                        policy_logits, action_mask
                    )

                dist = Categorical(logits=policy_logits)

                if deterministic:
                    action = policy_logits.argmax(dim=-1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

        def addestra_ppo(self, batch_data: Dict, game_state: str, entropy_coeff: Optional[float] = None) -> Dict[str, float]:
            """Training PPO con clipped surrogate objective e adaptive entropy."""
            rete, ottimizzatore = self.seleziona_rete(game_state)
            rete.train()

            # Usa entropy coefficient fornito o default
            if entropy_coeff is None:
                entropy_coeff = IPERPARAMETRI['PPO_ENTROPY_COEFF']

            states = torch.stack(batch_data['states']).to(self.device)
            actions = torch.tensor(batch_data['actions'], dtype=torch.long).to(self.device)
            old_log_probs = torch.tensor(batch_data['old_log_probs'], dtype=torch.float32).to(self.device)
            advantages = torch.tensor(batch_data['advantages'], dtype=torch.float32).to(self.device)
            returns = torch.tensor(batch_data['returns'], dtype=torch.float32).to(self.device)

            # Normalizza advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            n_updates = 0

            dataset_size = len(states)
            minibatch_size = IPERPARAMETRI['PPO_MINIBATCH_SIZE']

            for epoch in range(IPERPARAMETRI['PPO_EPOCHS']):
                indices = torch.randperm(dataset_size)

                for start in range(0, dataset_size, minibatch_size):
                    end = min(start + minibatch_size, dataset_size)
                    mb_indices = indices[start:end]

                    mb_states = states[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]

                    policy_logits, values = rete(mb_states)
                    dist = Categorical(logits=policy_logits)

                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - IPERPARAMETRI['PPO_CLIP_EPSILON'],
                                       1.0 + IPERPARAMETRI['PPO_CLIP_EPSILON']) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values.squeeze(), mb_returns)

                    loss = (policy_loss +
                           IPERPARAMETRI['PPO_VALUE_COEFF'] * value_loss -
                           entropy_coeff * entropy)

                    ottimizzatore.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(rete.parameters(), IPERPARAMETRI['PPO_MAX_GRAD_NORM'])
                    ottimizzatore.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    n_updates += 1

            return {
                'policy_loss': total_policy_loss / n_updates,
                'value_loss': total_value_loss / n_updates,
                'entropy': total_entropy / n_updates
            }

class AgentePokemonAI:
    """Agente AI Pokemon con PPO."""
    def __init__(self, rom_path: str, headless: bool = False) -> None:
        if not TORCH_AVAILABLE or not DEPS_AVAILABLE:
            raise ErroreAIPokemon("Missing dependencies. Install: torch, pyboy, keyboard, PIL, cv2")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Supporto per training parallelo
        self.use_shared_buffer = False  # Impostato a True dal parallel_trainer
        self.shared_buffer = None
        self.rank = 0  # Rank del worker (0 = visibile, >0 = headless)

        # PyBoy con finestra ottimizzata
        window_type = "headless" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type)

        # Imposta velocita emulazione (0=illimitata, 1=normale, 2=2x, etc.)
        emulation_speed = IPERPARAMETRI['EMULATION_SPEED']
        self.pyboy.set_emulation_speed(emulation_speed)
        speed_desc = "illimitata" if emulation_speed == 0 else f"{emulation_speed}x"
        print(f"[INFO] Velocita emulazione: {speed_desc}")

        # PyBoy 2.6.0+ usa API basata su stringhe per i bottoni
        self.actions = [
            None,      # noop
            'up',      # up arrow
            'down',    # down arrow
            'left',    # left arrow
            'right',   # right arrow
            'a',       # A button
            'b',       # B button
            'start',   # start button
            'select'   # select button
        ]

        rom_name = os.path.splitext(os.path.basename(rom_path))[0]
        self.save_dir = f"pokemon_ai_saves_{rom_name}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_path = os.path.join(self.save_dir, "model_ppo.pth")
        self.stats_path = os.path.join(self.save_dir, "stats_ppo.json")
        self.game_state_path = os.path.join(self.save_dir, "game_state.state")

        input_channels = IPERPARAMETRI['FRAME_STACK']
        self.gruppo_reti = GruppoRetiPPO(len(self.actions), self.device, input_channels)

        self.trajectory_buffer = BufferTraiettorie()
        self.frame_stack = FrameStack(IPERPARAMETRI['FRAME_STACK'])
        self.cache_immagini = CacheImmagini()
        self.lettore_memoria = LettoreMemoriaGioco(self.pyboy)
        self.rilevatore_stato = RilevatorStatoGioco()
        self.salvatore_asincrono = SalvatoreAsincrono()

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
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        return {'episodes': 0, 'total_frames': 0, 'best_reward': float('-inf')}

    def _save_stats(self):
        final_state = self.lettore_memoria.ottieni_stato_corrente()
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

    def _load_checkpoint(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.gruppo_reti.rete_esplorazione.load_state_dict(checkpoint['explorer_state'])
                self.gruppo_reti.rete_battaglia.load_state_dict(checkpoint['battle_state'])
                self.gruppo_reti.rete_menu.load_state_dict(checkpoint['menu_state'])
                self.episode_count = checkpoint.get('episode', 0)
                self.frame_count = checkpoint.get('frame', 0)
                print(f"[LOAD] Checkpoint caricato: episodio {self.episode_count}, frame {self.frame_count}")

                # Carica stato del gioco se esistente
                if os.path.exists(self.game_state_path):
                    try:
                        with open(self.game_state_path, 'rb') as f:
                            self.pyboy.load_state(f)
                        print(f"[LOAD] Stato del gioco caricato da {self.game_state_path}")
                    except Exception as e:
                        print(f"[WARN] Impossibile caricare stato gioco: {e}")
                        print("[INFO] Partenza da inizio gioco")
            except Exception as e:
                print(f"[ERROR] Errore caricamento checkpoint: {e}")

    def _save_checkpoint(self):
        checkpoint = {
            'explorer_state': self.gruppo_reti.rete_esplorazione.state_dict(),
            'battle_state': self.gruppo_reti.rete_battaglia.state_dict(),
            'menu_state': self.gruppo_reti.rete_menu.state_dict(),
            'episode': self.episode_count,
            'frame': self.frame_count
        }
        self.salvatore_asincrono.salva_asincrono(lambda d: torch.save(d, self.model_path), checkpoint)

        # Salva stato del gioco
        try:
            with open(self.game_state_path, 'wb') as f:
                self.pyboy.save_state(f)
        except Exception as e:
            print(f"[WARN] Impossibile salvare stato gioco: {e}")

    def _get_screen_tensor(self):
        """Ottieni frame corrente preprocessato."""
        screen = self.pyboy.screen.image  # PyBoy 2.6.0+ API
        gray = np.array(screen.convert('L'))
        self.last_screen_array = gray.copy()

        cached = self.cache_immagini.ottieni(gray)
        if cached is not None:
            normalized = cached
        else:
            normalized = gray.astype(np.float32) / 255.0
            self.cache_immagini.salva(gray, normalized)

        tensor = torch.from_numpy(normalized).unsqueeze(0)
        return tensor.to(self.device)

    def _detect_game_state(self, screen_array):
        if self.rilevatore_stato.rileva_battaglia(screen_array):
            self.current_game_state = "battle"
        elif self.rilevatore_stato.rileva_dialogo(screen_array):
            self.current_game_state = "dialogue"
        elif self.rilevatore_stato.rileva_menu(screen_array):
            self.current_game_state = "menu"
        else:
            self.current_game_state = "exploring"

        return self.current_game_state

    def _calculate_reward(self):
        """
        Calcola reward basato SOLO su progressione di gioco (memoria).
        Niente reward per movimento generico - previene convergenza prematura.
        """
        reward = 0

        memory_state = self.lettore_memoria.ottieni_stato_corrente()
        memory_reward = self.lettore_memoria.calcola_ricompense_eventi(memory_state)
        reward += memory_reward

        # Anti-Loop System: DISABILITATO se flag è False
        if IPERPARAMETRI['ANTI_LOOP_ENABLED']:
            # Anti-Loop System: penalità per comportamenti ripetitivi
            loop_penalty = self.anti_loop_buffer.calculate_loop_penalty()
            if loop_penalty < 0:
                reward += loop_penalty
                # Log quando detectato loop (solo occasionalmente per non spammare)
                if self.frame_count % 5000 == 0:  # Ogni 5000 frame invece di 100
                    print(f"[ANTI-LOOP] Loop detectato! Penalità: {loop_penalty:.2f}")

            # Exploration bonus: reward per esplorare attivamente
            exploration_bonus = self.anti_loop_buffer.get_exploration_bonus()
            if exploration_bonus > 0:
                reward += exploration_bonus

        self.reward_history.append(reward)
        self.total_reward += reward
        return reward

    def avvia_training(self) -> None:
        """Loop principale di training PPO con rendering fluido."""
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

                # Controlli velocita: + e - per aumentare/diminuire
                if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                    current_speed = IPERPARAMETRI['EMULATION_SPEED']
                    IPERPARAMETRI['EMULATION_SPEED'] = min(current_speed + 1, 10)
                    self.pyboy.set_emulation_speed(IPERPARAMETRI['EMULATION_SPEED'])
                    print(f"[INFO] Velocita: {IPERPARAMETRI['EMULATION_SPEED']}x")
                    time.sleep(0.2)
                if keyboard.is_pressed('-') or keyboard.is_pressed('_'):
                    current_speed = IPERPARAMETRI['EMULATION_SPEED']
                    IPERPARAMETRI['EMULATION_SPEED'] = max(current_speed - 1, 0)
                    self.pyboy.set_emulation_speed(IPERPARAMETRI['EMULATION_SPEED'])
                    speed_desc = "illimitata" if IPERPARAMETRI['EMULATION_SPEED'] == 0 else f"{IPERPARAMETRI['EMULATION_SPEED']}x"
                    print(f"[INFO] Velocita: {speed_desc}")
                    time.sleep(0.2)

                if paused:
                    self.pyboy.tick()
                    continue

                # Cattura stato corrente
                single_frame = self._get_screen_tensor()
                self.frame_stack.aggiungi(single_frame)
                stacked_state = self.frame_stack.ottieni_stack()

                # Context-Aware Action Masking
                action_mask = self.action_filter.get_action_mask(self.current_game_state)

                # L'AI decide l'azione da eseguire (con action masking)
                action, log_prob, value = self.gruppo_reti.scegli_azione(
                    stacked_state, self.current_game_state, action_mask=action_mask
                )

                # Esegui azione
                button = self.actions[action]
                if button is not None:
                    self.pyboy.button_press(button)

                # Frameskip adattivo basato sullo stato di gioco
                frameskip_map = {
                    "dialogue": IPERPARAMETRI['FRAMESKIP_DIALOGUE'],
                    "battle": IPERPARAMETRI['FRAMESKIP_BATTLE'],
                    "menu": IPERPARAMETRI['FRAMESKIP_MENU'],
                    "exploring": IPERPARAMETRI['FRAMESKIP_EXPLORING']
                }
                frameskip = frameskip_map.get(self.current_game_state,
                                             IPERPARAMETRI['FRAMESKIP_BASE'])

                # Rendering fluido senza lag
                if IPERPARAMETRI['RENDER_ENABLED']:
                    # Renderizza ogni N frame per fluidita
                    render_freq = IPERPARAMETRI['RENDER_EVERY_N_FRAMES']
                    for i in range(frameskip):
                        should_render = (i % render_freq == 0)
                        self.pyboy.tick(1, render=should_render)
                        render_counter += 1
                else:
                    # Modalita headless: massima velocita
                    self.pyboy.tick(count=frameskip, render=False)

                # Rilascia bottone
                if button is not None:
                    self.pyboy.button_release(button)

                # Cattura prossimo stato
                next_single_frame = self._get_screen_tensor()

                # Aggiorna Anti-Loop Buffer solo se abilitato
                if IPERPARAMETRI['ANTI_LOOP_ENABLED']:
                    mem_state = self.lettore_memoria.ottieni_stato_corrente()
                    self.anti_loop_buffer.add_state(
                        mem_state.get('pos_x', 0),
                        mem_state.get('pos_y', 0),
                        mem_state.get('id_mappa', 0),
                        action
                    )

                reward = self._calculate_reward()

                # Monitoraggio performance
                perf_frame_count += 1
                if perf_frame_count % IPERPARAMETRI['PERFORMANCE_LOG_INTERVAL'] == 0:
                    elapsed = time.time() - perf_start_time
                    fps = perf_frame_count / elapsed
                    avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0

                    # Calcola entropy corrente
                    current_entropy = self.entropy_scheduler.get_entropy(self.frame_count)

                    # Stato gioco dettagliato
                    mem_state = self.lettore_memoria.ottieni_stato_corrente()
                    print(f"[PERF] {fps:.1f} FPS | Frame: {self.frame_count} | "
                          f"Stato: {self.current_game_state} | Reward Medio: {avg_reward:.2f}")
                    print(f"[GAME] Medaglie: {mem_state.get('medaglie', 0)} | "
                          f"Pokedex: {mem_state.get('pokedex_posseduti', 0)}/{mem_state.get('pokedex_visti', 0)} | "
                          f"Mappa: {mem_state.get('id_mappa', 0)} | "
                          f"Pos: ({mem_state.get('pos_x', 0)},{mem_state.get('pos_y', 0)})")
                    print(f"[ADAPTIVE] Entropy: {current_entropy:.4f} | "
                          f"Exploration: {'High' if current_entropy > 0.03 else 'Medium' if current_entropy > 0.01 else 'Low'}")

                    perf_start_time = time.time()
                    perf_frame_count = 0

                # Aggiungi a trajectory buffer
                self.trajectory_buffer.aggiungi(
                    stacked_state.cpu(), action, reward, value,
                    log_prob, False, self.current_game_state
                )

                # Training quando buffer è pieno
                if self.trajectory_buffer.is_full():
                    self.frame_stack.aggiungi(next_single_frame)
                    next_stacked_state = self.frame_stack.ottieni_stack()
                    _, _, next_value = self.gruppo_reti.scegli_azione(
                        next_stacked_state, self.current_game_state, deterministic=True
                    )

                    advantages, returns = self.trajectory_buffer.calcola_vantaggi_gae(next_value)

                    # Calcola entropy coefficient adattivo basato sui frame totali
                    current_entropy_coeff = self.entropy_scheduler.get_entropy(self.frame_count)

                    # Train per game state
                    game_states_in_buffer = set(self.trajectory_buffer.game_states)
                    for gs in game_states_in_buffer:
                        indices = [i for i, g in enumerate(self.trajectory_buffer.game_states) if g == gs]
                        if len(indices) < IPERPARAMETRI['PPO_MINIBATCH_SIZE']:
                            continue

                        batch_data = {
                            'states': [self.trajectory_buffer.states[i] for i in indices],
                            'actions': [self.trajectory_buffer.actions[i] for i in indices],
                            'old_log_probs': [self.trajectory_buffer.log_probs[i] for i in indices],
                            'advantages': [advantages[i] for i in indices],
                            'returns': [returns[i] for i in indices],
                            'game_states': [gs] * len(indices)
                        }

                        # Training con entropy coefficient adattivo
                        metrics = self.gruppo_reti.addestra_ppo(batch_data, gs, entropy_coeff=current_entropy_coeff)
                        self.loss_history.append(metrics['policy_loss'])

                        if self.frame_count % 100 == 0:
                            print(f"[TRAIN] Frame {self.frame_count} [{gs}] - "
                                  f"Policy Loss: {metrics['policy_loss']:.4f}, "
                                  f"Value Loss: {metrics['value_loss']:.4f}, "
                                  f"Entropy: {metrics['entropy']:.4f}")

                    self.trajectory_buffer.reset()

                # Salvataggio checkpoint
                if self.frame_count - last_save_frame >= IPERPARAMETRI['FREQUENZA_SALVATAGGIO']:
                    self._save_checkpoint()
                    self._save_stats()
                    last_save_frame = self.frame_count
                    print(f"[SAVE] Checkpoint salvato al frame {self.frame_count}")

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\n[INFO] Training interrotto dall'utente")
        finally:
            self._save_checkpoint()
            self._save_stats()
            self.pyboy.stop()
            print(f"[INFO] Training completato. Frame: {self.frame_count}, Reward Totale: {self.total_reward:.2f}")

def principale() -> None:
    while True:
        rom_path = input("Pokemon ROM path (.gb/.gbc): ").strip().strip('"')
        if os.path.exists(rom_path) and (rom_path.lower().endswith('.gbc') or rom_path.lower().endswith('.gb')) and os.path.getsize(rom_path) > 0:
            break

    headless = input("Headless mode (y/N): ").lower().strip() == 'y'

    try:
        agente = AgentePokemonAI(rom_path, headless=headless)
        agente.avvia_training()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    principale()
