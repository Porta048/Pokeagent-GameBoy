import os, sys, time, random, threading, json, pickle, hashlib
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
    'PPO_ENTROPY_COEFF': 0.01,
    'PPO_GAE_LAMBDA': 0.95,
    'PPO_GAE_GAMMA': 0.99,
    'PPO_EPOCHS': 3,
    'PPO_MINIBATCH_SIZE': 32,
    'PPO_TRAJECTORY_LENGTH': 512,
    'PPO_LR': 3e-4,
    'PPO_MAX_GRAD_NORM': 0.5,
    'FRAME_STACK': 4,
    # Emulator Optimizations
    'FRAMESKIP_BASE': 6,              # Base frameskip (4-8 recommended)
    'FRAMESKIP_DIALOGUE': 3,          # Lower for dialogue (need to catch text)
    'FRAMESKIP_BATTLE': 8,            # Higher for battle (faster)
    'FRAMESKIP_EXPLORING': 6,         # Balanced for exploration
    'FRAMESKIP_MENU': 4,              # Moderate for menu navigation
    'RENDER_ENABLED': False,          # Disable rendering for speed
    'PERFORMANCE_LOG_INTERVAL': 1000  # Log performance every N frames
}

class ErroreAIPokemon(Exception): pass

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

        ricompensa = 0.0

        # Sistema reward multi-componente sofisticato
        ricompensa += self._calcola_ricompense_medaglie(stato_corrente)
        ricompensa += self._calcola_ricompense_pokemon(stato_corrente)
        ricompensa += self._calcola_ricompense_livelli_bilanciato(stato_corrente)  # NEW: Balanced
        ricompensa += self._calcola_ricompense_soldi(stato_corrente)
        ricompensa += self._calcola_ricompense_esplorazione(stato_corrente)
        ricompensa += self._calcola_ricompense_battaglia(stato_corrente)

        # NEW: Advanced rewards
        ricompensa += self._calcola_ricompense_event_flags(stato_corrente)  # Event flags
        ricompensa += self._calcola_ricompense_navigation(stato_corrente)   # Navigation exploration
        ricompensa += self._calcola_ricompense_healing(stato_corrente)      # Healing rewards

        self.stato_precedente = stato_corrente.copy()
        return ricompensa

    def _calcola_ricompense_medaglie(self, s: Dict[str, Any]) -> float:
        return 1000 if s.get('medaglie', 0) > self.stato_precedente.get('medaglie', 0) else 0

    def _calcola_ricompense_pokemon(self, s: Dict[str, Any]) -> float:
        r = 0
        if s.get('pokedex_posseduti', 0) > self.stato_precedente.get('pokedex_posseduti', 0):
            r += 100 * (s.get('pokedex_posseduti', 0) - self.stato_precedente.get('pokedex_posseduti', 0))
        if s.get('pokedex_visti', 0) > self.stato_precedente.get('pokedex_visti', 0):
            r += 10 * (s.get('pokedex_visti', 0) - self.stato_precedente.get('pokedex_visti', 0))
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
        r = 30 if s.get('id_mappa', 0) != self.stato_precedente.get('id_mappa', 0) else 0
        diff_pos = abs(s.get('pos_x', 0) - self.stato_precedente.get('pos_x', 0)) + abs(s.get('pos_y', 0) - self.stato_precedente.get('pos_y', 0))
        return r + (2 if diff_pos > 5 else 0)

    def _calcola_ricompense_battaglia(self, s: Dict[str, Any]) -> float:
        prev_battle = self.stato_precedente.get('in_battaglia', False)
        curr_battle = s.get('in_battaglia', False)
        if prev_battle and not curr_battle:
            return 50 if any(hp > 0 for hp in s.get('hp_squadra', [])) else -100
        return 2 if not prev_battle and curr_battle else 0

    def _calcola_ricompense_event_flags(self, s: Dict[str, Any]) -> float:
        """
        Event Flags Reward: +2 per ogni evento completato (trainer battles, quest progress, gym badges).
        Traccia event flags e trainer flags per incentivare progressione nella storia.
        """
        reward = 0.0

        # Event flags (quest progress, items, story events)
        event_flags_curr = s.get('event_flags', set())
        new_events = event_flags_curr - self.event_flags_precedenti

        if new_events:
            reward += 2.0 * len(new_events)
            self.event_flags_precedenti = event_flags_curr.copy()

        # Trainer flags (trainer battles vinti)
        trainer_flags_curr = s.get('trainer_flags', set())
        new_trainers = trainer_flags_curr - self.trainer_flags_precedenti

        if new_trainers:
            # Trainer battles valgono di più (non grinding)
            reward += 2.0 * len(new_trainers)
            self.trainer_flags_precedenti = trainer_flags_curr.copy()

        return reward

    def _calcola_ricompense_navigation(self, s: Dict[str, Any]) -> float:
        """
        Navigation Reward: +0.005 per ogni nuova coordinata visitata.
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
            return 0.005

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

        def scegli_azione(self, state: torch.Tensor, game_state: str, deterministic: bool = False):
            """Scelta azione con policy stocastica."""
            rete, _ = self.seleziona_rete(game_state)
            rete.eval()

            with torch.no_grad():
                state_batch = state.unsqueeze(0) if state.dim() == 3 else state
                policy_logits, value = rete(state_batch)

                dist = Categorical(logits=policy_logits)

                if deterministic:
                    action = policy_logits.argmax(dim=-1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

        def addestra_ppo(self, batch_data: Dict, game_state: str) -> Dict[str, float]:
            """Training PPO con clipped surrogate objective."""
            rete, ottimizzatore = self.seleziona_rete(game_state)
            rete.train()

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
                           IPERPARAMETRI['PPO_ENTROPY_COEFF'] * entropy)

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

        self.pyboy = PyBoy(rom_path, window="headless" if headless else "SDL2")
        self.pyboy.set_emulation_speed(0)

        # PyBoy 2.6.0+ uses string-based button API
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

        input_channels = IPERPARAMETRI['FRAME_STACK']
        self.gruppo_reti = GruppoRetiPPO(len(self.actions), self.device, input_channels)

        self.trajectory_buffer = BufferTraiettorie()
        self.frame_stack = FrameStack(IPERPARAMETRI['FRAME_STACK'])
        self.cache_immagini = CacheImmagini()
        self.lettore_memoria = LettoreMemoriaGioco(self.pyboy)
        self.rilevatore_stato = RilevatorStatoGioco()
        self.salvatore_asincrono = SalvatoreAsincrono()

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
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.gruppo_reti.rete_esplorazione.load_state_dict(checkpoint['explorer_state'])
                self.gruppo_reti.rete_battaglia.load_state_dict(checkpoint['battle_state'])
                self.gruppo_reti.rete_menu.load_state_dict(checkpoint['menu_state'])
                self.episode_count = checkpoint.get('episode', 0)
                self.frame_count = checkpoint.get('frame', 0)
                print(f"Checkpoint loaded: episode {self.episode_count}, frame {self.frame_count}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

    def _save_checkpoint(self):
        checkpoint = {
            'explorer_state': self.gruppo_reti.rete_esplorazione.state_dict(),
            'battle_state': self.gruppo_reti.rete_battaglia.state_dict(),
            'menu_state': self.gruppo_reti.rete_menu.state_dict(),
            'episode': self.episode_count,
            'frame': self.frame_count
        }
        self.salvatore_asincrono.salva_asincrono(lambda d: torch.save(d, self.model_path), checkpoint)

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

    def _calculate_reward(self, screen_tensor, previous_screen):
        reward = 0

        if hasattr(self, 'last_screen_array'):
            game_state = self._detect_game_state(self.last_screen_array)
        else:
            game_state = "exploring"

        memory_state = self.lettore_memoria.ottieni_stato_corrente()
        memory_reward = self.lettore_memoria.calcola_ricompense_eventi(memory_state)
        reward += memory_reward

        if previous_screen is not None:
            diff = torch.abs(screen_tensor - previous_screen).mean().item()

            movement_threshold = 0.02
            if game_state == "dialogue":
                movement_threshold = 0.005
            elif game_state == "battle":
                movement_threshold = 0.01

            if diff > movement_threshold:
                reward += min(diff * 10, 2.0)
            elif diff < movement_threshold * 0.1:
                reward -= 0.05

        self.reward_history.append(reward)
        self.total_reward += reward
        return reward

    def avvia_training(self) -> None:
        """Main PPO training loop with emulator optimizations."""
        paused = False
        previous_single_frame = None
        last_save_frame = 0

        # Performance tracking
        perf_start_time = time.time()
        perf_frame_count = 0

        # Initialize frame stack
        initial_frame = self._get_screen_tensor()
        self.frame_stack.reset(initial_frame)

        try:
            while True:
                if keyboard.is_pressed('escape'):
                    break

                if keyboard.is_pressed('space'):
                    paused = not paused
                    time.sleep(0.3)

                if paused:
                    self.pyboy.tick()
                    continue

                single_frame = self._get_screen_tensor()
                self.frame_stack.aggiungi(single_frame)
                stacked_state = self.frame_stack.ottieni_stack()

                action, log_prob, value = self.gruppo_reti.scegli_azione(
                    stacked_state, self.current_game_state
                )

                # PyBoy 2.6.0+ API: use button_press/button_release
                button = self.actions[action]
                if button is not None:  # Skip noop
                    self.pyboy.button_press(button)

                # OPTIMIZATION: Adaptive frameskipping based on game state
                frameskip_map = {
                    "dialogue": IPERPARAMETRI['FRAMESKIP_DIALOGUE'],
                    "battle": IPERPARAMETRI['FRAMESKIP_BATTLE'],
                    "menu": IPERPARAMETRI['FRAMESKIP_MENU'],
                    "exploring": IPERPARAMETRI['FRAMESKIP_EXPLORING']
                }
                frameskip = frameskip_map.get(self.current_game_state, IPERPARAMETRI['FRAMESKIP_BASE'])

                # OPTIMIZATION: Use tick(count=X, render=False) for speed
                self.pyboy.tick(count=frameskip, render=IPERPARAMETRI['RENDER_ENABLED'])

                # Release button after ticking
                if button is not None:
                    self.pyboy.button_release(button)

                next_single_frame = self._get_screen_tensor()
                reward = self._calculate_reward(next_single_frame, previous_single_frame)

                # Performance monitoring
                perf_frame_count += 1
                if perf_frame_count % IPERPARAMETRI['PERFORMANCE_LOG_INTERVAL'] == 0:
                    elapsed = time.time() - perf_start_time
                    fps = perf_frame_count / elapsed
                    print(f"Performance: {fps:.1f} FPS (avg), Frame: {self.frame_count}, State: {self.current_game_state}")
                    perf_start_time = time.time()
                    perf_frame_count = 0

                self.trajectory_buffer.aggiungi(
                    stacked_state.cpu(),
                    action,
                    reward,
                    value,
                    log_prob,
                    False,
                    self.current_game_state
                )

                if self.trajectory_buffer.is_full():
                    # Calcola GAE e addestra
                    self.frame_stack.aggiungi(next_single_frame)
                    next_stacked_state = self.frame_stack.ottieni_stack()
                    _, _, next_value = self.gruppo_reti.scegli_azione(
                        next_stacked_state, self.current_game_state, deterministic=True
                    )

                    advantages, returns = self.trajectory_buffer.calcola_vantaggi_gae(next_value)

                    # Train su ciascun game state presente nel buffer
                    game_states_in_buffer = set(self.trajectory_buffer.game_states)
                    for gs in game_states_in_buffer:
                        # Filtra batch per game state
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

                        metrics = self.gruppo_reti.addestra_ppo(batch_data, gs)
                        self.loss_history.append(metrics['policy_loss'])

                        if self.frame_count % 100 == 0:
                            print(f"Frame {self.frame_count} [{gs}] - "
                                  f"Policy Loss: {metrics['policy_loss']:.4f}, "
                                  f"Value Loss: {metrics['value_loss']:.4f}, "
                                  f"Entropy: {metrics['entropy']:.4f}, "
                                  f"Reward: {self.total_reward:.2f}")

                    self.trajectory_buffer.reset()

                if self.frame_count - last_save_frame >= IPERPARAMETRI['FREQUENZA_SALVATAGGIO']:
                    self._save_checkpoint()
                    self._save_stats()
                    last_save_frame = self.frame_count
                    print(f"Checkpoint saved at frame {self.frame_count}")

                previous_single_frame = next_single_frame
                self.frame_count += 1

        except KeyboardInterrupt:
            print("Training interrupted by user")

        finally:
            self._save_checkpoint()
            self._save_stats()
            self.pyboy.stop()
            print(f"Training completed. Total frames: {self.frame_count}, Total reward: {self.total_reward:.2f}")

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
