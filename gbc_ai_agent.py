import os, sys, time, random, threading, json, pickle, hashlib
from typing import Union, List, Dict, Optional, Tuple, Any
from collections import deque
from functools import lru_cache
import numpy as np

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
    'DIMENSIONE_BATCH': 64,
    'FATTORE_SCONTO': 0.99,
    'EPSILON_MINIMO': 0.05,
    'DECADIMENTO_EPSILON': 0.9995,
    'TASSO_APPRENDIMENTO': 0.00025,
    'DIMENSIONE_MEMORIA': 50000,
    'FREQUENZA_AGGIORNAMENTO_TARGET': 100,
    'FREQUENZA_SALVATAGGIO': 10000,
    'ALPHA_PRIORITA': 0.6,
    'BETA_PRIORITA': 0.4,
    'EPSILON_PRIORITA': 1e-6,
    'DIMENSIONE_CACHE': 1000
}

class ErroreAIPokemon(Exception): pass

class BufferRiproduzioneConPriorita:
    """Buffer di riproduzione esperienza con priorità per DQN."""
    def __init__(self, capacita: int, alpha: float = 0.6):
        self.capacita = capacita
        self.alpha = alpha
        self.buffer = []
        self.priorita = []
        self.posizione = 0
        self.dimensione = 0

    def aggiungi(self, esperienza, priorita: float = None):
        priorita = priorita or (max(self.priorita) if self.priorita else 1.0)

        if self.dimensione < self.capacita:
            self.buffer.append(esperienza)
            self.priorita.append(priorita ** self.alpha)
            self.dimensione += 1
        else:
            self.buffer[self.posizione] = esperienza
            self.priorita[self.posizione] = priorita ** self.alpha

        self.posizione = (self.posizione + 1) % self.capacita

    def campiona(self, dim_batch: int, beta: float = 0.4):
        if self.dimensione < dim_batch: return [], [], []

        arr_p = np.array(self.priorita[:self.dimensione])
        p_tot = arr_p.sum()

        if self.dimensione > 10000:
            idx_top = np.argpartition(arr_p, -dim_batch//2)[-dim_batch//2:]
            idx_rand = np.random.choice(self.dimensione, dim_batch//2, replace=False)
            indici = np.concatenate([idx_top, idx_rand])
        else:
            indici = np.random.choice(self.dimensione, dim_batch, replace=False) if p_tot == 0 else \
                     np.random.choice(self.dimensione, dim_batch, p=arr_p / p_tot)

        esperienze = [self.buffer[i] for i in indici]

        if self.dimensione > 10000 or p_tot == 0:
            pesi = np.ones(dim_batch).tolist()
        else:
            prob = arr_p[indici] / p_tot
            pesi = ((self.dimensione * prob) ** (-beta) / ((self.dimensione * prob) ** (-beta)).max()).tolist()

        return esperienze, pesi, indici

    def aggiorna_priorita(self, indici, nuove_priorita):
        for idx, p in zip(indici, nuove_priorita):
            self.priorita[idx] = (p + IPERPARAMETRI['EPSILON_PRIORITA']) ** self.alpha

    def __len__(self): return self.dimensione

class CacheImmagini:
    """Cache LRU per immagini schermo preprocessate con pulizia basata su frequenza."""
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

    def _processa_coda(self):
        with self.lock: self.attivo = True
        while True:
            with self.lock:
                if not self.coda:
                    self.attivo = False
                    break
                func, dati = self.coda.popleft()
            try:
                func(dati)
            except Exception as e:
                print(f"Errore salvataggio: {e}")

def controlla_dipendenze() -> bool:
    """Controlla e installa le dipendenze richieste."""
    try:
        import torch
        return True
    except ImportError:
        os.system(f"{sys.executable} -m pip install torch pyboy numpy opencv-python keyboard")
        return True

torch_disponibile = controlla_dipendenze()

import pyboy
from pyboy.utils import WindowEvent
import keyboard
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LettoreMemoriaGioco:
    """Legge e monitora lo stato della memoria del gioco Pokemon."""
    def __init__(self, pyboy: Any) -> None:
        self.pyboy = pyboy
        self.tipo_gioco = self._rileva_tipo_gioco()
        self.indirizzi_memoria = self._ottieni_indirizzi_memoria()
        self.stato_precedente = {'soldi_giocatore': 0, 'medaglie': 0, 'pokedex_posseduti': 0, 'pokedex_visti': 0, 'livelli_squadra': [0] * 6, 'hp_squadra': [0] * 6, 'pos_x': 0, 'pos_y': 0, 'id_mappa': 0}

    def _rileva_tipo_gioco(self) -> str:
        t = self.pyboy.cartridge_title.strip().upper()
        return 'rb' if 'RED' in t or 'BLUE' in t else 'yellow' if 'YELLOW' in t else \
               'gs' if 'GOLD' in t or 'SILVER' in t else 'crystal' if 'CRYSTAL' in t else 'generic'

    def _ottieni_indirizzi_memoria(self) -> Dict[str, int]:
        """Ottieni gli indirizzi di memoria per il tipo di gioco corrente."""
        base = {'soldi_giocatore': 0xD347, 'medaglie': 0xD356, 'conteggio_squadra': 0xD163, 'pos_x': 0xD362, 'pos_y': 0xD361, 'id_mappa': 0xD35E, 'pokedex_posseduti': 0xD2F7, 'pokedex_visti': 0xD30A, 'livelli_squadra': 0xD18C, 'hp_squadra': 0xD16C, 'tipo_battaglia': 0xD057}
        if self.tipo_gioco in ['gs', 'crystal']:
            base.update({'soldi_giocatore': 0xD84E, 'medaglie': 0xD857, 'pokedex_posseduti': 0xDE99, 'conteggio_squadra': 0xDCD7, 'livelli_squadra': 0xDCFF, 'hp_squadra': 0xDD01, 'pos_x': 0xDCB8, 'pos_y': 0xDCB7, 'id_mappa': 0xDCB5})
        return base

    def leggi_memoria(self, addr: int, lung: int = 1) -> Union[int, List[int]]:
        try:
            return self.pyboy.memory[addr] if lung == 1 else [self.pyboy.memory[addr + i] for i in range(lung)]
        except:
            return 0 if lung == 1 else [0] * lung

    def ottieni_stato_corrente(self) -> Dict[str, Any]:
        """Ottieni lo stato corrente del gioco dalla memoria."""
        stato = {}
        try:
            for chiave in ['soldi_giocatore', 'medaglie', 'pokedex_posseduti', 'pokedex_visti', 'pos_x', 'pos_y', 'id_mappa']:
                if chiave in self.indirizzi_memoria:
                    if chiave == 'soldi_giocatore':
                        stato[chiave] = self._bcd_a_int(self.leggi_memoria(self.indirizzi_memoria[chiave], 3))
                    elif chiave == 'medaglie':
                        stato[chiave] = bin(self.leggi_memoria(self.indirizzi_memoria[chiave])).count('1')
                    elif chiave in ['pokedex_posseduti', 'pokedex_visti']:
                        stato[chiave] = sum(bin(b).count('1') for b in self.leggi_memoria(self.indirizzi_memoria[chiave], 19))
                    else:
                        stato[chiave] = self.leggi_memoria(self.indirizzi_memoria[chiave])

            conteggio_squadra = min(self.leggi_memoria(self.indirizzi_memoria.get('conteggio_squadra', 0xD163)), 6)
            stato['conteggio_squadra'] = conteggio_squadra
            if conteggio_squadra > 0:
                livelli = self.leggi_memoria(self.indirizzi_memoria.get('livelli_squadra', 0xD18C), conteggio_squadra * 48)
                stato['livelli_squadra'] = [livelli[i * 48] if i * 48 < len(livelli) else 0 for i in range(6)]
                dati_hp = self.leggi_memoria(self.indirizzi_memoria.get('hp_squadra', 0xD16C), conteggio_squadra * 96)
                stato['hp_squadra'] = [(dati_hp[i*96] << 8) | dati_hp[i*96+1] if i*96+1 < len(dati_hp) else 0 for i in range(conteggio_squadra)]

            stato['in_battaglia'] = self.leggi_memoria(self.indirizzi_memoria.get('tipo_battaglia', 0xD057)) != 0
        except:
            pass
        return stato

    def _bcd_a_int(self, bcd: List[int]) -> int:
        return sum(((b >> 4) * 10 + (b & 0x0F)) * (100 ** i) for i, b in enumerate(reversed(bcd)))

    def calcola_ricompense_eventi(self, stato_corrente: Dict[str, Any]) -> float:
        """Calcola la ricompensa basata sugli eventi di gioco."""
        ricompensa = 0
        if not self.stato_precedente:
            self.stato_precedente = stato_corrente.copy()
            return ricompensa

        ricompensa += self._calcola_ricompense_medaglie(stato_corrente)
        ricompensa += self._calcola_ricompense_pokemon(stato_corrente)
        ricompensa += self._calcola_ricompense_livelli(stato_corrente)
        ricompensa += self._calcola_ricompense_soldi(stato_corrente)
        ricompensa += self._calcola_ricompense_esplorazione(stato_corrente)
        ricompensa += self._calcola_ricompense_battaglia(stato_corrente)

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

    def _calcola_ricompense_livelli(self, s: Dict[str, Any]) -> float:
        lv_curr = s.get('livelli_squadra', [0] * 6)
        lv_prev = self.stato_precedente.get('livelli_squadra', [0] * 6)
        return sum(50 * (lv_curr[i] - lv_prev[i]) for i in range(min(len(lv_curr), len(lv_prev))) if lv_curr[i] > lv_prev[i])

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

    def rileva_movimento_bloccato(self, curr: np.ndarray, prev: Optional[np.ndarray]) -> bool:
        if prev is None or curr is None or curr.size == 0 or prev.size == 0 or curr.shape != (144, 160) or prev.shape != (144, 160):
            return False
        try:
            diff = np.mean(np.abs(curr - prev))
            center_var = np.var(curr[REGIONI_SCHERMO['REGIONE_CENTRALE']])
            return diff < IPERPARAMETRI['SOGLIA_MOVIMENTO'] or center_var < IPERPARAMETRI['SOGLIA_BLOCCATO']
        except: return False

def calcola_dimensioni_output_conv(altezza_input, larghezza_input, layer_conv):
    """Calcola le dimensioni output dopo i layer convoluzionali."""
    h, w = altezza_input, larghezza_input
    for dimensione_kernel, stride in layer_conv:
        h = (h - dimensione_kernel) // stride + 1
        w = (w - dimensione_kernel) // stride + 1
    return h, w

DIMENSIONI_CONV_ESPLORAZIONE = calcola_dimensioni_output_conv(144, 160, [(8, 4), (4, 2), (3, 1)])
DIMENSIONI_CONV_MENU = calcola_dimensioni_output_conv(144, 160, [(4, 2), (3, 2)])

class ReteEsplorazione(nn.Module):
    """DQN per esplorazione e navigazione."""
    def __init__(self, n_azioni):
        super(ReteEsplorazione, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        altezza_conv, larghezza_conv = DIMENSIONI_CONV_ESPLORAZIONE
        dimensione_input_lineare = altezza_conv * larghezza_conv * 64
        self.fc1 = nn.Linear(dimensione_input_lineare, 256)
        self.fc2 = nn.Linear(256, n_azioni)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReteBattaglia(nn.Module):
    def __init__(self, n_azioni):
        super(ReteBattaglia, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        altezza_conv, larghezza_conv = DIMENSIONI_CONV_ESPLORAZIONE
        dimensione_input_lineare = altezza_conv * larghezza_conv * 64
        self.fc1 = nn.Linear(dimensione_input_lineare, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_azioni)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReteMenu(nn.Module):
    def __init__(self, n_azioni):
        super(ReteMenu, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        altezza_conv, larghezza_conv = DIMENSIONI_CONV_MENU
        dimensione_input_lineare = altezza_conv * larghezza_conv * 32
        self.fc1 = nn.Linear(dimensione_input_lineare, 128)
        self.fc2 = nn.Linear(128, n_azioni)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GruppoRetiMultiple:
    def __init__(self, n_azioni: int, device: Any) -> None:
        self.device = device
        self.n_azioni = n_azioni

        self.rete_esplorazione = ReteEsplorazione(n_azioni).to(device)
        self.rete_battaglia = ReteBattaglia(n_azioni).to(device)
        self.rete_menu = ReteMenu(n_azioni).to(device)

        self.target_esplorazione = ReteEsplorazione(n_azioni).to(device)
        self.target_battaglia = ReteBattaglia(n_azioni).to(device)
        self.target_menu = ReteMenu(n_azioni).to(device)

        # Perfeziona
        self.ottimizzatore_esplorazione = optim.Adam(self.rete_esplorazione.parameters(), lr=IPERPARAMETRI['TASSO_APPRENDIMENTO'])
        self.ottimizzatore_battaglia = optim.Adam(self.rete_battaglia.parameters(), lr=0.0003)
        self.ottimizzatore_menu = optim.Adam(self.rete_menu.parameters(), lr=0.0002)

        # Adatta
        self.scheduler_esplorazione = optim.lr_scheduler.ReduceLROnPlateau(self.ottimizzatore_esplorazione, patience=1000)
        self.scheduler_battaglia = optim.lr_scheduler.ReduceLROnPlateau(self.ottimizzatore_battaglia, patience=500)
        self.scheduler_menu = optim.lr_scheduler.ReduceLROnPlateau(self.ottimizzatore_menu, patience=500)

        self.target_esplorazione.load_state_dict(self.rete_esplorazione.state_dict())
        self.target_battaglia.load_state_dict(self.rete_battaglia.state_dict())
        self.target_menu.load_state_dict(self.rete_menu.state_dict())

        # Mescolanza
        self.usa_precisione_mista = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        if self.usa_precisione_mista:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def scegli_azione(self, state: Any, game_state: str, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_azioni - 1)
        
        with torch.no_grad():
            state_batch = state.unsqueeze(0)
            if game_state == "battle":
                q_values = self.rete_battaglia(state_batch)
            elif game_state == "menu":
                q_values = self.rete_menu(state_batch)
            else:
                q_values = self.rete_esplorazione(state_batch)
            return q_values.argmax().item()
    
    def addestra_rete(self, batch: List[Tuple], game_state: str, pesi_importanza: List[float] = None) -> Optional[Tuple[float, List[float]]]:
        # Allena
        if len(batch) < 4:
            return None, []

        states = []
        next_states = []
        for exp in batch:
            if isinstance(exp[0], np.ndarray) and exp[0].shape == (144, 160):
                states.append(exp[0])
                next_states.append(exp[3])

        if len(states) < 4:
            return None, []

        states_np = np.expand_dims(np.array(states), axis=1)
        next_states_np = np.expand_dims(np.array(next_states), axis=1)

        states_t = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch[:len(states)]]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch[:len(states)]]).to(self.device)
        next_states_t = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch[:len(states)]]).to(self.device)

        # Bilancia
        if pesi_importanza is not None:
            pesi_t = torch.FloatTensor(pesi_importanza[:len(states)]).to(self.device)
        else:
            pesi_t = torch.ones(len(states)).to(self.device)
        
        # Scegli
        if game_state == "battle":
            rete = self.rete_battaglia
            rete_target = self.target_battaglia
            ottimizzatore = self.ottimizzatore_battaglia
            scheduler = self.scheduler_battaglia
        elif game_state == "menu":
            rete = self.rete_menu
            rete_target = self.target_menu
            ottimizzatore = self.ottimizzatore_menu
            scheduler = self.scheduler_menu
        else:
            rete = self.rete_esplorazione
            rete_target = self.target_esplorazione
            ottimizzatore = self.ottimizzatore_esplorazione
            scheduler = self.scheduler_esplorazione

        # Addestra
        if self.usa_precisione_mista:
            with torch.cuda.amp.autocast():
                current_q_values = rete(states_t).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = rete_target(next_states_t).max(1)[0]
                    target_q_values = rewards + (1 - dones) * IPERPARAMETRI['FATTORE_SCONTO'] * next_q_values

                td_errors = current_q_values.squeeze() - target_q_values
                loss = (pesi_t * (td_errors ** 2)).mean()

            ottimizzatore.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(ottimizzatore)
            torch.nn.utils.clip_grad_norm_(rete.parameters(), max_norm=1.0)
            self.scaler.step(ottimizzatore)
            self.scaler.update()
        else:
            current_q_values = rete(states_t).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = rete_target(next_states_t).max(1)[0]
                target_q_values = rewards + (1 - dones) * IPERPARAMETRI['FATTORE_SCONTO'] * next_q_values

            td_errors = current_q_values.squeeze() - target_q_values
            loss = (pesi_t * (td_errors ** 2)).mean()

            ottimizzatore.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rete.parameters(), max_norm=1.0)
            ottimizzatore.step()

        # Aggiorna
        scheduler.step(loss)

        # Restituisci
        td_errors_abs = torch.abs(td_errors).detach().cpu().numpy().tolist()
        return loss.item(), td_errors_abs
    
    def aggiorna_reti_obiettivo(self) -> None:
        tau = 0.005
        for target_param, param in zip(self.target_esplorazione.parameters(), self.rete_esplorazione.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_battaglia.parameters(), self.rete_battaglia.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_menu.parameters(), self.rete_menu.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class AgentePokemonAI:
    def __init__(self, rom_path: str, headless: bool = False) -> None:
        self.rom_name = os.path.splitext(os.path.basename(rom_path))[0]
        self.save_dir = f"pokemon_ai_saves_{self.rom_name}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_path = os.path.join(self.save_dir, "model.pth")
        self.memory_path = os.path.join(self.save_dir, "memory.pkl")
        self.stats_path = os.path.join(self.save_dir, "stats.json")

        self.pyboy = pyboy.PyBoy(rom_path, window="null" if headless else "SDL2", debug=False)
        self.rilevatore_stato = RilevatorStatoGioco()
        self.lettore_memoria = LettoreMemoriaGioco(self.pyboy)

        self.cache_immagini = CacheImmagini()
        self.salvatore_asincrono = SalvatoreAsincrono()
        self.beta_priorita = IPERPARAMETRI['BETA_PRIORITA']

        WE = WindowEvent
        self.actions = [[], [WE.PRESS_ARROW_UP], [WE.PRESS_ARROW_DOWN], [WE.PRESS_ARROW_LEFT], [WE.PRESS_ARROW_RIGHT],
                        [WE.PRESS_BUTTON_A], [WE.PRESS_BUTTON_B], [WE.PRESS_BUTTON_START], [WE.PRESS_BUTTON_SELECT]]
        self.release_map = {WE.PRESS_ARROW_UP: WE.RELEASE_ARROW_UP, WE.PRESS_ARROW_DOWN: WE.RELEASE_ARROW_DOWN,
                           WE.PRESS_ARROW_LEFT: WE.RELEASE_ARROW_LEFT, WE.PRESS_ARROW_RIGHT: WE.RELEASE_ARROW_RIGHT,
                           WE.PRESS_BUTTON_A: WE.RELEASE_BUTTON_A, WE.PRESS_BUTTON_B: WE.RELEASE_BUTTON_B,
                           WE.PRESS_BUTTON_START: WE.RELEASE_BUTTON_START, WE.PRESS_BUTTON_SELECT: WE.RELEASE_BUTTON_SELECT}
        
        self.stats = self._load_stats()
        self.epsilon = max(IPERPARAMETRI['EPSILON_MINIMO'], 0.9 * (0.995 ** (self.stats.get('total_frames', 0) / 10000)))
        # Prioritizza
        self.memory = BufferRiproduzioneConPriorita(IPERPARAMETRI['DIMENSIONE_MEMORIA'], IPERPARAMETRI['ALPHA_PRIORITA'])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gruppo_reti = GruppoRetiMultiple(len(self.actions), self.device)
        
        self.frame_count = 0
        self.episode_count = self.stats.get('episodes', 0)
        self.total_reward = 0
        self.best_reward = self.stats.get('best_reward', float('-inf'))
        self.current_game_state = "exploring"
        self.last_memory_check = 0
        self.memory_check_interval = IPERPARAMETRI['INTERVALLO_CONTROLLO_MEMORIA']
        self.reward_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=100)
        self.early_stopping_patience = 2000
        self.best_avg_reward = float('-inf')
        self.patience_counter = 0
        self.min_episodes_before_stopping = 50
        
        self._load_memory()

    def _has_torch_multiagent(self) -> bool:
        return torch_disponibile and hasattr(self, 'multi_agent')

    def _safe_choose_action(self, state: Any, game_state: str) -> int:
        if self._has_torch_multiagent():
            return self.gruppo_reti.scegli_azione(state, game_state, self.epsilon)
        else:
            return random.randint(0, len(self.actions) - 1)

    def _safe_replay(self) -> None:
        # Ripeti
        if not self._has_torch_multiagent() or len(self.memory) < IPERPARAMETRI['DIMENSIONE_BATCH']:
            return
        try:
            # Incrementa
            self.beta_priorita = min(1.0, self.beta_priorita + 0.0001)

            # Campiona
            esperienze, pesi, indici = self.memory.campiona(IPERPARAMETRI['DIMENSIONE_BATCH'], self.beta_priorita)

            if not esperienze:
                return

            # Organizza
            explorer_batch, explorer_pesi, explorer_indici = [], [], []
            battle_batch, battle_pesi, battle_indici = [], [], []
            menu_batch, menu_pesi, menu_indici = [], [], []

            for i, (exp, peso, idx) in enumerate(zip(esperienze, pesi, indici)):
                if len(exp) >= 6:
                    game_state = exp[5]
                    if game_state == "battle":
                        battle_batch.append(exp)
                        battle_pesi.append(peso)
                        battle_indici.append(idx)
                    elif game_state == "menu":
                        menu_batch.append(exp)
                        menu_pesi.append(peso)
                        menu_indici.append(idx)
                    else:
                        explorer_batch.append(exp)
                        explorer_pesi.append(peso)
                        explorer_indici.append(idx)

            loss_total = 0
            losses_count = 0
            tutti_td_errors = []
            tutti_indici = []

            # Addestra
            batch_configs = [
                (explorer_batch, explorer_pesi, explorer_indici, 'exploring'),
                (battle_batch, battle_pesi, battle_indici, 'battle'),
                (menu_batch, menu_pesi, menu_indici, 'menu')
            ]

            for batch, pesi_batch, indici_batch, game_state in batch_configs:
                if len(batch) >= 4:
                    risultato = self.gruppo_reti.addestra_rete(batch, game_state, pesi_batch)
                    if risultato:
                        loss, td_errors = risultato
                        loss_total += loss
                        losses_count += 1
                        tutti_td_errors.extend(td_errors)
                        tutti_indici.extend(indici_batch)

            # Riordina
            if tutti_td_errors and tutti_indici:
                self.memory.aggiorna_priorita(tutti_indici, tutti_td_errors)

            if losses_count > 0:
                avg_loss = loss_total / losses_count
                self.loss_history.append(avg_loss)

            if self.epsilon > IPERPARAMETRI['EPSILON_MINIMO']:
                self.epsilon *= IPERPARAMETRI['DECADIMENTO_EPSILON']

        except Exception as e:
            raise ErroreAIPokemon(f"Training fallito: {e}")

    def _safe_update_target_model(self) -> None:
        if self._has_torch_multiagent():
            self.gruppo_reti.aggiorna_reti_obiettivo()

    def _safe_save_model(self) -> None:
        if self._has_torch_multiagent():
            try:
                checkpoint = {'explorer_state': self.gruppo_reti.rete_esplorazione.state_dict(), 'battle_state': self.gruppo_reti.rete_battaglia.state_dict(), 'menu_state': self.gruppo_reti.rete_menu.state_dict(), 'epsilon': self.epsilon, 'episode': self.episode_count, 'frame': self.frame_count}
                torch.save(checkpoint, self.model_path)
            except Exception as e:
                raise ErroreAIPokemon(f"Save error: {e}")

    def _get_screen_tensor(self):
        # Ottieni
        screen = self.pyboy.screen.image
        gray = np.array(screen.convert('L'))
        self.last_screen_array = gray.copy()

        # Controlla
        cached_result = self.cache_immagini.ottieni(gray)
        if cached_result is not None:
            if self._has_torch_multiagent():
                return torch.from_numpy(cached_result).unsqueeze(0).to(self.device)
            return cached_result

        # Preprocessa
        normalized = gray.astype(np.float32) / 255.0
        self.cache_immagini.salva(gray, normalized)

        if self._has_torch_multiagent():
            tensor = torch.from_numpy(normalized).unsqueeze(0)
            return tensor.to(self.device)
        return normalized

    def _detect_game_state(self, screen_array):
        old_state = self.current_game_state
        
        if self.rilevatore_stato.rileva_battaglia(screen_array):
            self.current_game_state = "battle"
        elif self.rilevatore_stato.rileva_dialogo(screen_array):
            self.current_game_state = "dialogue"
        elif self.rilevatore_stato.rileva_menu(screen_array):
            self.current_game_state = "menu"
        else:
            self.current_game_state = "exploring"
            
        return self.current_game_state

    def _calculate_reward(self, screen_tensor, previous_screen, action):
        reward = 0

        if hasattr(self, 'last_screen_array'):
            game_state = self._detect_game_state(self.last_screen_array)
        else:
            game_state = "exploring"

        # Controlla
        memory_state = self.lettore_memoria.ottieni_stato_corrente()
        memory_reward = self.lettore_memoria.calcola_ricompense_eventi(memory_state)
        reward += memory_reward

        # Movimento
        if previous_screen is not None:
            if self._has_torch_multiagent():
                diff = torch.abs(screen_tensor - previous_screen).mean().item()
            else:
                diff = np.mean(np.abs(screen_tensor - previous_screen))

            # Scala
            movement_threshold = 0.02
            if game_state == "dialogue":
                movement_threshold = 0.005  # Dialogo
            elif game_state == "battle":
                movement_threshold = 0.01   # Battaglia

            if diff > movement_threshold:
                reward += min(diff * 10, 2.0)  # Limita
            elif diff < movement_threshold * 0.1:
                reward -= 0.05  # Penalizza

        self.reward_history.append(reward)
        self.total_reward += reward
        return reward

    def scegli_azione(self, state):
        game_state = self.current_game_state if hasattr(self, 'current_game_state') else "exploring"
        return self._safe_choose_action(state, game_state)

    def memorizza_esperienza(self, state, action, reward, next_state, done):
        # Salva
        if self._has_torch_multiagent():
            state_np = state.squeeze(0).cpu().numpy()
            next_state_np = next_state.squeeze(0).cpu().numpy()
        else:
            state_np = state
            next_state_np = next_state

        game_state = self.current_game_state if hasattr(self, 'current_game_state') else "exploring"
        experience = (state_np, action, reward, next_state_np, done, game_state)

        # Calcola
        priorita_iniziale = abs(reward) + IPERPARAMETRI['EPSILON_PRIORITA']
        self.memory.aggiungi(experience, priorita_iniziale)

    def esegui_replay(self):
        self._safe_replay()

    def aggiorna_rete_target(self):
        self._safe_update_target_model()

    def _load_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        return {'episodes': 0, 'total_frames': 0, 'best_reward': float('-inf')}

    def _save_stats(self):
        final_state = self.lettore_memoria.ottieni_stato_corrente()
        if final_state:
            self.stats['final_state'] = final_state
        
        self.stats.update({'episodes': self.episode_count, 'total_frames': self.stats.get('total_frames', 0) + self.frame_count, 'best_reward': max(self.best_reward, self.total_reward)})
        
        if len(self.reward_history) > 0:
            self.stats['avg_reward_last_1000'] = float(np.mean(list(self.reward_history)))
        
        with open(self.stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def _save_memory(self):
        memory_data = {'memory': list(self.memory)[-10000:]}
        with open(self.memory_path, 'wb') as f:
            pickle.dump(memory_data, f)

    def _load_memory(self):
        # Carica
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'rb') as f:
                    memory_data = pickle.load(f)
                for exp in memory_data.get('memory', [])[-5000:]:
                    if len(exp) >= 5 and isinstance(exp[0], np.ndarray):
                        # Inizializza
                        priorita = abs(exp[2]) + IPERPARAMETRI['EPSILON_PRIORITA'] if len(exp) > 2 else 1.0
                        self.memory.aggiungi(exp, priorita)
            except Exception as e:
                print(f"Errore nel caricamento memoria: {e}")

    def avvia_training(self) -> None:
        paused = False
        previous_screen = None
        last_save_frame = 0
        
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
                
                state = self._get_screen_tensor()
                action = self.scegli_azione(state)

                for btn in self.actions[action]: self.pyboy.send_input(btn)

                wait_frames = {"dialogue": 2, "battle": 6}.get(self.current_game_state, 4)
                for _ in range(wait_frames): self.pyboy.tick()

                for btn in self.actions[action]:
                    if btn in self.release_map: self.pyboy.send_input(self.release_map[btn])

                next_state = self._get_screen_tensor()
                reward = self._calculate_reward(next_state, previous_screen, action)

                if previous_screen is not None:
                    self.memorizza_esperienza(state, action, reward, next_state, False)

                if self.frame_count % 4 == 0 and len(self.memory) >= IPERPARAMETRI['DIMENSIONE_BATCH']:
                    self.esegui_replay()

                if self.frame_count % IPERPARAMETRI['FREQUENZA_AGGIORNAMENTO_TARGET'] == 0:
                    self.aggiorna_rete_target()

                if self.frame_count - last_save_frame >= IPERPARAMETRI['FREQUENZA_SALVATAGGIO']:
                    self._save_all()
                    last_save_frame = self.frame_count
                    if self._should_early_stop():
                        print(f"Early stopping at frame {self.frame_count}")
                        break
                
                previous_screen = next_state
                self.frame_count += 1
                    
        except KeyboardInterrupt:
            pass
        
        finally:
            self._save_all()
            self.pyboy.stop()

    def _should_early_stop(self) -> bool:
        min_frames = self.min_episodes_before_stopping * 1000
        if self.frame_count < min_frames or len(self.reward_history) < 100: return False

        curr_avg = np.mean(list(self.reward_history)[-100:])
        if curr_avg > self.best_avg_reward * 1.01:
            self.best_avg_reward = curr_avg
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.early_stopping_patience

    def _save_all(self) -> None:
        dati_modello = None
        if self._has_torch_multiagent():
            dati_modello = {
                'explorer_state': self.gruppo_reti.rete_esplorazione.state_dict(),
                'battle_state': self.gruppo_reti.rete_battaglia.state_dict(),
                'menu_state': self.gruppo_reti.rete_menu.state_dict(),
                'epsilon': self.epsilon,
                'episode': self.episode_count,
                'frame': self.frame_count
            }

        dati_memoria = {'memory': list(self.memory.buffer)[-10000:]}

        final_state = self.lettore_memoria.ottieni_stato_corrente()
        dati_stats = self.stats.copy()
        dati_stats.update({
            'episodes': self.episode_count,
            'total_frames': self.stats.get('total_frames', 0) + self.frame_count,
            'best_reward': max(self.best_reward, self.total_reward)
        })
        if final_state:
            dati_stats['final_state'] = final_state
        if len(self.reward_history) > 0:
            dati_stats['avg_reward_last_1000'] = float(np.mean(list(self.reward_history)))

        if dati_modello:
            self.salvatore_asincrono.salva_asincrono(lambda d: torch.save(d, self.model_path), dati_modello)

        self.salvatore_asincrono.salva_asincrono(lambda d: pickle.dump(d, open(self.memory_path, 'wb')), dati_memoria)
        self.salvatore_asincrono.salva_asincrono(lambda d: json.dump(d, open(self.stats_path, 'w'), indent=2), dati_stats)

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

if __name__ == "__main__":
    principale()