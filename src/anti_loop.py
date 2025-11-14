import time
from collections import deque
from hyperparameters import HYPERPARAMETERS


class AdaptiveEntropyScheduler:
    """
    Scheduler per coefficiente entropia dinamico.
    Riduce gradualmente esplorazione casuale man mano che l'agente impara.

    Formula: entropy = start + (end - start) * min(1.0, frames / decay_frames)
    """
    def __init__(self):
        self.start_entropy = HYPERPARAMETERS['PPO_ENTROPY_START']
        self.end_entropy = HYPERPARAMETERS['PPO_ENTROPY_END']
        self.decay_frames = HYPERPARAMETERS['PPO_ENTROPY_DECAY_FRAMES']

    def get_entropy(self, current_frame: int) -> float:
        """Calcola coefficiente entropia corrente basato su frame."""
        progress = min(1.0, current_frame / self.decay_frames)
        entropy = self.start_entropy + (self.end_entropy - self.start_entropy) * progress
        return entropy


class AntiLoopMemoryBuffer:
    """
    Buffer per rilevamento e prevenzione di loop comportamentali.
    Traccia stati recenti e penalizza pattern ripetitivi.

    Strategia pattern: algoritmi di rilevamento differenti per tipi diversi di loop.
    """
    def __init__(self):
        self.buffer_size = HYPERPARAMETERS['ANTI_LOOP_BUFFER_SIZE']
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_history = deque(maxlen=20)  # Ultime 20 azioni
        self.position_history = deque(maxlen=50)  # Ultime 50 posizioni

    def add_state(self, pos_x: int, pos_y: int, id_map: int, action: int):
        """Aggiungi stato corrente al buffer."""
        state_key = (id_map, pos_x, pos_y)
        self.state_buffer.append(state_key)
        self.action_history.append(action)
        self.position_history.append(state_key)

    def detect_position_loop(self) -> bool:
        """
        Rileva loop di posizione (es. avanti-indietro ripetutamente).
        Ritorna True se l'agente è in loop.
        """
        if len(self.state_buffer) < 20:
            return False

        # Conta occorrenze degli ultimi 20 stati (erano 10)
        recent_states = list(self.state_buffer)[-20:]
        unique_states = set(recent_states)

        # Se visita meno di 2 posizioni uniche negli ultimi 20 step = loop (era <3 in 10)
        # Molto più permissivo
        return len(unique_states) <= 2

    def detect_action_loop(self) -> bool:
        """
        Rileva loop di azioni (es. premere A ripetutamente senza progressione).
        Ritorna True se l'agente ripete la stessa azione troppo.
        """
        if len(self.action_history) < HYPERPARAMETERS['ACTION_REPEAT_MAX']:
            return False

        recent_actions = list(self.action_history)[-HYPERPARAMETERS['ACTION_REPEAT_MAX']:]
        # Se ultime N azioni sono identiche = loop
        return len(set(recent_actions)) == 1

    def detect_oscillation(self) -> bool:
        """
        Rileva oscillazione (es. su-giù-su-giù).
        Ritorna True se pattern oscillatorio rilevato.
        """
        if len(self.position_history) < 16:  # Aumentato da 8 a 16
            return False

        # Verifica se posizioni alternano tra 2 valori
        recent_pos = list(self.position_history)[-16:]  # Aumentato da 8 a 16
        unique_pos = set(recent_pos)

        if len(unique_pos) == 2:
            # Verifica alternanza perfetta per almeno 12 step (erano 8)
            alternating_count = sum(
                1 for i in range(len(recent_pos)-1)
                if recent_pos[i] != recent_pos[i+1]
            )
            # Solo se alterna per più del 90% dei casi (molto restrittivo)
            return alternating_count >= 14  # 14 su 15 transizioni

        return False

    def calculate_loop_penalty(self) -> float:
        """
        Calcola penalità basata su loop rilevati.
        Ritorna penalità negativa se loop rilevato, 0 altrimenti.
        """
        penalty = 0.0

        if self.detect_position_loop():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY']

        if self.detect_action_loop():
            penalty += HYPERPARAMETERS['ACTION_REPEAT_PENALTY']

        if self.detect_oscillation():
            penalty += HYPERPARAMETERS['ANTI_LOOP_PENALTY'] * 0.5

        return penalty

    def get_exploration_bonus(self) -> float:
        """
        Bonus per comportamento esplorativo (opposto di loop).
        Ritorna bonus positivo se l'agente esplora attivamente.
        """
        if len(self.state_buffer) < 20:
            return 0.0

        # Conta stati unici negli ultimi 20 step
        recent_states = list(self.state_buffer)[-20:]
        unique_ratio = len(set(recent_states)) / len(recent_states)

        # Bonus proporzionale a diversità di stati visitati (più facile da raggiungere)
        if unique_ratio > 0.6:  # >60% stati unici = esplorazione eccellente (era 0.7)
            return 1.5  # Ridotto da 2.0
        elif unique_ratio > 0.4:  # >40% stati unici = buona esplorazione (era 0.5)
            return 0.8  # Ridotto da 1.0

        return 0.0