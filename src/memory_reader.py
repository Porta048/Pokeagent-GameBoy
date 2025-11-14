import time
from collections import deque
import numpy as np


class GameMemoryReader:
    """Legge memoria di gioco per reward ed eventi con sistema sofisticato multi-componente."""
    MEMORY_ADDRESSES = {
        'MONEY': (0xD347, 0xD349),
        'BADGES': 0xD356,
        'POKEDEX_CAUGHT': 0xD2F7,
        'POKEDEX_SEEN': 0xD30A,
        'TEAM_LEVELS': [(0xD18C + i*44, 0xD18C + i*44 + 1) for i in range(6)],
        'HP_TEAM': [(0xD16D + i*44, 0xD16E + i*44) for i in range(6)],
        'HP_MAX_TEAM': [(0xD18D + i*44, 0xD18E + i*44) for i in range(6)],
        'MAP_ID': 0xD35E,
        'POS_X': 0xD361,
        'POS_Y': 0xD362,
        'EVENT_FLAGS': (0xD747, 0xD886),  # Range flag eventi (320 bytes)
        'TRAINER_FLAGS': (0xD5A0, 0xD5F7)  # Flag sconfitta allenatore (88 bytes)
    }

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.previous_state = {}

        # Tracciamento navigazione
        self.visited_coordinates = set()
        self.last_position = None

        # Tracciamento eventi
        self.previous_event_flags = set()
        self.previous_trainer_flags = set()

        # Tracciamento anti-grinding
        self.wild_battle_count = 0
        self.last_average_level = 0
        self.recent_level_ups = deque(maxlen=10)  # Traccia level-up recenti

    def get_current_state(self) -> dict:
        try:
            # PyBoy modern API: use memory array indexing
            mem = lambda addr: self.pyboy.memory[addr]

            money_bcd = [mem(addr) for addr in range(self.MEMORY_ADDRESSES['MONEY'][0], self.MEMORY_ADDRESSES['MONEY'][1] + 1)]
            money = sum(((b >> 4) * 10 + (b & 0xF)) * (100 ** i) for i, b in enumerate(reversed(money_bcd)))

            badges = bin(mem(self.MEMORY_ADDRESSES['BADGES'])).count('1')

            pokedex_caught = sum(bin(mem(addr)).count('1') for addr in range(self.MEMORY_ADDRESSES['POKEDEX_CAUGHT'], self.MEMORY_ADDRESSES['POKEDEX_CAUGHT'] + 19))
            pokedex_seen = sum(bin(mem(addr)).count('1') for addr in range(self.MEMORY_ADDRESSES['POKEDEX_SEEN'], self.MEMORY_ADDRESSES['POKEDEX_SEEN'] + 19))

            team_levels = [mem(addr[0]) for addr in self.MEMORY_ADDRESSES['TEAM_LEVELS']]
            hp_team = [mem(addr[0]) * 256 + mem(addr[1]) for addr in self.MEMORY_ADDRESSES['HP_TEAM']]
            hp_max_team = [mem(addr[0]) * 256 + mem(addr[1]) for addr in self.MEMORY_ADDRESSES['HP_MAX_TEAM']]

            in_battle = any(hp > 0 for hp in hp_team[:3])

            # Leggi flag eventi
            event_flags = set()
            for addr in range(self.MEMORY_ADDRESSES['EVENT_FLAGS'][0], self.MEMORY_ADDRESSES['EVENT_FLAGS'][1] + 1):
                byte_val = mem(addr)
                for bit in range(8):
                    if byte_val & (1 << bit):
                        event_flags.add((addr - self.MEMORY_ADDRESSES['EVENT_FLAGS'][0]) * 8 + bit)

            # Leggi flag allenatori
            trainer_flags = set()
            for addr in range(self.MEMORY_ADDRESSES['TRAINER_FLAGS'][0], self.MEMORY_ADDRESSES['TRAINER_FLAGS'][1] + 1):
                byte_val = mem(addr)
                for bit in range(8):
                    if byte_val & (1 << bit):
                        trainer_flags.add((addr - self.MEMORY_ADDRESSES['TRAINER_FLAGS'][0]) * 8 + bit)

            return {
                'player_money': money,
                'badges': badges,
                'pokedex_caught': pokedex_caught,
                'pokedex_seen': pokedex_seen,
                'team_levels': team_levels,
                'hp_team': hp_team,
                'hp_max_team': hp_max_team,
                'in_battle': in_battle,
                'map_id': mem(self.MEMORY_ADDRESSES['MAP_ID']),
                'pos_x': mem(self.MEMORY_ADDRESSES['POS_X']),
                'pos_y': mem(self.MEMORY_ADDRESSES['POS_Y']),
                'event_flags': event_flags,
                'trainer_flags': trainer_flags
            }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error reading game memory state: {str(e)}")
            return self.previous_state.copy() if self.previous_state else {}

    def calculate_event_rewards(self, current_state: dict) -> float:
        if not self.previous_state or not current_state:
            self.previous_state = current_state.copy()
            return 0.0

        total_reward = 0.0
        reward_details = {}

        # Sistema reward sofisticato multi-componente
        r = self._calculate_badge_rewards(current_state)
        if r != 0: reward_details['badges'] = r
        total_reward += r

        r = self._calculate_pokemon_rewards(current_state)
        if r != 0: reward_details['pokemon'] = r
        total_reward += r

        r = self._calculate_balanced_level_rewards(current_state)
        if r != 0: reward_details['levels'] = r
        total_reward += r

        r = self._calculate_money_rewards(current_state)
        if r != 0: reward_details['money'] = r
        total_reward += r

        r = self._calculate_exploration_rewards(current_state)
        if r != 0: reward_details['exploration'] = r
        total_reward += r

        r = self._calculate_battle_rewards(current_state)
        if r != 0: reward_details['battle'] = r
        total_reward += r

        # Modulo Curiosità Intrinseca (ICM)
        r = self.calculate_curiosity_reward(current_state)
        if r != 0: reward_details['curiosity'] = r
        total_reward += r

        # Reward avanzati
        r = self._calculate_event_flags_rewards(current_state)
        if r != 0: reward_details['events'] = r
        total_reward += r

        r = self._calculate_navigation_rewards(current_state)
        if r != 0: reward_details['navigation'] = r
        total_reward += r

        r = self._calculate_healing_rewards(current_state)
        if r != 0: reward_details['healing'] = r
        total_reward += r

        # Log reward significativi
        if reward_details:
            print(f"[REWARD] {reward_details} = {total_reward:.2f}")

        self.previous_state = current_state.copy()
        return total_reward

    def _calculate_badge_rewards(self, s: dict) -> float:
        # Medaglie = progressione principale di gioco
        return 2000 if s.get('badges', 0) > self.previous_state.get('badges', 0) else 0

    def _calculate_pokemon_rewards(self, s: dict) -> float:
        r = 0
        if s.get('pokedex_caught', 0) > self.previous_state.get('pokedex_caught', 0):
            # Catturare Pokemon = obiettivo importante
            r += 150 * (s.get('pokedex_caught', 0) - self.previous_state.get('pokedex_caught', 0))
        if s.get('pokedex_seen', 0) > self.previous_state.get('pokedex_seen', 0):
            # Vedere nuovi Pokemon = esplorazione utile
            r += 20 * (s.get('pokedex_seen', 0) - self.previous_state.get('pokedex_seen', 0))
        return r

    def _calculate_balanced_level_rewards(self, s: dict) -> float:
        """
        Reward livello bilanciato: incentiva level-up precoci, scoraggia grinding eccessivo.
        Formula: reward = base_reward * diminishing_factor
        """
        lv_curr = s.get('team_levels', [0] * 6)
        lv_prev = self.previous_state.get('team_levels', [0] * 6)

        reward = 0.0

        for i in range(min(len(lv_curr), len(lv_prev))):
            if lv_curr[i] > lv_prev[i]:
                level_up_amount = lv_curr[i] - lv_prev[i]

                # Reward base diminuisce con livello (anti-grinding)
                if lv_curr[i] <= 15:
                    base_reward = 50  # Gioco iniziale: reward pieno
                elif lv_curr[i] <= 30:
                    base_reward = 30  # Metà gioco: ridotto
                elif lv_curr[i] <= 45:
                    base_reward = 15  # Fine gioco: minimo
                else:
                    base_reward = 5   # Endgame: molto piccolo

                # Penalizza grinding su Pokemon selvatici (se troppi level-up recenti)
                self.recent_level_ups.append(time.time())
                level_ups_last_minute = sum(1 for t in self.recent_level_ups if time.time() - t < 60)

                if level_ups_last_minute > 5:
                    # Troppi level-up in poco tempo = grinding
                    grinding_penalty = 0.3
                else:
                    grinding_penalty = 1.0

                reward += base_reward * level_up_amount * grinding_penalty

        return reward

    def _calculate_money_rewards(self, s: dict) -> float:
        diff = s.get('player_money', 0) - self.previous_state.get('player_money', 0)
        return min(diff / 100, 20) if diff > 0 else -20 if diff < -100 else 0

    def _calculate_exploration_rewards(self, s: dict) -> float:
        # Reward maggiore per cambio mappa (importante per progressione)
        r = 80 if s.get('map_id', 0) != self.previous_state.get('map_id', 0) else 0
        # Reward per movimento significativo (incoraggia esplorazione attiva)
        diff_pos = abs(s.get('pos_x', 0) - self.previous_state.get('pos_x', 0)) + abs(s.get('pos_y', 0) - self.previous_state.get('pos_y', 0))
        return r + (8 if diff_pos > 5 else 0)

    def _calculate_battle_rewards(self, s: dict) -> float:
        prev_battle = self.previous_state.get('in_battle', False)
        curr_battle = s.get('in_battle', False)
        if prev_battle and not curr_battle:
            return 50 if any(hp > 0 for hp in s.get('hp_team', [])) else -100
        return 2 if not prev_battle and curr_battle else 0

    def _calculate_event_flags_rewards(self, s: dict) -> float:
        """
        Reward flag eventi: premia eventi completati (battaglie allenatori, progressione quest, medaglie palestra).
        Traccia flag eventi e allenatori per incentivare progressione storia.
        """
        reward = 0.0

        # Flag eventi (progressione quest, oggetti, eventi storia)
        event_flags_curr = s.get('event_flags', set())
        new_events = event_flags_curr - self.previous_event_flags

        if new_events:
            # Eventi storia = progressione importante
            reward += 5.0 * len(new_events)
            self.previous_event_flags = event_flags_curr.copy()

        # Flag allenatori (battaglie allenatori vinte)
        trainer_flags_curr = s.get('trainer_flags', set())
        new_trainers = trainer_flags_curr - self.previous_trainer_flags

        if new_trainers:
            # Battaglie allenatori sono preziose (non grinding, progressione obbligatoria)
            reward += 100.0 * len(new_trainers)
            self.previous_trainer_flags = trainer_flags_curr.copy()

        return reward

    def _calculate_navigation_rewards(self, s: dict) -> float:
        """
        Reward navigazione: premia esplorazione di nuove coordinate.
        Incoraggia esplorazione sistematica senza grinding nella stessa area.
        """
        pos_x = s.get('pos_x', 0)
        pos_y = s.get('pos_y', 0)
        map_id = s.get('map_id', 0)

        # Crea chiave unica per posizione (mappa, x, y)
        coord_key = (map_id, pos_x, pos_y)

        # Reward solo se nuova coordinata
        if coord_key not in self.visited_coordinates:
            self.visited_coordinates.add(coord_key)
            return 2.0  # Aumentato per incoraggiare esplorazione sistematica

        return 0.0

    def _calculate_healing_rewards(self, s: dict) -> float:
        """
        Reward cura: proporzionale agli HP recuperati.
        Premia uso di Centri Pokemon e pozioni per mantenere team in salute.
        """
        hp_curr = s.get('hp_team', [0] * 6)
        hp_max_curr = s.get('hp_max_team', [0] * 6)
        hp_prev = self.previous_state.get('hp_team', [0] * 6)

        reward = 0.0

        for i in range(min(len(hp_curr), len(hp_prev), len(hp_max_curr))):
            if hp_max_curr[i] > 0:  # Pokemon esiste
                hp_recovered = hp_curr[i] - hp_prev[i]

                # Reward solo se HP aumentati (cura)
                if hp_recovered > 0:
                    # Reward proporzionale a % HP recuperati
                    recovery_percent = hp_recovered / hp_max_curr[i]
                    reward += recovery_percent * 5.0  # Max 5 reward per cura completa

        return reward

    def calculate_curiosity_reward(self, current_state: dict) -> float:
        """
        Reward Modulo Curiosità Intrinseca (ICM).
        Premia esplorazione di nuovi stati di gioco.
        """
        reward = 0.0

        # Curiosità per nuove mappe visitate
        current_map = current_state.get('map_id', 0)
        if not hasattr(self, 'maps_visited'):
            self.maps_visited = set()

        if current_map not in self.maps_visited:
            self.maps_visited.add(current_map)
            reward += 100.0  # Bonus grande per nuova mappa (aumentato da 50 a 100)

        # Curiosità per nuovi Pokemon visti/catturati
        pokedex_seen = current_state.get('pokedex_seen', 0)
        if pokedex_seen > self.previous_state.get('pokedex_seen', 0):
            reward += 50.0 * (pokedex_seen - self.previous_state.get('pokedex_seen', 0))

        return reward