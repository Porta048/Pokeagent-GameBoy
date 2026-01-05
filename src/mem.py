import time
from collections import deque
import numpy as np
class GameMemoryReader:
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
        'EVENT_FLAGS': (0xD747, 0xD886),
        'TRAINER_FLAGS': (0xD5A0, 0xD5F7),
        'BATTLE_TYPE': 0xD057,  # 0 = no battle, 1 = wild, 2 = trainer
    }
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.previous_state = {}
        self.visited_coordinates = set()
        self.last_position = None
        self.previous_event_flags = set()
        self.previous_trainer_flags = set()
        self.recent_level_ups = deque(maxlen=10)  
    def get_current_state(self) -> dict:
        try:
            mem = lambda addr: self.pyboy.memory[addr]
            money_bcd = [mem(addr) for addr in range(self.MEMORY_ADDRESSES['MONEY'][0], self.MEMORY_ADDRESSES['MONEY'][1] + 1)]
            money = sum(((b >> 4) * 10 + (b & 0xF)) * (100 ** i) for i, b in enumerate(reversed(money_bcd)))
            badges = bin(mem(self.MEMORY_ADDRESSES['BADGES'])).count('1')
            pokedex_caught = sum(bin(mem(addr)).count('1') for addr in range(self.MEMORY_ADDRESSES['POKEDEX_CAUGHT'], self.MEMORY_ADDRESSES['POKEDEX_CAUGHT'] + 19))
            pokedex_seen = sum(bin(mem(addr)).count('1') for addr in range(self.MEMORY_ADDRESSES['POKEDEX_SEEN'], self.MEMORY_ADDRESSES['POKEDEX_SEEN'] + 19))
            team_levels = [mem(addr[0]) for addr in self.MEMORY_ADDRESSES['TEAM_LEVELS']]
            hp_team = [mem(addr[0]) * 256 + mem(addr[1]) for addr in self.MEMORY_ADDRESSES['HP_TEAM']]
            hp_max_team = [mem(addr[0]) * 256 + mem(addr[1]) for addr in self.MEMORY_ADDRESSES['HP_MAX_TEAM']]
            battle_type = mem(self.MEMORY_ADDRESSES['BATTLE_TYPE'])
            in_battle = battle_type != 0  # 0 = no battle, 1 = wild, 2 = trainer
            event_flags = set()
            for addr in range(self.MEMORY_ADDRESSES['EVENT_FLAGS'][0], self.MEMORY_ADDRESSES['EVENT_FLAGS'][1] + 1):
                byte_val = mem(addr)
                for bit in range(8):
                    if byte_val & (1 << bit):
                        event_flags.add((addr - self.MEMORY_ADDRESSES['EVENT_FLAGS'][0]) * 8 + bit)
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
        r = self._calculate_event_flags_rewards(current_state)
        if r != 0: reward_details['events'] = r
        total_reward += r
        r = self._calculate_navigation_rewards(current_state)
        if r != 0: reward_details['navigation'] = r
        total_reward += r
        r = self._calculate_healing_rewards(current_state)
        if r != 0: reward_details['healing'] = r
        total_reward += r
        if reward_details:
            print(f"[REWARD] {reward_details} = {total_reward:.2f}")
        self.previous_state = current_state.copy()
        return total_reward
    def _calculate_badge_rewards(self, s: dict) -> float:
        return 2000 if s.get('badges', 0) > self.previous_state.get('badges', 0) else 0
    def _calculate_pokemon_rewards(self, s: dict) -> float:
        r = 0
        caught_diff = s.get('pokedex_caught', 0) - self.previous_state.get('pokedex_caught', 0)
        if caught_diff > 0:
            r += 300 * caught_diff  
        seen_diff = s.get('pokedex_seen', 0) - self.previous_state.get('pokedex_seen', 0)
        if seen_diff > 0:
            r += 30 * seen_diff  
        return r
    def _calculate_balanced_level_rewards(self, s: dict) -> float:
        lv_curr = s.get('team_levels', [0] * 6)
        lv_prev = self.previous_state.get('team_levels', [0] * 6)
        reward = 0.0
        for i in range(min(len(lv_curr), len(lv_prev))):
            if lv_curr[i] > lv_prev[i]:
                level_up_amount = lv_curr[i] - lv_prev[i]
                if lv_curr[i] <= 15:
                    base_reward = 50  
                elif lv_curr[i] <= 30:
                    base_reward = 30  
                elif lv_curr[i] <= 45:
                    base_reward = 15  
                else:
                    base_reward = 5   
                self.recent_level_ups.append(time.time())
                level_ups_last_minute = sum(1 for t in self.recent_level_ups if time.time() - t < 60)
                if level_ups_last_minute > 5:
                    grinding_penalty = 0.3
                else:
                    grinding_penalty = 1.0
                reward += base_reward * level_up_amount * grinding_penalty
        return reward
    def _calculate_money_rewards(self, s: dict) -> float:
        diff = s.get('player_money', 0) - self.previous_state.get('player_money', 0)
        return min(diff / 100, 20) if diff > 0 else -20 if diff < -100 else 0
    def _calculate_exploration_rewards(self, s: dict) -> float:
        current_map = s.get('map_id', 0)
        map_reward = 0
        if current_map != self.previous_state.get('map_id', 0):
            if not hasattr(self, 'maps_visited'):
                self.maps_visited = set()
            if current_map not in self.maps_visited:
                map_reward = 250  
                self.maps_visited.add(current_map)
            else:
                map_reward = 30  
        return map_reward
    def _calculate_battle_rewards(self, s: dict) -> float:
        prev_battle = self.previous_state.get('in_battle', False)
        curr_battle = s.get('in_battle', False)
        if prev_battle and not curr_battle:
            team_alive = any(hp > 0 for hp in s.get('hp_team', []))
            if team_alive:
                return 120
            else:
                return -200
        if not prev_battle and curr_battle:
            return 10
        return 0
    def _calculate_event_flags_rewards(self, s: dict) -> float:
        reward = 0.0
        event_flags_curr = s.get('event_flags', set())
        new_events = event_flags_curr - self.previous_event_flags
        if new_events:
            reward += 50.0 * len(new_events)
            self.previous_event_flags = event_flags_curr.copy()
        trainer_flags_curr = s.get('trainer_flags', set())
        new_trainers = trainer_flags_curr - self.previous_trainer_flags
        if new_trainers:
            reward += 200.0 * len(new_trainers)
            self.previous_trainer_flags = trainer_flags_curr.copy()
        return reward
    def _calculate_navigation_rewards(self, s: dict) -> float:
        pos_x = s.get('pos_x', 0)
        pos_y = s.get('pos_y', 0)
        map_id = s.get('map_id', 0)
        coord_key = (map_id, pos_x, pos_y)
        if coord_key not in self.visited_coordinates:
            self.visited_coordinates.add(coord_key)
            return 5.0
        return 0.0
    def _calculate_healing_rewards(self, s: dict) -> float:
        hp_curr = s.get('hp_team', [0] * 6)
        hp_max_curr = s.get('hp_max_team', [0] * 6)
        hp_prev = self.previous_state.get('hp_team', [0] * 6)
        reward = 0.0
        for i in range(min(len(hp_curr), len(hp_prev), len(hp_max_curr))):
            if hp_max_curr[i] > 0:
                hp_recovered = hp_curr[i] - hp_prev[i]
                if hp_recovered > 0:
                    recovery_percent = hp_recovered / hp_max_curr[i]
                    reward += recovery_percent * 5.0
        return reward