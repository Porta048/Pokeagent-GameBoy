import math
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class RewardWeights:
    alpha: float = 0.3
    beta: float = 0.4
    gamma: float = 0.3
    alpha_0: float = 0.3
    beta_0: float = 0.4
    gamma_0: float = 0.3
    rho_progress: float = 0.1
    rho_stall: float = 0.5
    rho_exploration: float = 0.2
    tau_stall: float = 300.0


class SubgoalDetector:
    SUBGOAL_WEIGHTS = {
        'talk_to_npc': 0.1, 'buy_pokeball': 0.2, 'enter_pokecenter': 0.1,
        'heal_pokemon': 0.1, 'defeat_trainer': 0.15
    }
    POKECENTER_MAPS = [0x01, 0x02, 0x03, 0x04, 0x05]

    def __init__(self, llm_client=None):
        self.completed_subgoals = set()
        self.subgoal_history = deque(maxlen=100)
        self.llm_client = llm_client
        self.last_llm_call_time = 0
        self.llm_call_interval = 5

    def detect_subgoals(self, current: Dict, previous: Dict, screen_array=None) -> Dict[str, float]:
        rewards = self._detect_traditional(current, previous)

        if self.llm_client and time.time() - self.last_llm_call_time > self.llm_call_interval:
            rewards.update(self._detect_llm_guided(current, screen_array))
            self.last_llm_call_time = time.time()

        filtered = {}
        for subgoal, reward in rewards.items():
            key = f"{subgoal}_{current.get('map_id', 0)}_{current.get('pos_x', 0)}_{current.get('pos_y', 0)}"
            if key not in self.completed_subgoals:
                filtered[subgoal] = reward
                self.completed_subgoals.add(key)
                self.subgoal_history.append((time.time(), subgoal, reward))
        return filtered

    def _detect_traditional(self, current: Dict, previous: Dict) -> Dict[str, float]:
        rewards = {}
        new_events = current.get('event_flags', set()) - previous.get('event_flags', set())
        npc_flags = [f for f in new_events if 100 <= f <= 200]
        if npc_flags:
            rewards['talk_to_npc'] = self.SUBGOAL_WEIGHTS['talk_to_npc'] * len(npc_flags)

        if current.get('player_money', 0) - previous.get('player_money', 0) < -100:
            rewards['buy_pokeball'] = self.SUBGOAL_WEIGHTS['buy_pokeball']

        if current.get('map_id') in self.POKECENTER_MAPS and previous.get('map_id') not in self.POKECENTER_MAPS:
            rewards['enter_pokecenter'] = self.SUBGOAL_WEIGHTS['enter_pokecenter']

        hp_curr, hp_prev = current.get('hp_team', [0]*6), previous.get('hp_team', [0]*6)
        hp_max = current.get('hp_max_team', [0]*6)
        for i in range(min(len(hp_curr), len(hp_prev), len(hp_max))):
            if hp_max[i] > 0 and hp_curr[i] > hp_prev[i] and hp_prev[i] < hp_max[i] * 0.5:
                rewards['heal_pokemon'] = self.SUBGOAL_WEIGHTS['heal_pokemon']
                break

        new_trainers = current.get('trainer_flags', set()) - previous.get('trainer_flags', set())
        if new_trainers:
            rewards['defeat_trainer'] = self.SUBGOAL_WEIGHTS['defeat_trainer'] * len(new_trainers)

        return rewards

    def _detect_llm_guided(self, current: Dict, screen_array=None) -> Dict[str, float]:
        if not self.llm_client:
            return {}
        try:
            context = {
                'map_id': current.get('map_id', 0),
                'position': (current.get('pos_x', 0), current.get('pos_y', 0)),
                'badges': current.get('badges', 0),
                'pokedex_caught': current.get('pokedex_caught', 0),
            }
            prompt = f"Pokemon state: {context}. Identify strategic subgoals. Reply with: explore/train/item"
            response = self.llm_client.get_subgoal_analysis(prompt, screen_array)
            if isinstance(response, str):
                resp_lower = response.lower()
                if 'explore' in resp_lower:
                    return {'llm_exploration': 0.15}
                if 'train' in resp_lower or 'gym' in resp_lower:
                    return {'llm_training': 0.2}
                if 'item' in resp_lower:
                    return {'llm_item_search': 0.25}
            return {}
        except Exception:
            return {}


class HierarchicalRewardCalculator:
    def __init__(self, llm_client=None):
        self.subgoal_detector = SubgoalDetector(llm_client)
        self.weights = RewardWeights()
        self.frame_count = 0
        self.explored_states = set()
        self.state_visit_counts = defaultdict(int)
        self.last_progress_time = time.time()
        self.progress_checkpoints = {'badges': 0, 'pokedex': 0}

    def calculate_total_reward(self, current_state: Dict, previous_state: Dict, loop_penalty: float = 0.0, screen_array=None) -> Tuple[float, Dict[str, float]]:
        self.frame_count += 1
        self._update_weights(current_state)

        primary = self._calc_primary(current_state, previous_state)
        secondary = self.subgoal_detector.detect_subgoals(current_state, previous_state, screen_array)
        intrinsic = self._calc_intrinsic(current_state)
        penalties = self._calc_penalties(current_state, previous_state, loop_penalty)

        total = 0.0
        details = {}

        for prefix, rewards, weight in [
            ('primary', primary, self.weights.alpha),
            ('secondary', secondary, self.weights.beta),
            ('intrinsic', intrinsic, self.weights.gamma),
            ('penalty', penalties, 1.0)
        ]:
            for k, v in rewards.items():
                weighted = weight * v if prefix != 'penalty' else v
                total += weighted
                details[f'{prefix}_{k}'] = round(weighted, 4)

        return np.clip(total, -1.0, 2.0), details

    def _update_weights(self, state: Dict):
        badges, pokedex = state.get('badges', 0), state.get('pokedex_caught', 0)
        d_badges = max(0, badges - self.progress_checkpoints['badges'])
        d_pokedex = max(0, pokedex - self.progress_checkpoints['pokedex'])

        if d_badges > 0:
            self.progress_checkpoints['badges'] = badges
            self.last_progress_time = time.time()
        if d_pokedex > 0:
            self.progress_checkpoints['pokedex'] = pokedex
            self.last_progress_time = time.time()

        t = time.time() - self.last_progress_time
        progress = d_badges + d_pokedex * 0.1

        self.weights.alpha = min(0.8, max(0.1, self.weights.alpha_0 * (1 + self.weights.rho_progress * progress)))
        self.weights.beta = min(0.7, max(0.1, self.weights.beta_0 * (1 + self.weights.rho_stall * math.exp(-t / self.weights.tau_stall))))
        self.weights.gamma = min(0.6, max(0.1, self.weights.gamma_0 * (1 + self.weights.rho_exploration * t / self.weights.tau_stall)))

    def _calc_primary(self, current: Dict, previous: Dict) -> Dict[str, float]:
        rewards = {}
        if current.get('badges', 0) > previous.get('badges', 0):
            rewards['badge'] = 2.0

        in_battle_prev = previous.get('in_battle', False)
        in_battle_curr = current.get('in_battle', False)

        if in_battle_prev and not in_battle_curr:
            alive = any(hp > 0 for hp in current.get('hp_team', []))
            rewards['battle_end'] = 1.0 if alive else -0.8

        if in_battle_curr:
            enemy_hp_prev = previous.get('enemy_hp', 100)
            enemy_hp_curr = current.get('enemy_hp', 100)
            if enemy_hp_curr < enemy_hp_prev:
                damage_dealt = (enemy_hp_prev - enemy_hp_curr) / max(enemy_hp_prev, 1)
                rewards['damage_dealt'] = damage_dealt * 0.3

            team_hp_prev = sum(previous.get('hp_team', [0]))
            team_hp_curr = sum(current.get('hp_team', [0]))
            if team_hp_curr < team_hp_prev:
                damage_taken = (team_hp_prev - team_hp_curr) / max(team_hp_prev, 1)
                rewards['damage_taken'] = -damage_taken * 0.15

            if current.get('pokemon_caught', 0) > previous.get('pokemon_caught', 0):
                rewards['catch'] = 0.8

        exp_curr = sum(current.get('team_exp', [0]))
        exp_prev = sum(previous.get('team_exp', [0]))
        if exp_curr > exp_prev:
            rewards['exp_gain'] = min((exp_curr - exp_prev) / 1000, 0.2)

        return rewards

    def _calc_intrinsic(self, state: Dict) -> Dict[str, float]:
        rewards = {}
        key = (state.get('map_id', 0), state.get('pos_x', 0), state.get('pos_y', 0))
        self.state_visit_counts[key] += 1
        count = self.state_visit_counts[key]
        rewards['novelty'] = 0.05 if count == 1 else (0.02 if count <= 3 else 0.005 / count)

        map_id = state.get('map_id', 0)
        if map_id not in self.explored_states:
            self.explored_states.add(map_id)
            rewards['new_map'] = 0.08
        return rewards

    def _calc_penalties(self, current: Dict, previous: Dict, loop_penalty: float) -> Dict[str, float]:
        penalties = {}
        t = time.time() - self.last_progress_time
        if t > 600:
            penalties['no_progress'] = -0.1
        elif t > 300:
            penalties['no_progress'] = -0.05
        if loop_penalty < 0:
            penalties['loop'] = loop_penalty / 10.0
        if current.get('in_battle') and current.get('map_id') == previous.get('map_id'):
            penalties['stuck_battle'] = -0.02
        return penalties

    def get_current_weights(self) -> RewardWeights:
        return self.weights
