"""
Hierarchical Reward System for Pokemon AI Agent
Based on the mathematical improvements proposed:
- Primary: Task Critical (final goal signals, dense but scaled)
- Secondary: Subgoal Strategic (complex sequence guidance) 
- Intrinsic: Exploration & Learning (novelty and uncertainty)
- Penalty: Behavior Correction (loop and inefficiency penalties)

Formula: R_total = α * R_primary + β * R_secondary + γ * R_intrinseco + R_penalty
"""
import time
import numpy as np
from collections import deque, defaultdict
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
from .cfg import config
from .hyp import HYPERPARAMETERS


@dataclass
class RewardWeights:
    """Dynamic weights for the hierarchical reward system"""
    alpha: float = 0.3  # Primary (starts low, increases with progress)
    beta: float = 0.4   # Secondary (adaptive, increases when agent stuck)
    gamma: float = 0.3  # Intrinsic (adaptive, exploration)
    # Penalty is always applied (no weight needed)


class SubgoalDetector:
    """Detects and rewards strategic subgoals based on game state analysis"""

    def __init__(self, llm_client=None):
        self.completed_subgoals = set()
        self.subgoal_history = deque(maxlen=100)
        self.subgoal_weights = {
            'talk_to_npc': 0.1,
            'buy_pokeball': 0.2,
            'use_item_outside_battle': 0.15,
            'enter_pokecenter': 0.1,
            'heal_pokemon': 0.1,
            'train_in_gym': 0.25,
            'complete_puzzle': 0.3,
            'find_hidden_item': 0.25,
            'catch_specific_pokemon': 0.2,
            'defeat_trainer': 0.15
        }
        self.llm_client = llm_client
        self.llm_subgoal_cache = {}
        self.last_llm_call_time = 0
        self.llm_call_interval = 5  # Minimum seconds between LLM calls

    def detect_subgoals(self, current_state: Dict, previous_state: Dict, screen_array=None) -> Dict[str, float]:
        """Detect strategic subgoals and return rewards, with optional LLM guidance"""
        rewards = {}

        # Traditional subgoal detection
        rewards.update(self._detect_traditional_subgoals(current_state, previous_state))

        # LLM-guided subgoal detection (rate-limited)
        if self.llm_client and time.time() - self.last_llm_call_time > self.llm_call_interval:
            llm_rewards = self._detect_llm_guided_subgoals(current_state, screen_array)
            rewards.update(llm_rewards)
            self.last_llm_call_time = time.time()

        # Filter out already completed subgoals to avoid repetitive rewards
        filtered_rewards = {}
        for subgoal, reward in rewards.items():
            subgoal_key = f"{subgoal}_{current_state.get('map_id', 0)}_{current_state.get('pos_x', 0)}_{current_state.get('pos_y', 0)}"
            if subgoal_key not in self.completed_subgoals:
                filtered_rewards[subgoal] = reward
                self.completed_subgoals.add(subgoal_key)
                self.subgoal_history.append((time.time(), subgoal, reward))

        return filtered_rewards

    def _detect_traditional_subgoals(self, current_state: Dict, previous_state: Dict) -> Dict[str, float]:
        """Traditional subgoal detection based on memory values"""
        rewards = {}

        # Talk to NPC - detect event flag changes that indicate NPC interaction
        new_event_flags = current_state.get('event_flags', set()) - previous_state.get('event_flags', set())
        if new_event_flags:
            # Check if new event flags correspond to NPC conversations
            npc_related_flags = [flag for flag in new_event_flags if 100 <= flag <= 200]  # Example range
            if npc_related_flags:
                rewards['talk_to_npc'] = self.subgoal_weights['talk_to_npc'] * len(npc_related_flags)

        # Buy Pokeball - detect money decrease with item increase
        money_diff = current_state.get('player_money', 0) - previous_state.get('player_money', 0)
        if money_diff < -100:  # Significant money spent
            # This could indicate buying items, reward if it's strategic
            rewards['buy_pokeball'] = self.subgoal_weights['buy_pokeball']

        # Enter Pokecenter - detect map changes to Pokecenter maps
        pokecenter_maps = [0x01, 0x02, 0x03, 0x04, 0x05]  # Example Pokecenter map IDs
        if current_state.get('map_id') in pokecenter_maps and previous_state.get('map_id') not in pokecenter_maps:
            rewards['enter_pokecenter'] = self.subgoal_weights['enter_pokecenter']

        # Heal Pokemon - detect HP restoration
        hp_curr = current_state.get('hp_team', [0] * 6)
        hp_prev = previous_state.get('hp_team', [0] * 6)
        hp_max_curr = current_state.get('hp_max_team', [0] * 6)

        for i in range(min(len(hp_curr), len(hp_prev), len(hp_max_curr))):
            if hp_max_curr[i] > 0 and hp_curr[i] > hp_prev[i] and hp_prev[i] < hp_max_curr[i] * 0.5:
                # Significant healing when HP was low
                rewards['heal_pokemon'] = self.subgoal_weights['heal_pokemon']
                break

        # Defeat trainer - detect trainer flag changes
        new_trainer_flags = current_state.get('trainer_flags', set()) - previous_state.get('trainer_flags', set())
        if new_trainer_flags:
            rewards['defeat_trainer'] = self.subgoal_weights['defeat_trainer'] * len(new_trainer_flags)

        return rewards

    def _detect_llm_guided_subgoals(self, current_state: Dict, screen_array=None) -> Dict[str, float]:
        """Use LLM to detect contextually relevant subgoals based on screen and state"""
        if not self.llm_client:
            return {}

        try:
            # Create a context for the LLM to analyze
            context = {
                'map_id': current_state.get('map_id', 0),
                'position': (current_state.get('pos_x', 0), current_state.get('pos_y', 0)),
                'badges': current_state.get('badges', 0),
                'pokedex_caught': current_state.get('pokedex_caught', 0),
                'pokedex_seen': current_state.get('pokedex_seen', 0),
                'money': current_state.get('player_money', 0),
                'in_battle': current_state.get('in_battle', False),
                'team_levels': current_state.get('team_levels', []),
                'event_flags_count': len(current_state.get('event_flags', set()))
            }

            # Create a prompt for the LLM to identify strategic subgoals
            prompt = f"""
            Analyze this Pokemon game state and identify strategic subgoals that would be beneficial for the AI agent to pursue.
            Current state: {context}

            Consider:
            - Areas of the map that haven't been explored
            - Pokemon that haven't been caught
            - Gym locations that haven't been challenged
            - Items that could be useful
            - NPCs that might provide useful information or items
            - Strategic positioning for progression

            Respond with a JSON object containing potential subgoals and their strategic importance (0.0 to 0.5):
            Example: {{"explore_route_4": 0.2, "find_hidden_item": 0.3, "train_team": 0.15}}
            """

            # Call the LLM to get strategic subgoal suggestions
            response = self.llm_client.get_subgoal_analysis(prompt, screen_array)

            # Parse the LLM response (simplified - in practice, you'd want more robust parsing)
            if isinstance(response, str):
                # This is a simplified parsing - in practice you'd want to use proper JSON parsing
                # and handle potential errors more robustly
                if 'explore' in response.lower():
                    return {'llm_guided_exploration': 0.15}
                elif 'train' in response.lower() or 'gym' in response.lower():
                    return {'llm_guided_training': 0.2}
                elif 'item' in response.lower():
                    return {'llm_guided_item_search': 0.25}

            return {}

        except Exception as e:
            # If LLM call fails, return empty rewards but continue execution
            print(f"[LLM] Subgoal analysis failed: {e}")
            return {}


class HierarchicalRewardCalculator:
    """Main class for calculating hierarchical rewards"""

    def __init__(self, llm_client=None):
        self.subgoal_detector = SubgoalDetector(llm_client=llm_client)
        self.weights = RewardWeights()
        self.frame_count = 0

        # Track exploration for intrinsic rewards
        self.explored_states = set()
        self.state_visit_counts = defaultdict(int)
        self.exploration_memory = deque(maxlen=1000)

        # Track progress for adaptive weights
        self.badge_progress = 0
        self.pokedex_progress = 0

        # Track time-based penalties
        self.last_progress_time = time.time()
        self.progress_checkpoints = {
            'badges': 0,
            'pokedex': 0,
            'money': 0
        }
    
    def update_weights_adaptively(self, current_state: Dict):
        """Update reward weights based on agent progress and stuck status"""
        # Check for progress
        current_badges = current_state.get('badges', 0)
        current_pokedex = current_state.get('pokedex_caught', 0)
        current_money = current_state.get('player_money', 0)
        
        # Increase alpha (primary weight) as progress is made
        if current_badges > self.progress_checkpoints['badges']:
            self.weights.alpha = min(0.7, self.weights.alpha + 0.05)
            self.progress_checkpoints['badges'] = current_badges
            self.last_progress_time = time.time()
        
        if current_pokedex > self.progress_checkpoints['pokedex']:
            self.weights.alpha = min(0.7, self.weights.alpha + 0.02)
            self.progress_checkpoints['pokedex'] = current_pokedex
            self.last_progress_time = time.time()
        
        # Increase beta/gamma if agent is stuck (no progress for a while)
        time_since_progress = time.time() - self.last_progress_time
        if time_since_progress > 300:  # 5 minutes without progress
            self.weights.beta = min(0.6, self.weights.beta + 0.05)  # More subgoal guidance
            self.weights.gamma = min(0.5, self.weights.gamma + 0.05)  # More exploration
        else:
            # Gradually decrease if making progress
            if self.weights.beta > 0.4:
                self.weights.beta = max(0.4, self.weights.beta - 0.001)
            if self.weights.gamma > 0.3:
                self.weights.gamma = max(0.3, self.weights.gamma - 0.001)
    
    def calculate_primary_rewards(self, current_state: Dict, previous_state: Dict) -> Dict[str, float]:
        """Calculate primary (task-critical) rewards - normalized to [0, 1.0] range"""
        rewards = {}
        
        # Badge reward - normalized from 2000 to 1.0
        badge_diff = current_state.get('badges', 0) - previous_state.get('badges', 0)
        if badge_diff > 0:
            rewards['badge'] = 1.0  # Maximum primary reward
        
        # Victory reward - normalized from 120 to 0.8
        # This would be handled in battle context
        in_battle_prev = previous_state.get('in_battle', False)
        in_battle_curr = current_state.get('in_battle', False)
        
        if in_battle_prev and not in_battle_curr:
            team_alive = any(hp > 0 for hp in current_state.get('hp_team', []))
            if team_alive:
                rewards['battle_victory'] = 0.8
            else:
                rewards['battle_defeat'] = -0.5  # Penalty, not primary reward
        
        return rewards
    
    def calculate_secondary_rewards(self, current_state: Dict, previous_state: Dict, screen_array=None) -> Dict[str, float]:
        """Calculate secondary (subgoal) rewards - normalized to [0, 0.3] range"""
        return self.subgoal_detector.detect_subgoals(current_state, previous_state, screen_array=screen_array)
    
    def calculate_intrinsic_rewards(self, current_state: Dict) -> Dict[str, float]:
        """Calculate intrinsic (exploration) rewards - normalized to [0, 0.1] range"""
        rewards = {}
        
        # State novelty bonus
        state_key = (
            current_state.get('map_id', 0),
            current_state.get('pos_x', 0),
            current_state.get('pos_y', 0),
            tuple(sorted(current_state.get('event_flags', set()) & set(range(0, 50))))  # First 50 flags
        )
        
        self.state_visit_counts[state_key] += 1
        visit_count = self.state_visit_counts[state_key]
        
        if visit_count == 1:
            # First visit to this state - high novelty bonus
            rewards['novelty'] = 0.05
        elif visit_count <= 3:
            # Low visit count - medium novelty bonus
            rewards['novelty'] = 0.02
        else:
            # Frequently visited - low novelty bonus
            rewards['novelty'] = 0.005 / visit_count  # Diminishing returns
        
        # Map exploration bonus
        map_key = current_state.get('map_id', 0)
        if map_key not in self.explored_states:
            self.explored_states.add(map_key)
            rewards['new_map'] = 0.08
        
        return rewards
    
    def calculate_penalty_rewards(self, current_state: Dict, previous_state: Dict, 
                                loop_penalty: float = 0.0) -> Dict[str, float]:
        """Calculate penalty rewards - negative values"""
        penalties = {}
        
        # Time-based penalty for no progress
        time_since_progress = time.time() - self.last_progress_time
        if time_since_progress > 600:  # 10 minutes without progress
            penalties['no_progress'] = -0.1
        elif time_since_progress > 300:  # 5 minutes without progress
            penalties['no_progress'] = -0.05
        
        # Loop penalty from anti-loop system
        if loop_penalty < 0:
            penalties['loop'] = loop_penalty / 10.0  # Normalize the existing penalty
        
        # Stuck in battle penalty
        if current_state.get('in_battle', False) and current_state.get('map_id', 0) == previous_state.get('map_id', 0):
            # Been in same battle for too long
            penalties['stuck_in_battle'] = -0.02
        
        return penalties
    
    def calculate_total_reward(self, current_state: Dict, previous_state: Dict,
                             loop_penalty: float = 0.0, screen_array=None) -> Tuple[float, Dict[str, float]]:
        """Calculate total hierarchical reward using the mathematical formula"""
        self.frame_count += 1

        # Update adaptive weights based on progress
        self.update_weights_adaptively(current_state)

        # Calculate all reward components
        primary_rewards = self.calculate_primary_rewards(current_state, previous_state)
        secondary_rewards = self.calculate_secondary_rewards(current_state, previous_state, screen_array=screen_array)
        intrinsic_rewards = self.calculate_intrinsic_rewards(current_state)
        penalty_rewards = self.calculate_penalty_rewards(current_state, previous_state, loop_penalty)

        # Apply weights and sum up
        total_reward = 0.0
        reward_details = {}

        # Primary rewards (scaled by alpha)
        primary_total = sum(primary_rewards.values())
        total_reward += self.weights.alpha * primary_total
        for key, value in primary_rewards.items():
            reward_details[f'primary_{key}'] = round(self.weights.alpha * value, 4)

        # Secondary rewards (scaled by beta)
        secondary_total = sum(secondary_rewards.values())
        total_reward += self.weights.beta * secondary_total
        for key, value in secondary_rewards.items():
            reward_details[f'secondary_{key}'] = round(self.weights.beta * value, 4)

        # Intrinsic rewards (scaled by gamma)
        intrinsic_total = sum(intrinsic_rewards.values())
        total_reward += self.weights.gamma * intrinsic_total
        for key, value in intrinsic_rewards.items():
            reward_details[f'intrinsic_{key}'] = round(self.weights.gamma * value, 4)

        # Penalty rewards (no scaling, applied directly)
        penalty_total = sum(penalty_rewards.values())
        total_reward += penalty_total
        for key, value in penalty_rewards.items():
            reward_details[f'penalty_{key}'] = round(value, 4)

        # Normalize total to prevent extreme values
        total_reward = np.clip(total_reward, -1.0, 2.0)

        return total_reward, reward_details
    
    def get_current_weights(self) -> RewardWeights:
        """Return current reward weights for monitoring"""
        return self.weights