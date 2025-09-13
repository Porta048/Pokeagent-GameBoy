import os
import sys
import time
import random
from typing import Union, List, Dict, Optional, Tuple, Any
import numpy as np
import json
import pickle
from collections import deque
import hashlib

REGIONS = {
    'HP': (slice(100, 120), slice(90, 150)),
    'MENU': (slice(110, 140), slice(0, 80)),
    'DIALOGUE': (slice(100, 140), slice(10, 150)),
    'CENTER': (slice(60, 100), slice(70, 90))
}

CONST = {
    'HP_THRESH': 500, 'MENU_THRESH': 0.15, 'DIALOGUE_THRESH': 30,
    'MOVE_THRESH': 0.02, 'STUCK_THRESH': 50, 'MEM_INTERVAL': 30,
    'BATCH_SIZE': 64, 'GAMMA': 0.99, 'EPS_MIN': 0.05, 'EPS_DECAY': 0.9995,
    'LR': 0.00025, 'MEM_SIZE': 50000, 'TARGET_FREQ': 100, 'SAVE_FREQ': 10000
}

class PokemonAIError(Exception): pass

def check_dependencies() -> bool:
    try:
        import torch
        return True
    except ImportError:
        os.system(f"{sys.executable} -m pip install torch pyboy numpy opencv-python keyboard")
        return True

torch_available = check_dependencies()

import pyboy
from pyboy.utils import WindowEvent
import keyboard
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PokemonMemoryReader:
    def __init__(self, pyboy: Any) -> None:
        self.pyboy = pyboy
        self.game_type = self._detect_game_type()
        self.memory_addresses = self._get_memory_addresses()
        self.prev_state = {'player_money': 0, 'badges': 0, 'pokedex_owned': 0, 'pokedex_seen': 0, 'party_levels': [0] * 6, 'party_hp': [0] * 6, 'x_pos': 0, 'y_pos': 0, 'map_id': 0}

    def _detect_game_type(self) -> str:
        title = self.pyboy.cartridge_title.strip().upper()
        if 'RED' in title or 'BLUE' in title: return 'rb'
        elif 'YELLOW' in title: return 'yellow'
        elif 'GOLD' in title or 'SILVER' in title: return 'gs'
        elif 'CRYSTAL' in title: return 'crystal'
        else: return 'generic'

    def _get_memory_addresses(self) -> Dict[str, int]:
        base = {'player_money': 0xD347, 'badges': 0xD356, 'party_count': 0xD163, 'x_pos': 0xD362, 'y_pos': 0xD361, 'map_id': 0xD35E, 'pokedex_owned': 0xD2F7, 'pokedex_seen': 0xD30A, 'party_levels': 0xD18C, 'party_hp': 0xD16C, 'battle_type': 0xD057}
        if self.game_type in ['gs', 'crystal']:
            base.update({'player_money': 0xD84E, 'badges': 0xD857, 'pokedex_owned': 0xDE99, 'party_count': 0xDCD7, 'party_levels': 0xDCFF, 'party_hp': 0xDD01, 'x_pos': 0xDCB8, 'y_pos': 0xDCB7, 'map_id': 0xDCB5})
        return base

    def read_memory(self, address: int, length: int = 1) -> Union[int, List[int]]:
        try:
            return self.pyboy.memory[address] if length == 1 else [self.pyboy.memory[address + i] for i in range(length)]
        except:
            return 0 if length == 1 else [0] * length

    def get_current_state(self) -> Dict[str, Any]:
        state = {}
        try:
            for key in ['player_money', 'badges', 'pokedex_owned', 'pokedex_seen', 'x_pos', 'y_pos', 'map_id']:
                if key in self.memory_addresses:
                    if key == 'player_money':
                        state[key] = self._bcd_to_int(self.read_memory(self.memory_addresses[key], 3))
                    elif key == 'badges':
                        state[key] = bin(self.read_memory(self.memory_addresses[key])).count('1')
                    elif key in ['pokedex_owned', 'pokedex_seen']:
                        state[key] = sum(bin(b).count('1') for b in self.read_memory(self.memory_addresses[key], 19))
                    else:
                        state[key] = self.read_memory(self.memory_addresses[key])
            
            party_count = min(self.read_memory(self.memory_addresses.get('party_count', 0xD163)), 6)
            state['party_count'] = party_count
            if party_count > 0:
                levels = self.read_memory(self.memory_addresses.get('party_levels', 0xD18C), party_count * 48)
                state['party_levels'] = [levels[i * 48] if i * 48 < len(levels) else 0 for i in range(6)]
                hp_data = self.read_memory(self.memory_addresses.get('party_hp', 0xD16C), party_count * 96)
                state['party_hp'] = [(hp_data[i*96] << 8) | hp_data[i*96+1] if i*96+1 < len(hp_data) else 0 for i in range(party_count)]
            
            state['in_battle'] = self.read_memory(self.memory_addresses.get('battle_type', 0xD057)) != 0
        except:
            pass
        return state

    def _bcd_to_int(self, bcd_bytes: List[int]) -> int:
        return sum(((b >> 4) * 10 + (b & 0x0F)) * (100 ** i) for i, b in enumerate(reversed(bcd_bytes)))

    def calculate_reward_events(self, current_state: Dict[str, Any]) -> float:
        reward = 0
        if not self.prev_state:
            self.prev_state = current_state.copy()
            return reward
            
        reward += self._calculate_badge_rewards(current_state)
        reward += self._calculate_pokemon_rewards(current_state)
        reward += self._calculate_level_rewards(current_state)
        reward += self._calculate_money_rewards(current_state)
        reward += self._calculate_exploration_rewards(current_state)
        reward += self._calculate_battle_rewards(current_state)
        
        self.prev_state = current_state.copy()
        return reward
    
    def _calculate_badge_rewards(self, current_state: Dict[str, Any]) -> float:
        return 1000 if current_state.get('badges', 0) > self.prev_state.get('badges', 0) else 0
    
    def _calculate_pokemon_rewards(self, current_state: Dict[str, Any]) -> float:
        reward = 0
        if current_state.get('pokedex_owned', 0) > self.prev_state.get('pokedex_owned', 0):
            reward += 100 * (current_state.get('pokedex_owned', 0) - self.prev_state.get('pokedex_owned', 0))
        if current_state.get('pokedex_seen', 0) > self.prev_state.get('pokedex_seen', 0):
            reward += 10 * (current_state.get('pokedex_seen', 0) - self.prev_state.get('pokedex_seen', 0))
        return reward
    
    def _calculate_level_rewards(self, current_state: Dict[str, Any]) -> float:
        reward = 0
        current_levels = current_state.get('party_levels', [0] * 6)
        prev_levels = self.prev_state.get('party_levels', [0] * 6)
        for i in range(min(len(current_levels), len(prev_levels))):
            if current_levels[i] > prev_levels[i]:
                reward += 50 * (current_levels[i] - prev_levels[i])
        return reward
    
    def _calculate_money_rewards(self, current_state: Dict[str, Any]) -> float:
        money_diff = current_state.get('player_money', 0) - self.prev_state.get('player_money', 0)
        if money_diff > 0: return min(money_diff / 100, 20)
        elif money_diff < -100: return -20
        return 0
    
    def _calculate_exploration_rewards(self, current_state: Dict[str, Any]) -> float:
        reward = 0
        if current_state.get('map_id', 0) != self.prev_state.get('map_id', 0): reward += 30
        x_diff = abs(current_state.get('x_pos', 0) - self.prev_state.get('x_pos', 0))
        y_diff = abs(current_state.get('y_pos', 0) - self.prev_state.get('y_pos', 0))
        if x_diff + y_diff > 5: reward += 2
        return reward
    
    def _calculate_battle_rewards(self, current_state: Dict[str, Any]) -> float:
        reward = 0
        if self.prev_state.get('in_battle', False) and not current_state.get('in_battle', False):
            party_hp = current_state.get('party_hp', [])
            if any(hp > 0 for hp in party_hp): reward += 50
            else: reward -= 100
        elif not self.prev_state.get('in_battle', False) and current_state.get('in_battle', False):
            reward += 2
        return reward

class PokemonStateDetector:
    def __init__(self) -> None:
        pass
        
    def detect_battle(self, screen_array: np.ndarray) -> bool:
        hp_region = screen_array[REGIONS['HP']]
        hp_variance = np.var(hp_region)
        menu_region = screen_array[REGIONS['MENU']]
        menu_edges = cv2.Canny(menu_region.astype(np.uint8), 50, 150)
        edge_density = np.sum(menu_edges > 0) / menu_edges.size
        return hp_variance > CONST['HP_THRESH'] and edge_density > 0.1
    
    def detect_menu(self, screen_array: np.ndarray) -> bool:
        edges = cv2.Canny(screen_array.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        num_lines = 0 if lines is None else len(lines)
        return edge_density > CONST['MENU_THRESH'] and num_lines > 5
    
    def detect_dialogue(self, screen_array: np.ndarray) -> bool:
        dialogue_region = screen_array[REGIONS['DIALOGUE']]
        contrast = np.std(dialogue_region)
        edges = cv2.Canny(dialogue_region.astype(np.uint8), 50, 150)
        has_box = np.sum(edges[0, :]) > 20 and np.sum(edges[-1, :]) > 20
        return contrast > CONST['DIALOGUE_THRESH'] and has_box
    
    def detect_blocked_movement(self, current_screen: np.ndarray, previous_screen: Optional[np.ndarray]) -> bool:
        if previous_screen is None: return False
        diff = np.mean(np.abs(current_screen - previous_screen))
        is_blocked = diff < CONST['MOVE_THRESH']
        center_region = current_screen[REGIONS['CENTER']]
        center_variance = np.var(center_region)
        stuck_against_wall = center_variance < CONST['STUCK_THRESH']
        return is_blocked or stuck_against_wall

def calculate_conv_output_size(input_height, input_width, conv_layers):
    h, w = input_height, input_width
    for kernel_size, stride in conv_layers:
        h = (h - kernel_size) // stride + 1
        w = (w - kernel_size) // stride + 1
    return h, w

STANDARD_CONV_DIMS = calculate_conv_output_size(144, 160, [(8, 4), (4, 2), (3, 1)])
MENU_CONV_DIMS = calculate_conv_output_size(144, 160, [(4, 2), (3, 2)])

class ExplorerDQN(nn.Module):
    def __init__(self, n_actions):
        super(ExplorerDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        convh, convw = STANDARD_CONV_DIMS
        linear_input_size = convh * convw * 64
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class BattleDQN(nn.Module):
    def __init__(self, n_actions):
        super(BattleDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        convh, convw = STANDARD_CONV_DIMS
        linear_input_size = convh * convw * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MenuDQN(nn.Module):
    def __init__(self, n_actions):
        super(MenuDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        convh, convw = MENU_CONV_DIMS
        linear_input_size = convh * convw * 32
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PokemonMultiAgent:
    def __init__(self, n_actions: int, device: Any) -> None:
        self.device = device
        self.n_actions = n_actions
        
        self.explorer_agent = ExplorerDQN(n_actions).to(device)
        self.battle_agent = BattleDQN(n_actions).to(device)
        self.menu_agent = MenuDQN(n_actions).to(device)
        
        self.explorer_target = ExplorerDQN(n_actions).to(device)
        self.battle_target = BattleDQN(n_actions).to(device)
        self.menu_target = MenuDQN(n_actions).to(device)
        
        self.explorer_optimizer = optim.Adam(self.explorer_agent.parameters(), lr=CONST['LR'])
        self.battle_optimizer = optim.Adam(self.battle_agent.parameters(), lr=0.0003)
        self.menu_optimizer = optim.Adam(self.menu_agent.parameters(), lr=0.0002)
        
        self.explorer_target.load_state_dict(self.explorer_agent.state_dict())
        self.battle_target.load_state_dict(self.battle_agent.state_dict())
        self.menu_target.load_state_dict(self.menu_agent.state_dict())
        
    def choose_action(self, state: Any, game_state: str, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_batch = state.unsqueeze(0)
            if game_state == "battle":
                q_values = self.battle_agent(state_batch)
            elif game_state == "menu":
                q_values = self.menu_agent(state_batch)
            else:
                q_values = self.explorer_agent(state_batch)
            return q_values.argmax().item()
    
    def train_agent(self, batch: List[Tuple], game_state: str) -> Optional[float]:
        if len(batch) < 4:
            return
        
        states = []
        next_states = []
        for exp in batch:
            if isinstance(exp[0], np.ndarray) and exp[0].shape == (144, 160):
                states.append(exp[0])
                next_states.append(exp[3])
        
        if len(states) < 4:
            return
        
        states_np = np.expand_dims(np.array(states), axis=1)
        next_states_np = np.expand_dims(np.array(next_states), axis=1)
        
        states_t = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch[:len(states)]]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch[:len(states)]]).to(self.device)
        next_states_t = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch[:len(states)]]).to(self.device)
        
        if game_state == "battle":
            current_q_values = self.battle_agent(states_t).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.battle_target(next_states_t).max(1)[0]
                target_q_values = rewards + (1 - dones) * CONST['GAMMA'] * next_q_values
            
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            self.battle_optimizer.zero_grad()
            loss.backward()
            self.battle_optimizer.step()
            
        elif game_state == "menu":
            current_q_values = self.menu_agent(states_t).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.menu_target(next_states_t).max(1)[0]
                target_q_values = rewards + (1 - dones) * CONST['GAMMA'] * next_q_values
            
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            self.menu_optimizer.zero_grad()
            loss.backward()
            self.menu_optimizer.step()
            
        else:
            current_q_values = self.explorer_agent(states_t).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.explorer_target(next_states_t).max(1)[0]
                target_q_values = rewards + (1 - dones) * CONST['GAMMA'] * next_q_values
            
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            self.explorer_optimizer.zero_grad()
            loss.backward()
            self.explorer_optimizer.step()
        
        return loss.item()
    
    def update_target_networks(self) -> None:
        tau = 0.005
        for target_param, param in zip(self.explorer_target.parameters(), self.explorer_agent.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.battle_target.parameters(), self.battle_agent.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.menu_target.parameters(), self.menu_agent.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class PokemonAI:
    def __init__(self, rom_path: str, headless: bool = False) -> None:
        self.rom_name = os.path.splitext(os.path.basename(rom_path))[0]
        self.save_dir = f"pokemon_ai_saves_{self.rom_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.save_dir, "model.pth")
        self.memory_path = os.path.join(self.save_dir, "memory.pkl")
        self.stats_path = os.path.join(self.save_dir, "stats.json")
        
        self.pyboy = pyboy.PyBoy(rom_path, window="null" if headless else "SDL2", debug=False)
        self.state_detector = PokemonStateDetector()
        self.memory_reader = PokemonMemoryReader(self.pyboy)
        
        self.actions = [[], [WindowEvent.PRESS_ARROW_UP], [WindowEvent.PRESS_ARROW_DOWN], [WindowEvent.PRESS_ARROW_LEFT], [WindowEvent.PRESS_ARROW_RIGHT], [WindowEvent.PRESS_BUTTON_A], [WindowEvent.PRESS_BUTTON_B], [WindowEvent.PRESS_BUTTON_START], [WindowEvent.PRESS_BUTTON_SELECT]]
        self.release_map = {WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A, WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B, WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START, WindowEvent.PRESS_BUTTON_SELECT: WindowEvent.RELEASE_BUTTON_SELECT}
        
        self.stats = self._load_stats()
        self.epsilon = max(CONST['EPS_MIN'], 0.9 * (0.995 ** (self.stats.get('total_frames', 0) / 10000)))
        self.memory = deque(maxlen=CONST['MEM_SIZE'])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_agent = PokemonMultiAgent(len(self.actions), self.device)
        
        self.frame_count = 0
        self.episode_count = self.stats.get('episodes', 0)
        self.total_reward = 0
        self.best_reward = self.stats.get('best_reward', float('-inf'))
        self.current_game_state = "exploring"
        self.last_memory_check = 0
        self.reward_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=100)
        
        self._load_memory()

    def _has_torch_multiagent(self) -> bool:
        return torch_available and hasattr(self, 'multi_agent')

    def _safe_choose_action(self, state: Any, game_state: str) -> int:
        if self._has_torch_multiagent():
            return self.multi_agent.choose_action(state, game_state, self.epsilon)
        else:
            return random.randint(0, len(self.actions) - 1)

    def _safe_replay(self) -> None:
        if not self._has_torch_multiagent() or len(self.memory) < CONST['BATCH_SIZE']:
            return
        try:
            explorer_batch = []
            battle_batch = []
            menu_batch = []
            
            memory_size = len(self.memory)
            sample_size = min(int(CONST['BATCH_SIZE'] * 2), memory_size)
            sampled_experiences = random.sample(self.memory, sample_size)
            
            for exp in sampled_experiences:
                if len(exp) >= 6:
                    game_state = exp[5]
                    if game_state == "battle" and len(battle_batch) < CONST['BATCH_SIZE']:
                        battle_batch.append(exp)
                    elif game_state == "menu" and len(menu_batch) < CONST['BATCH_SIZE']:
                        menu_batch.append(exp)
                    elif len(explorer_batch) < CONST['BATCH_SIZE']:
                        explorer_batch.append(exp)
                
                if len(explorer_batch) >= CONST['BATCH_SIZE'] and len(battle_batch) >= CONST['BATCH_SIZE'] and len(menu_batch) >= CONST['BATCH_SIZE']:
                    break
            
            loss_total = 0
            losses_count = 0
            
            for batch, game_state in [(explorer_batch, 'exploring'), (battle_batch, 'battle'), (menu_batch, 'menu')]:
                if len(batch) >= 4:
                    loss = self.multi_agent.train_agent(batch[:CONST['BATCH_SIZE']], game_state)
                    if loss:
                        loss_total += loss
                        losses_count += 1
            
            if losses_count > 0:
                avg_loss = loss_total / losses_count
                self.loss_history.append(avg_loss)
            
            if self.epsilon > CONST['EPS_MIN']:
                self.epsilon *= CONST['EPS_DECAY']
                
        except Exception as e:
            raise PokemonAIError(f"Training failed: {e}")

    def _safe_update_target_model(self) -> None:
        if self._has_torch_multiagent():
            self.multi_agent.update_target_networks()

    def _safe_save_model(self) -> None:
        if self._has_torch_multiagent():
            try:
                checkpoint = {'explorer_state': self.multi_agent.explorer_agent.state_dict(), 'battle_state': self.multi_agent.battle_agent.state_dict(), 'menu_state': self.multi_agent.menu_agent.state_dict(), 'epsilon': self.epsilon, 'episode': self.episode_count, 'frame': self.frame_count}
                torch.save(checkpoint, self.model_path)
            except Exception as e:
                raise PokemonAIError(f"Save error: {e}")

    def _get_screen_tensor(self):
        screen = self.pyboy.screen.image
        gray = np.array(screen.convert('L'))
        self.last_screen_array = gray.copy()
        normalized = gray.astype(np.float32) / 255.0
        
        if self._has_torch_multiagent():
            tensor = torch.from_numpy(normalized).unsqueeze(0)
            return tensor.to(self.device)
        return normalized

    def _detect_game_state(self, screen_array):
        old_state = self.current_game_state
        
        if self.state_detector.detect_battle(screen_array):
            self.current_game_state = "battle"
        elif self.state_detector.detect_dialogue(screen_array):
            self.current_game_state = "dialogue"
        elif self.state_detector.detect_menu(screen_array):
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
        
        if self.frame_count - self.last_memory_check >= CONST['MEM_INTERVAL']:
            self.last_memory_check = self.frame_count
            memory_state = self.memory_reader.get_current_state()
            memory_reward = self.memory_reader.calculate_reward_events(memory_state)
            reward += memory_reward
        
        if previous_screen is not None:
            if self._has_torch_multiagent():
                diff = torch.abs(screen_tensor - previous_screen).mean().item()
            else:
                diff = np.mean(np.abs(screen_tensor - previous_screen))
            
            if diff > 0.02:
                reward += 1.0
            else:
                reward -= 0.1
        
        self.reward_history.append(reward)
        self.total_reward += reward
        return reward

    def choose_action(self, state):
        game_state = self.current_game_state if hasattr(self, 'current_game_state') else "exploring"
        return self._safe_choose_action(state, game_state)

    def remember(self, state, action, reward, next_state, done):
        if self._has_torch_multiagent():
            state_np = state.squeeze(0).cpu().numpy()
            next_state_np = next_state.squeeze(0).cpu().numpy()
        else:
            state_np = state
            next_state_np = next_state
        
        game_state = self.current_game_state if hasattr(self, 'current_game_state') else "exploring"
        experience = (state_np, action, reward, next_state_np, done, game_state)
        self.memory.append(experience)

    def replay(self):
        self._safe_replay()

    def update_target_model(self):
        self._safe_update_target_model()

    def _load_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        return {'episodes': 0, 'total_frames': 0, 'best_reward': float('-inf')}

    def _save_stats(self):
        final_state = self.memory_reader.get_current_state()
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
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'rb') as f:
                    memory_data = pickle.load(f)
                for exp in memory_data.get('memory', [])[-5000:]:
                    if len(exp) == 5 and isinstance(exp[0], np.ndarray):
                        self.memory.append(exp)
            except:
                pass

    def play(self) -> None:
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
                action = self.choose_action(state)
                
                for button in self.actions[action]:
                    self.pyboy.send_input(button)
                
                wait_frames = 4
                if self.current_game_state == "dialogue":
                    wait_frames = 2
                elif self.current_game_state == "battle":
                    wait_frames = 6
                
                for _ in range(wait_frames):
                    self.pyboy.tick()
                
                for button in self.actions[action]:
                    if button in self.release_map:
                        self.pyboy.send_input(self.release_map[button])
                
                next_state = self._get_screen_tensor()
                reward = self._calculate_reward(next_state, previous_screen, action)
                
                if previous_screen is not None:
                    self.remember(state, action, reward, next_state, False)
                
                if self.frame_count % 4 == 0 and len(self.memory) >= CONST['BATCH_SIZE']:
                    self.replay()
                
                if self.frame_count % CONST['TARGET_FREQ'] == 0:
                    self.update_target_model()
                
                if self.frame_count - last_save_frame >= CONST['SAVE_FREQ']:
                    self._save_all()
                    last_save_frame = self.frame_count
                
                previous_screen = next_state
                self.frame_count += 1
                    
        except KeyboardInterrupt:
            pass
        
        finally:
            self._save_all()
            self.pyboy.stop()

    def _save_all(self) -> None:
        self._safe_save_model()
        self._save_memory()
        self._save_stats()

def main() -> None:
    while True:
        rom_path = input("Pokemon ROM path (.gb/.gbc): ").strip().strip('"')
        if os.path.exists(rom_path) and (rom_path.lower().endswith('.gbc') or rom_path.lower().endswith('.gb')) and os.path.getsize(rom_path) > 0:
            break
    
    headless = input("Headless mode (y/N): ").lower().strip() == 'y'
    
    try:
        ai = PokemonAI(rom_path, headless=headless)
        ai.play()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()