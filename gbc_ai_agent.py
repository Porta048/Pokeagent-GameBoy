
"""
Pokemon Game Boy AI Agent - Deep Learning Optimized
Specialized version for autonomous Pokemon gameplay
"""

import os
import sys
import time
import random
import numpy as np
import json
import pickle
from collections import deque
from datetime import datetime
import hashlib


def check_and_install_dependencies():
    """Check and install required dependencies"""
    required = {
        'pyboy': 'pyboy',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'keyboard': 'keyboard',
        'torch': 'torch torchvision --index-url https://download.pytorch.org/whl/cpu',
        'opencv-python': 'opencv-python'
    }
    
    print("Checking dependencies...")
    missing = []
    
    
    for module, package in required.items():
        if module != 'torch':
            try:
                if module == 'opencv-python':
                    __import__('cv2')
                else:
                    __import__(module)
                print(f"{module} already installed")
            except ImportError:
                missing.append(package)
    
  
    try:
        import torch
        print(f"PyTorch already installed - version {torch.__version__}")
        torch_available = True
    except ImportError:
        print("PyTorch not found - installing...")
        print("This may take a few minutes...")
        result = os.system(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        if result == 0:
            print("PyTorch installed successfully!")
            torch_available = True
        else:
            print("Error installing PyTorch")
            torch_available = False
    
    
    if missing:
        print(f"Installing other dependencies: {', '.join(missing)}")
        for package in missing:
            os.system(f"{sys.executable} -m pip install {package}")
    
    return torch_available


torch_available = check_and_install_dependencies()


import pyboy
from pyboy.utils import WindowEvent
from PIL import Image
import keyboard
import cv2


if torch_available:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    print(f"PyTorch {torch.__version__} loaded successfully!")


class PokemonMemoryReader:
    """Reads game memory for advanced reward shaping"""
    
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.game_type = self._detect_game_type()
        
        
        self.memory_addresses = self._get_memory_addresses()
        
        
        self.prev_state = {
            'player_money': 0,
            'badges': 0,
            'pokedex_owned': 0,
            'pokedex_seen': 0,
            'party_levels': [0] * 6,
            'party_hp': [0] * 6,
            'x_pos': 0,
            'y_pos': 0,
            'map_id': 0,
            'battle_turns': 0,
            'items_count': 0
        }
    
    def _detect_game_type(self):
        """Detects which Pokemon game is loaded"""
       
        title = self.pyboy.cartridge_title.strip()
        
        if 'RED' in title.upper() or 'BLUE' in title.upper():
            return 'rb'
        elif 'YELLOW' in title.upper():
            return 'yellow'
        elif 'GOLD' in title.upper() or 'SILVER' in title.upper():
            return 'gs'
        elif 'CRYSTAL' in title.upper():
            return 'crystal'
        else:
            print(f"Unrecognized game: {title}, using generic addresses")
            return 'generic'
    
    def _get_memory_addresses(self):
        """Gets memory addresses for the specific game"""
        gen1_base = {
            'player_money': 0xD347,
            'badges': 0xD356,
            'party_count': 0xD163,
            'x_pos': 0xD362,
            'y_pos': 0xD361,
            'map_id': 0xD35E,
            'items_count': 0xD31D,
            'player_name': 0xD158,
            'pokedex_owned': 0xD2F7,
            'pokedex_seen': 0xD30A,
            'party_species': 0xD164,
            'party_levels': 0xD18C,
            'party_hp': 0xD16C,
            'battle_type': 0xD057
        }
        
        if self.game_type == 'rb':
            return {
                **gen1_base,
                'rival_name': 0xD34A,
                'party_max_hp': 0xD18D,
                'enemy_hp': 0xCFE6,
                'enemy_max_hp': 0xCFF4,
                'current_box_items': 0xD53A
            }
       
        elif self.game_type == 'yellow':
            return {
                **gen1_base,
                'pikachu_happiness': 0xD46F
            }
        
        elif self.game_type in ['gs', 'crystal']:
            gen2_base = {
                'player_money': 0xD84E,
                'badges': 0xD857,
                'player_name': 0xD47B,
                'pokedex_owned': 0xDE99,
                'party_count': 0xDCD7,
                'party_species': 0xDCD8,
                'party_levels': 0xDCFF,
                'party_hp': 0xDD01,
                'x_pos': 0xDCB8,
                'y_pos': 0xDCB7,
                'map_id': 0xDCB5,
                'johto_badges': 0xD857,
                'kanto_badges': 0xD858,
                'battle_type': 0xD0EE
            }
            
            if self.game_type == 'gs':
                return {
                    **gen2_base,
                    'pokedex_seen': 0xDEA9,
                    'items_pocket': 0xD892,
                    'key_items_pocket': 0xD8BC,
                    'time_played': 0xD4A0
                }
            else:  # crystal
                return {
                    **gen2_base,
                    'player_gender': 0xD472,
                    'pokedex_seen': 0xDEB9,
                    'unown_dex': 0xDE41,
                    'mom_money': 0xD851
                }
        else:
            return {
                'player_money': 0xD347,
                'badges': 0xD356,
                'party_count': 0xD163,
                'x_pos': 0xD362,
                'y_pos': 0xD361,
                'map_id': 0xD35E
            }
    
    def read_memory(self, address, length=1):
        """Reads bytes from Game Boy memory"""
        try:
            if length == 1:
                return self.pyboy.memory[address]
            else:
                return [self.pyboy.memory[address + i] for i in range(length)]
        except:
            return 0 if length == 1 else [0] * length
    
    def get_current_state(self):
        """Gets current game state from memory"""
        state = {}
        
        try:
            
            if 'player_money' in self.memory_addresses:
                money_bytes = self.read_memory(self.memory_addresses['player_money'], 3)
                state['player_money'] = self._bcd_to_int(money_bytes)
            
            
            if 'badges' in self.memory_addresses:
                state['badges'] = bin(self.read_memory(self.memory_addresses['badges'])).count('1')
            
            
            if 'pokedex_owned' in self.memory_addresses:
                owned_bytes = self.read_memory(self.memory_addresses['pokedex_owned'], 19)
                state['pokedex_owned'] = sum(bin(b).count('1') for b in owned_bytes)
            
            if 'pokedex_seen' in self.memory_addresses:
                seen_bytes = self.read_memory(self.memory_addresses['pokedex_seen'], 19)
                state['pokedex_seen'] = sum(bin(b).count('1') for b in seen_bytes)
            
           
            if 'party_count' in self.memory_addresses:
                party_count = min(self.read_memory(self.memory_addresses['party_count']), 6)
                state['party_count'] = party_count
                
                
                if 'party_levels' in self.memory_addresses and party_count > 0:
                    levels = self.read_memory(self.memory_addresses['party_levels'], party_count * 0x30)
                    state['party_levels'] = [levels[i * 0x30] if i * 0x30 < len(levels) else 0 for i in range(6)]
                
                
                if 'party_hp' in self.memory_addresses and party_count > 0:
                    hp_data = self.read_memory(self.memory_addresses['party_hp'], party_count * 2 * 0x30)
                    state['party_hp'] = []
                    for i in range(party_count):
                        if i * 0x30 * 2 + 1 < len(hp_data):
                            hp = (hp_data[i * 0x30 * 2] << 8) | hp_data[i * 0x30 * 2 + 1]
                            state['party_hp'].append(hp)
                        else:
                            state['party_hp'].append(0)
            
           
            if 'x_pos' in self.memory_addresses:
                state['x_pos'] = self.read_memory(self.memory_addresses['x_pos'])
            if 'y_pos' in self.memory_addresses:
                state['y_pos'] = self.read_memory(self.memory_addresses['y_pos'])
            if 'map_id' in self.memory_addresses:
                state['map_id'] = self.read_memory(self.memory_addresses['map_id'])
            
           
            if 'battle_type' in self.memory_addresses:
                state['in_battle'] = self.read_memory(self.memory_addresses['battle_type']) != 0
            
            
            if 'items_count' in self.memory_addresses:
                state['items_count'] = self.read_memory(self.memory_addresses['items_count'])
            
        except Exception as e:
            print(f"Memory read error: {e}")
        
        return state
    
    def _bcd_to_int(self, bcd_bytes):
        """Converts Binary Coded Decimal to integer"""
        result = 0
        for byte in bcd_bytes:
            result = result * 100 + ((byte >> 4) * 10) + (byte & 0x0F)
        return result
    
    def calculate_reward_events(self, current_state):
        """Calculates rewards based on specific game events"""
        reward = 0
        events = []
        
      
        if self.prev_state:
            
            
            
            if current_state.get('badges', 0) > self.prev_state.get('badges', 0):
                reward += 1000
                events.append(f"NEW BADGE! Total: {current_state.get('badges', 0)}")
            
           
            if current_state.get('pokedex_owned', 0) > self.prev_state.get('pokedex_owned', 0):
                new_pokemon = current_state.get('pokedex_owned', 0) - self.prev_state.get('pokedex_owned', 0)
                reward += 100 * new_pokemon
                events.append(f"NEW POKEMON CAUGHT! Total: {current_state.get('pokedex_owned', 0)}")
            
            
            if current_state.get('pokedex_seen', 0) > self.prev_state.get('pokedex_seen', 0):
                new_seen = current_state.get('pokedex_seen', 0) - self.prev_state.get('pokedex_seen', 0)
                reward += 10 * new_seen
                events.append(f"New Pokemon seen! Total: {current_state.get('pokedex_seen', 0)}")
            
          
            current_levels = current_state.get('party_levels', [0] * 6)
            prev_levels = self.prev_state.get('party_levels', [0] * 6)
            for i in range(min(len(current_levels), len(prev_levels))):
                if current_levels[i] > prev_levels[i]:
                    level_diff = current_levels[i] - prev_levels[i]
                    reward += 50 * level_diff
                    events.append(f"Pokemon #{i+1} leveled up to {current_levels[i]}!")
            
           
            money_diff = current_state.get('player_money', 0) - self.prev_state.get('player_money', 0)
            if money_diff > 0:
                reward += min(money_diff / 100, 20)  
                events.append(f"Earned ¥{money_diff}")
            
            
            if current_state.get('map_id', 0) != self.prev_state.get('map_id', 0):
                reward += 30
                events.append(f"New area! Map ID: {current_state.get('map_id', 0)}")
            
            
            x_diff = abs(current_state.get('x_pos', 0) - self.prev_state.get('x_pos', 0))
            y_diff = abs(current_state.get('y_pos', 0) - self.prev_state.get('y_pos', 0))
            if x_diff + y_diff > 5:
                reward += 2
            
          
            
            
            if self.prev_state.get('in_battle', False) and not current_state.get('in_battle', False):
               
                party_hp = current_state.get('party_hp', [])
                prev_party_hp = self.prev_state.get('party_hp', [])
                
                
                if any(hp > 0 for hp in party_hp):
                    
                    all_pokemon_ok = all(hp > 0 for hp in prev_party_hp if hp > 0)
                    if all_pokemon_ok:
                        reward += 50  
                        events.append("BATTLE WON!")
                    else:
                        reward += 20  
                        events.append("Battle won (with losses)")
                else:
                    
                    reward -= 100
                    events.append("DEFEAT! All Pokemon KO!")
            
           
            if not self.prev_state.get('in_battle', False) and current_state.get('in_battle', False):
                reward += 2
                events.append("Battle started!")
            
           
            current_hp = sum(current_state.get('party_hp', []))
            prev_hp = sum(self.prev_state.get('party_hp', []))
            if current_hp > prev_hp + 20:  
                reward += 5
                events.append("Pokemon healed!")
            
            
            if current_state.get('items_count', 0) > self.prev_state.get('items_count', 0):
                reward += 5
                events.append("New item obtained!")
            
            
            
          
            for i in range(min(len(current_state.get('party_hp', [])), len(self.prev_state.get('party_hp', [])))):
                if self.prev_state.get('party_hp', [])[i] > 0 and current_state.get('party_hp', [])[i] == 0:
                    reward -= 30
                    events.append(f"Pokemon #{i+1} defeated!")
            
           
            if money_diff < -100:
                reward -= 20
                events.append(f"Lost ¥{-money_diff}")
            
            
            if self.game_type == 'yellow' and 'pikachu_happiness' in self.memory_addresses:
                happiness = self.read_memory(self.memory_addresses['pikachu_happiness'])
                if happiness > self.prev_state.get('pikachu_happiness', 0):
                    reward += 10
                    events.append(f"Pikachu happier! Happiness: {happiness}")
                current_state['pikachu_happiness'] = happiness
            
         
            if self.game_type == 'crystal' and 'mom_money' in self.memory_addresses:
                mom_money = self.read_memory(self.memory_addresses['mom_money'], 3)
                mom_money_val = self._bcd_to_int(mom_money)
                if mom_money_val > self.prev_state.get('mom_money', 0):
                    reward += 5
                    events.append(f"Mom saved ¥{mom_money_val}")
                current_state['mom_money'] = mom_money_val
            
           
            if self.game_type in ['gs', 'crystal'] and 'kanto_badges' in self.memory_addresses:
                kanto_badges = bin(self.read_memory(self.memory_addresses['kanto_badges'])).count('1')
                if kanto_badges > self.prev_state.get('kanto_badges', 0):
                    reward += 1000
                    events.append(f"KANTO BADGE! Kanto Total: {kanto_badges}")
                current_state['kanto_badges'] = kanto_badges
                    
            
            if 'party_species' in self.memory_addresses and current_state.get('party_count', 0) > 0:
                species = self.read_memory(self.memory_addresses['party_species'], 6)
                prev_species = self.prev_state.get('party_species', [0] * 6)
                
                for i in range(min(len(species), len(prev_species))):
                    if species[i] != prev_species[i] and species[i] > 0 and prev_species[i] > 0:
                        
                        reward += 200
                        events.append(f"POKEMON EVOLVED! Slot #{i+1}")
                
                current_state['party_species'] = species
        
        
        for event in events:
            print(f"  {event}")
        
       
        self.prev_state = current_state.copy()
        
        return reward


class PokemonStateDetector:
    """Detects Pokemon game state"""
    
    def __init__(self):
        pass
        
    def detect_battle(self, screen_array):
        """Detects if we are in battle"""
     
        height, width = screen_array.shape
        
       
        hp_region = screen_array[100:120, 90:150]
        hp_variance = np.var(hp_region)
        
       
        menu_region = screen_array[110:140, 0:80]
        menu_edges = cv2.Canny(menu_region.astype(np.uint8), 50, 150)
        edge_density = np.sum(menu_edges > 0) / menu_edges.size
        
        
        is_battle = hp_variance > 500 and edge_density > 0.1
        
        return is_battle
    
    def detect_menu(self, screen_array):
        """Detects if we are in a menu"""
       
        edges = cv2.Canny(screen_array.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        num_lines = 0 if lines is None else len(lines)
        
        return edge_density > 0.15 and num_lines > 5
    
    def detect_dialogue(self, screen_array):
        """Detects if there is an active dialogue"""
        
        dialogue_region = screen_array[100:140, 10:150]
        
        
        contrast = np.std(dialogue_region)
        
       
        edges = cv2.Canny(dialogue_region.astype(np.uint8), 50, 150)
        has_box = np.sum(edges[0, :]) > 20 and np.sum(edges[-1, :]) > 20
        
        return contrast > 30 and has_box
    
    def detect_blocked_movement(self, current_screen, previous_screen, movement_threshold=0.02):
        """Detects if the player movement is blocked or stuck"""
        if previous_screen is None:
            return False
        
        # Calculate screen difference to detect movement
        diff = np.mean(np.abs(current_screen - previous_screen))
        
        # Movement is considered blocked if screen difference is below threshold
        is_blocked = diff < movement_threshold
        
        # Additional check: look for specific blocked movement patterns
        # Check for repeating screen patterns that indicate collision with walls/objects
        center_region = current_screen[60:100, 70:90]  # Player area
        center_variance = np.var(center_region)
        
        # Very low variance in center region might indicate player is stuck against wall
        stuck_against_wall = center_variance < 50
        
        return is_blocked or stuck_against_wall
    



if torch_available:
    def calculate_conv_output_size(input_height, input_width, conv_layers):
        """
        Calculate the output size after a series of convolutional layers
        
        Args:
            input_height: Initial height (e.g., 144)
            input_width: Initial width (e.g., 160) 
            conv_layers: List of (kernel_size, stride) tuples
        
        Returns:
            (output_height, output_width, linear_input_size_for_last_channels)
        """
        h, w = input_height, input_width
        
        for kernel_size, stride in conv_layers:
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
        
        return h, w
    
    # Pre-calculated common architectures
    STANDARD_CONV_DIMS = calculate_conv_output_size(144, 160, [(8, 4), (4, 2), (3, 1)])  # For Explorer & Battle DQN
    MENU_CONV_DIMS = calculate_conv_output_size(144, 160, [(4, 2), (3, 2)])  # For Menu DQN
    
    class ExplorerDQN(nn.Module):
        """DQN specializzato per esplorazione"""
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
        """DQN specializzato per battaglie"""
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
        """DQN specializzato per navigazione menu"""
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
        """Multi-Agent Architecture per Pokemon"""
        def __init__(self, n_actions, device):
            self.device = device
            self.n_actions = n_actions
            
            # Agenti specializzati
            self.explorer_agent = ExplorerDQN(n_actions).to(device)
            self.battle_agent = BattleDQN(n_actions).to(device)
            self.menu_agent = MenuDQN(n_actions).to(device)
            
            # Target networks
            self.explorer_target = ExplorerDQN(n_actions).to(device)
            self.battle_target = BattleDQN(n_actions).to(device)
            self.menu_target = MenuDQN(n_actions).to(device)
            
            # Ottimizzatori
            self.explorer_optimizer = optim.Adam(self.explorer_agent.parameters(), lr=0.0001)
            self.battle_optimizer = optim.Adam(self.battle_agent.parameters(), lr=0.0003)
            self.menu_optimizer = optim.Adam(self.menu_agent.parameters(), lr=0.0002)
            
            # Inizializza target networks
            self.explorer_target.load_state_dict(self.explorer_agent.state_dict())
            self.battle_target.load_state_dict(self.battle_agent.state_dict())
            self.menu_target.load_state_dict(self.menu_agent.state_dict())
            
        def choose_action(self, state, game_state, epsilon):
            """Sceglie azione basata sul game state"""
            if random.random() < epsilon:
                return random.randint(0, self.n_actions - 1)
            
            with torch.no_grad():
                state_batch = state.unsqueeze(0)
                
                if game_state == "battle":
                    q_values = self.battle_agent(state_batch)
                elif game_state == "menu":
                    q_values = self.menu_agent(state_batch)
                else:  # exploring
                    q_values = self.explorer_agent(state_batch)
                
                return q_values.argmax().item()
        
        def train_agent(self, batch, game_state):
            """Addestra l'agente appropriato"""
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
                    target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
                
                loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
                self.battle_optimizer.zero_grad()
                loss.backward()
                self.battle_optimizer.step()
                
            elif game_state == "menu":
                current_q_values = self.menu_agent(states_t).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = self.menu_target(next_states_t).max(1)[0]
                    target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
                
                loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
                self.menu_optimizer.zero_grad()
                loss.backward()
                self.menu_optimizer.step()
                
            else:  # exploring
                current_q_values = self.explorer_agent(states_t).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = self.explorer_target(next_states_t).max(1)[0]
                    target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
                
                loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
                self.explorer_optimizer.zero_grad()
                loss.backward()
                self.explorer_optimizer.step()
            
            return loss.item()
        
        def update_target_networks(self):
            """Aggiorna le target networks"""
            tau = 0.005
            
            for target_param, param in zip(self.explorer_target.parameters(), self.explorer_agent.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
            for target_param, param in zip(self.battle_target.parameters(), self.battle_agent.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                
            for target_param, param in zip(self.menu_target.parameters(), self.menu_agent.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class PokemonAI:
    """AI specialized for Pokemon"""
    
    def __init__(self, rom_path, headless=False):
        print(f"\nInitializing Pokemon AI for: {os.path.basename(rom_path)}")
        
        
        self.rom_name = os.path.splitext(os.path.basename(rom_path))[0]
        self.save_dir = f"pokemon_ai_saves_{self.rom_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.save_dir, "model.pth")
        self.memory_path = os.path.join(self.save_dir, "memory.pkl")
        self.stats_path = os.path.join(self.save_dir, "stats.json")
        self.checkpoints_path = os.path.join(self.save_dir, "checkpoints.pkl")
        
       
        self.pyboy = pyboy.PyBoy(
            rom_path,
            window="headless" if headless else "SDL2",
            debug=False
        )
        print("PyBoy emulator started!")
        
       
        self.state_detector = PokemonStateDetector()
        self.memory_reader = PokemonMemoryReader(self.pyboy)
        print(f"Game detected: {self.memory_reader.game_type.upper()}")
        
        
        self.actions = [
            [],  
            [WindowEvent.PRESS_ARROW_UP],      
            [WindowEvent.PRESS_ARROW_DOWN],    
            [WindowEvent.PRESS_ARROW_LEFT],    
            [WindowEvent.PRESS_ARROW_RIGHT],   
            [WindowEvent.PRESS_BUTTON_A],      
            [WindowEvent.PRESS_BUTTON_B],      
            [WindowEvent.PRESS_BUTTON_START],  
            [WindowEvent.PRESS_BUTTON_SELECT], 
           
            [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_B],   
            [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_B],  
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B], 
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B], 
            
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_A],    
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_B],     
        ]
        
        self.action_names = ['None', 'Up', 'Down', 'Left', 'Right', 
                            'A', 'B', 'Start', 'Select', 'Run Up', 'Run Down',
                            'Run Left', 'Run Right', 'Double A', 'Double B']
        
   
        self.release_map = {
            WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
        }
        
       
        self.stats = self._load_stats()
        
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = max(0.1, 0.9 * (0.995 ** (self.stats.get('total_frames', 0) / 10000)))
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.00025
        self.memory = deque(maxlen=50000)
        
        # Inizializza device prima di tutto
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch_available:
            self.multi_agent = PokemonMultiAgent(len(self.actions), self.device)
            # Statistiche training per bilanciamento
            self.training_stats = {
                'explorer_count': 0,
                'battle_count': 0,
                'menu_count': 0,
                'explorer_loss': [],
                'battle_loss': [],
                'menu_loss': []
            }
        
        
        
        self.frame_count = 0
        self.episode_count = self.stats.get('episodes', 0)
        self.total_reward = 0
        self.best_reward = self.stats.get('best_reward', float('-inf'))
        
        
        self.battles_won = self.stats.get('battles_won', 0)
        self.pokemon_caught = self.stats.get('pokemon_caught', 0)
        self.badges_earned = self.stats.get('badges_earned', 0)
        self.locations_discovered = set(self.stats.get('locations_discovered', []))
        
       
        self.action_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=100)
        
        
        self.stuck_counter = 0
        self.visited_states = set()
        self.checkpoint_states = {}
        self.last_checkpoint = None
        
        
        self.current_game_state = "exploring"  
        self.last_battle_frame = 0
        
        
        self.last_memory_check = 0
        self.memory_check_interval = 30  
        
       
        self._load_memory()
        self._load_checkpoints()
      
        initial_state = self.memory_reader.get_current_state()
        self.memory_reader.prev_state = initial_state
        
        print(f"Pokemon AI ready! Session #{self.episode_count + 1}")
        print(f"Initial statistics:")
        print(f"  - Total frames: {self.stats.get('total_frames', 0):,}")
        print(f"  - Battles won: {self.battles_won}")
        print(f"  - Pokemon caught: {self.pokemon_caught}")
        print(f"  - Badges: {self.badges_earned}")
        print(f"  - Locations discovered: {len(self.locations_discovered)}")
        if initial_state:
            print(f"  - Money: ¥{initial_state.get('player_money', 0)}")
            print(f"  - Pokemon in party: {initial_state.get('party_count', 0)}")
            print(f"  - Pokedex: {initial_state.get('pokedex_owned', 0)}/{initial_state.get('pokedex_seen', 0)}")
    
    def _has_torch_multiagent(self):
        """Check if torch and multi-agent are available"""
        return torch_available and hasattr(self, 'multi_agent')
    
    def _safe_choose_action(self, state, game_state):
        """Safely choose action with torch availability check"""
        if self._has_torch_multiagent():
            return self.multi_agent.choose_action(state, game_state, self.epsilon)
        else:
            return random.randint(0, len(self.actions) - 1)
    
    def _safe_replay(self):
        """Safely perform replay training with torch availability check"""
        if not self._has_torch_multiagent() or len(self.memory) < self.batch_size:
            return
        return self._do_replay_training()
    
    def _safe_update_target_model(self):
        """Safely update target networks with torch availability check"""
        if self._has_torch_multiagent():
            self.multi_agent.update_target_networks()
    
    def _safe_save_model(self):
        """Safely save model with torch availability check"""
        if self._has_torch_multiagent():
            self._do_model_save()
    
    def _is_torch_tensor(self, tensor):
        """Check if object is a torch tensor"""
        return torch_available and isinstance(tensor, torch.Tensor)
    
    def _safe_tensor_diff(self, screen_tensor, previous_screen):
        """Safely calculate tensor difference"""
        if self._is_torch_tensor(screen_tensor):
            return torch.abs(screen_tensor - previous_screen).mean().item()
        else:
            return np.mean(np.abs(screen_tensor - previous_screen))
    
    def _safe_get_screen_hash(self, screen_tensor):
        """Safely get screen hash handling torch tensors"""
        if self._is_torch_tensor(screen_tensor):
            small = F.interpolate(screen_tensor.unsqueeze(0), size=(36, 40), mode='bilinear')
            screen_bytes = small.cpu().numpy().tobytes()
        else:
            small = cv2.resize(screen_tensor, (40, 36))
            screen_bytes = small.tobytes()
        
        return hashlib.md5(screen_bytes).hexdigest()
    
    def _safe_tensor_to_numpy(self, state, next_state):
        """Safely convert tensors to numpy arrays"""
        if self._is_torch_tensor(state):
            state_np = state.squeeze(0).cpu().numpy()
            next_state_np = next_state.squeeze(0).cpu().numpy()
        else:
            state_np = state
            next_state_np = next_state
        return state_np, next_state_np
    
    def _do_replay_training(self):
        """Perform the actual replay training logic"""
        try:
            # Raggruppa esperienze per game state
            explorer_batch = []
            battle_batch = []
            menu_batch = []
            
            # Calcola bilanciamento per training equo
            total_training = (self.training_stats['explorer_count'] + 
                            self.training_stats['battle_count'] + 
                            self.training_stats['menu_count'])
            
            # Determina priorità per bilanciamento
            min_trained = min(self.training_stats['explorer_count'],
                            self.training_stats['battle_count'], 
                            self.training_stats['menu_count'])
            
            priority_explorer = self.training_stats['explorer_count'] == min_trained
            priority_battle = self.training_stats['battle_count'] == min_trained
            priority_menu = self.training_stats['menu_count'] == min_trained
            
            # More efficient sampling: sample enough for each agent type
            # Estimate we need ~batch_size per agent, so sample 2x to account for distribution
            sample_size = min(self.batch_size * 2, len(self.memory))
            sampled_experiences = random.sample(self.memory, sample_size)
            
            for exp in sampled_experiences:
                if len(exp) >= 6:  # Include game_state
                    game_state = exp[5]
                    # Stop collecting when we have enough for each batch
                    if game_state == "battle" and len(battle_batch) < self.batch_size:
                        battle_batch.append(exp)
                    elif game_state == "menu" and len(menu_batch) < self.batch_size:
                        menu_batch.append(exp)
                    elif len(explorer_batch) < self.batch_size:
                        explorer_batch.append(exp)
                
                # Early termination if all batches are full
                if (len(explorer_batch) >= self.batch_size and 
                    len(battle_batch) >= self.batch_size and 
                    len(menu_batch) >= self.batch_size):
                    break
            
            loss_total = 0
            losses_count = 0
            
            # Configuration for each agent type
            agent_configs = [
                {
                    'name': 'explorer',
                    'batch': explorer_batch,
                    'agent_type': 'exploring',
                    'priority': priority_explorer,
                    'count_key': 'explorer_count',
                    'loss_key': 'explorer_loss'
                },
                {
                    'name': 'battle',
                    'batch': battle_batch,
                    'agent_type': 'battle',
                    'priority': priority_battle,
                    'count_key': 'battle_count',
                    'loss_key': 'battle_loss'
                },
                {
                    'name': 'menu',
                    'batch': menu_batch,
                    'agent_type': 'menu',
                    'priority': priority_menu,
                    'count_key': 'menu_count',
                    'loss_key': 'menu_loss'
                }
            ]
            
            # Train agents with priority balancing - priority agents first
            for priority_first in [True, False]:
                for config in agent_configs:
                    should_train = (config['priority'] if priority_first else not config['priority'])
                    
                    if should_train and len(config['batch']) >= 4:
                        loss = self.multi_agent.train_agent(
                            config['batch'][:self.batch_size], 
                            config['agent_type']
                        )
                        
                        if loss:
                            loss_total += loss
                            losses_count += 1
                            self.training_stats[config['count_key']] += 1
                            self.training_stats[config['loss_key']].append(loss)
                            
                            # Keep only last 100 losses
                            if len(self.training_stats[config['loss_key']]) > 100:
                                self.training_stats[config['loss_key']].pop(0)
            
            if losses_count > 0:
                avg_loss = loss_total / losses_count
                self.loss_history.append(avg_loss)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            print(f"Multi-agent training error: {e}")
    
    def _get_screen_tensor(self):
        """Gets screen as PyTorch tensor"""
        screen = self.pyboy.screen.image
        gray = np.array(screen.convert('L'))
        
        self.last_screen_array = gray.copy()
        normalized = gray.astype(np.float32) / 255.0
        
        if self._has_torch_multiagent():
            tensor = torch.from_numpy(normalized).unsqueeze(0)
            return tensor.to(self.device)
        return normalized
    
    def _detect_game_state(self, screen_array):
        """Detects current game state"""
        old_state = self.current_game_state
        
        if self.state_detector.detect_battle(screen_array):
            self.current_game_state = "battle"
            if old_state != "battle":
                print("BATTLE STARTED!")
                self.last_battle_frame = self.frame_count
        elif self.state_detector.detect_dialogue(screen_array):
            self.current_game_state = "dialogue"
        elif self.state_detector.detect_menu(screen_array):
            self.current_game_state = "menu"
        else:
            self.current_game_state = "exploring"
            
        return self.current_game_state
    
    def _calculate_reward(self, screen_tensor, previous_screen, action):
        """Advanced reward system for Pokemon gameplay"""
        reward = 0
        
        if hasattr(self, 'last_screen_array'):
            game_state = self._detect_game_state(self.last_screen_array)
        else:
            game_state = "exploring"
        
        memory_state = None
        if self.frame_count - self.last_memory_check >= self.memory_check_interval:
            self.last_memory_check = self.frame_count
            memory_state = self.memory_reader.get_current_state()
            memory_reward = self.memory_reader.calculate_reward_events(memory_state)
            reward += memory_reward
            
            self.badges_earned = memory_state.get('badges', 0)
            self.pokemon_caught = memory_state.get('pokedex_owned', 0)
        
        if previous_screen is not None:
            diff = self._safe_tensor_diff(screen_tensor, previous_screen)
            
            if diff > 0.02:
                reward += 1.0
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
                reward -= 0.1
        
        
        self.reward_history.append(reward)
        self.total_reward += reward
        
        return reward
    
    
    # =====================================================
    # SAVE/LOAD SYSTEM - Persistence and Data Management
    # =====================================================
    
    # --- Utility Methods ---
    def _safe_pickle_save(self, data, path, description="data"):
        """Safely save data with pickle"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"{description} saved!")
        except Exception as e:
            print(f"{description} save error: {e}")
    
    def _safe_pickle_load(self, path, description="data"):
        """Safely load data with pickle"""
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"{description} load error: {e}")
            return None
    
    # --- Model Save/Load Methods ---
    def _do_model_save(self):
        """Perform the actual model saving logic"""
        try:
            checkpoint = {
                'explorer_state': self.multi_agent.explorer_agent.state_dict(),
                'battle_state': self.multi_agent.battle_agent.state_dict(),
                'menu_state': self.multi_agent.menu_agent.state_dict(),
                'epsilon': self.epsilon,
                'episode': self.episode_count,
                'frame': self.frame_count,
                'battles_won': self.battles_won,
                'pokemon_caught': self.pokemon_caught,
                'badges_earned': self.badges_earned
            }
            torch.save(checkpoint, self.model_path)
            
            self._safe_pickle_save(self.checkpoint_states, self.checkpoints_path, "Checkpoints")
                
            print("Multi-agent models saved!")
        except Exception as e:
            print(f"Save error: {e}")
    
    def _get_screen_hash(self, screen_tensor):
        """Calculates screen hash for tracking visited states"""
        return self._safe_get_screen_hash(screen_tensor)
    
    def choose_action(self, state):
        """Multi-Agent action selection"""
        game_state = self.current_game_state if hasattr(self, 'current_game_state') else "exploring"
        
        # MULTI-AGENT ARCHITECTURE
        action = self._safe_choose_action(state, game_state)
        
        self.action_history.append(action)
        return action
    
    
    def remember(self, state, action, reward, next_state, done):
        """Saves experience for multi-agent learning"""
        state_np, next_state_np = self._safe_tensor_to_numpy(state, next_state)
        
        # Aggiungi game_state all'esperienza
        game_state = self.current_game_state if hasattr(self, 'current_game_state') else "exploring"
        experience = (state_np, action, reward, next_state_np, done, game_state)
        self.memory.append(experience)
    
    def replay(self):
        """Multi-agent replay training"""
        self._safe_replay()
    
    def update_target_model(self):
        """Update target networks for multi-agent"""
        self._safe_update_target_model()
    
    
    # --- Checkpoint Save/Load Methods ---
    def _save_checkpoint(self, state):
        """Saves checkpoint for backtracking"""
        checkpoint_id = len(self.checkpoint_states)
        self.checkpoint_states[checkpoint_id] = {
            'state': state,
            'frame': self.frame_count,
            'reward': self.total_reward,
            'position': len(self.visited_states)
        }
        self.last_checkpoint = checkpoint_id
        print(f"Checkpoint #{checkpoint_id} saved!")
    
    def _load_checkpoints(self):
        """Loads saved checkpoints"""
        data = self._safe_pickle_load(self.checkpoints_path, "Checkpoints")
        self.checkpoint_states = data if data is not None else {}
        if self.checkpoint_states:
            print(f"Loaded {len(self.checkpoint_states)} checkpoints")
    
    # --- Game State Save/Load Methods ---
    def save_game_state(self, slot=0):
        """Saves current game state"""
        state_path = os.path.join(self.save_dir, f"savestate_{slot}.state")
        with open(state_path, "wb") as f:
            f.write(self.pyboy.save_state())
        print(f"Savestate saved in slot {slot}")
    
    def load_game_state(self, slot=0):
        """Loads a saved game state"""
        state_path = os.path.join(self.save_dir, f"savestate_{slot}.state")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)
            print(f"Savestate loaded from slot {slot}")
    
    def _save_model(self):
        """Saves multi-agent models"""
        self._safe_save_model()
    
    # --- Statistics and Memory Save/Load Methods ---
    def _load_stats(self):
        """Loads Pokemon statistics"""
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        return {
            'episodes': 0,
            'total_frames': 0,
            'best_reward': float('-inf'),
            'battles_won': 0,
            'pokemon_caught': 0,
            'badges_earned': 0,
            'locations_discovered': []
        }
    
    def _save_stats(self):
        """Saves Pokemon statistics with memory data"""
        
        if hasattr(self, 'memory_reader'):
            final_state = self.memory_reader.get_current_state()
            
           
            if final_state:
                self.badges_earned = final_state.get('badges', self.badges_earned)
                self.pokemon_caught = final_state.get('pokedex_owned', self.pokemon_caught)
                
               
                self.stats['final_state'] = {
                    'player_money': final_state.get('player_money', 0),
                    'pokedex_seen': final_state.get('pokedex_seen', 0),
                    'party_levels': final_state.get('party_levels', []),
                    'last_map_id': final_state.get('map_id', 0),
                    'items_count': final_state.get('items_count', 0)
                }
        
        self.stats.update({
            'episodes': self.episode_count,
            'total_frames': self.stats.get('total_frames', 0) + self.frame_count,
            'best_reward': max(self.best_reward, self.total_reward),
            'battles_won': self.battles_won,
            'pokemon_caught': self.pokemon_caught,
            'badges_earned': self.badges_earned,
            'locations_discovered': list(self.locations_discovered)
        })
        
      
        if len(self.reward_history) > 0:
            self.stats['avg_reward_last_1000'] = float(np.mean(list(self.reward_history)))
        
        with open(self.stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _save_memory(self):
        """Saves experience memory"""
        memory_data = {'memory': list(self.memory)[-10000:]}
        self._safe_pickle_save(memory_data, self.memory_path, f"Memory ({len(memory_data['memory'])} experiences)")
    
    def _load_memory(self):
        """Loads experience memory"""
        memory_data = self._safe_pickle_load(self.memory_path, "Memory")
        if memory_data is None:
            return
            
        for exp in memory_data.get('memory', [])[-5000:]:
            if len(exp) == 5 and isinstance(exp[0], np.ndarray):
                self.memory.append(exp)
        
        print(f"Memory loaded: {len(self.memory)} experiences")
    
    def play(self):
        """Main loop optimized for Pokemon"""
        print("\nPOKEMON AI STARTED!")
        print("Commands: ESC=Exit, SPACE=Pause, S=Save, R=Report\n")
        
        paused = False
        previous_screen = None
        last_save_frame = 0
        
        try:
            while True:
                
                if keyboard.is_pressed('escape'):
                    print("\nSaving and exiting...")
                    break
                
                if keyboard.is_pressed('space'):
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")
                    time.sleep(0.3)
                
                if keyboard.is_pressed('s'):
                    self._save_all()
                    time.sleep(0.3)
                
                if keyboard.is_pressed('r'):
                    self._print_report()
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
                
                
                if self.frame_count % 4 == 0 and len(self.memory) >= self.batch_size:
                    self.replay()
                
                if self.frame_count % 100 == 0:
                    self.update_target_model()
                
               
                if self.frame_count - last_save_frame >= 10000:
                    self._save_all()
                    last_save_frame = self.frame_count
                
                
                if self.frame_count % 1000 == 0:
                    self._print_stats()
                
                
                previous_screen = next_state
                self.frame_count += 1
                
                
                if self.frame_count % 100000 == 0:
                    self._new_episode()
                    
        except KeyboardInterrupt:
            print("\nInterrupting...")
        
        finally:
            self._save_all()
            self.pyboy.stop()
            self._print_final_report()
    
    # --- Comprehensive Save Method ---
    def _save_all(self):
        """Saves everything"""
        print("Complete save...")
        self._save_model()
        self._save_memory()
        self._save_stats()
        print("Save completed!")
    
    def _print_stats(self):
        """Prints multi-agent statistics"""
        avg_reward = np.mean(list(self.reward_history)[-100:]) if len(self.reward_history) >= 100 else 0
        avg_loss = np.mean(list(self.loss_history)) if len(self.loss_history) > 0 else 0
        
        memory_state = self.memory_reader.get_current_state()
        
        print(f"\nFrame {self.frame_count:,} | State: {self.current_game_state}")
        print(f"  Total reward: {self.total_reward:.2f} (avg: {avg_reward:.3f})")
        print(f"  Loss: {avg_loss:.4f} | ε: {self.epsilon:.3f}")
        
        # MULTI-AGENT STATUS
        if self._has_torch_multiagent() and hasattr(self, 'training_stats'):
            print(f"  MULTI-AGENT: 3 agenti specializzati (Explorer/Battle/Menu)")
            print(f"  Agente attivo: {self.current_game_state.upper()}")
            
            # Statistiche training bilanciamento
            total_training = (self.training_stats['explorer_count'] + 
                            self.training_stats['battle_count'] + 
                            self.training_stats['menu_count'])
            
            if total_training > 0:
                exp_pct = self.training_stats['explorer_count'] / total_training * 100
                bat_pct = self.training_stats['battle_count'] / total_training * 100
                men_pct = self.training_stats['menu_count'] / total_training * 100
                
                print(f"  Training bilanciamento:")
                print(f"    Explorer: {self.training_stats['explorer_count']} ({exp_pct:.1f}%)")
                print(f"    Battle: {self.training_stats['battle_count']} ({bat_pct:.1f}%)")
                print(f"    Menu: {self.training_stats['menu_count']} ({men_pct:.1f}%)")
                
                # Loss averages
                if self.training_stats['explorer_loss']:
                    exp_loss = np.mean(self.training_stats['explorer_loss'][-20:])
                    print(f"    Explorer Loss: {exp_loss:.4f}")
                if self.training_stats['battle_loss']:
                    bat_loss = np.mean(self.training_stats['battle_loss'][-20:])
                    print(f"    Battle Loss: {bat_loss:.4f}")
                if self.training_stats['menu_loss']:
                    men_loss = np.mean(self.training_stats['menu_loss'][-20:])
                    print(f"    Menu Loss: {men_loss:.4f}")
        
        if memory_state:
            print(f"  Badges: {memory_state.get('badges', 0)}/8 | Pokemon: {memory_state.get('pokedex_owned', 0)}")
        
        if self.total_reward > self.best_reward:
            self.best_reward = self.total_reward
            print(f"  NEW RECORD!")
    
    def _print_report(self):
        """Detailed report"""
        print("\n" + "="*60)
        print("POKEMON AI REPORT")
        print("="*60)
        print(f"Episode: {self.episode_count}")
        print(f"Total frames: {self.stats.get('total_frames', 0) + self.frame_count:,}")
        print(f"Locations explored: {len(self.visited_states)}")
        print(f"Checkpoints: {len(self.checkpoint_states)}")
        print(f"Epsilon: {self.epsilon:.3f}")
        
       
        if len(self.action_history) > 0:
            from collections import Counter
            action_counts = Counter(self.action_history)
            print("\nMost frequent actions:")
            for action, count in action_counts.most_common(5):
                print(f"   {self.action_names[action]}: {count} ({count/len(self.action_history)*100:.1f}%)")
    
    def _new_episode(self):
        """Starts new episode"""
        self.episode_count += 1
        print(f"\nNEW EPISODE #{self.episode_count}")
        print(f"  Episode reward: {self.total_reward:.2f}")
        print(f"  States visited: {len(self.visited_states)}")
        
        self.frame_count = 0
        self.total_reward = 0
        self.stuck_counter = 0
       
        if len(self.visited_states) > 10000:
            
            self.visited_states = set(list(self.visited_states)[-5000:])
    
    def _print_final_report(self):
        """Detailed final report with memory data"""
        
        final_state = self.memory_reader.get_current_state() if hasattr(self, 'memory_reader') else {}
        
        print(f"\n{'='*60}")
        print(f"FINAL POKEMON AI REPORT")
        print(f"{'='*60}")
        print(f"Completed episodes: {self.episode_count}")
        print(f"Total frames: {self.stats.get('total_frames', 0):,}")
        print(f"Best score: {self.best_reward:.2f}")
        
        print(f"\nGAME PROGRESS:")
        if final_state:
            print(f"  Final money: ¥{final_state.get('player_money', 0)}")
            print(f"  Badges earned: {final_state.get('badges', 0)}/8")
            print(f"  Pokedex: {final_state.get('pokedex_owned', 0)} caught, {final_state.get('pokedex_seen', 0)} seen")
            print(f"  Pokemon in party: {final_state.get('party_count', 0)}")
            
           
            levels = final_state.get('party_levels', [])
            if any(levels):
                print(f"  Final team levels: {[l for l in levels if l > 0]}")
            
            print(f"  Total items: {final_state.get('items_count', 0)}")
            print(f"  Last position: Map {final_state.get('map_id', 0)}, X:{final_state.get('x_pos', 0)}, Y:{final_state.get('y_pos', 0)}")
        
        print(f"\nMULTI-AGENT ARCHITECTURE:")
        if self._has_torch_multiagent():
            print(f"  3 agenti specializzati:")
            print(f"    - ExplorerDQN: Esplorazione")
            print(f"    - BattleDQN: Combattimenti") 
            print(f"    - MenuDQN: Navigazione menu")
        
        print(f"\nAI STATISTICS:")
        print(f"  Experiences in memory: {len(self.memory)}")
        print(f"  Final epsilon: {self.epsilon:.3f}")
        
        print(f"\nSaved files:")
        print(f"  - Model: {self.model_path}")
        print(f"  - Statistics: {self.stats_path}")
        print(f"  - Memory: {self.memory_path}")
        print(f"  - Checkpoints: {self.checkpoints_path}")


def main():
    """Main function"""
    print("="*60)
    print("POKEMON AI - DEEP LEARNING AGENT")
    print("="*60)
    print("\nAI specialized for playing Pokemon")
    print("Deep Q-Network with Dueling Architecture")
    print("Game state recognition")
    print("Pokemon-specific reward system\n")
    
    # ROM Selection
    while True:
        rom_path = input("Pokemon ROM path (.gbc): ").strip().strip('"')
        
        if os.path.exists(rom_path) and rom_path.lower().endswith('.gbc'):
            if os.path.getsize(rom_path) > 0:
                break
            else:
                print("Error: ROM file is empty. Select a valid ROM file.")
        else:
            print("Invalid file! Make sure it's a Pokemon .gbc file")
    
    print(f"\nROM loaded: {os.path.basename(rom_path)}")
    
   
    rom_name = os.path.splitext(os.path.basename(rom_path))[0]
    save_dir = f"pokemon_ai_saves_{rom_name}"
    
    if os.path.exists(save_dir):
        print(f"\nFound existing saves!")
        stats_path = os.path.join(save_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print(f"  - Episodes: {stats.get('episodes', 0)}")
            print(f"  - Total frames: {stats.get('total_frames', 0):,}")
            print(f"  - Battles won: {stats.get('battles_won', 0)}")
            print(f"  - Pokemon caught: {stats.get('pokemon_caught', 0)}")
            
            reset = input("\nDo you want to RESET everything? (y/N): ").lower().strip()
            if reset == 'y':
                import shutil
                shutil.rmtree(save_dir)
                print("Reset completed!")
    
   
    headless = input("\nHeadless mode (no window)? (y/N): ").lower().strip() == 'y'
    
    print("\nStarting Pokemon AI...")
    time.sleep(2)
    
    try:
        ai = PokemonAI(rom_path, headless=headless)
        ai.play()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress ENTER to exit...")


if __name__ == "__main__":
    main()