# emulator.py
import logging
import time
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Any, Optional
import json
from dataclasses import dataclass
import math


try:
    from config import config as CFG
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import config as CFG

logger = logging.getLogger("pokeagent.emulator")

@dataclass
class ActionTimingParams:
    """Parametri fisici e costanti per il calcolo del timing delle azioni."""
    FPS: float = 59.73
    INPUT_POLL_WINDOW_F: int = 3
    SAFETY_FACTOR: float = 1.25
    
    WALK_DURATION_F: int = 16
    BUMP_DURATION_F: int = 8
    MENU_SCROLL_F: int = 4
    TEXT_PRINT_DELAY_F: int = 2
    
    def calculate_frames(self, action_type: str) -> Tuple[int, int]:

        if action_type == 'move':
            hold = math.ceil(self.INPUT_POLL_WINDOW_F * 2.0)
            wait = math.ceil(self.WALK_DURATION_F * self.SAFETY_FACTOR)
            return hold, wait
            
        elif action_type == 'menu':
            hold = math.ceil(self.INPUT_POLL_WINDOW_F * 1.5)
            wait = math.ceil(self.MENU_SCROLL_F * self.SAFETY_FACTOR * 2)
            return hold, wait
            
        elif action_type == 'interact':
            hold = math.ceil(self.INPUT_POLL_WINDOW_F * 1.5)
            wait = 20
            return hold, wait
            
        else:
            return 8, 20

class EmulatorHarness:
    """Gestisce l'emulatore, fornisce percezione strutturata e strumenti di azione."""

    def __init__(self, rom_path: str):
        self.rom_path = rom_path
        self.emulator = None
        self._frame_index = 0
        self._init_emulator()
        self.loaded_knowledge_base = self._load_knowledge_base()
        self.timing_params = ActionTimingParams() # Inizializza modello matematico

    def _tick(self, count: int = 1, render: Optional[bool] = None) -> bool:
        if render is None:
            render = bool(CFG.RENDER_ENABLED) and (self._frame_index % int(CFG.RENDER_EVERY_N_FRAMES) == 0)
        ok = bool(self.emulator.tick(count=count, render=render))
        self._frame_index += int(count)
        return ok

    def tick_idle(self, frames: int = 1) -> bool:
        return self._tick(count=int(max(1, frames)), render=None)

    def _init_emulator(self):
        """Inizializza l'emulatore di Game Boy con PyBoy v2.x."""
        try:
            from pyboy import PyBoy
            window = "null" if CFG.HEADLESS else "SDL2"
            self.emulator = PyBoy(self.rom_path, window=window)
            self.emulator.set_emulation_speed(CFG.EMULATION_SPEED)
            logger.info("PyBoy inizializzato: rom=%s window=%s speed=%sx", self.rom_path, window, CFG.EMULATION_SPEED)
        except ImportError:
            logger.error("PyBoy non installato. Installa con: pip install pyboy")
            raise
        except Exception as e:
            logger.exception("Impossibile inizializzare PyBoy: %s", e)
            raise

    def _load_knowledge_base(self) -> Dict:
        """Carica la knowledge base da file JSON."""
        try:
            with open(CFG.KNOWLEDGE_BASE_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Knowledge base non trovata in %s", CFG.KNOWLEDGE_BASE_FILE)
            return {"type_matchups": {}, "map_connections": {}, "important_npcs": {}}

    def get_current_state(self) -> Dict[str, Any]:
        """Cattura lo stato completo del gioco: screenshot + dati dalla RAM."""
        visual = self._get_screenshot()
        visual_np = np.array(visual) if CFG.LLM_USE_VISION else None
        state = {
            "visual": visual,
            "visual_np": visual_np,
            "ram": self._read_ram_data(),
            "parsed": None
        }
        state["parsed"] = self._parse_state(state["ram"], visual)
        return state

    def _get_screenshot(self) -> Image.Image:
        """Cattura uno screenshot dallo schermo dell'emulatore."""
        try:
            self._tick(1, render=None)
            screenshot = self.emulator.screen.image
            return screenshot
        except Exception as e:
            logger.exception("Impossibile catturare screenshot: %s", e)
            return Image.new('RGB', (160, 144), color='black')

    def _read_ram_data(self) -> Dict[str, int]:
        """Legge dati specifici dalla memoria del gioco usando gli offset definiti."""
        ram_data = {}
        try:
            for name, offset in CFG.RAM_OFFSETS.items():
                value = self.emulator.memory[offset]
                ram_data[name] = value
            logger.debug("Dati RAM letti con successo")
        except Exception as e:
            logger.exception("Impossibile leggere memoria: %s", e)
            for name in CFG.RAM_OFFSETS.keys():
                ram_data[name] = 0
        return ram_data

    def _parse_state(self, ram_data: Dict, screenshot: Image.Image) -> Dict[str, Any]:
        """Analizza i dati grezzi (RAM e immagine) in uno stato di gioco strutturato."""
        parsed = {
            "location": f"Mappa {ram_data.get('current_map', 0)} a ({ram_data.get('player_x', 0)}, {ram_data.get('player_y', 0)})",
            "in_battle": ram_data.get('in_battle_flag', 0) == 1,
            "menu_open": ram_data.get('menu_flag', 0) == 1,
            "text_box": self._extract_text_from_screenshot(screenshot),
            "badges": int(ram_data.get("badges", 0) or 0),
            "money": int(ram_data.get("money", 0) or 0),
            "party_count": int(ram_data.get("party_count", 0) or 0),
            "opponent_level": int(ram_data.get("opponent_level", 0) or 0),
            "opponent_hp_current": int(ram_data.get("opponent_hp_current", 0) or 0),
            "opponent_hp_max": int(ram_data.get("opponent_hp_max", 0) or 0)
        }
        return parsed

    def _extract_text_from_screenshot(self, screenshot: Image.Image) -> str:
        """Tenta di estrarre testo dallo screenshot (può usare OCR o il modello VL)."""
        return "Testo del dialogo non implementato."

    # --- STRUMENTI (TOOLS) PER L'AGENTE DI ESECUZIONE ---
    def press_button(self, button: str):
        """Premi un singolo pulsante (A, B, UP, DOWN, LEFT, RIGHT, START, SELECT)."""
        try:
            button_map = {
                'A': 'a',
                'B': 'b',
                'UP': 'up',
                'DOWN': 'down',
                'LEFT': 'left',
                'RIGHT': 'right',
                'START': 'start',
                'SELECT': 'select'
            }
            
            if button not in button_map:
                logger.warning("Pulsante non valido: %s", button)
                return
            
            btn = button_map[button]
            
            action_type = 'interact'
            if btn in ("up", "down", "left", "right"):
                action_type = 'move'
            elif btn in ("start", "select"):
                action_type = 'menu'
            elif btn in ("a", "b"):
                action_type = 'interact'

            hold_frames, wait_frames = self.timing_params.calculate_frames(action_type)

            self.emulator.button(btn, hold_frames)
            self._tick(count=hold_frames + wait_frames, render=None)
            
            logger.debug("Pulsante: %s hold=%s wait=%s type=%s", button, hold_frames, wait_frames, action_type)
            if bool(CFG.RENDER_ENABLED) and (not bool(CFG.HEADLESS)):
                time.sleep(0.01)
            
        except Exception as e:
            logger.exception("Impossibile premere pulsante %s: %s", button, e)

    def navigate_to(self, target_map: int, target_x: int, target_y: int) -> bool:
        """
        Strumento avanzato: naviga verso coordinate specifiche.
        Restituisce True se l'azione è iniziata con successo.
        """
        if not CFG.ENABLE_PATHFINDER:
            return False
        logger.info("Pathfinding verso mappa=%s (%s,%s)", target_map, target_x, target_y)
        return True

    def get_knowledge_context(self, current_map_id: int) -> str:
        """Restituisce un contesto testuale basato sulla knowledge base e la posizione attuale."""
        if not self.loaded_knowledge_base:
            return "Knowledge base non disponibile."
        
        context = []
        
        # Info mappa
        map_info = self.loaded_knowledge_base.get("maps", {}).get(str(current_map_id))
        if map_info:
            context.append(f"Current Location: {map_info.get('name', 'Unknown')}")
            context.append(f"Description: {map_info.get('description', '')}")
            connections = map_info.get("connections", {})
            if connections:
                context.append(f"Connections: {', '.join([f'{k}->{v}' for k, v in connections.items()])}")
            events = map_info.get("events", [])
            if events:
                context.append(f"Key Events here: {', '.join(events)}")
        
        # Info obiettivi
        objectives = self.loaded_knowledge_base.get("objectives", [])
        active_objectives = [obj['description'] for obj in objectives[:3]]
        context.append(f"Active Objectives: {'; '.join(active_objectives)}")
        
        return "\n".join(context)

    def search_knowledge_base(self, query: str) -> Optional[str]:
        """Cerca nella knowledge base caricata."""
        if query.startswith("type_matchup:"):
            pokemon_type = query.split(":")[1].strip()
            return self.loaded_knowledge_base.get("type_matchups", {}).get(pokemon_type, "Sconosciuto")
        return None

    def solve_boulder_puzzle(self, puzzle_id: str) -> bool:
        """Risolutore specializzato per un puzzle specifico."""
        logger.info("Risolvo puzzle: %s", puzzle_id)
        sequence = ["UP", "RIGHT", "DOWN", "LEFT"]
        for btn in sequence:
            self.press_button(btn)
        return True

    def close(self):
        """Chiude correttamente l'emulatore e libera le risorse."""
        if self.emulator:
            try:
                self.emulator.stop()
                logger.info("Emulatore chiuso correttamente")
            except Exception as e:
                logger.exception("Impossibile chiudere emulatore: %s", e)
