import logging
import base64
import io
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from queue import Queue, Empty

import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "qwen3-vl:2b"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 5.0
    min_interval_ms: int = 500
    max_calls_per_minute: int = 30
    cache_size: int = 128
    cache_ttl_seconds: int = 60
    use_vision: bool = True
    use_for_exploration: bool = True
    use_for_battle: bool = True
    use_for_menu: bool = False
    fallback_on_error: bool = True


class LLMRateLimiter:
    def __init__(self, min_interval_ms: int = 500, max_per_minute: int = 30):
        self.min_interval_ms = min_interval_ms
        self.max_per_minute = max_per_minute
        self.last_call_time = 0.0
        self.call_times: deque = deque(maxlen=max_per_minute)
        self._lock = threading.Lock()

    def can_call(self) -> bool:
        with self._lock:
            now = time.time()
            if (now - self.last_call_time) * 1000 < self.min_interval_ms:
                return False
            recent_calls = sum(1 for t in self.call_times if t > now - 60)
            return recent_calls < self.max_per_minute

    def record_call(self) -> None:
        with self._lock:
            now = time.time()
            self.last_call_time = now
            self.call_times.append(now)


class ResponseCache:
    def __init__(self, max_size: int = 128, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, game_state: str, context_hash: str) -> Optional[Dict]:
        with self._lock:
            key = f"{game_state}:{context_hash}"
            if key not in self.cache:
                return None
            response, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                return None
            return response

    def set(self, game_state: str, context_hash: str, response: Dict) -> None:
        with self._lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            key = f"{game_state}:{context_hash}"
            self.cache[key] = (response, time.time())


class AsyncLLMWorker:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.request_queue: Queue = Queue(maxsize=1)
        self.latest_response: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def submit_request(self, prompt: str, image_b64: Optional[str] = None) -> None:
        try:
            self.request_queue.get_nowait()
        except Empty:
            pass
        try:
            self.request_queue.put_nowait((prompt, image_b64))
        except Exception:
            pass

    def get_latest_response(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.latest_response

    def _worker_loop(self):
        while self._running:
            try:
                prompt, image_b64 = self.request_queue.get(timeout=0.1)
                response = self._make_request(prompt, image_b64)
                if response:
                    with self._lock:
                        self.latest_response = response
            except Empty:
                continue
            except Exception as e:
                logger.debug(f"[LLM] Worker error: {e}")

    def _make_request(self, prompt: str, image_b64: Optional[str]) -> Optional[Dict]:
        try:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }
            if image_b64:
                payload["images"] = [image_b64]

            response = requests.post(
                f"{self.config.host}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            if response.status_code != 200:
                return None

            result = response.json()
            return {
                'raw_response': result.get('response', '').strip(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.debug(f"[LLM] Request failed: {e}")
            return None


class OllamaLLMClient:
    ACTION_MAP = {
        'UP': 'up', 'DOWN': 'down', 'LEFT': 'left', 'RIGHT': 'right',
        'NORTH': 'up', 'SOUTH': 'down', 'WEST': 'left', 'EAST': 'right',
        'ATTACK': 'a', 'CONFIRM': 'a', 'INTERACT': 'a', 'YES': 'a', 'SELECT': 'a',
        'CANCEL': 'b', 'RUN': 'b', 'BACK': 'b', 'NO': 'b', 'FLEE': 'b',
        'SWITCH': 'right', 'ITEM': 'down', 'MENU': 'start', 'START': 'start',
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.rate_limiter = LLMRateLimiter(
            min_interval_ms=self.config.min_interval_ms,
            max_per_minute=self.config.max_calls_per_minute
        )
        self.cache = ResponseCache(
            max_size=self.config.cache_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        self.available = self._check_availability()
        self.stats = {'calls': 0, 'cache_hits': 0, 'errors': 0, 'total_latency_ms': 0, 'async_requests': 0}
        self._async_worker = AsyncLLMWorker(self.config)
        if self.available:
            self._async_worker.start()
        self._last_parsed_response: Optional[Dict[str, Any]] = None
        self._last_game_state: str = ""
        self._last_response_time: float = 0

    def _check_availability(self) -> bool:
        if not REQUESTS_AVAILABLE or not self.config.enabled:
            return False
        try:
            response = requests.get(f"{self.config.host}/api/tags", timeout=2.0)
            if response.status_code != 200:
                return False
            models = response.json().get('models', [])
            model_base = self.config.model.split(':')[0]
            return any(model_base in m.get('name', '') for m in models)
        except Exception:
            return False

    def _encode_image(self, screen_array: np.ndarray) -> Optional[str]:
        if not PIL_AVAILABLE or not self.config.use_vision:
            return None
        try:
            mode = 'L' if len(screen_array.shape) == 2 else None
            img = Image.fromarray(screen_array, mode=mode)
            img = img.resize((160, 144), Image.Resampling.NEAREST)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception:
            return None

    def _build_prompt(self, game_state: str, memory_state: Dict, screen_array: Optional[np.ndarray] = None, action_history: Optional[list] = None) -> Tuple[str, Optional[str]]:
        badges = memory_state.get('badges', 0)
        pokemon_caught = memory_state.get('pokedex_caught', 0)
        map_id = memory_state.get('map_id', 0)
        pos_x = memory_state.get('pos_x', 0)
        pos_y = memory_state.get('pos_y', 0)
        in_battle = memory_state.get('in_battle', False)
        team_levels = [lv for lv in memory_state.get('team_levels', []) if lv > 0]
        hp_team = memory_state.get('hp_team', [])
        hp_max = memory_state.get('hp_max_team', [])
        enemy_hp = memory_state.get('enemy_hp', 0)

        team_hp_pct = int((sum(hp_team) / max(sum(hp_max), 1)) * 100)
        avg_lv = int(np.mean(team_levels)) if team_levels else 0

        recent_actions = ""
        if action_history and len(action_history) >= 5:
            recent = action_history[-5:]
            recent_actions = f"\nLast 5 actions: {recent}"
            if len(set(recent)) <= 2:
                recent_actions += " [WARNING: You are repeating actions! Try something different]"

        if in_battle or game_state == "battle":
            prompt = f"""Pokemon Battle. Look at the screen.
Team: {len(team_levels)} pokemon, avg Lv{avg_lv}, HP:{team_hp_pct}%
Enemy HP: {enemy_hp}%{recent_actions}

Choose the BEST action. Reply with ONLY one word:
- ATTACK (select fight/first move)
- SWITCH (change pokemon)
- ITEM (use potion/ball)
- RUN (flee from wild pokemon)

Your answer:"""
        elif game_state == "exploring":
            prompt = f"""Pokemon exploration. Look at the screen.
Badges: {badges}/8 | Pokemon: {pokemon_caught} | Map:{map_id} | Pos:({pos_x},{pos_y}){recent_actions}

You control the player. Choose direction to PROGRESS in the game.
Reply with ONLY one word:
- UP/DOWN/LEFT/RIGHT (move)
- INTERACT (talk to NPC, read sign, enter door)

Your answer:"""
        else:
            prompt = f"""Pokemon menu/dialogue. Look at the screen.{recent_actions}

Navigate the menu. Reply with ONLY one word:
- CONFIRM (press A, select option)
- CANCEL (press B, go back)
- UP/DOWN (navigate menu)

Your answer:"""

        image_b64 = self._encode_image(screen_array) if screen_array is not None else None
        return prompt, image_b64

    def get_action(self, game_state: str, memory_state: Dict, screen_array: Optional[np.ndarray] = None, action_history: Optional[list] = None) -> Optional[int]:
        """
        Get action directly from LLM. Returns action index or None if unavailable.
        This is the PRIMARY decision maker - RL network is fallback.
        """
        if not self.available:
            return None

        if not self.rate_limiter.can_call():
            if self._last_parsed_response and self._last_parsed_response.get('suggested_action'):
                return self._action_to_index(self._last_parsed_response['suggested_action'])
            return None

        prompt, image_b64 = self._build_prompt(game_state, memory_state, screen_array, action_history)
        self._async_worker.submit_request(prompt, image_b64)
        self.rate_limiter.record_call()
        self.stats['async_requests'] += 1
        self._last_game_state = game_state

        worker_response = self._async_worker.get_latest_response()
        if worker_response and worker_response.get('raw_response'):
            response_time = worker_response.get('timestamp', 0)
            if response_time > self._last_response_time:
                self._last_response_time = response_time
                parsed = self._parse_response(worker_response['raw_response'], game_state)
                self._last_parsed_response = parsed
                self.stats['calls'] += 1

                action = parsed.get('suggested_action')
                if action:
                    return self._action_to_index(action)

        if self._last_parsed_response and self._last_parsed_response.get('suggested_action'):
            return self._action_to_index(self._last_parsed_response['suggested_action'])

        return None

    def _action_to_index(self, action_name: str) -> Optional[int]:
        """Convert action name to index for game input."""
        mapping = {
            None: 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4,
            'a': 5, 'b': 6, 'start': 7, 'select': 8
        }
        return mapping.get(action_name)

    def get_action_bias(self, game_state: str, memory_state: Dict, screen_array: Optional[np.ndarray] = None, force: bool = False) -> Optional[Dict[str, Any]]:
        """Legacy method for soft bias. Use get_action() for direct control."""
        if not self.available:
            return None
        return self._last_parsed_response

    def _parse_response(self, raw: str, game_state: str) -> Dict[str, Any]:
        result = {'suggested_action': None, 'confidence': 0.5, 'reasoning': raw[:100] if raw else "", 'action_weights': None}
        if not raw:
            return result

        words = raw.upper().split()
        if not words:
            return result

        first_word = words[0].strip('.,!?:')
        suggested = self.ACTION_MAP.get(first_word)
        if suggested:
            result['suggested_action'] = suggested
            result['confidence'] = 0.7
            if game_state == "exploring" and suggested in ['up', 'down', 'left', 'right']:
                weights = {'up': 1.0, 'down': 1.0, 'left': 1.0, 'right': 1.0, 'a': 1.0}
                weights[suggested] = 2.0
                result['action_weights'] = weights

        return result

    def apply_action_bias(self, action_probs: np.ndarray, actions: List[Optional[str]], llm_response: Optional[Dict]) -> np.ndarray:
        if not llm_response:
            return action_probs

        suggested = llm_response.get('suggested_action')
        if not suggested:
            return action_probs

        try:
            action_idx = actions.index(suggested)
            biased = action_probs.copy()
            biased[action_idx] *= 1.0 + llm_response.get('confidence', 0.5)
            return biased / biased.sum()
        except ValueError:
            return action_probs

    def get_subgoal_analysis(self, prompt: str, screen_array: Optional[np.ndarray] = None) -> str:
        if not self.available:
            return ""
        try:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens}
            }
            if screen_array is not None and self.config.use_vision:
                image_b64 = self._encode_image(screen_array)
                if image_b64:
                    payload["images"] = [image_b64]

            response = requests.post(f"{self.config.host}/api/generate", json=payload, timeout=self.config.timeout)
            if response.status_code == 200:
                self.stats['calls'] += 1
                return response.json().get('response', '').strip()
            return ""
        except Exception:
            return ""

    def get_stats(self) -> Dict[str, Any]:
        total_calls = max(self.stats['calls'] + self.stats['cache_hits'], 1)
        return {
            **self.stats,
            'avg_latency_ms': self.stats['total_latency_ms'] / max(self.stats['calls'], 1),
            'cache_hit_rate': (self.stats['cache_hits'] / total_calls) * 100,
            'available': self.available,
            'model': self.config.model
        }

    def shutdown(self):
        self._async_worker.stop()


_llm_client: Optional[OllamaLLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> OllamaLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = OllamaLLMClient(config)
    return _llm_client


def reset_llm_client() -> None:
    global _llm_client
    if _llm_client:
        _llm_client.shutdown()
    _llm_client = None
