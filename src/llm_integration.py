"""
LLM Integration for Pokemon AI Agent.

Uses Ollama for:
- Strategic decision-making (exploration direction, battle tactics)
- Action bias based on game context reasoning
- Optional: Screenshot analysis (multimodal)

Architecture:
- Async-friendly with background thread for non-blocking calls
- Caching to reduce inference calls
- Graceful degradation if LLM unavailable
"""

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
    """Configuration for LLM integration."""
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "qwen3-vl:2b"

    # Inference settings
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 5.0  # seconds

    # Rate limiting
    min_interval_ms: int = 500  # Minimum ms between LLM calls
    max_calls_per_minute: int = 30

    # Caching
    cache_size: int = 128
    cache_ttl_seconds: int = 60

    # Feature flags
    use_vision: bool = True  # Enable screenshot analysis
    use_for_exploration: bool = True
    use_for_battle: bool = True
    use_for_menu: bool = False  # Menu is fast, LLM adds latency

    # Fallback
    fallback_on_error: bool = True  # Continue without LLM if error


class LLMRateLimiter:
    """Rate limiter for LLM API calls."""

    def __init__(self, min_interval_ms: int = 500, max_per_minute: int = 30):
        self.min_interval_ms = min_interval_ms
        self.max_per_minute = max_per_minute
        self.last_call_time = 0.0
        self.call_times: deque = deque(maxlen=max_per_minute)
        self._lock = threading.Lock()

    def can_call(self) -> bool:
        """Check if we can make an LLM call."""
        with self._lock:
            now = time.time()

            # Check minimum interval
            if (now - self.last_call_time) * 1000 < self.min_interval_ms:
                return False

            # Check calls per minute
            one_minute_ago = now - 60
            recent_calls = sum(1 for t in self.call_times if t > one_minute_ago)
            if recent_calls >= self.max_per_minute:
                return False

            return True

    def record_call(self) -> None:
        """Record that a call was made."""
        with self._lock:
            now = time.time()
            self.last_call_time = now
            self.call_times.append(now)


class ResponseCache:
    """Thread-safe TTL cache for LLM responses."""

    def __init__(self, max_size: int = 128, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def _make_key(self, game_state: str, context_hash: str) -> str:
        return f"{game_state}:{context_hash}"

    def get(self, game_state: str, context_hash: str) -> Optional[Dict]:
        """Get cached response if valid."""
        with self._lock:
            key = self._make_key(game_state, context_hash)
            if key not in self.cache:
                return None

            response, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                return None

            return response

    def set(self, game_state: str, context_hash: str, response: Dict) -> None:
        """Cache a response."""
        with self._lock:
            # Evict old entries if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            key = self._make_key(game_state, context_hash)
            self.cache[key] = (response, time.time())


class AsyncLLMWorker:
    """Background worker for non-blocking LLM calls."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.request_queue: Queue = Queue(maxsize=1)  # Only keep latest request
        self.latest_response: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.debug("[LLM] Async worker started")

    def stop(self):
        """Stop the background worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def submit_request(self, prompt: str, image_b64: Optional[str] = None) -> None:
        """Submit a request to be processed in background (non-blocking)."""
        # Clear old request if queue is full
        try:
            self.request_queue.get_nowait()
        except Empty:
            pass

        try:
            self.request_queue.put_nowait((prompt, image_b64))
        except:
            pass  # Queue full, skip this request

    def get_latest_response(self) -> Optional[Dict[str, Any]]:
        """Get the most recent LLM response (non-blocking)."""
        with self._lock:
            return self.latest_response

    def _worker_loop(self):
        """Background worker loop."""
        while self._running:
            try:
                # Wait for request with timeout
                try:
                    prompt, image_b64 = self.request_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Make the API call
                response = self._make_request(prompt, image_b64)
                if response:
                    with self._lock:
                        self.latest_response = response

            except Exception as e:
                logger.debug(f"[LLM] Worker error: {e}")

    def _make_request(self, prompt: str, image_b64: Optional[str]) -> Optional[Dict]:
        """Make the actual HTTP request to Ollama."""
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

            if image_b64 is not None:
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
    """
    Client for Ollama LLM integration with async support.

    Provides strategic reasoning for the Pokemon AI agent.
    Uses background thread to avoid blocking the game loop.
    """

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
        self.stats = {
            'calls': 0,
            'cache_hits': 0,
            'errors': 0,
            'total_latency_ms': 0,
            'async_requests': 0
        }

        # Async worker for non-blocking calls
        self._async_worker = AsyncLLMWorker(self.config)
        if self.available:
            self._async_worker.start()

        # Store last parsed response
        self._last_parsed_response: Optional[Dict[str, Any]] = None
        self._last_game_state: str = ""

    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        if not REQUESTS_AVAILABLE:
            logger.warning("[LLM] requests library not installed")
            return False

        if not self.config.enabled:
            logger.info("[LLM] Disabled by config")
            return False

        try:
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=2.0
            )
            if response.status_code != 200:
                logger.warning(f"[LLM] Ollama not responding: {response.status_code}")
                return False

            # Check if model exists
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]

            # Check for exact match or base name match
            model_base = self.config.model.split(':')[0]
            if not any(model_base in name for name in model_names):
                logger.warning(f"[LLM] Model {self.config.model} not found. "
                              f"Available: {model_names}")
                logger.info(f"[LLM] Run: ollama pull {self.config.model}")
                return False

            logger.info(f"[LLM] Ollama ready with {self.config.model}")
            return True

        except requests.exceptions.ConnectionError:
            logger.warning("[LLM] Ollama not running. Start with: ollama serve")
            return False
        except Exception as e:
            logger.warning(f"[LLM] Availability check failed: {e}")
            return False

    def _encode_image(self, screen_array: np.ndarray) -> Optional[str]:
        """Encode screenshot to base64 for vision model."""
        if not PIL_AVAILABLE or not self.config.use_vision:
            return None

        try:
            # Convert grayscale to RGB if needed
            if len(screen_array.shape) == 2:
                img = Image.fromarray(screen_array, mode='L')
            else:
                img = Image.fromarray(screen_array)

            # Resize for efficiency (Game Boy is 160x144)
            img = img.resize((160, 144), Image.Resampling.NEAREST)

            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.debug(f"[LLM] Image encoding failed: {e}")
            return None

    def _build_prompt(
        self,
        game_state: str,
        memory_state: Dict,
        screen_array: Optional[np.ndarray] = None
    ) -> Tuple[str, Optional[str]]:
        """Build prompt based on game state and context."""

        # Extract key info from memory
        badges = memory_state.get('badges', 0)
        pokemon_caught = memory_state.get('pokedex_caught', 0)
        pokemon_seen = memory_state.get('pokedex_seen', 0)
        map_id = memory_state.get('map_id', 0)
        pos_x = memory_state.get('pos_x', 0)
        pos_y = memory_state.get('pos_y', 0)
        in_battle = memory_state.get('in_battle', False)
        team_levels = memory_state.get('team_levels', [])
        hp_team = memory_state.get('hp_team', [])
        hp_max = memory_state.get('hp_max_team', [])

        # Calculate team status
        active_pokemon = sum(1 for lv in team_levels if lv > 0)
        avg_level = np.mean([lv for lv in team_levels if lv > 0]) if active_pokemon > 0 else 0
        team_hp_percent = (sum(hp_team) / max(sum(hp_max), 1)) * 100

        base_context = f"""Pokemon Red - AI Agent Status:
- Badges: {badges}/8
- Pokedex: {pokemon_caught} caught, {pokemon_seen} seen
- Team: {active_pokemon} Pokemon, avg level {avg_level:.0f}
- Team HP: {team_hp_percent:.0f}%
- Location: Map {map_id}, Position ({pos_x}, {pos_y})
"""

        if game_state == "battle" or in_battle:
            prompt = f"""{base_context}
BATTLE MODE - Quick tactical decision needed.

Given the current battle, suggest the best action:
- ATTACK: Press A to attack
- SWITCH: Navigate to switch Pokemon
- ITEM: Navigate to use item
- RUN: Try to flee (wild only)

Respond with ONE word: ATTACK, SWITCH, ITEM, or RUN
Then briefly explain why (max 10 words)."""

        elif game_state == "exploring":
            prompt = f"""{base_context}
EXPLORATION MODE - Suggest direction to explore.

Consider:
- Unexplored areas are valuable
- Need to find gyms for badges
- Wild Pokemon for training
- Avoid backtracking

Suggest direction: UP, DOWN, LEFT, RIGHT, or INTERACT (press A)
Respond with ONE word then brief reason (max 10 words)."""

        else:  # menu or dialogue
            prompt = f"""{base_context}
MENU/DIALOGUE MODE - Quick navigation.

Suggest: CONFIRM (A), CANCEL (B), or NAVIGATE (arrow)
ONE word response."""

        # Encode image if available
        image_b64 = None
        if screen_array is not None and self.config.use_vision:
            image_b64 = self._encode_image(screen_array)

        return prompt, image_b64

    def _compute_context_hash(self, game_state: str, memory_state: Dict) -> str:
        """Compute hash for caching based on relevant state."""
        # Only cache based on slowly-changing state
        key_parts = [
            game_state,
            str(memory_state.get('map_id', 0)),
            str(memory_state.get('badges', 0)),
            str(memory_state.get('in_battle', False)),
        ]
        return ':'.join(key_parts)

    def get_action_bias(
        self,
        game_state: str,
        memory_state: Dict,
        screen_array: Optional[np.ndarray] = None,
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get LLM suggestion for action bias (NON-BLOCKING).

        Returns the latest available response without waiting for LLM.
        Submits new requests to background worker when rate limit allows.

        Returns dict with:
        - 'suggested_action': str (e.g., 'up', 'a', 'b')
        - 'confidence': float 0-1
        - 'reasoning': str
        - 'action_weights': Optional[Dict[str, float]] for soft bias

        Returns None if LLM unavailable or no response ready.
        """
        if not self.available:
            return None

        # Check feature flags
        if game_state == "exploring" and not self.config.use_for_exploration:
            return None
        if game_state == "battle" and not self.config.use_for_battle:
            return None
        if game_state == "menu" and not self.config.use_for_menu:
            return None

        # Check cache first
        context_hash = self._compute_context_hash(game_state, memory_state)
        cached = self.cache.get(game_state, context_hash)
        if cached is not None:
            self.stats['cache_hits'] += 1
            return cached

        # Submit new request if rate limit allows (non-blocking)
        if force or self.rate_limiter.can_call():
            prompt, image_b64 = self._build_prompt(game_state, memory_state, screen_array)
            self._async_worker.submit_request(prompt, image_b64)
            self.rate_limiter.record_call()
            self.stats['async_requests'] += 1
            self._last_game_state = game_state

        # Check for new response from worker (non-blocking)
        worker_response = self._async_worker.get_latest_response()
        if worker_response and worker_response.get('raw_response'):
            # Check if this is a new response
            response_time = worker_response.get('timestamp', 0)
            last_time = getattr(self, '_last_response_time', 0)

            if response_time > last_time:
                self._last_response_time = response_time
                raw_response = worker_response['raw_response']
                parsed = self._parse_response(raw_response, self._last_game_state or game_state)
                self._last_parsed_response = parsed
                self.stats['calls'] += 1

                # Cache the result
                self.cache.set(game_state, context_hash, parsed)

                return parsed

        # Return last known response if available
        return self._last_parsed_response

    def _parse_response(self, raw: str, game_state: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        # Default response
        result = {
            'suggested_action': None,
            'confidence': 0.5,
            'reasoning': raw[:100] if raw else "No response",
            'action_weights': None
        }

        if not raw:
            return result

        # Extract first word as action
        words = raw.upper().split()
        if not words:
            return result

        first_word = words[0].strip('.,!?:')

        # Map LLM suggestions to game actions
        action_map = {
            # Directions
            'UP': 'up',
            'DOWN': 'down',
            'LEFT': 'left',
            'RIGHT': 'right',
            'NORTH': 'up',
            'SOUTH': 'down',
            'WEST': 'left',
            'EAST': 'right',
            # Confirmations
            'ATTACK': 'a',
            'CONFIRM': 'a',
            'INTERACT': 'a',
            'YES': 'a',
            'SELECT': 'a',
            # Cancellations
            'CANCEL': 'b',
            'RUN': 'b',
            'BACK': 'b',
            'NO': 'b',
            'FLEE': 'b',
            # Special
            'SWITCH': 'right',  # Navigate to switch menu
            'ITEM': 'down',     # Navigate to items
            'NAVIGATE': None,   # No specific action
            'MENU': 'start',
            'START': 'start',
        }

        suggested = action_map.get(first_word)
        if suggested:
            result['suggested_action'] = suggested
            result['confidence'] = 0.7

        # Extract reasoning (everything after first word)
        if len(words) > 1:
            result['reasoning'] = ' '.join(words[1:])[:100]

        # Build soft action weights for exploration
        if game_state == "exploring" and suggested in ['up', 'down', 'left', 'right']:
            weights = {'up': 1.0, 'down': 1.0, 'left': 1.0, 'right': 1.0, 'a': 1.0}
            weights[suggested] = 2.0  # Boost suggested direction
            result['action_weights'] = weights

        return result

    def apply_action_bias(
        self,
        action_probs: np.ndarray,
        actions: List[Optional[str]],
        llm_response: Optional[Dict]
    ) -> np.ndarray:
        """
        Apply LLM suggestion as soft bias to action probabilities.

        Args:
            action_probs: Original action probabilities from policy network
            actions: List of action names (None, 'up', 'down', etc.)
            llm_response: Response from get_action_bias()

        Returns:
            Biased action probabilities (still sums to 1)
        """
        if llm_response is None:
            return action_probs

        suggested = llm_response.get('suggested_action')
        confidence = llm_response.get('confidence', 0.5)

        if suggested is None:
            return action_probs

        # Find index of suggested action
        try:
            action_idx = actions.index(suggested)
        except ValueError:
            return action_probs

        # Apply soft bias (blend with original)
        biased = action_probs.copy()

        # Boost suggested action by confidence
        boost_factor = 1.0 + confidence
        biased[action_idx] *= boost_factor

        # Renormalize
        biased = biased / biased.sum()

        return biased

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        avg_latency = (
            self.stats['total_latency_ms'] / max(self.stats['calls'], 1)
        )
        cache_rate = (
            self.stats['cache_hits'] /
            max(self.stats['calls'] + self.stats['cache_hits'], 1)
        ) * 100

        return {
            **self.stats,
            'avg_latency_ms': avg_latency,
            'cache_hit_rate': cache_rate,
            'available': self.available,
            'model': self.config.model
        }

    def shutdown(self):
        """Shutdown the async worker."""
        self._async_worker.stop()


# Singleton instance for easy import
_llm_client: Optional[OllamaLLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> OllamaLLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OllamaLLMClient(config)
    return _llm_client


def reset_llm_client() -> None:
    """Reset the LLM client (useful for testing)."""
    global _llm_client
    if _llm_client:
        _llm_client.shutdown()
    _llm_client = None
