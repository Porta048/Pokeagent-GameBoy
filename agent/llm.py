import time
import requests
import json
import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import base64
import numpy as np
from io import BytesIO

@dataclass
class LLMConfig:
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:0.5b"
    temperature: float = 0.3
    timeout: float = 60.0
    min_interval_ms: int = 2000
    max_calls_per_minute: int = 30
    cache_ttl_seconds: int = 20
    use_vision: bool = True
    use_token_bucket: bool = True
    use_fast_cache_key: bool = True
    use_for_exploration: bool = True
    use_for_battle: bool = True
    use_for_menu: bool = False
    fallback_on_error: bool = True
    retry_attempts: int = 2
    consecutive_failure_threshold: int = 5
    failure_cooldown_seconds: int = 60

class OllamaLLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.available = self.config.enabled
        self.last_call_time = 0.0
        self.call_count = 0
        self.reset_time = time.monotonic()
        self.cache: Dict[str, Any] = {}
        self.consecutive_failures = 0
        self.failure_cooldown_until = 0.0
        self._bucket_capacity = float(max(1, self.config.max_calls_per_minute))
        self._bucket_tokens = self._bucket_capacity
        self._bucket_refill_rate = self._bucket_capacity / 60.0
        self._bucket_last_refill = time.monotonic()
        self._degraded = False
        self._latency_ema_s = 0.0
        self._last_latency_s = 0.0

    def _enter_cooldown(self, error_kind: str) -> None:
        self.available = False
        self._degraded = True
        self.cache.clear()
        current_time = time.time()
        self.failure_cooldown_until = current_time + float(self.config.failure_cooldown_seconds)
        print(
            f"[LLM] Too many consecutive {error_kind} ({self.consecutive_failures}), "
            f"disabling for {self.config.failure_cooldown_seconds} seconds until {time.ctime(self.failure_cooldown_until)}"
        )

    def _request_timeout_arg(self, is_action: bool = False) -> Tuple[float, float]:
        try:
            total_timeout_s = float(self.config.timeout)
        except Exception:
            total_timeout_s = 60.0
        
        connect_timeout_s = 2.0 if not is_action else 1.5
        default_read = 30.0 if is_action else 10.0
        read_timeout_s = min(default_read, max(1.0, total_timeout_s))
        
        return (connect_timeout_s, read_timeout_s)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "consecutive_failures": self.consecutive_failures,
            "call_count": self.call_count,
            "degraded": self._degraded,
            "last_latency_s": self._last_latency_s,
            "latency_ema_s": self._latency_ema_s
        }

    def reset_state(self) -> None:
        self.consecutive_failures = 0
        self.failure_cooldown_until = 0.0
        self._degraded = False
        self._latency_ema_s = 0.0
        self.available = True
        self._degraded = False
        print("[LLM] Stato resettato - forzando utilizzo dell'intelligenza")
        print(f"[LLM] Config enabled: {self.config.enabled}, Available: {self.available}")

    def _cache_key(self, game_state: str, game_context: Dict, action_history: List[str]) -> str:
        try:
            payload = {
                "game_state": game_state,
                "game_context": game_context,
                "action_history": action_history
            }
            raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            raw = repr((game_state, action_history))
        if self.config.use_fast_cache_key:
            return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()
        return raw

    def _check_rate_limit(self) -> bool:
        if not self.config.use_token_bucket:
            now = time.monotonic()
            if (now - self.reset_time) > 60.0:
                self.reset_time = now
                self.call_count = 0

            if self.call_count >= self.config.max_calls_per_minute:
                return False

            min_interval_s = max(0.0, float(self.config.min_interval_ms) / 1000.0)
            since_last = now - self.last_call_time
            if since_last < min_interval_s:
                time.sleep(min_interval_s - since_last)
            return True

        now = time.monotonic()
        elapsed = now - self._bucket_last_refill
        if elapsed > 0:
            self._bucket_tokens = min(self._bucket_capacity, self._bucket_tokens + elapsed * self._bucket_refill_rate)
            self._bucket_last_refill = now

        min_interval_s = max(0.0, float(self.config.min_interval_ms) / 1000.0)
        since_last = now - self.last_call_time
        if since_last < min_interval_s:
            time.sleep(min_interval_s - since_last)
            now = time.monotonic()

        if self._bucket_tokens < 1.0:
            return False

        self._bucket_tokens -= 1.0
        return True

    def _check_availability(self) -> bool:
        current_time = time.time()

        if self.consecutive_failures >= 3:
            print(f"[LLM] Forcing availability reset (consecutive_failures: {self.consecutive_failures})")
            self.consecutive_failures = 0
            self.available = True
            self._degraded = False
            self.failure_cooldown_until = 0

        if self.failure_cooldown_until > 0 and current_time >= self.failure_cooldown_until:
            self.consecutive_failures = 0
            self.available = self.config.enabled
            self.failure_cooldown_until = 0
            print(f"[LLM] Cooldown expired, re-enabling LLM")

        return self.available and self.config.enabled

    def _get_system_prompt(self, game_state: str) -> str:
        if game_state == "battle":
            return "You are a highly intelligent AI assisting a Pokémon trainer in a battle. Your goal is to choose the best action to win the battle. Consider Pokémon types, moves, and current HP."
        elif game_state == "exploring":
            return "You are an AI guiding a Pokémon trainer through the world. Your goal is to explore, find new Pokémon, battle trainers, and progress through the story. Avoid getting stuck in loops."
        elif game_state == "menu":
            return "You are an AI assisting a Pokémon trainer navigate menus. Your goal is to perform requested actions within the menu system."
        return "You are a highly intelligent AI assisting a Pokémon trainer in their adventure."

    def _format_game_context(self, game_context: Dict) -> str:
        parts = ["Current Game State:"]
        for key, value in game_context.items():
            parts.append(f"- {key}: {value}")
        return "\n".join(parts) + "\n"

    def _encode_image(self, image_array: np.ndarray) -> str:
        from PIL import Image
        pil_image = Image.fromarray(image_array)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if not self._check_availability():
            return None
        if not self._check_rate_limit():
            print("[LLM] Rate limit exceeded, skipping text generation.")
            return None
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False
        }
        timeout_arg = self._request_timeout_arg()
        for attempt in range(self.config.retry_attempts + 1):
            try:
                t0 = time.monotonic()
                response = requests.post(f"{self.config.host}/api/chat", json=payload, timeout=timeout_arg)
                response.raise_for_status()
                data = response.json()
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"].strip()
                elif "response" in data:
                    content = data["response"].strip()
                else:
                    print(f"[LLM] Unexpected response format during text generation: {data}")
                    continue
                self.consecutive_failures = 0
                self._last_latency_s = time.monotonic() - t0
                self._latency_ema_s = self._latency_ema_s * 0.9 + self._last_latency_s * 0.1
                self.last_call_time = time.monotonic()
                self.call_count += 1
                return content
            except requests.exceptions.ReadTimeout as e:
                print(f"[LLM] Text generation timeout: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("text timeouts")
                break
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                print(f"[LLM] Text generation error on attempt {attempt+1}/{self.config.retry_attempts + 1}: {e}")

                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("text errors")
                    break
                if attempt < self.config.retry_attempts:
                    time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[LLM] Unexpected text generation error: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("text errors")
                    break
                break
        return None

    def _construct_prompt(self, game_state: str, game_context: Dict, screen_image: Optional[np.ndarray], action_history: List[str]) -> List[Dict]:
        prompt_content = self._format_game_context(game_context) + \
             f"\nAction History (last 20): {', '.join(action_history)}" + \
             "\n\nTask: Analyze the situation and decide the next move." + \
             "\nProvide your response in JSON format with two fields:" + \
             "\n1. 'thought': A brief reasoning of why you chose this action." + \
             "\n2. 'action': The action name from [None, 'up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']." + \
             "\n\nExample response: {\"thought\": \"I need to move up to reach the door.\", \"action\": \"up\"}"
        
        messages = [
            {"role": "system", "content": self._get_system_prompt(game_state) + "\nYou are a Pokemon Red player. Always think before you act."},
            {"role": "user", "content": prompt_content}
        ]

        return messages

    def get_action(self, game_state: str, game_context: Dict, screen_image: Optional[np.ndarray], action_history: List[str]) -> Optional[int]:
        if not self._check_availability():
            return None

        if not self.config.use_for_exploration and game_state == "exploring":
            return None
        if not self.config.use_for_battle and game_state == "battle":
            return None
        if not self.config.use_for_menu and game_state == "menu":
            return None

        cache_key = self._cache_key(game_state, game_context, action_history)
        stale_cached_response: Optional[int] = None
        now_mono = time.monotonic()
        if cache_key in self.cache:
            cached_response, cache_time = self.cache[cache_key]
            if (now_mono - cache_time) < float(self.config.cache_ttl_seconds):
                self.consecutive_failures = 0
                return cached_response
            stale_cached_response = cached_response

        if not self._check_rate_limit():
            print("[LLM] Rate limit exceeded, falling back to RL.")
            return None

        messages = self._construct_prompt(game_state, game_context, screen_image, action_history)

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False
        }

        if self._latency_ema_s >= 2.5:
            self._degraded = True
        if self.config.use_vision and not self._degraded and screen_image is not None:
            encoded_image = self._encode_image(screen_image)
            payload["images"] = [encoded_image]

        timeout_arg = self._request_timeout_arg(is_action=True)

        for attempt in range(1):
            try:
                t0 = time.monotonic()
                response = requests.post(f"{self.config.host}/api/chat", json=payload, timeout=timeout_arg)
                response.raise_for_status()
                response_data = response.json()

                self.consecutive_failures = 0
                self._last_latency_s = time.monotonic() - t0
                self._latency_ema_s = self._latency_ema_s * 0.9 + self._last_latency_s * 0.1

                if 'message' in response_data and 'content' in response_data['message']:
                    content = response_data['message']['content'].strip()
                elif 'response' in response_data:
                    content = response_data['response'].strip()
                else:
                    print(f"[LLM] Unexpected response format: {response_data}")
                    continue

                try:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)
                        
                        thought = data.get('thought', 'No thought provided')
                        action_str = data.get('action', '').lower()
                        
                        print(f"[LLM Reasoning] {thought}")
                        print(f"[LLM Decision] Action: {action_str}")
                        
                        action_map = {
                            'none': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4, 
                            'a': 5, 'b': 6, 'start': 7, 'select': 8
                        }
                        
                        if action_str in action_map:
                            action_index = action_map[action_str]
                            self.cache[cache_key] = (action_index, time.monotonic())
                            self.last_call_time = time.monotonic()
                            self.call_count += 1
                            return action_index
                        else:
                            print(f"[LLM] Invalid action string: {action_str}")
                    else:
                        print(f"[LLM] No JSON found in response: {content}")
                        
                except Exception as e:
                    print(f"[LLM] Error parsing JSON response: {e}, Content: {content}")

                numbers = re.findall(r'\d+', content)
                if numbers:
                    print(f"[LLM] Fallback parsing used (found numbers)")
                    action_index = int(numbers[0])
                    if 0 <= action_index <= 8:
                        return action_index
                else:
                    print(f"[LLM] No valid action index found in response: {content}")
                    content_lower = content.lower()
                    if 'up' in content_lower:
                        action_index = 1
                    elif 'down' in content_lower:
                        action_index = 2
                    elif 'left' in content_lower:
                        action_index = 3
                    elif 'right' in content_lower:
                        action_index = 4
                    elif 'a' in content_lower:
                        action_index = 5
                    elif 'b' in content_lower:
                        action_index = 6
                    elif 'start' in content_lower:
                        action_index = 7
                    elif 'select' in content_lower:
                        action_index = 8
                    else:
                        return None

                    if 0 <= action_index <= 8:
                        self.cache[cache_key] = (action_index, time.monotonic())
                        self.last_call_time = time.monotonic()
                        self.call_count += 1
                        return action_index

                return None
            except requests.exceptions.ReadTimeout as e:
                print(f"[LLM] Timeout in get_action: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("timeouts")
                if stale_cached_response is not None:
                    return stale_cached_response
                break
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                print(f"[LLM] Error on attempt {attempt+1}/{self.config.retry_attempts + 1}: {e}")

                self.consecutive_failures += 1

                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("errors")
                    break
            except Exception as e:
                print(f"[LLM] Unexpected error: {e}")

                self.consecutive_failures += 1

                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("errors")
                    break
                break

        return None
