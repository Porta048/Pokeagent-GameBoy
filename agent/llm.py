import logging
import time
import requests
import json
import re
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import base64
import numpy as np
from io import BytesIO

logger = logging.getLogger("pokeagent.llm")

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
    # Nuove configurazioni per performance
    enable_smart_caching: bool = True
    cache_similarity_threshold: float = 0.85
    adaptive_timeout: bool = True
    fast_model_fallback: str = "qwen2.5:0.5b"
    enable_response_compression: bool = True
    max_response_length: int = 150

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
        self._rate_limit_lock = threading.Lock()
        self._state_lock = threading.Lock()
        
        # Statistiche per performance monitoring
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'fallbacks_used': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        # Setup logging dettagliato se abilitato
        if hasattr(config, 'enable_detailed_logging') and config.enable_detailed_logging:
            logger.setLevel(logging.DEBUG)
            
        logger.info("OllamaLLMClient inizializzato con modello: %s", config.model)

    def _enter_cooldown(self, error_kind: str) -> None:
        self.available = False
        self._degraded = True
        self.cache.clear()
        current_time = time.time()
        self.failure_cooldown_until = current_time + float(self.config.failure_cooldown_seconds)
        logger.warning(
            "Troppi %s consecutivi (%s), disabilito per %ss fino a %s",
            error_kind,
            self.consecutive_failures,
            self.config.failure_cooldown_seconds,
            time.ctime(self.failure_cooldown_until),
        )

    def _request_timeout_arg(self, is_action: bool = False) -> Tuple[float, float]:
        """Timeout adattivo basato su performance recenti"""
        try:
            base_timeout = float(self.config.timeout)
        except Exception:
            base_timeout = 60.0
        
        # Timeout adattivo basato su latenza media
        if self.config.adaptive_timeout and self._latency_ema_s > 0:
            # Se la latenza media è alta, usiamo timeout più lungo ma meno retry
            if self._latency_ema_s > 4.0:
                total_timeout_s = min(base_timeout, self._latency_ema_s * 1.5)
            else:
                total_timeout_s = min(base_timeout, max(3.0, self._latency_ema_s * 2.0))
        else:
            total_timeout_s = base_timeout
            
        # Timeout più aggressivi per azioni rapide
        if is_action and self.config.adaptive_timeout:
            if self._latency_ema_s > 0 and self._latency_ema_s < 2.0:
                total_timeout_s = min(total_timeout_s, 4.0)  # Timeout breve per risposte veloci
            else:
                total_timeout_s = min(total_timeout_s, 6.0)  # Timeout moderato

        read_timeout_s = max(1.0, total_timeout_s)
        connect_timeout_s = min((1.0 if is_action else 1.5), read_timeout_s * 0.3)

        return (connect_timeout_s, read_timeout_s)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "consecutive_failures": self.consecutive_failures,
            "call_count": self.call_count,
            "degraded": self._degraded,
            "last_latency_s": self._last_latency_s,
            "latency_ema_s": self._latency_ema_s,
            "performance_stats": self._stats.copy(),
            "cache_size": len(self.cache),
            "cache_hit_rate": self._stats['cache_hits'] / max(1, self._stats['total_requests']),
            "success_rate": self._stats['successful_requests'] / max(1, self._stats['total_requests'])
        }

    def reset_state(self) -> None:
        self.consecutive_failures = 0
        self.failure_cooldown_until = 0.0
        self._degraded = False
        self._latency_ema_s = 0.0
        self._last_latency_s = 0.0
        self.available = bool(self.config.enabled)
        logger.debug("LLM reset_state: enabled=%s available=%s", self.config.enabled, self.available)

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

    def _find_similar_cache_entry(self, game_state: str, game_context: Dict, action_history: List[str]) -> Optional[int]:
        """Trova voci in cache simili basate su similarità di contesto"""
        if not self.config.enable_smart_caching or not self.cache:
            return None
            
        # Estraiamo feature chiave per similarità
        current_features = {
            'state': game_state,
            'key_context': {k: v for k, v in game_context.items() if k in ['location', 'battle_state', 'menu_type']},
            'recent_actions': action_history[-3:] if action_history else []
        }
        
        best_similarity = 0.0
        best_action = None
        
        for cache_key, (cached_action, cache_time) in self.cache.items():
            try:
                # Decodifica la chiave cache per confronto
                if self.config.use_fast_cache_key:
                    # Per chiavi hashate, usiamo un confronto approssimato
                    similarity = self._calculate_context_similarity(current_features, cache_key)
                else:
                    # Per chiavi in chiaro, confronto diretto
                    similarity = self._calculate_direct_similarity(current_features, cache_key)
                
                if similarity > best_similarity and similarity >= self.config.cache_similarity_threshold:
                    best_similarity = similarity
                    best_action = cached_action
                    
            except Exception as e:
                logger.debug("Errore calcolo similarità cache: %s", e)
                continue
                
        if best_action is not None:
            logger.debug("Cache hit con similarità %.2f per stato %s", best_similarity, game_state)
            
        return best_action
    
    def _calculate_context_similarity(self, current_features: Dict, cache_key: str) -> float:
        """Calcola similarità approssimata tra contesti"""
        # Implementazione semplificata - può essere migliorata
        key_str = str(cache_key)
        similarity = 0.0
        
        # Controlla se lo stato di gioco è lo stesso
        if current_features['state'] in key_str:
            similarity += 0.4
            
        # Controlla feature chiave del contesto
        for key, value in current_features['key_context'].items():
            if str(value) in key_str:
                similarity += 0.2
                
        # Controlla azioni recenti
        for action in current_features['recent_actions']:
            if action in key_str:
                similarity += 0.1
                
        return min(similarity, 1.0)
    
    def _calculate_direct_similarity(self, current_features: Dict, cache_data: str) -> float:
        """Calcola similarità diretta tra contesti"""
        try:
            # Per implementazioni future con dati in chiaro
            return 0.8  # Placeholder
        except Exception:
            return 0.0

    def _get_fallback_action(self, game_state: str, game_context: Dict, action_history: List[str]) -> Optional[int]:
        """Fallback deterministico quando LLM non è disponibile"""
        logger.debug("Utilizzo fallback deterministico per %s", game_state)
        
        # Fallback basato su stato di gioco
        if game_state == "battle":
            return self._get_battle_fallback(game_context, action_history)
        elif game_state == "exploring":
            return self._get_exploration_fallback(game_context, action_history)
        elif game_state == "menu":
            return self._get_menu_fallback(game_context, action_history)
        else:
            # Default: nessuna azione
            return 0
    
    def _get_battle_fallback(self, game_context: Dict, action_history: List[str]) -> int:
        """Fallback per battaglie - strategia semplice"""
        try:
            # Se HP bassi, prova a scappare o usare pozione
            if 'player_hp' in game_context and isinstance(game_context['player_hp'], (int, float)):
                if game_context['player_hp'] < 20:
                    return 6  # B per tornare indietro/proteggersi
            
            # Alterna tra attacco (A) e movimento
            if action_history and action_history[-1] == 'a':
                return 0  # Nessuna azione per cambiare strategia
            else:
                return 5  # A per attaccare
                
        except Exception as e:
            logger.debug("Errore fallback battaglia: %s", e)
            return 5  # Default: attacca
    
    def _get_exploration_fallback(self, game_context: Dict, action_history: List[str]) -> int:
        """Fallback per esplorazione - movimento base"""
        try:
            # Evita loop movimenti ripetitivi
            if len(action_history) >= 3 and len(set(action_history[-3:])) == 1:
                # Cambia direzione se stai andando nel loop
                last_action = action_history[-1]
                if last_action == 'up':
                    return 2  # Down
                elif last_action == 'down':
                    return 1  # Up
                elif last_action == 'left':
                    return 4  # Right
                elif last_action == 'right':
                    return 3  # Left
            
            # Movimento predefinito: su/giù alternati
            if len(action_history) % 2 == 0:
                return 1  # Up
            else:
                return 2  # Down
                
        except Exception as e:
            logger.debug("Errore fallback esplorazione: %s", e)
            return 1  # Default: su
    
    def _get_menu_fallback(self, game_context: Dict, action_history: List[str]) -> int:
        """Fallback per menu - navigazione base"""
        try:
            # In menu, usa A per selezionare o down per navigare
            if len(action_history) == 0 or action_history[-1] == 'a':
                return 2  # Down per navigare
            else:
                return 5  # A per selezionare
                
        except Exception as e:
            logger.debug("Errore fallback menu: %s", e)
            return 5  # Default: A

    def _check_rate_limit(self) -> bool:
        with self._rate_limit_lock:
            return self._check_rate_limit_locked()

    def _check_rate_limit_locked(self) -> bool:
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

        if not bool(self.config.enabled):
            return False

        if self.failure_cooldown_until > 0 and current_time >= self.failure_cooldown_until:
            self.consecutive_failures = 0
            self.available = True
            self._degraded = False
            self.failure_cooldown_until = 0
            logger.info("Cooldown terminato, riabilito LLM")

        if self.failure_cooldown_until > 0 and current_time < self.failure_cooldown_until:
            return False

        return bool(self.available)

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

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": float(self.config.temperature if temperature is None else temperature),
            "stream": False
        }
        options: Dict[str, Any] = {}
        if isinstance(num_predict, int) and num_predict > 0:
            options["num_predict"] = int(num_predict)
        if isinstance(seed, int) and seed >= 0:
            options["seed"] = int(seed)
        if options:
            payload["options"] = options
        return payload

    def _parse_action_index(self, content: str) -> Optional[int]:
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                action_str = str(data.get('action', '')).lower().strip()
                thought = data.get('thought', None)
                if isinstance(thought, str) and thought.strip():
                    logger.debug("LLM reasoning: %s", thought.strip())
                logger.debug("LLM decision action: %s", action_str)

                action_map = {
                    'none': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4,
                    'a': 5, 'b': 6, 'start': 7, 'select': 8
                }
                if action_str in action_map:
                    return int(action_map[action_str])
        except Exception as e:
            logger.debug("Errore parsing JSON LLM: %s", e)

        numbers = re.findall(r'\d+', content)
        if numbers:
            idx = int(numbers[0])
            if 0 <= idx <= 8:
                return idx

        content_lower = content.lower()
        if 'up' in content_lower:
            return 1
        if 'down' in content_lower:
            return 2
        if 'left' in content_lower:
            return 3
        if 'right' in content_lower:
            return 4
        if re.search(r'(^|\W)a(\W|$)', content_lower):
            return 5
        if re.search(r'(^|\W)b(\W|$)', content_lower):
            return 6
        if 'start' in content_lower:
            return 7
        if 'select' in content_lower:
            return 8
        return None

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Optional[str]:
        if not self._check_availability():
            return None
        if not self._check_rate_limit():
            logger.debug("Rate limit LLM superato, salto generate_text")
            return None
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = self._build_payload(messages, temperature=temperature, num_predict=num_predict, seed=seed)
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
                    logger.warning("Formato risposta LLM inatteso in generate_text: %s", data)
                    continue
                self.consecutive_failures = 0
                self._last_latency_s = time.monotonic() - t0
                self._latency_ema_s = self._latency_ema_s * 0.9 + self._last_latency_s * 0.1
                with self._state_lock:
                    self.last_call_time = time.monotonic()
                    self.call_count += 1
                return content
            except requests.exceptions.ReadTimeout as e:
                logger.warning("Timeout generate_text: %s", e)
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("text timeouts")
                break
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                logger.warning(
                    "Errore generate_text tentativo %s/%s: %s",
                    attempt + 1,
                    self.config.retry_attempts + 1,
                    e,
                )

                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("text errors")
                    break
                if attempt < self.config.retry_attempts:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("Errore inatteso generate_text: %s", e)
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

    def get_action(
        self,
        game_state: str,
        game_context: Dict,
        screen_image: Optional[np.ndarray],
        action_history: List[str],
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        use_cache: bool = True
    ) -> Optional[int]:
        # Tracciamento statistiche
        self._stats['total_requests'] += 1
        
        if not self._check_availability():
            self._stats['failed_requests'] += 1
            return None

        if not self.config.use_for_exploration and game_state == "exploring":
            self._stats['failed_requests'] += 1
            return None
        if not self.config.use_for_battle and game_state == "battle":
            self._stats['failed_requests'] += 1
            return None
        if not self.config.use_for_menu and game_state == "menu":
            self._stats['failed_requests'] += 1
            return None

        cache_key = self._cache_key(game_state, game_context, action_history)
        stale_cached_response: Optional[int] = None
        now_mono = time.monotonic()
        
        # Caching intelligente con similarità
        if use_cache:
            with self._state_lock:
                # Prima cerca cache esatta
                if cache_key in self.cache:
                    cached_response, cache_time = self.cache[cache_key]
                    if (now_mono - cache_time) < float(self.config.cache_ttl_seconds):
                        self.consecutive_failures = 0
                        self._stats['cache_hits'] += 1
                        self._stats['successful_requests'] += 1
                        logger.debug("Cache hit esatto per %s", game_state)
                        return cached_response
                    stale_cached_response = cached_response
                
                # Poi cerca cache simile
                similar_action = self._find_similar_cache_entry(game_state, game_context, action_history)
                if similar_action is not None:
                    self._stats['cache_hits'] += 1
                    self._stats['successful_requests'] += 1
                    logger.debug("Cache hit simile per %s", game_state)
                    return similar_action
        
        self._stats['cache_misses'] += 1

        if not self._check_rate_limit():
            logger.debug("Rate limit LLM superato, fallback")
            self._stats['fallbacks_used'] += 1
            # Fallback deterministico basato su contesto
            return self._get_fallback_action(game_state, game_context, action_history)

        messages = self._construct_prompt(game_state, game_context, screen_image, action_history)

        payload = self._build_payload(messages, temperature=temperature, num_predict=num_predict, seed=seed)

        if self._latency_ema_s >= 2.5:
            self._degraded = True
        if self.config.use_vision and not self._degraded and screen_image is not None:
            encoded_image = self._encode_image(screen_image)
            payload["images"] = [encoded_image]

        timeout_arg = self._request_timeout_arg(is_action=True)

        # Retry intelligente con backoff esponenziale adattivo
        max_attempts = max(1, self.config.retry_attempts + 1)
        base_delay = 0.5  # Delay base per retry
        
        for attempt in range(max_attempts):
            try:
                t0 = time.monotonic()
                logger.debug("Tentativo LLM %d/%d per stato %s", attempt + 1, max_attempts, game_state)
                
                response = requests.post(f"{self.config.host}/api/chat", json=payload, timeout=timeout_arg)
                response.raise_for_status()
                response_data = response.json()

                self.consecutive_failures = 0
                self._last_latency_s = time.monotonic() - t0
                self._latency_ema_s = self._latency_ema_s * 0.9 + self._last_latency_s * 0.1
                
                logger.debug("Risposta LLM ricevuta in %.3fs", self._last_latency_s)

                if 'message' in response_data and 'content' in response_data['message']:
                    content = response_data['message']['content'].strip()
                elif 'response' in response_data:
                    content = response_data['response'].strip()
                else:
                    logger.warning("Formato risposta LLM inatteso in get_action: %s", response_data)
                    continue

                # Limita lunghezza risposta se configurato
                if self.config.enable_response_compression and len(content) > self.config.max_response_length:
                    content = content[:self.config.max_response_length] + "..."

                action_index = self._parse_action_index(content)
                if isinstance(action_index, int) and 0 <= action_index <= 8:
                    if use_cache:
                        with self._state_lock:
                            self.cache[cache_key] = (action_index, time.monotonic())
                    with self._state_lock:
                        self.last_call_time = time.monotonic()
                        self.call_count += 1
                    self._stats['successful_requests'] += 1
                    
                    # Aggiorna tempo medio risposta
                    if self._stats['avg_response_time'] == 0:
                        self._stats['avg_response_time'] = self._last_latency_s
                    else:
                        self._stats['avg_response_time'] = (self._stats['avg_response_time'] * 0.9 + self._last_latency_s * 0.1)
                    
                    logger.info("Azione LLM valida: %d per stato %s (tempo: %.3fs)", action_index, game_state, self._last_latency_s)
                    return action_index
                else:
                    logger.warning("Indice azione non valido: %s (contenuto: %s)", action_index, content[:100])

            except requests.exceptions.ReadTimeout as e:
                logger.warning("Timeout get_action (tentativo %d/%d): %s", attempt + 1, max_attempts, e)
                self.consecutive_failures += 1
                
                # Fallback immediato su timeout
                if stale_cached_response is not None:
                    logger.info("Utilizzo cache obsoleta per timeout")
                    return stale_cached_response
                    
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("timeouts")
                    break
                    
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                logger.warning(
                    "Errore get_action tentativo %d/%d: %s",
                    attempt + 1, max_attempts, e,
                )
                self.consecutive_failures += 1

                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("errors")
                    break
                    
                # Retry con backoff esponenziale adattivo
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    delay = min(delay, 8.0)  # Max 8 secondi
                    logger.debug("Attesa retry: %.1fs", delay)
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error("Errore inatteso get_action (tentativo %d/%d): %s", attempt + 1, max_attempts, e, exc_info=True)
                self.consecutive_failures += 1
                
                if self.consecutive_failures >= self.config.consecutive_failure_threshold:
                    self._enter_cooldown("errors")
                    break
                    
                if attempt < max_attempts - 1:
                    time.sleep(base_delay * (2 ** attempt))

        # Tutti i tentativi falliti - usa fallback
        logger.warning("Tutti i tentativi LLM falliti per %s, uso fallback", game_state)
        self._stats['failed_requests'] += 1
        self._stats['fallbacks_used'] += 1
        return self._get_fallback_action(game_state, game_context, action_history)

    def get_action_candidates(
        self,
        game_state: str,
        game_context: Dict,
        screen_image: Optional[np.ndarray],
        action_history: List[str],
        n_candidates: int = 3,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[int]:
        n = int(n_candidates)
        if n < 1:
            return []

        seeds = [random.randint(0, 2**31 - 1) for _ in range(n)]
        candidates: List[int] = []
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(
                    self.get_action,
                    game_state,
                    game_context,
                    screen_image,
                    action_history,
                    num_predict,
                    temperature,
                    seeds[i],
                    False
                )
                for i in range(n)
            ]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if isinstance(res, int) and 0 <= res <= 8:
                        candidates.append(res)
                except Exception as e:
                    logger.debug("Errore generazione candidati LLM: %s", e)
        return candidates
