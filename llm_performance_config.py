#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OptimizedLLMConfig:
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:0.5b"
    temperature: float = 0.2
    timeout: float = 30.0
    min_interval_ms: int = 100
    max_calls_per_minute: int = 300
    cache_ttl_seconds: int = 60
    
    use_vision: bool = False
    use_token_bucket: bool = True
    use_fast_cache_key: bool = True
    use_for_exploration: bool = True
    use_for_battle: bool = True
    use_for_menu: bool = True
    fallback_on_error: bool = True
    
    retry_attempts: int = 1
    consecutive_failure_threshold: int = 2
    failure_cooldown_seconds: int = 30
    
    enable_smart_caching: bool = True
    cache_similarity_threshold: float = 0.80
    adaptive_timeout: bool = True
    fast_model_fallback: str = "qwen2.5:0.5b"
    enable_response_compression: bool = True
    max_response_length: int = 100
    enable_detailed_logging: bool = False

class PerformanceOptimizedLLM:
    
    @staticmethod
    def get_config_for_state(game_state: str) -> OptimizedLLMConfig:
        base_config = OptimizedLLMConfig()
        
        if game_state == "battle":
            config = OptimizedLLMConfig(
                temperature=0.1,
                timeout=20.0,
                max_calls_per_minute=200,
                cache_ttl_seconds=90,
                retry_attempts=2,
                max_response_length=120
            )
        elif game_state == "menu":
            config = OptimizedLLMConfig(
                temperature=0.1,
                timeout=15.0,
                max_calls_per_minute=400,
                cache_ttl_seconds=120,
                retry_attempts=1,
                max_response_length=50
            )
        elif game_state == "exploring":
            config = OptimizedLLMConfig(
                temperature=0.25,
                timeout=25.0,
                max_calls_per_minute=250,
                cache_ttl_seconds=45,
                retry_attempts=1,
                max_response_length=80
            )
        else:
            config = base_config
            
        return config
    
    @staticmethod
    def get_performance_tips() -> list[str]:
        return [
            "1. Usa qwen2.5:0.5b o llama3.2:1b per velocità massima",
            "2. Disabilita vision se non strettamente necessario",
            "3. Riduci temperature per coerenza maggiore",
            "4. Aumenta cache TTL per scenari ricorrenti",
            "5. Usa timeout adattivi per bilanciare velocità/affidabilità",
            "6. Implementa fallback deterministici per errori comuni",
            "7. Monitora statistiche per identificare colli di bottiglia",
            "8. Considera modello locale più piccolo per testing"
        ]

FAST_CONFIG = OptimizedLLMConfig(
    model="qwen2.5:0.5b",
    temperature=0.1,
    timeout=15.0,
    max_calls_per_minute=500,
    enable_detailed_logging=False
)

BALANCED_CONFIG = OptimizedLLMConfig(
    model="llama3.2:1b",
    temperature=0.2,
    timeout=25.0,
    max_calls_per_minute=300,
    enable_detailed_logging=False
)

QUALITY_CONFIG = OptimizedLLMConfig(
    model="llama3.2:3b",
    temperature=0.3,
    timeout=45.0,
    max_calls_per_minute=150,
    retry_attempts=2,
    enable_detailed_logging=True
)

if __name__ == "__main__":
    print("Configurazioni Performance LLM Ottimizzate")
    print("=" * 50)
    
    for state in ["battle", "menu", "exploring"]:
        config = PerformanceOptimizedLLM.get_config_for_state(state)
        print(f"\nConfigurazione per {state.upper()}:")
        print(f"  Modello: {config.model}")
        print(f"  Timeout: {config.timeout}s")
        print(f"  Max calls/min: {config.max_calls_per_minute}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Retry attempts: {config.retry_attempts}")
    
    print("\n" + "=" * 50)
    print("Suggerimenti Performance:")
    for tip in PerformanceOptimizedLLM.get_performance_tips():
        print(f"  {tip}")
