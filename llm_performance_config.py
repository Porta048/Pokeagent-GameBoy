#!/usr/bin/env python3
"""
Configurazione Performance Ottimizzata per LLM Integration
Basata sui risultati dei test di performance eseguiti.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OptimizedLLMConfig:
    """Configurazione ottimizzata per performance massime"""
    
    # Configurazioni base (ottimizzate)
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:0.5b"  # Modello più veloce
    temperature: float = 0.2  # Ridotto per coerenza maggiore
    timeout: float = 30.0  # Ridotto da 60s
    min_interval_ms: int = 100  # Aumentato da 50ms per stabilità
    max_calls_per_minute: int = 300  # Ridotto da 600 per qualità
    cache_ttl_seconds: int = 60  # Aumentato da 20s
    
    # Feature flags ottimizzate
    use_vision: bool = False  # Disabilitato per velocità
    use_token_bucket: bool = True
    use_fast_cache_key: bool = True
    use_for_exploration: bool = True
    use_for_battle: bool = True
    use_for_menu: bool = True
    fallback_on_error: bool = True
    
    # Retry e error handling ottimizzati
    retry_attempts: int = 1  # Ridotto da 2 per velocità
    consecutive_failure_threshold: int = 2  # Ridotto da 3
    failure_cooldown_seconds: int = 30  # Ridotto da 60s
    
    # Nuove feature per performance
    enable_smart_caching: bool = True
    cache_similarity_threshold: float = 0.80  # Leggermente ridotto
    adaptive_timeout: bool = True
    fast_model_fallback: str = "qwen2.5:0.5b"
    enable_response_compression: bool = True
    max_response_length: int = 100  # Ridotto da 150
    enable_detailed_logging: bool = False  # Disabilitato in produzione

class PerformanceOptimizedLLM:
    """Wrapper per configurazioni multiple in base allo stato di gioco"""
    
    @staticmethod
    def get_config_for_state(game_state: str) -> OptimizedLLMConfig:
        """Restituisce configurazione ottimizzata per lo stato di gioco"""
        
        base_config = OptimizedLLMConfig()
        
        if game_state == "battle":
            # Configurazione aggressiva per battaglie
            config = OptimizedLLMConfig(
                temperature=0.1,  # Massima coerenza
                timeout=20.0,  # Timeout breve
                max_calls_per_minute=200,  # Più lento per qualità
                cache_ttl_seconds=90,  # Cache lunga per strategie
                retry_attempts=2,  # Più retry per battaglie importanti
                max_response_length=120  # Più dettagliato
            )
        elif game_state == "menu":
            # Configurazione veloce per menu
            config = OptimizedLLMConfig(
                temperature=0.1,  # Massima coerenza
                timeout=15.0,  # Timeout molto breve
                max_calls_per_minute=400,  # Più veloce
                cache_ttl_seconds=120,  # Cache lunga per menu
                retry_attempts=1,  # Meno retry
                max_response_length=50  # Risposte brevi
            )
        elif game_state == "exploring":
            # Configurazione bilanciata per esplorazione
            config = OptimizedLLMConfig(
                temperature=0.25,  # Bilanciato
                timeout=25.0,  # Timeout medio
                max_calls_per_minute=250,  # Velocità media
                cache_ttl_seconds=45,  # Cache media
                retry_attempts=1,  # Retry medio
                max_response_length=80  # Dettaglio medio
            )
        else:
            # Default - configurazione base
            config = base_config
            
        return config
    
    @staticmethod
    def get_performance_tips() -> list[str]:
        """Restituisce suggerimenti per performance migliori"""
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

# Configurazioni predefinite
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