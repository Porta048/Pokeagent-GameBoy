import logging

from config import config as CFG
from agent.agent import MasterAgent
from agent.emulator import EmulatorHarness
from agent.llm import LLMConfig, OllamaLLMClient


def _build_llm_client() -> OllamaLLMClient:
    llm_config = LLMConfig(
        enabled=bool(CFG.LLM_ENABLED),
        host=str(CFG.LLM_HOST),
        model=str(CFG.LLM_MODEL),
        temperature=float(CFG.LLM_TEMPERATURE),
        timeout=float(CFG.LLM_TIMEOUT),
        min_interval_ms=int(CFG.LLM_MIN_INTERVAL_MS),
        max_calls_per_minute=int(CFG.LLM_MAX_CALLS_PER_MINUTE),
        cache_ttl_seconds=int(CFG.LLM_CACHE_TTL_SECONDS),
        use_vision=bool(CFG.LLM_USE_VISION),
        use_token_bucket=bool(CFG.LLM_USE_TOKEN_BUCKET),
        use_fast_cache_key=bool(CFG.LLM_USE_FAST_CACHE_KEY),
        use_for_exploration=bool(CFG.LLM_USE_FOR_EXPLORATION),
        use_for_battle=bool(CFG.LLM_USE_FOR_BATTLE),
        use_for_menu=bool(CFG.LLM_USE_FOR_MENU),
        fallback_on_error=bool(CFG.LLM_FALLBACK_ON_ERROR),
        retry_attempts=int(CFG.LLM_RETRY_ATTEMPTS),
        consecutive_failure_threshold=int(CFG.LLM_CONSECUTIVE_FAILURE_THRESHOLD),
        failure_cooldown_seconds=int(CFG.LLM_FAILURE_COOLDOWN_SECONDS)
    )
    return OllamaLLMClient(llm_config)


def main() -> None:
    logging.basicConfig(level=getattr(logging, str(CFG.LOG_LEVEL).upper(), logging.INFO))

    emulator = EmulatorHarness(CFG.ROM_PATH)
    llm_client = _build_llm_client()
    agent = MasterAgent(emulator, llm_client)

    try:
        while True:
            agent.run_step()
    except KeyboardInterrupt:
        pass
    finally:
        emulator.close()


if __name__ == "__main__":
    main()

