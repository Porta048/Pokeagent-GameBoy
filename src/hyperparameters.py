"""Iperparametri per l'agente AI Pokemon."""
from .config import config

# Iperparametri per l'agente AI
HYPERPARAMETERS = {
    'HP_THRESHOLD': config.HP_THRESHOLD,
    'MENU_THRESHOLD': config.MENU_THRESHOLD,
    'DIALOGUE_THRESHOLD': config.DIALOGUE_THRESHOLD,
    'MOVEMENT_THRESHOLD': 0.02,
    'BLOCKED_THRESHOLD': 50,
    'MEMORY_CHECK_INTERVAL': 30,
    'SAVE_FREQUENCY': config.SAVE_FREQUENCY,
    'CACHE_SIZE': 1000,
    # Iperparametri PPO
    'PPO_CLIP_EPSILON': 0.2,
    'PPO_VALUE_COEFF': 0.5,
    'PPO_ENTROPY_COEFF': 0.01,  # Dinamico - gestito da AdaptiveEntropyScheduler
    'PPO_ENTROPY_START': 0.1,    # Aumentato da 0.05 a 0.1 (più esplorazione)
    'PPO_ENTROPY_END': 0.01,     # Aumentato da 0.005 a 0.01 (mantiene esplorazione)
    'PPO_ENTROPY_DECAY_FRAMES': 1000000,  # Aumentato da 500k a 1M (decay più lento)
    'PPO_GAE_LAMBDA': 0.95,
    'PPO_GAE_GAMMA': 0.99,
    'PPO_EPOCHS': 3,
    'PPO_MINIBATCH_SIZE': 32,
    'PPO_TRAJECTORY_LENGTH': 512,
    'PPO_LR': 3e-4,
    'PPO_MAX_GRAD_NORM': 0.5,
    'FRAME_STACK': config.FRAME_STACK_SIZE,
    # Sistema anti-confusione
    'ANTI_LOOP_ENABLED': config.ANTI_LOOP_ENABLED,
    'ANTI_LOOP_BUFFER_SIZE': 100,        # Traccia ultimi 100 stati
    'ANTI_LOOP_THRESHOLD': 8,            # Penalità se >8 stati simili
    'ANTI_LOOP_PENALTY': -2.0,           # Penalità ridotta (era -5.0)
    'ACTION_REPEAT_MAX': 10,             # Aumentato a 10 (era 5) - più permissivo
    'ACTION_REPEAT_PENALTY': -1.0,       # Penalità ridotta (era -2.0)
    # Ottimizzazioni emulatore
    'FRAMESKIP_BASE': config.FRAMESKIP_MAP["base"],
    'FRAMESKIP_DIALOGUE': config.FRAMESKIP_MAP["dialogue"],
    'FRAMESKIP_BATTLE': config.FRAMESKIP_MAP["battle"],
    'FRAMESKIP_EXPLORING': config.FRAMESKIP_MAP["exploring"],
    'FRAMESKIP_MENU': config.FRAMESKIP_MAP["menu"],
    'TURN_BASED_MODE': False,         # True = stile ClaudePlayer
    'RENDER_ENABLED': config.RENDER_ENABLED,
    'RENDER_EVERY_N_FRAMES': config.RENDER_EVERY_N_FRAMES,
    'PERFORMANCE_LOG_INTERVAL': config.PERFORMANCE_LOG_INTERVAL,
    # Velocità emulazione
    'EMULATION_SPEED': config.EMULATION_SPEED
}