"""Iperparametri per l'agente AI Pokemon."""
from config import config

# Iperparametri per l'agente AI
HYPERPARAMETERS = {
    'HP_THRESHOLD': config.hp_threshold,
    'MENU_THRESHOLD': config.menu_threshold,
    'DIALOGUE_THRESHOLD': config.dialogue_threshold,
    'MOVEMENT_THRESHOLD': 0.02,
    'BLOCKED_THRESHOLD': 50,
    'MEMORY_CHECK_INTERVAL': 30,
    'SAVE_FREQUENCY': config.save_frequency,
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
    'FRAME_STACK': config.frame_stack_size,
    # Sistema anti-confusione
    'ANTI_LOOP_ENABLED': config.anti_loop_enabled,
    'ANTI_LOOP_BUFFER_SIZE': 100,        # Traccia ultimi 100 stati
    'ANTI_LOOP_THRESHOLD': 8,            # Penalità se >8 stati simili
    'ANTI_LOOP_PENALTY': -2.0,           # Penalità ridotta (era -5.0)
    'ACTION_REPEAT_MAX': 10,             # Aumentato a 10 (era 5) - più permissivo
    'ACTION_REPEAT_PENALTY': -1.0,       # Penalità ridotta (era -2.0)
    # Ottimizzazioni emulatore
    'FRAMESKIP_BASE': config.frameskip_map["base"],
    'FRAMESKIP_DIALOGUE': config.frameskip_map["dialogue"],
    'FRAMESKIP_BATTLE': config.frameskip_map["battle"],
    'FRAMESKIP_EXPLORING': config.frameskip_map["exploring"],
    'FRAMESKIP_MENU': config.frameskip_map["menu"],
    'TURN_BASED_MODE': False,         # True = stile ClaudePlayer
    'RENDER_ENABLED': config.render_enabled,
    'RENDER_EVERY_N_FRAMES': config.render_every_n_frames,
    'PERFORMANCE_LOG_INTERVAL': config.performance_log_interval,
    # Velocità emulazione
    'EMULATION_SPEED': config.emulation_speed
}