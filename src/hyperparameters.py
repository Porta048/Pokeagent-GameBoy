from config import config

# Hyperparameters for the AI agent
HYPERPARAMETERS = {
    'HP_THRESHOLD': config.HP_THRESHOLD,
    'MENU_THRESHOLD': config.MENU_THRESHOLD,
    'DIALOGUE_THRESHOLD': config.DIALOGUE_THRESHOLD,
    'MOVEMENT_THRESHOLD': 0.02,
    'BLOCKED_THRESHOLD': 50,
    'MEMORY_CHECK_INTERVAL': 30,
    'SAVE_FREQUENCY': config.SAVE_FREQUENCY,
    'CACHE_SIZE': 1000,
    # PPO Hyperparameters
    'PPO_CLIP_EPSILON': 0.2,
    'PPO_VALUE_COEFF': 0.5,
    'PPO_ENTROPY_COEFF': 0.01,  # Dynamic - managed by AdaptiveEntropyScheduler
    'PPO_ENTROPY_START': 0.1,    # Increased from 0.05 to 0.1 (MUCH more exploration)
    'PPO_ENTROPY_END': 0.01,     # Increased from 0.005 to 0.01 (maintains exploration)
    'PPO_ENTROPY_DECAY_FRAMES': 1000000,  # Increased from 500k to 1M (slower decay)
    'PPO_GAE_LAMBDA': 0.95,
    'PPO_GAE_GAMMA': 0.99,
    'PPO_EPOCHS': 3,
    'PPO_MINIBATCH_SIZE': 32,
    'PPO_TRAJECTORY_LENGTH': 512,
    'PPO_LR': 3e-4,
    'PPO_MAX_GRAD_NORM': 0.5,
    'FRAME_STACK': config.FRAME_STACK_SIZE,
    # Anti-Confusion System
    'ANTI_LOOP_ENABLED': config.ANTI_LOOP_ENABLED,
    'ANTI_LOOP_BUFFER_SIZE': 100,        # Tracks last 100 states
    'ANTI_LOOP_THRESHOLD': 8,            # Penalty if >8 similar states
    'ANTI_LOOP_PENALTY': -2.0,           # Reduced penalty (was -5.0)
    'ACTION_REPEAT_MAX': 10,             # Increased to 10 (was 5) - more permissive
    'ACTION_REPEAT_PENALTY': -1.0,       # Reduced penalty (was -2.0)
    # Emulator Optimizations
    'FRAMESKIP_BASE': config.FRAMESKIP_MAP["base"],
    'FRAMESKIP_DIALOGUE': config.FRAMESKIP_MAP["dialogue"],
    'FRAMESKIP_BATTLE': config.FRAMESKIP_MAP["battle"],
    'FRAMESKIP_EXPLORING': config.FRAMESKIP_MAP["exploring"],
    'FRAMESKIP_MENU': config.FRAMESKIP_MAP["menu"],
    'TURN_BASED_MODE': False,         # True = ClaudePlayer style
    'RENDER_ENABLED': config.RENDER_ENABLED,
    'RENDER_EVERY_N_FRAMES': config.RENDER_EVERY_N_FRAMES,
    'PERFORMANCE_LOG_INTERVAL': config.PERFORMANCE_LOG_INTERVAL,
    # Emulation Speed
    'EMULATION_SPEED': config.EMULATION_SPEED
}