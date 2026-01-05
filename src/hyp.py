from .cfg import config
HYPERPARAMETERS = {
    'MOVEMENT_THRESHOLD': 0.02,
    'BLOCKED_THRESHOLD': 30,
    'MEMORY_CHECK_INTERVAL': 20,
    'SAVE_FREQUENCY': config.SAVE_FREQUENCY,
    'CACHE_SIZE': 1000,
    'PPO_CLIP_EPSILON': 0.15,
    'PPO_VALUE_COEFF': 0.5,
    'PPO_ENTROPY_COEFF': 0.02,
    'PPO_ENTROPY_START': 0.15,
    'PPO_ENTROPY_END': 0.005,
    'PPO_ENTROPY_DECAY_FRAMES': 300000,
    'PPO_GAE_LAMBDA': 0.92,
    'PPO_GAE_GAMMA': 0.995,
    'PPO_EPOCHS': 4,
    'PPO_MINIBATCH_SIZE': 64,
    'PPO_TRAJECTORY_LENGTH': 256,
    'PPO_LR': 2.5e-4,
    'PPO_MAX_GRAD_NORM': 0.5,
    'FRAME_STACK': config.FRAME_STACK_SIZE,
    'DEEPSEEK_VL2_SHUFFLE_FACTOR': 2,
    'DEEPSEEK_VL2_EXPLORATION_KV_RANK': 48,
    'DEEPSEEK_VL2_BATTLE_KV_RANK': 80,
    'DEEPSEEK_VL2_MENU_KV_RANK': 32,
    'DEEPSEEK_VL2_EXPLORATION_EMBED_DIM': 192,
    'DEEPSEEK_VL2_BATTLE_EMBED_DIM': 320,
    'DEEPSEEK_VL2_MENU_EMBED_DIM': 128,
    'DEEPSEEK_VL2_EXPLORATION_NUM_HEADS': 3,
    'DEEPSEEK_VL2_BATTLE_NUM_HEADS': 5,
    'DEEPSEEK_VL2_MENU_NUM_HEADS': 2,
    'DEEPSEEK_VL2_EXPLORATION_MLA_LAYERS': 1,
    'DEEPSEEK_VL2_BATTLE_MLA_LAYERS': 3,
    'DEEPSEEK_VL2_MENU_MLA_LAYERS': 1,
    'ANTI_LOOP_ENABLED': config.ANTI_LOOP_ENABLED,
    'ANTI_LOOP_BUFFER_SIZE': 64,
    'ANTI_LOOP_THRESHOLD': 4,
    'ANTI_LOOP_PENALTY': -5.0,
    'ACTION_REPEAT_MAX': 6,
    'ACTION_REPEAT_PENALTY': -3.0,
    'OSCILLATION_PENALTY': -4.0,
    'CURIOSITY_COEFF': 0.1,
    'EXPLORATION_BONUS': 0.5,
    'FRAMESKIP_BASE': config.FRAMESKIP_MAP["base"],
    'FRAMESKIP_DIALOGUE': config.FRAMESKIP_MAP["dialogue"],
    'FRAMESKIP_BATTLE': config.FRAMESKIP_MAP["battle"],
    'FRAMESKIP_EXPLORING': config.FRAMESKIP_MAP["exploring"],
    'FRAMESKIP_MENU': config.FRAMESKIP_MAP["menu"],
    'TURN_BASED_MODE': False,
    'RENDER_ENABLED': config.RENDER_ENABLED,
    'RENDER_EVERY_N_FRAMES': config.RENDER_EVERY_N_FRAMES,
    'PERFORMANCE_LOG_INTERVAL': config.PERFORMANCE_LOG_INTERVAL,
    'EMULATION_SPEED': config.EMULATION_SPEED,
    # GRPO (Group Relative Policy Optimization) - DeepSeek-R1 January 2025
    'GRPO_ENABLED': True,                    # Enable group-relative optimization
    'GRPO_GROUP_BY': 'game_state',          # 'game_state' | 'none' (fallback to PPO)
    'GRPO_MIN_GROUP_SIZE': 2,               # Minimum samples for group normalization
    'GRPO_LOG_GROUP_STATS': True,           # Log per-group statistics
    # World Model (Dreamer-style imagination training)
    'WORLD_MODEL_ENABLED': True,
    'WORLD_MODEL_START_FRAME': 10_000,      # Start training after 10k frames
    'WORLD_MODEL_LR': 1e-4,                 # Learning rate
    'WORLD_MODEL_IMAGINATION_HORIZON': 10,  # Steps to imagine ahead
    'WORLD_MODEL_LATENT_DIM': 192,          # Match exploration network
}