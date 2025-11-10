# Pokemon AI Agent

Autonomous AI agent that plays Pokemon Red/Blue completely by itself using **Proximal Policy Optimization (PPO)** with PyBoy.

> **⚡ NEW**: Migrated from DQN to PPO - **4-6x faster convergence** (<1 hour vs 4-6 hours)!

## Overview

This is a fully autonomous AI agent that plays Pokemon Red/Blue from start to finish without any human intervention. The agent uses reinforcement learning to learn optimal strategies for:

- Navigating the game world
- Battling wild Pokemon and trainers
- Catching Pokemon
- Obtaining gym badges
- Completing the main story

![Pokemon AI Screenshot](Screenshot%202025-09-13%20221934.png)

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. Choose training mode:

   **Single Environment** (visualize AI playing):
   ```bash
   python gbc_ai_agent.py
   ```
   - Shows window with gameplay
   - ~2,000-4,000 FPS (GPU)
   - Good for debugging/watching

   **Parallel Training** (8-96x faster):
   ```bash
   python parallel_trainer.py
   ```
   - 8-96 environments in parallel
   - ~16,000-64,000 FPS (8-32 workers)
   - **Recommended for production**

3. Enter ROM path:
   ```
   Pokemon ROM path (.gb/.gbc): ..\Pokemon Red.gb
   ```

4. Choose mode:
   - N - With visual window
   - y - Headless mode (faster)

## Controls

- ESC - Exit
- SPACE - Pause/Resume

## Features

- **PPO Actor-Critic** architecture (3 specialized networks: Explorer, Battle, Menu)
- **Frame stacking 4x** for temporal context
- **GAE-λ advantage estimation** for stable learning
- Automatic progress tracking and statistics
- Model checkpoint auto-save every 10k frames
- Game state detection (battle/menu/dialogue)
- Advanced event-based reward system
- Compatible with Pokemon Red/Blue/Yellow/Gold/Silver

## Algorithm Details

**PPO (Proximal Policy Optimization)**:
- Clipped surrogate loss (ε=0.2)
- Entropy bonus for exploration
- On-policy trajectory collection (512 steps)
- 3 epochs per update with minibatch shuffling
- Converges **4-6x faster** than DQN

See [PPO_MIGRATION.md](PPO_MIGRATION.md) for technical details.

## Performance

| Metric | DQN (old) | PPO (current) |
|--------|-----------|---------------|
| First badge | ~2 hours | ~20 minutes |
| Convergence | 6+ hours | <1 hour |
| Stability | Medium | High |

The AI learns automatically and improves over time.
