# Pokemon AI Agent

AI agent that learns to play Pokemon Red/Blue using Deep Q-Learning with PyBoy.

![Pokemon AI Screenshot](Screenshot%202025-09-13%20221934.png)

## Quick Start

1. Install dependencies:
   ```bash
   pip install torch pyboy numpy opencv-python keyboard
   ```

2. Run the AI:
   ```bash
   python gbc_ai_agent.py
   ```

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

- Multi-agent DQN (Explorer, Battle, Menu)
- Automatic progress tracking and statistics
- Model and memory auto-save
- Game state detection (battle/menu/dialogue)
- Advanced reward system
- Compatible with Pokemon Red/Blue/Yellow/Gold/Silver

The AI learns automatically and improves over time.