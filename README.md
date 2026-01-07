# PokéAgent GameBoy

An autonomous AI agent designed to play Pokémon Red on Game Boy using **PyBoy** for emulation and local **LLMs (Large Language Models)** (via Ollama) for decision reasoning.

The agent uses a hybrid approach combining:
- **Computer Vision**: Analysis of the game screen.
- **LLM Reasoning (Chain of Thought)**: Strategic planning based on context.
- **Knowledge Base**: Database of game knowledge (maps, objectives, type weaknesses).
- **Hybrid Memory**: Short-term memory for immediate context and long-term memory for progress tracking.

## Key Features

*   **3-Phase Architecture**:
    *   **Planning**: Formulates short-term goals based on current state and Knowledge Base.
    *   **Execution**: Translates goals into precise action sequences, handling movement and menus.
    *   **Critique**: Periodically evaluates action success and adapts strategy.
*   **Local LLM Integration**: Support for models via Ollama (default: `llama3.2:3b` for a strong quality/speed trade-off).
*   **Chain of Thought (CoT)**: The agent "thinks" before acting, explaining the reasoning behind its choices (visible in logs).
*   **Test-Time Compute Scaling**: Dynamically increases parallel LLM action sampling (best-of-N) for harder situations.
*   **Semantic Knowledge Base**: The system understands the game map, area connections, and main objectives.
*   **Intelligent Fallback Management**: If the LLM is slow or unavailable, the agent uses heuristic logic to avoid getting stuck.
*   **GUI Synchronization**: Optimized to synchronize agent actions with game animations.

## Model Gameplay Logic

### How the Agent Makes Decisions

The agent operates through a continuous cycle of **perception → planning → action → reflection**:

1. **Context Analysis**: Every 25 steps, the agent analyzes:
   - Player position in the current map
   - Pokémon team state (HP, levels, conditions)
   - Active game mode (exploration, battle, menu, dialogue)
   - Long-term goals (e.g., "Defeat the Pokémon League")

2. **Goal Formulation**: Based on context, the agent generates specific objectives such as:
   - *"Explore the area to find useful items"*
   - *"Heal injured Pokémon at the Pokémon Center"*
   - *"Battle the Trainer on Route 24"*

3. **Action Selection**: The agent considers 9 possible actions:
   - Movement: up, down, left, right
   - Interactions: A, B, START, SELECT
   - No action (wait)

### Intelligent Gameplay Strategies

#### Mode Recognition
The agent adapts its behavior based on the situation:

- **Exploration Mode**: Varied movements, item searching, NPC interaction
- **Battle Mode**: Type strategies, HP management, Pokémon switching
- **Menu Mode**: Efficient navigation through options
- **Dialogue Mode**: Progression through conversations

#### Test-Time Compute Scaling (Best-of-N)
For each action decision, the agent can sample multiple candidate actions in parallel from the LLM and pick the best one according to heuristics. The number of candidates is adjusted dynamically based on game mode and estimated difficulty.

When the LLM becomes slow or degraded, the agent reduces parallel sampling to keep the loop responsive.

#### Anti-Loop System
To avoid getting stuck, the agent implements:
- **Pattern Detection**: Recognizes repetitive sequences (e.g., up-down-up-down)
- **Circular Movement Detection**: Identifies circular movements indicating being stuck
- **Alternating Pattern Prevention**: Avoids A-B-A-B patterns that yield no progress
- **Exploration Boost**: After 3-4 failed attempts, increases action variety

#### Hybrid Fallback System
When the LLM is unavailable or slow:

1. **Contextual Strategy**: Selects actions appropriate to the mode
   - In battle: priority to "A" to attack, "up/down" to select moves
   - In menus: "B" to go back, directional navigation
   - In exploration: cardinal movements with preference for new directions

2. **Pattern Memory**: Remembers which actions worked in similar contexts

3. **Exploration Heuristic**: Prefers directions not recently explored

### Timing and Synchronization

The agent respects Game Boy timing:
- **Frame Rate**: 59.73 FPS (original Game Boy speed)
- **Animation Duration**: Waits for animation completion (16 frames per step)
- **Input Windows**: Holds buttons for 3-6 frames to ensure registration
- **Safety Factor**: 25% margin to compensate for emulation lag

### Adaptive Learning

The agent improves over time through:
- **Success Memory**: Remembers sequences that led to progress
- **Effective Patterns**: Identifies winning strategies for specific situations
- **Periodic Critique**: Every 50 steps, evaluates if the approach is working
- **Strategic Adaptation**: Modifies goals if no progress is being made

### Error Management

The system includes robustness through:
- **Rate Limiting**: Max 600 LLM calls per minute to avoid overload
- **Timeout Management**: 60 seconds max for LLM responses
- **Hierarchical Fallback**: 4 levels of fallback (LLM → Template → Heuristics → Random Action)
- **Crash Recovery**: State restoration after critical errors

## Requirements

*   **Python 3.10+**
*   **Ollama**: Installed and running locally.
*   **Pokémon Red ROM**: `.gb` file (to be placed in `roms/`).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Pokeagent-GameBoy.git
    cd Pokeagent-GameBoy
    ```

2.  **Create a virtual environment (optional but recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Ollama**:
    Ensure Ollama is installed and pull the required model (or modify it in `config.py`):
    ```bash
    ollama pull llama3.2:3b
    ```

5.  **Add the ROM**:
    Copy your Pokémon Red ROM (e.g., `Pokemon Red.gb`) into the `roms/` folder.

## Usage

Ensure the Ollama server is running, then start the agent:

```bash
python3 main.py
```

## Advanced Configuration (`config.py`)

You can customize the agent's behavior by modifying `config.py`:

*   `LLM_MODEL`: Ollama model to use (default: `qwen2.5:0.5b`).
*   `EMULATION_SPEED`: Emulation speed (default: `1`x).
*   `HEADLESS`: Set to `True` to run without a graphical window.
*   `RENDER_EVERY_N_FRAMES`: Rendering frequency (to speed up headless mode).
*   `LLM_MAX_CALLS_PER_MINUTE`: API call limit to avoid overload.
*   `ADAPTIVE_COMPUTE_ENABLED`: Enables dynamic action compute allocation.
*   `PARALLEL_SAMPLING_ENABLED`: Enables best-of-N parallel action sampling.
*   `PARALLEL_SAMPLING_MIN_CANDIDATES` / `PARALLEL_SAMPLING_MAX_CANDIDATES`: Bounds for parallel sampling.
*   `LLM_ACTION_BUDGET_*`: Token budgets per mode (exploring/menu/battle/boss).

## Contributing

Feel free to open Issues or Pull Requests to improve the agent, add new features to the Knowledge Base, or optimize prompts!

## License

[Insert your license here, e.g., MIT]
