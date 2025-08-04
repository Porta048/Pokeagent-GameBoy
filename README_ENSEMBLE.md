# Pokemon Ensemble AI - Multi-Agent System

## Overview

The Pokemon Ensemble AI is an advanced parallel training system that uses multiple AI agent instances to accelerate learning and improve performance in Pokemon Gold/Silver/Crystal games.

### Key Features

- **Parallel Training**: 3-6 agents learning simultaneously
- **Ensemble Learning**: Intelligent combination of knowledge
- **Specialization**: Each agent focuses on different aspects of the game
- **Synchronization**: Periodic knowledge sharing between agents
- **Flexible Configuration**: Customizable parameters for each agent

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pyboy numpy pillow keyboard
```

### 2. ROM Preparation

Make sure you have a Pokemon ROM file (.gbc) in the project directory:
- Pokemon Gold
- Pokemon Silver  
- Pokemon Crystal

### 3. Launch Ensemble

#### Interactive Mode (Recommended)
```bash
python launch_ensemble.py
```

#### Automatic Mode
```bash
python launch_ensemble.py --auto --agents 3 --method soft_merge --episodes 30
```

#### Advanced Mode
```bash
python launch_ensemble.py --rom "Pokemon Silver.gbc" --agents 4 --method majority_vote --episodes 50
```

## Ensemble Methods

### 1. Soft Merge (Recommended)
- **Description**: Gradually combines Q-values from different agents
- **Advantages**: Stable and continuous learning
- **Usage**: Ideal for long-term training

### 2. Majority Vote
- **Description**: Chooses the action voted by the majority of agents
- **Advantages**: Robust and conservative decisions
- **Usage**: Good for critical situations

## Agent Configuration

### Agent 1: Explorer
- **Focus**: Exploration and discovery
- **Parameters**: High epsilon, reward scaling for exploration
- **Objective**: Find new areas and Pokemon Centers

### Agent 2: Battler
- **Focus**: Combat and battles
- **Parameters**: High gamma, extended memory
- **Objective**: Optimize combat strategies

### Agent 3: Navigator
- **Focus**: Navigation and menu management
- **Parameters**: High learning rate, optimized batch size
- **Objective**: Efficiency in menus and navigation

## Advanced Configuration

### ensemble_config.json File

```json
{
  "ensemble_settings": {
    "num_agents": 3,
    "ensemble_method": "soft_merge",
    "sync_interval": 1000,
    "max_episodes_per_agent": 50
  },
  "agent_configurations": [
    {
      "agent_id": 0,
      "name": "Explorer",
      "learning_rate": 0.0003,
      "focus": "exploration"
    }
  ]
}
```

### Customizable Parameters

- **learning_rate**: Learning speed (0.0001-0.001)
- **epsilon_decay**: Exploration reduction speed (0.995-0.9999)
- **gamma**: Future reward discount factor (0.95-0.999)
- **batch_size**: Training batch size (16-128)
- **memory_size**: Replay memory size (5000-50000)
- **reward_scaling**: Reward multiplier (0.5-2.0)

## Performance Monitoring

### Real-time Output
```
Ensemble Progress (t=120s):
   Active agents: 3/3
   Average reward: 45.67
   Total frames: 15,420
Ensemble weights updated: ['0.350', '0.425', '0.225']
```

### Save Files
- `ai_saves_[ROM]_agent_0/`: Explorer Agent data
- `ai_saves_[ROM]_agent_1/`: Battler Agent data
- `ai_saves_[ROM]_agent_2/`: Navigator Agent data
- `ensemble_state.json`: General ensemble state

## Using the Trained Ensemble Model

### Loading Trained Ensemble

```python
from ensemble_pokemon_ai import EnsemblePokemonAI

# Load existing ensemble
ensemble = EnsemblePokemonAI("Pokemon Silver.gbc", num_agents=3)
ensemble.initialize_agents()

# Get action from ensemble
state = get_current_game_state()
action = ensemble.get_ensemble_action(state)
```

### Continue Training

```python
# Continue training from checkpoint
ensemble.start_parallel_training(max_episodes_per_agent=20)
```

## Troubleshooting

### "ROM not found" Error
- Verify that the ROM file is in the correct directory
- Check the file extension (.gbc, .gb)
- Use full path if necessary

### "PyTorch not available" Error
```bash
pip install torch torchvision torchaudio
```

### Low Performance
- Reduce number of agents (2-3)
- Decrease resolution if possible
- Use headless mode for non-primary agents

### Insufficient Memory
- Reduce `memory_size` in configuration
- Decrease `batch_size`
- Use fewer simultaneous agents

## Performance Optimization

### Recommended Hardware
- **CPU**: 4+ cores for 3 agents
- **RAM**: 8GB+ for complete ensemble
- **GPU**: Optional but accelerates training

### Optimal Configurations

#### Quick Setup (2 agents)
```json
{
  "num_agents": 2,
  "sync_interval": 500,
  "max_episodes_per_agent": 20
}
```

#### Balanced Setup (3 agents)
```json
{
  "num_agents": 3,
  "sync_interval": 1000,
  "max_episodes_per_agent": 30
}
```

#### Intensive Setup (4+ agents)
```json
{
  "num_agents": 4,
  "sync_interval": 1500,
  "max_episodes_per_agent": 50
}
```

## Training Strategies

### Incremental Training
1. Start with 2 agents for 20 episodes
2. Add the third agent
3. Gradually increase episodes
4. Optimize parameters based on results

### Specialized Training
1. Configure agents with different focuses
2. Monitor individual performance
3. Adjust ensemble weights dynamically
4. Save winning configurations

## Practical Examples

### Example 1: Quick Training
```bash
# Quick training for testing
python launch_ensemble.py --auto --agents 2 --episodes 15
```

### Example 2: Intensive Training
```bash
# Complete overnight training
python launch_ensemble.py --agents 4 --method soft_merge --episodes 100
```

### Example 3: Specialized Training
```python
# Custom configuration for exploration
config = {
    'agent_id': 0,
    'focus': 'exploration',
    'reward_scaling': 1.5,
    'epsilon_min': 0.15
}
```

## Contributions

To contribute to the project:
1. Fork the repository
2. Create branch for new features
3. Test changes with ensemble
4. Submit pull request

## License

This project is released under MIT license. See LICENSE for details.

---

**Note**: Make sure you legally own the Pokemon ROMs used. This software is for educational and research purposes only.