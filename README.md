# Pokeagent GameBoy

AI agent that learns to play Pokemon Red using Behavioral Cloning from human gameplay.

## How It Works

1. **Record** your gameplay sessions
2. **Train** the model on your recordings
3. **Play** and watch the AI imitate your playstyle

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp "Pokemon Red.gb" roms/
```

## Usage

### Record Gameplay

Play the game and record your actions:

```bash
python main.py --mode record
```

Controls:
- WASD or Arrow keys: Move
- Z or J: A button
- X or K: B button
- Enter: START
- Space: SELECT
- ESC: Save and exit

### Train Model

Train on your recorded gameplay:

```bash
python main.py --mode train
python main.py --mode train --epochs 100
python main.py --mode train --batch-size 128
```

### Watch AI Play

Run inference with the trained model:

```bash
python main.py --mode play
python main.py --mode play --steps 50000
python main.py --mode play --temperature 0.3
```

Temperature controls randomness:
- 0.0 = always pick best action
- 0.5 = balanced (default)
- 1.0 = more random

## Configuration

Edit `config.py` to adjust:

- `BC_LEARNING_RATE`: Learning rate (default: 1e-4)
- `BC_EPOCHS`: Training epochs (default: 50)
- `BC_BATCH_SIZE`: Batch size (default: 64)
- `INFERENCE_TEMPERATURE`: Play randomness (default: 0.5)

## Data Format

Recordings are saved as `.npz` files in `recordings/`:
- `observations`: 84x84 grayscale screenshots
- `actions`: Integer action indices (0-8)

## Architecture

- CNN feature extractor (Nature DQN style)
- MLP policy head with dropout
- Cross-entropy loss for action prediction

## Files

- `main.py` - Entry point
- `config.py` - Configuration
- `agent/model.py` - Neural network and dataset
- `agent/environment.py` - Game environment
- `agent/trainer.py` - Recording and training
- `agent/emulator.py` - PyBoy wrapper

## Requirements

- Python 3.8+
- PyTorch
- PyBoy
- pygame (for recording)
- Pokemon Red ROM

## License

MIT
