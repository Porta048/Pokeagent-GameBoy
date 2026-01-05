# Pokemon AI Agent - Versione 5.0

[![CI](https://github.com/yourusername/Pokeagent-GameBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/Pokeagent-GameBoy/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GRPO](https://img.shields.io/badge/RL-GRPO-green.svg)](https://arxiv.org/abs/2501.12948)
[![LLM](https://img.shields.io/badge/LLM-Ollama-purple.svg)](https://ollama.ai/)

Agente AI autonomo che gioca a Pokemon Rosso/Blu usando **GRPO** (Group Relative Policy Optimization) con **World Model** e **integrazione LLM** per decision-making strategico.

> **VERSIONE 5.0 - GRPO + WORLD MODEL + LLM** (Gennaio 2026):
>
> - **LLM Integration**: Qwen3-VL per reasoning strategico (async, non-blocking)
> - **Vision-Language Model**: Screenshot analysis in tempo reale
> - **GRPO**: DeepSeek-R1 group-relative advantage normalization
> - **World Model**: Dreamer-style imagination training
> - **DeepSeek-VL2 Vision**: PixelShuffleAdaptor + Multi-head Latent Attention
> - **3.2M parametri**: Exploration (454K) + Battle (2.5M) + Menu (257K)
> - **Anti-loop system**: Menu spam detection + temporal reasoning

<p align="center">
  <a href="Screenshot%202025-09-13%20221934.png">
    <img src="Screenshot%202025-09-13%20221934.png" alt="Screenshot del progetto" width="800"/>
  </a>
  <br>
  <em>Screenshot del progetto</em>
</p>

## Panoramica

Agente AI completamente autonomo che impara a giocare Pokemon Rosso/Blu dall'inizio alla fine senza intervento umano. L'agente combina:

- **Reinforcement Learning (GRPO)**: Apprendimento da esperienza con policy optimization
- **World Model**: Immaginazione e pianificazione a lungo termine
- **LLM Reasoning**: Decision-making strategico con Qwen3-VL via Ollama

## Architettura

### Sistema Ibrido RL + LLM

```
                    +-------------------+
                    |   Qwen3-VL LLM    |
                    | (Strategic Bias)  |
                    +--------+----------+
                             |
                             v
+----------+    +------------+------------+    +-----------+
| PyBoy    | -> | PPO Network             | -> | Actions   |
| Emulator |    | (ExplorationPPO/        |    | (A/B/D-pad|
| (screen) |    |  BattlePPO/MenuPPO)     |    |  etc.)    |
+----------+    +------------+------------+    +-----------+
                             ^
                             |
                    +--------+----------+
                    |    World Model    |
                    | (Imagination)     |
                    +-------------------+
```

### LLM Integration (Novita V5.0)

L'LLM funziona come **strategic advisor** che fornisce bias alle azioni:

| Componente | Descrizione | Beneficio |
|------------|-------------|-----------|
| **Async Worker** | Thread separato per chiamate LLM | Zero lag nel game loop |
| **Vision Analysis** | Screenshot encoding base64 | Contesto visivo per decisioni |
| **Rate Limiting** | Max 60 chiamate/min | Bilanciamento latenza/frequenza |
| **Response Cache** | TTL 15s | Riduce chiamate ripetitive |
| **Soft Bias** | Boost logit azione suggerita | Non override policy, solo guida |

**Configurazione LLM** ([src/config.py](src/config.py)):

```python
# LLM Integration (Ollama + qwen3-vl:2b)
LLM_ENABLED: bool = True
LLM_HOST: str = "http://localhost:11434"
LLM_MODEL: str = "qwen3-vl:2b"
LLM_TIMEOUT: float = 10.0  # Async non blocca
LLM_MIN_INTERVAL_MS: int = 1000  # 1 chiamata/sec
LLM_USE_VISION: bool = True  # Screenshot analysis
LLM_USE_FOR_EXPLORATION: bool = True
LLM_USE_FOR_BATTLE: bool = True
LLM_USE_FOR_MENU: bool = False
```

### Vision Encoder

Architettura basata su DeepSeek-VL2 ([arXiv:2412.10302](https://arxiv.org/abs/2412.10302)):

```
Input Frame (144x160)
    |
CNN Backbone (Conv + BatchNorm + GELU)
    |
PixelShuffleAdaptor (compressione 2x2)
    |
Multi-head Latent Attention (MLA)
    |
Policy Head -> 9 azioni
Value Head  -> stima valore
```

### Tre Reti Specializzate

| Rete | Parametri | Embed Dim | MLA Layers | Uso |
|------|-----------|-----------|------------|-----|
| **ExplorationPPO** | 454K | 192 | 1 | Navigazione veloce |
| **BattlePPO** | 2.5M | 320 | 3 | Strategie complesse |
| **MenuPPO** | 257K | 128 | 1 | UI semplici |

**Totale: 3.2M parametri**

## Avvio Rapido

### Prerequisiti

1. **Python 3.8+**
2. **Ollama** (per LLM integration):
   ```bash
   # Installa Ollama: https://ollama.ai/download
   # Scarica il modello vision
   ollama pull qwen3-vl:2b

   # Avvia Ollama server
   ollama serve
   ```

### Installazione

```bash
# Clona il repository
git clone https://github.com/yourusername/Pokeagent-GameBoy.git
cd Pokeagent-GameBoy

# Installa con pip
pip install -e .

# Per supporto GPU (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Esecuzione dell'Agente

**IMPORTANTE**: Devi fornire il tuo file ROM Pokemon (`.gb` o `.gbc`).

```bash
# Specifica il percorso alla tua ROM
python -m src --rom-path pokemon_red.gb

# Con opzioni aggiuntive
python -m src --rom-path pokemon_red.gb --headless --speed 2
python -m src --rom-path pokemon.gb --log-level DEBUG
```

**Opzioni CLI disponibili**:
- `--rom-path PATH` - **[RICHIESTO]** Percorso al file ROM Pokemon
- `--headless` - Esegui senza finestra grafica (migliora FPS)
- `--speed N` - Velocita emulazione (0=illimitata, 1=normale, 2=2x). Default: 0
- `--log-level LEVEL` - Livello logging (DEBUG, INFO, WARNING, ERROR)

### Controlli

Durante il training:
- `ESC` - Esci e salva progresso
- `SPACE` - Pausa/Riprendi training
- `+` / `=` - Aumenta velocita emulazione
- `-` / `_` - Diminuisci velocita emulazione

## Configurazione

### Velocita e Rendering ([src/config.py](src/config.py))

```python
EMULATION_SPEED: int = 0  # 0=illimitata, 1=normale, 2=2x, etc.
RENDER_ENABLED: bool = True
FRAMESKIP_MAP: Dict[str, int] = {
    "dialogue": 6,
    "battle": 10,
    "menu": 6,
    "exploring": 8,
    "base": 6
}
```

### LLM Configuration

```python
# Disabilita LLM per training puro RL
LLM_ENABLED: bool = False

# Cambia modello
LLM_MODEL: str = "llama3.2-vision:latest"  # Alternativa

# Aumenta frequenza chiamate
LLM_MIN_INTERVAL_MS: int = 500  # Ogni 500ms

# Disabilita vision per latenza minore
LLM_USE_VISION: bool = False
```

### Modelli LLM Compatibili

| Modello | Vision | Latenza | Qualita |
|---------|--------|---------|---------|
| `qwen3-vl:2b` | Si | ~500ms | Buona |
| `llama3.2-vision:latest` | Si | ~800ms | Ottima |
| `mistral:7b` | No | ~200ms | Buona (solo testo) |

## Caratteristiche

### Reinforcement Learning
- **GRPO** (Group Relative Policy Optimization) con normalizzazione advantage
- **PPO** con clipped surrogate loss
- **World Model** per imagination training
- **GAE-lambda** per stima vantaggi
- **Entropy scheduling** adattivo

### LLM Integration
- **Async non-blocking**: Thread separato per chiamate HTTP
- **Vision analysis**: Screenshot encoding per contesto visivo
- **Strategic reasoning**: Suggerimenti per esplorazione e battaglia
- **Soft bias**: Guida policy senza override completo
- **Rate limiting**: Bilanciamento tra frequenza e latenza
- **Caching**: Riuso risposte per stati simili

### Intelligenza di Gioco
- **Lettura memoria** per tracciamento stato di gioco
- **Rilevamento eventi** (medaglie, battaglie, Pokemon catturati)
- **Rilevamento automatico stato** (battaglia/menu/dialogo/esplorazione)
- **Esplorazione guidata dalla curiosita**
- **Meccanismi anti-grinding**

## Confronto Performance

| Metrica | V4.0 (GRPO) | **V5.0 (GRPO + LLM)** |
|---------|-------------|----------------------|
| **Prima medaglia** | ~8 min | **~6 min** |
| **Primo Pokemon** | ~5 min | **~4 min** |
| **Memoria GPU** | ~600MB | **~650MB** |
| **FPS (con LLM)** | N/A | **~1000 FPS** |
| **Decision quality** | RL only | **RL + Strategic reasoning** |

## Output Esempio

```
[INFO] ROM: C:\Games\Pokemon Red.gb
[INFO] Emulation speed: unlimited
[LLM] Ollama ready with qwen3-vl:2b
[LLM] qwen3-vl:2b ready for strategic reasoning
[LOAD] Checkpoint loaded: episode 0, frame 131022

[PERF] 1524.2 FPS | Frame: 135000 | State: exploring | Avg Reward: 18.32
[GAME] Badges: 1 | Pokedex: 8/15 | Map: 54 | Pos: (12,8)
[ADAPTIVE] Entropy: 0.0456 | Exploration: High
[LLM] Calls: 45 | Cache: 67% | Avg latency: 523ms
[REWARD] {'badges': 2000, 'pokemon': 150, 'exploration': 88} = 2238.00
```

## Struttura Progetto

```
Pokeagent-GameBoy/
|-- src/
|   |-- main.py              # Entry point e loop training
|   |-- config.py            # Configurazione
|   |-- models.py            # PPONetworkGroup
|   |-- vision_encoder.py    # Vision Encoder (PixelShuffle + MLA)
|   |-- llm_integration.py   # LLM client asincrono
|   |-- memory_reader.py     # Sistema ricompense
|   |-- anti_loop.py         # Prevenzione loop
|   |-- action_filter.py     # Mascheramento azioni
|   |-- trajectory_buffer.py # Raccolta traiettorie PPO
|   |-- simple_world_model.py # World Model per immaginazione
|   `-- hyperparameters.py   # Iperparametri
|-- tests/                   # Test unitari
|-- docs/                    # Documentazione
`-- README.md
```

## Risoluzione Problemi

### LLM non funziona

**Problema**: `[LLM] Ollama not running`
```bash
# Avvia Ollama server
ollama serve
```

**Problema**: `[LLM] Model not found`
```bash
# Scarica il modello
ollama pull qwen3-vl:2b
```

**Problema**: Lag durante il gioco
```python
# In config.py, aumenta intervallo chiamate
LLM_MIN_INTERVAL_MS: int = 3000  # Ogni 3 secondi

# Oppure disabilita vision
LLM_USE_VISION: bool = False
```

### Problemi Performance

**Problema**: FPS basso
- Usa `--headless` per disabilitare rendering
- Disabilita LLM: `LLM_ENABLED: bool = False`
- Aumenta `LLM_MIN_INTERVAL_MS`

**Problema**: Memoria GPU elevata
- Riduci `FRAME_STACK_SIZE`
- Usa modello LLM piu piccolo

## Gestione Modelli Ollama

```bash
# Lista modelli installati
ollama list

# Scarica nuovo modello
ollama pull qwen3-vl:2b

# Rimuovi modello
ollama rm qwen3-vl:2b

# Info modello
ollama show qwen3-vl:2b
```

## Licenza

Licenza MIT - vedi [LICENSE](LICENSE)

**Importante**: Gli utenti sono responsabili di ottenere legalmente i file ROM. Pokemon e un marchio di Nintendo/Game Freak/The Pokemon Company.

## Riconoscimenti

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Emulatore Game Boy
- [PyTorch](https://pytorch.org/) - Framework deep learning
- [Ollama](https://ollama.ai/) - LLM locale
- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) - Vision-Language Model
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [DeepSeek-VL2 Paper](https://arxiv.org/abs/2412.10302) - Vision Encoder
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) - GRPO

---

**Progetto educativo per ricerca reinforcement learning. Performance varia in base a hardware e configurazione.**
