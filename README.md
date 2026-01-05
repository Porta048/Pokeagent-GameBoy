# Pokemon AI Agent - Versione 5.1

[![CI](https://github.com/yourusername/Pokeagent-GameBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/Pokeagent-GameBoy/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GRPO](https://img.shields.io/badge/RL-GRPO-green.svg)](https://arxiv.org/abs/2501.12948)
[![LLM](https://img.shields.io/badge/LLM-Ollama-purple.svg)](https://ollama.ai/)

Agente AI autonomo che gioca a Pokemon Rosso/Blu usando **GRPO** (Group Relative Policy Optimization) con **World Model** e **integrazione LLM** per decision-making strategico.

> **VERSIONE 5.1 - GRPO + WORLD MODEL + LLM** (Gennaio 2026):
>
> - **LLM Integration**: Qwen3-VL per reasoning strategico (async, non-blocking)
> - **Vision-Language Model**: Screenshot analysis in tempo reale
> - **GRPO**: DeepSeek-R1 group-relative advantage normalization
> - **World Model**: Dreamer-style imagination training
> - **DeepSeek-VL2 Vision**: PixelShuffleAdaptor + Multi-head Latent Attention
> - **1.76M parametri**: ExplorationPPO + BattlePPO + MenuPPO
> - **Anti-loop system**: Menu spam detection + temporal reasoning
> - **Graceful checkpoint handling**: Migrazione automatica tra architetture

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

### Sistema di Reward Gerarchico

Formula matematica avanzata per bilanciare obiettivi a breve e lungo termine:

```
R_totale = α(t) * R_primario + β(t) * R_secondario + γ(t) * R_intrinseco + R_penalità
```

Dove i pesi adattivi si aggiornano dinamicamente secondo le seguenti formule:

- **α(t) = α₀ * (1 + ρ_progressi)** - Aumenta con i progressi (badge, pokedex)
- **β(t) = β₀ * (1 + ρ_stallo * e^(-Δt_senza_progressi/τ))** - Aumenta quando l'agente è bloccato
- **γ(t) = γ₀ * (1 + ρ_esplorazione * H(s))** - Aumenta con l'entropia dello stato

Dove: α₀=0.3, β₀=0.4, γ₀=0.3, ρ_progressi=0.1, ρ_stallo=0.5, ρ_esplorazione=0.2, τ=300s.

### LLM Integration (Novita V5.1)

L'LLM e' il **primary decision maker** - la rete RL e' usata come fallback:

```text
Decisione = LLM disponibile? -> Usa azione LLM
                            -> Altrimenti: Usa rete RL (fallback)
```

**Cambiamento da V5.0**: Prima l'LLM era un "soft bias advisor" che aggiungeva un boost ai logit della policy. Ora l'LLM decide direttamente l'azione, con la rete RL che interviene solo quando l'LLM non risponde in tempo.

| Componente | Descrizione | Beneficio |
| ---------- | ----------- | --------- |
| **Async Worker** | Thread separato per chiamate LLM | Zero lag nel game loop |
| **Vision Analysis** | Screenshot encoding base64 | Contesto visivo per decisioni |
| **Rate Limiting** | Max 60 chiamate/min | Bilanciamento latenza/frequenza |
| **Response Cache** | TTL 15s | Riduce chiamate ripetitive |
| **Primary Control** | LLM decide azione direttamente | Decisioni strategiche immediate |
| **RL Fallback** | Rete PPO quando LLM non risponde | Continuita' senza interruzioni |

**Configurazione LLM** ([src/config.py](src/config.py)):

```python
# LLM Integration (Ollama + qwen3-vl:2b)
LLM_ENABLED: bool = True
LLM_HOST: str = "http://localhost:11434"
LLM_MODEL: str = "qwen3-vl:2b"
LLM_TEMPERATURE: float = 0.3  # Temperature più bassa per risposte consistenti
LLM_TIMEOUT: float = 15.0  # Tempo aumentato per elaborazione vision
LLM_MIN_INTERVAL_MS: int = 2000  # Intervallo aumentato a 2 secondi
LLM_MAX_CALLS_PER_MINUTE: int = 30  # Chiamate ridotte per evitare sovraccarico
LLM_CACHE_TTL_SECONDS: int = 20  # Cache aumentata per riutilizzo risposte
LLM_USE_VISION: bool = True  # Screenshot analysis
LLM_USE_FOR_EXPLORATION: bool = True
LLM_USE_FOR_BATTLE: bool = True
LLM_USE_FOR_MENU: bool = False
LLM_RETRY_ATTEMPTS: int = 2  # Numero di tentativi per richieste fallite
```

### Setup LLM

Per utilizzare correttamente l'integrazione LLM, segui questi passaggi:

1. **Installa Ollama**: Visita [ollama.com](https://ollama.com/) e scarica Ollama per il tuo sistema
2. **Avvia il servizio Ollama**:
   ```bash
   ollama serve
   ```
3. **Scarica il modello richiesto**:
   ```bash
   ollama pull qwen3-vl:2b
   ```
4. **Testa la connessione**:
   ```bash
   python test_llm_connection.py
   ```

Se il LLM non e' disponibile, il sistema utilizzera automaticamente la rete RL come fallback, garantendo che l'agente continui a funzionare anche senza LLM.

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
| ---- | --------- | --------- | ---------- | --- |
| **ExplorationPPO** | ~500K | 192 | 1 | Navigazione veloce |
| **BattlePPO** | ~900K | 320 | 3 | Strategie complesse |
| **MenuPPO** | ~350K | 128 | 1 | UI semplici |

Totale: 1.76M parametri

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

- **Primary decision maker**: LLM decide le azioni, RL e' fallback
- **Async non-blocking**: Thread separato per chiamate HTTP
- **Vision analysis**: Screenshot encoding per contesto visivo
- **Strategic reasoning**: Analisi situazione per esplorazione e battaglia
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

```text
[INFO] ROM: C:\Games\Pokemon Red.gb
[INFO] Emulation speed: unlimited
[LLM] Connected to Ollama, model: qwen3-vl:2b
[LLM] Ready
[LOAD] Checkpoint loaded: episode 0, frame 131022

[PERF] 1524.2fps Frame:135000 Reward:18.32
[GAME] Badges:1 Pokemon:8 Map:54 Pos:(12,8)
[DECISION] LLM:892 (74%) | RL fallback:315
[REWARD] {'badges': 2000, 'pokemon': 150, 'exploration': 88} = 2238.00
```


## Risoluzione Problemi

### LLM non funziona

**Problema**: `[LLM] Cannot connect to Ollama at http://localhost:11434`

```bash
# Avvia Ollama server
ollama serve
```

**Problema**: `[LLM] Model 'qwen3-vl:2b' not found. Available: [...]`

```bash
# Scarica il modello
ollama pull qwen3-vl:2b
```

**Problema**: LLM non prende decisioni (0 su 2000), con molti fallback RL

Possibili cause:
1. Ollama non e' in esecuzione
2. Modello non disponibile
3. Richieste LLM falliscono ripetutamente

Soluzione:
```bash
# Testa la connessione LLM
python test_llm_connection.py

# Verifica che Ollama sia in esecuzione
ollama serve

# Verifica che il modello sia disponibile
ollama list
```

**Problema**: Richieste LLM falliscono con messaggi tipo "Request returned no response"

Questo indica che le richieste all'LLM stanno fallendo. Il sistema ha ora un meccanismo di retry configurabile:
- Controlla la connessione a Ollama
- Aumenta `LLM_RETRY_ATTEMPTS` in `src/cfg.py` se necessario
- Controlla i log per dettagli sui fallimenti

**Problema**: Lag durante il gioco

```python
# In config.py, aumenta intervallo chiamate
LLM_MIN_INTERVAL_MS: int = 3000  # Ogni 3 secondi

# Oppure disabilita vision
LLM_USE_VISION: bool = False
```

### Checkpoint incompatibile

**Problema**: `exploration network architecture changed, training from scratch`

Questo messaggio indica che l'architettura della rete e' cambiata rispetto al checkpoint salvato. Il sistema gestisce automaticamente questa situazione avviando un nuovo training. Per forzare un reset completo, elimina il file checkpoint:

```bash
del "pokemon_ai_saves_Pokemon Red\model_ppo.pth"
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
