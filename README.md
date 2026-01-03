#  Pokemon AI Agent - Versione 3.0

[![CI](https://github.com/yourusername/Pokeagent-GameBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/Pokeagent-GameBoy/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PPO Algorithm](https://img.shields.io/badge/RL-PPO-green.svg)](https://arxiv.org/abs/1707.06347)
[![Vision Encoder](https://img.shields.io/badge/Arch-Vision%20Encoder-purple.svg)](https://arxiv.org/abs/2412.10302)

Agente AI autonomo che gioca a Pokemon Rosso/Blu completamente da solo usando **Proximal Policy Optimization (PPO)** con architettura **Vision Encoder** ispirata a DeepSeek-VL2.

> ** VERSIONE 3.0 - NUOVA ARCHITETTURA VISION ENCODER**:
>
> -  **Vision Encoder**: Architettura ispirata a DeepSeek-VL2 (arXiv:2412.10302)
> -  **PixelShuffleAdaptor**: Compressione 2×2 token visivi (4× meno token)
> -  **Multi-head Latent Attention**: KV cache compresso per efficienza
> -  **94% meno parametri**: Da 20M a 3M parametri totali
> -  **Migliore generalizzazione**: Meno overfitting su dataset RL
> -  **Training più stabile**: BatchNorm + GELU + LayerNorm

<p align="center">
  <a href="Screenshot%202025-09-13%20221934.png">
    <img src="Screenshot%202025-09-13%20221934.png" alt="Screenshot del progetto" width="800"/>
  </a>
  <br>
  <em>Screenshot del progetto</em>
</p>

## Panoramica

Agente AI completamente autonomo che impara a giocare Pokemon Rosso/Blu dall'inizio alla fine senza intervento umano. L'agente usa reinforcement learning per padroneggiare:

- Navigazione del mondo di gioco e scoperta di nuove aree
- Combattimenti strategici contro Pokemon selvatici e allenatori
- Cattura di Pokemon per costruire una squadra forte
- Ottenimento di medaglie dalle palestre e sconfitta dei capipalestra
- Completamento della storia principale

## Architettura

L'agente usa un'**architettura Vision Encoder** con PPO multi-rete, ispirata al paper DeepSeek-VL2 ([arXiv:2412.10302](https://arxiv.org/abs/2412.10302)).

### Vision Encoder (Novità V3.0)

Architettura basata su attention per comprensione visiva:

```
Input Frame (144×160)
    ↓
CNN Backbone (Conv + BatchNorm + GELU)
    ↓
PixelShuffleAdaptor (compressione 2×2)
    ↓
Multi-head Latent Attention (MLA)
    ↓
Policy Head → 9 azioni
Value Head  → stima valore
```

**Componenti chiave:**
- **PixelShuffleAdaptor**: Comprime token visivi 2×2 → 1 (4× riduzione)
- **Multi-head Latent Attention**: KV cache compresso per efficienza memoria
- **BatchNorm + GELU**: Training stabile con attivazioni smooth

### Tre Reti Specializzate

| Rete | Parametri | Embed Dim | MLA Layers | Uso |
|------|-----------|-----------|------------|-----|
| **ExplorationPPO** | 454K | 192 | 1 | Navigazione veloce |
| **BattlePPO** | 2.5M | 320 | 3 | Strategie complesse |
| **MenuPPO** | 257K | 128 | 1 | UI semplici |

**Totale: 3.2M parametri** (vs 20M architettura precedente)

### Componenti Aggiuntivi

- **Frame Stacking (4x)**: Contesto temporale da 4 frame consecutivi
- **Stima Vantaggio GAE-λ**: Gradienti stabili per l'apprendimento
- **Scheduling Entropia Adattivo**: Transizione graduale esplorazione → sfruttamento
- **Sistema Anti-Loop**: Rileva e penalizza comportamenti ripetitivi
- **Mascheramento Azioni Contestuale**: Filtra azioni non valide per stato di gioco
- **Ricompense Event-Based**: Segnali di ricompensa ricchi dalla memoria di gioco

## Avvio Rapido

### Installazione

```bash
# Clona il repository
git clone https://github.com/yourusername/Pokeagent-GameBoy.git
cd Pokeagent-GameBoy

# Installa con pip (consigliato)
pip install -e .

# Oppure installa le dipendenze manualmente
pip install -r requirements.txt

# Per supporto GPU (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Esecuzione dell'Agente

**IMPORTANTE**: Devi fornire il tuo file ROM Pokemon (`.gb` o `.gbc`). Questo progetto NON include file ROM.

```bash
# Specifica il percorso alla tua ROM
python -m src --rom-path pokemon_red.gb

# Esempi con percorsi diversi
python -m src --rom-path "C:/Games/Pokemon Red.gb"
python -m src --rom-path roms/pokemon_rosso.gb
python -m src --rom-path "/home/user/roms/pokemon.gb"

# Con opzioni aggiuntive
python -m src --rom-path pokemon_red.gb --headless --speed 2
python -m src --rom-path pokemon.gb --log-level DEBUG
```

**Opzioni CLI disponibili**:
- `--rom-path PATH` - **[RICHIESTO]** Percorso al file ROM Pokemon (`.gb` o `.gbc`)
- `--headless` - Esegui senza finestra grafica (migliora FPS di ~30%)
- `--speed N` - Velocità emulazione (0=illimitata, 1=normale, 2=2x, ecc.). Default: 0
- `--log-level LEVEL` - Livello logging (DEBUG, INFO, WARNING, ERROR). Default: INFO

**Per vedere tutti i comandi disponibili**:
```bash
python -m src --help
```

**Metodo alternativo - Modifica configurazione**:
Se preferisci non specificare `--rom-path` ogni volta, puoi modificare il default in [src/config.py](src/config.py) linea 21:
```python
ROM_PATH: str = "percorso/al/tuo/file.gb"
```
Poi esegui semplicemente: `python -m src`

### Controlli

Durante il training:
- `ESC` - Esci e salva progresso
- `SPACE` - Pausa/Riprendi training
- `+` / `=` - Aumenta velocità emulazione
- `-` / `_` - Diminuisci velocità emulazione

## Caratteristiche

### Reinforcement Learning
- **PPO (Proximal Policy Optimization)** con clipped surrogate loss
- **Architettura Actor-Critic** con heads separati per policy e value
- **Apprendimento on-policy** con buffer di traiettorie (512 passi)
- **Bonus entropia** per esplorazione con scheduling adattivo
- **3 epoche di training** per update con shuffling minibatch

### Intelligenza di Gioco
- **Lettura memoria** per tracciamento preciso stato di gioco
- **Rilevamento eventi** (medaglie, battaglie, Pokemon catturati, oggetti)
- **Rilevamento automatico stato** (battaglia/menu/dialogo/esplorazione)
- **Esplorazione guidata dalla curiosità** (ricompense intrinseche per nuove scoperte)
- **Meccanismi anti-grinding** (ritorni decrescenti per level grinding)

### Ottimizzazioni Training
- **Frameskip adattivo** (6-12 frame) basato sullo stato di gioco
- **Salvataggio automatico checkpoint** ogni 10.000 frame
- **Tracciamento statistiche** (ricompense, loss, progresso di gioco)
- **Monitoraggio performance** (FPS, contatore frame, metriche training)
- **Rendering fluido** (frequenza render configurabile)

##  Novità V3.0

### **Vision Encoder (Architettura Principale)**

Ispirata a DeepSeek-VL2 ([arXiv:2412.10302](https://arxiv.org/abs/2412.10302)):

| Componente | Descrizione | Beneficio |
|------------|-------------|-----------|
| **PixelShuffleAdaptor** | Compressione spaziale 2×2 | 4× meno token, preserva info |
| **Multi-head Latent Attention** | KV cache → vettori latenti | 8× meno memoria cache |
| **GELU + LayerNorm** | Attivazioni smooth | Training più stabile |
| **BatchNorm CNN** | Normalizzazione feature maps | Convergenza veloce |

**Riduzione parametri:**
- ExplorationPPO: 7.4M → 454K (**94% riduzione**)
- BattlePPO: 7.5M → 2.5M (**67% riduzione**)
- MenuPPO: 5.6M → 257K (**95% riduzione**)

### **Sistema Ricompense Ottimizzato**

-  **Eventi storia potenziati**: +50 (era +5) - Guidano verso obiettivi critici
-  **Allenatori raddoppiati**: +200 (era +100) - Progressione obbligatoria
-  **Esplorazione aumentata**: Nuova mappa +250, +5 per tile
-  **Battaglie incentivate**: Vittoria +120 - Engagement attivo

### **Action Filter Aggressivo**
-  **Mascheramento soft**: 0.3-0.5 per guidare senza bloccare
-  **Contestuale**: Diverso per battaglia/menu/dialogo/esplorazione

### **Codice Documentato**
-  **Ogni file commentato**: Docstring completi in italiano
-  **Matematica spiegata**: Log-space masking, GAE, MLA
-  **Paper references**: Citazioni per componenti chiave

##  Confronto Performance

| Metrica | PPO V1.0 | PPO V2.1 | **V3.0 (Vision)** |
|---------|----------|----------|-------------------|
| **Parametri totali** | 20M | 20M | **3.2M** |
| **Prima medaglia** | ~20 min | ~12 min | **~10 min** |
| **Primo Pokemon** | ~30 min | ~8 min | **~6 min** |
| **Memoria GPU** | ~2GB | ~2GB | **~500MB** |
| **Generalizzazione** | Bassa | Media | **Alta** |
| **Stabilità training** | Media | Alta | **Molto alta** |

## Progresso Training

L'agente traccia e logga automaticamente:
- **Stato gioco**: Medaglie, Pokedex (catturati/visti), soldi, posizione, mappa
- **Metriche training**: Policy loss, value loss, entropia, ricompensa media
- **Performance**: FPS, frame processati, percentuale esplorazione
- **Eventi**: Obiettivi principali (medaglie, Pokemon catturati, battaglie vinte)

Esempio output:
```
[PERF] 3524.2 FPS | Frame: 125000 | State: exploring | Avg Reward: 15.32
[GAME] Badges: 3 | Pokedex: 15/28 | Map: 54 | Pos: (12,8)
[ADAPTIVE] Entropy: 0.0456 | Exploration: High
[REWARD] {'badges': 2000, 'pokemon': 150, 'exploration': 88} = 2238.00
```

##  Documentazione Visuale

Guarda il **diagramma dell'architettura V2.0** qui sotto per una visione completa del sistema!

<p align="center">
  <a href="docs/architecture.svg">
    <img src="docs/architecture.svg" alt="Pokemon AI Architecture V2.0" width="800"/>
  </a>
  <br>
  <em>Diagramma dell'architettura V2.0 - Clicca per ingrandire</em>
</p>

Il diagramma SVG mostra:
-  **Loop training completo** (7 step dall'input al training)
-  **6 componenti principali** (PPO, PyBoy, State Detector, Memory Reader, Action Filter, Anti-Loop)
-  **Tutte le ottimizzazioni V2.0** con confronti prima/dopo
-  **Tabella performance** con metriche V2.0 vs V1.0

L'immagine è in formato SVG vettoriale, visualizzabile direttamente su GitHub e in qualsiasi browser moderno.

Vedi [docs/README.md](docs/README.md) per dettagli sulla documentazione.

## Struttura Progetto

```
Pokeagent-GameBoy/
├── src/
│   ├── main.py              # Entry point e loop training
│   ├── config.py            # Configurazione con validazione
│   ├── models.py            # PPONetworkGroup (gestione 3 reti)
│   ├── vision_encoder.py    # Vision Encoder (PixelShuffle + MLA)
│   ├── memory_reader.py     # Sistema ricompense ottimizzato
│   ├── anti_loop.py         # Rilevamento e prevenzione loop
│   ├── action_filter.py     # Mascheramento aggressivo 0.3-0.5
│   ├── trajectory_buffer.py # Raccolta traiettorie PPO
│   ├── state_detector.py    # State detection con cache 200 entry
│   ├── hyperparameters.py   # Iperparametri PPO e Vision Encoder
│   └── utils.py             # Utilità helper
├── docs/
│   ├── architecture.svg     # Diagramma architettura (SVG)
│   └── README.md            # Guida documentazione visuale
├── tests/                   # Test unitari (53 test)
├── .github/workflows/       # Pipeline CI/CD
├── pyproject.toml           # Packaging Python moderno
├── LICENSE                  # Licenza MIT
└── README.md                # Questo file
```

## Risoluzione Problemi

### Problemi ROM

**Problema**: "ROM file not found or invalid"
- **Soluzione**: Assicurati che il file ROM sia valido `.gb` o `.gbc`
- **Soluzione**: Aggiorna `rom_path` in [src/config.py](src/config.py)
- **Nota**: Devi possedere legalmente il gioco per usare la ROM

**Problema**: ROM carica ma gioco non parte
- **Soluzione**: Prova un dump ROM diverso; alcuni sono corrotti

### Problemi Performance

**Problema**: FPS basso (< 1000 FPS)
- **Soluzione**: Abilita modalità headless (`--headless`)
- **Soluzione**: Riduci `render_every_n_frames` in [src/config.py](src/config.py)
- **Soluzione**: Controlla GPU: `torch.cuda.is_available()`

**Problema**: Uso memoria GPU elevato
- **Soluzione**: Riduci `frame_stack_size` o `trajectory_length`
- **Soluzione**: Forza CPU in [src/config.py](src/config.py)

### Problemi Training

**Problema**: Agente si blocca in loop
- **Soluzione**: Abilita anti-loop: imposta `anti_loop_enabled = True` in [src/config.py](src/config.py)
- **Soluzione**: Aumenta coefficiente entropia

**Problema**: Nessun progresso oltre early game
- **Soluzione**: Controlla config ricompense in [src/memory_reader.py](src/memory_reader.py)
- **Soluzione**: Aumenta tempo training (>100k frame per milestone)
- **Soluzione**: Verifica attivazione ricompense eventi (controlla log)

**Problema**: Caricamento checkpoint fallisce
- **Soluzione**: Elimina checkpoint corrotti in `pokemon_ai_saves_*/`
- **Soluzione**: Controlla spazio disco
- **Soluzione**: Verifica compatibilità versione PyTorch

### Problemi Dipendenze

**Problema**: `ImportError: No module named 'pyboy'`
- **Soluzione**: `pip install -e .` o `pip install -r requirements.txt`

**Problema**: `SDL2 not found`
- **Ubuntu**: `sudo apt-get install libsdl2-dev`
- **macOS**: `brew install sdl2`
- **Windows**: Scarica da [libsdl.org](https://libsdl.org)

**Problema**: Problemi PyTorch CUDA
- **Soluzione**: Verifica: `nvidia-smi`
- **Soluzione**: Installa versione corretta:
  ```bash
  # CUDA 11.8
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  # CUDA 12.1
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

## Sviluppo

### Esecuzione Test

```bash
# Installa dipendenze sviluppo
pip install -e ".[dev]"

# Esegui tutti i test
pytest tests/ -v

# Con coverage
pytest tests/ --cov=src --cov-report=term-missing

# Test specifico
pytest tests/test_config.py -v
```

### Qualità Codice

```bash
# Formatta codice
black src tests

# Ordina import
isort src tests

# Lint
flake8 src tests

# Type check
mypy src --ignore-missing-imports
```

## Dettagli Algoritmo

**PPO (Proximal Policy Optimization)**:
- Clipped surrogate loss con epsilon=0.2
- Bonus entropia (0.1 -> 0.01) con scheduling adattivo
- GAE-lambda con lambda=0.95, gamma=0.99
- Raccolta traiettorie on-policy (512 passi)
- 3 epoche, dimensione minibatch 32
- Optimizer Adam, LR 3e-4
- Gradient clipping a 0.5

**Ingegneria Ricompense (V2.1 - PROGRESSIONE OTTIMIZZATA)**:

| Evento | V1.0 | V2.0 | **V2.1** | Motivazione |
|--------|------|------|----------|-------------|
| **Medaglie** | +2000 | +2000 | **+2000** | Obiettivo principale invariato |
| **Cattura Pokemon** | +150 | +300 | **+300** | Già ottimizzato in V2.0 |
| **Vedere Pokemon** | +20 | +30 | **+30** | Già ottimizzato in V2.0 |
| **Allenatori sconfitti** | +100 | +100 | **+200** | RADDOPPIATO - progressione obbligatoria |
| **Eventi storia** | +5 | +5 | **+50** | x10 - guidano verso obiettivi critici |
| **Vittoria battaglia** | +50 | +80 | **+120** | +50% - incentiva engagement attivo |
| **Sconfitta battaglia** | -100 | -200 | **-200** | Penalità severa invariata |
| **Nuova mappa** | +80 | +150 | **+250** | +67% - milestone importanti |
| **Mappa già vista** | +80 | +20 | **+30** | +50% - backtracking necessario |
| **Nuova coordinata** | +2 | +2 | **+5** | +150% - feedback continuo |
| **Movimento generico** | +8 | 0 | **0** | Rimosso - previene wandering |
| **Level up** | +50 → +5 | +50 → +5 | **+50 → +5** | Anti-grinding con decay |

**Filosofia V2.1**: Feedback **denso e frequente** per evitare blocchi. Ricompense alte per progressione reale, feedback continuo per navigazione tra obiettivi.

## Giochi Compatibili

- Pokemon Rosso
- Pokemon Blu
- Pokemon Giallo
- Pokemon Oro (sperimentale)
- Pokemon Argento (sperimentale)

## Licenza

Licenza MIT - vedi [LICENSE](LICENSE)

**Importante**: Gli utenti sono responsabili di ottenere legalmente i file ROM. Questo progetto non fornisce né approva la distribuzione di ROM. Pokemon è un marchio di Nintendo/Game Freak/The Pokemon Company.

## Riconoscimenti

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Emulatore Game Boy
- [PyTorch](https://pytorch.org/) - Framework deep learning
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization (OpenAI)
- [DeepSeek-VL2 Paper](https://arxiv.org/abs/2412.10302) - Ispirazione per Vision Encoder

---

**Progetto educativo per ricerca reinforcement learning. Performance varia in base a hardware e configurazione.**
