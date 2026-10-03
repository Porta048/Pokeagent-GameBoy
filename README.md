# Pokeagent GameBoy

Agente che impara a giocare a Pokemon Rosso (versione italiana) con Reinforcement Learning (PPO).

## Componenti

| File | Cosa fa |
|---|---|
| `agent/game_state.py` | Legge dalla RAM mappa, posizione, squadra, medaglie, Pokedex ed eventi |
| `agent/env.py` | Ambiente Gymnasium: 6 tasti, osserva schermo + caselle visitate + statistiche, premia esplorazione, livelli, medaglie, eventi e cure |
| `agent/train.py` | Training PPO su GPU con partite parallele, checkpoint e statistiche TensorBoard |
| `agent/play.py` | Mostra a schermo l'agente che gioca con l'ultimo checkpoint |

## Uso

Richiede `roms/Pokemon Red.gb` e lo stato iniziale `roms/Pokemon Red.gb.state`.

```bash
python main.py --mode train
tensorboard --logdir runs/ppo/tb
python main.py --mode play
```
