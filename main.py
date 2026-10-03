#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente PPO per Pokemon Rosso")
    parser.add_argument("--mode", choices=["train", "play"], default="train", help="train: addestra con PPO, play: guarda l'agente giocare")
    parser.add_argument("--rom", type=str, default="roms/Pokemon Red.gb", help="ROM (lo stato iniziale e' <rom>.state)")
    parser.add_argument("--run-dir", type=str, default="runs/ppo", help="Cartella per checkpoint e TensorBoard")
    parser.add_argument("--envs", type=int, default=10, help="Partite parallele per il training")
    parser.add_argument("--timesteps", type=int, default=50_000_000, help="Passi di training da eseguire in questa sessione")
    parser.add_argument("--episode-steps", type=int, default=20480, help="Durata massima di una partita in training")
    parser.add_argument("--steps", type=int, default=20480, help="Passi da giocare in modalita' play")
    parser.add_argument("--speed", type=int, default=1, help="Velocita' emulatore in play (0 = massima)")
    args = parser.parse_args()
    for path in (args.rom, args.rom + ".state"):
        if not Path(path).is_file():
            parser.error(f"file non trovato: {path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        if args.mode == "train":
            from agent.train import train
            train(args.rom, args.run_dir, args.envs, args.timesteps, args.episode_steps)
        else:
            from agent.play import play
            play(args.rom, args.run_dir, args.steps, args.speed)
    except KeyboardInterrupt:
        logging.info("Interrotto dall'utente")


if __name__ == "__main__":
    main()
