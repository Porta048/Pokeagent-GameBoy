#!/usr/bin/env python3
import argparse
import logging

from config import config as CFG


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon Red BC Agent")
    parser.add_argument(
        "--mode",
        choices=["record", "train", "play"],
        default="record",
        help="Mode: record (capture gameplay), train (BC training), play (inference)"
    )
    parser.add_argument(
        "--rom",
        type=str,
        default=CFG.ROM_PATH,
        help="Path to Pokemon Red ROM"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=CFG.RECORDINGS_DIR,
        help="Directory for recordings"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CFG.BC_EPOCHS,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CFG.BC_BATCH_SIZE,
        help="Training batch size"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Steps for play mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=CFG.INFERENCE_TEMPERATURE,
        help="Sampling temperature for play mode"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, CFG.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        if args.mode == "record":
            from agent.trainer import HumanRecorder
            recorder = HumanRecorder(rom_path=args.rom)
            recorder.record_session()
        elif args.mode == "train":
            from agent.trainer import BCTrainer
            trainer = BCTrainer(rom_path=args.rom)
            trainer.train(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            trainer.close()
        elif args.mode == "play":
            from agent.trainer import InferenceRunner
            runner = InferenceRunner(rom_path=args.rom)
            runner.play(num_steps=args.steps, temperature=args.temperature)
            runner.close()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
