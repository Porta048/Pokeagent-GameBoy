import logging
from pathlib import Path

from stable_baselines3 import PPO

from agent.env import PokemonRedEnv
from agent.train import load_checkpoint

logger = logging.getLogger("pokeagent.play")


def play(rom_path: str, run_dir: str, steps: int, speed: int) -> None:
    checkpoints = sorted((Path(run_dir) / "checkpoints").glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        logger.error("Nessun checkpoint in %s: lancia prima --mode train", run_dir)
        return
    env = PokemonRedEnv(rom_path, max_steps=steps, headless=False)
    env.pyboy.set_emulation_speed(speed)
    model = PPO("MultiInputPolicy", env, device="cpu")
    load_checkpoint(model, checkpoints[-1])
    logger.info("Gioco con %s (%d passi di training)", checkpoints[-1].name, model.num_timesteps)

    obs, info = env.reset()
    truncated = False
    try:
        while not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, _, truncated, info = env.step(int(action))
            if env.steps % 500 == 0:
                logger.info("Passo %d: %s", env.steps, info)
    finally:
        logger.info("Fine partita: %s", info)
        env.close()
