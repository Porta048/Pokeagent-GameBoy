import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import orjson
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from agent.env import PokemonRedEnv

logger = logging.getLogger("pokeagent.train")

INFO_KEYS = ("score", "tiles", "levels", "badges", "events", "dex")


class GameStatsCallback(BaseCallback):
    # Media delle statistiche di gioco delle partite a fine rollout, per TensorBoard
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals["infos"]
        for key in INFO_KEYS:
            self.logger.record(f"game/{key}_mean", float(np.mean([i[key] for i in infos])))
            self.logger.record(f"game/{key}_max", float(np.max([i[key] for i in infos])))


def load_checkpoint(model: PPO, path: Path) -> None:
    # PPO.load passa a torch.load lo stream dello zip, che torch 2.11 non legge: carico i pesi dai byte
    with zipfile.ZipFile(path) as z:
        model.policy.load_state_dict(torch.load(io.BytesIO(z.read("policy.pth")), map_location=model.device))
        model.policy.optimizer.load_state_dict(torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location=model.device))
        model.num_timesteps = orjson.loads(z.read("data"))["num_timesteps"]


def train(rom_path: str, run_dir: str, n_envs: int, total_steps: int, max_steps: int) -> None:
    run = Path(run_dir)
    ckpt_dir = run / "checkpoints"
    env = VecMonitor(SubprocVecEnv([lambda: PokemonRedEnv(rom_path, max_steps=max_steps) for _ in range(n_envs)]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=2048,
        batch_size=2048,
        n_epochs=3,
        gamma=0.998,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log=str(run / "tb"),
        device=device,
        verbose=1,
    )
    if checkpoints:
        logger.info("Riprendo da %s", checkpoints[-1])
        load_checkpoint(model, checkpoints[-1])
    logger.info("Training su %s con %d partite parallele", device, n_envs)

    callbacks = [
        CheckpointCallback(save_freq=max(1, 1_000_000 // n_envs), save_path=str(ckpt_dir), name_prefix="ppo"),
        GameStatsCallback(),
    ]
    try:
        model.learn(total_timesteps=total_steps, callback=callbacks, reset_num_timesteps=not checkpoints, tb_log_name="ppo")
    finally:
        model.save(ckpt_dir / "ppo_last")
        env.close()
