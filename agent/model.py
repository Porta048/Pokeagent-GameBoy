import logging
import math
import time
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import config as CFG

logger = logging.getLogger("pokeagent.model")


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.activation = nn.ReLU()
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, 84, 84)
            self.feature_dim = self._forward_conv(sample).shape[1]
        self._init_weights()

    def _init_weights(self):
        for module in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self._forward_conv(x)


class BCPolicy(nn.Module):
    def __init__(self, n_actions: int = 9, input_channels: int = 1):
        super().__init__()
        self.n_actions = n_actions
        self.feature_extractor = CNNFeatureExtractor(input_channels)
        feature_dim = self.feature_extractor.feature_dim
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.policy.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)
        return self.policy(features)

    def predict(self, x: torch.Tensor, temperature: float = 1.0) -> int:
        with torch.no_grad():
            logits = self.forward(x)
            if temperature <= 0:
                return torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).item()


class BCAgent:
    def __init__(self, n_actions: int = 9, device: torch.device = None, learning_rate: float = None):
        self.device = device or CFG.DEVICE
        self.n_actions = n_actions
        self.lr = learning_rate or CFG.BC_LEARNING_RATE
        self.policy = BCPolicy(n_actions=n_actions).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.total_steps = 0
        logger.info("BC Agent initialized: device=%s, lr=%.2e", self.device, self.lr)

    def select_action(self, obs: np.ndarray, temperature: float = 0.5) -> int:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        return self.policy.predict(obs_tensor, temperature)

    def train_step(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> Dict[str, float]:
        self.policy.train()
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        logits = self.policy(obs_batch)
        loss = self.criterion(logits, action_batch)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.total_steps += 1
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == action_batch).float().mean().item()
        return {"loss": loss.item(), "accuracy": accuracy}

    def validate(self, dataloader) -> Dict[str, float]:
        self.policy.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for obs_batch, action_batch in dataloader:
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                logits = self.policy(obs_batch)
                loss = self.criterion(logits, action_batch)
                total_loss += loss.item() * len(action_batch)
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == action_batch).sum().item()
                total_samples += len(action_batch)
        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        self.scheduler.step(avg_loss)
        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        logger.info("Model loaded from %s (step %d)", path, self.total_steps)


class GameplayDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._load_data()

    def _load_data(self):
        if not self.data_dir.exists():
            logger.warning("Data directory not found: %s", self.data_dir)
            return
        for npz_file in sorted(self.data_dir.glob("*.npz")):
            try:
                data = np.load(npz_file)
                observations = data["observations"]
                actions = data["actions"]
                for i in range(len(actions)):
                    self.samples.append((observations[i], actions[i]))
                logger.info("Loaded %d samples from %s", len(actions), npz_file.name)
            except Exception as e:
                logger.warning("Failed to load %s: %s", npz_file, e)
        logger.info("Total samples loaded: %d", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action = self.samples[idx]
        obs_tensor = torch.from_numpy(obs).float()
        action_tensor = torch.tensor(action, dtype=torch.long)
        return obs_tensor, action_tensor


class GameplayRecorder:
    def __init__(self, save_dir: str, max_samples_per_file: int = 500):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples_per_file
        self.observations = []
        self.actions = []
        self.file_counter = self._get_next_file_counter()
        self.last_save_time = time.time()
        logger.info("Recorder initialized: save_dir=%s", self.save_dir)

    def _get_next_file_counter(self) -> int:
        existing = list(self.save_dir.glob("gameplay_*.npz"))
        if not existing:
            return 0
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split("_")[1])
                numbers.append(num)
            except (ValueError, IndexError):
                pass
        return max(numbers) + 1 if numbers else 0

    def record(self, observation: np.ndarray, action: int):
        self.observations.append(observation.copy())
        self.actions.append(action)
        if len(self.actions) >= self.max_samples:
            self.save()
        elif time.time() - self.last_save_time > 60 and len(self.actions) > 0:
            self.save()

    def save(self):
        if not self.actions:
            return
        filename = self.save_dir / f"gameplay_{self.file_counter:04d}.npz"
        np.savez_compressed(
            filename,
            observations=np.array(self.observations),
            actions=np.array(self.actions),
        )
        logger.info("Saved %d samples to %s", len(self.actions), filename)
        self.file_counter += 1
        self.observations.clear()
        self.actions.clear()
        self.last_save_time = time.time()

    def get_stats(self) -> Dict[str, int]:
        return {
            "current_buffer": len(self.actions),
            "files_saved": self.file_counter,
            "total_recorded": self.file_counter * self.max_samples + len(self.actions),
        }
