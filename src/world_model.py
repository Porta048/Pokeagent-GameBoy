from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
class DreamerV3WorldModelEncoder(nn.Module):
    """
    Enhanced encoder for DreamerV3 world model with improved architecture.
    """
    def __init__(
        self,
        input_channels: int = 4,
        embed_dim: int = 1024,
        depth: int = 4
    ):
        super().__init__()
        layers = []
        in_channels = input_channels
        for i in range(depth):
            out_channels = 32 * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU()
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size after convolutions
        # Each conv layer reduces size by half, so after 4 layers: 144x160 -> 72x80 -> 36x40 -> 18x20 -> 9x10
        self._conv_out_size = out_channels * 9 * 10

        self.projection = nn.Sequential(
            nn.Linear(self._conv_out_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h = self.conv_layers(x)
        h = h.view(batch_size, -1)
        z = self.projection(h)
        return z


class DreamerV3WorldModelDecoder(nn.Module):
    """
    Enhanced decoder for DreamerV3 world model with improved architecture.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        output_channels: int = 1,
        depth: int = 4
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 512 * 9 * 10),  # Start from 9x10 feature map
            nn.LayerNorm(512 * 9 * 10),
            nn.SiLU()
        )

        layers = []
        in_channels = 512
        for i in range(depth):
            out_channels = 32 * (2 ** (depth - 1 - i))
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.GroupNorm(8, out_channels),
                nn.SiLU()
            ])
            in_channels = out_channels

        # Final layer to output the desired number of channels
        layers.append(
            nn.Conv2d(out_channels, output_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.Sigmoid())

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.projection(z)
        h = h.view(-1, 512, 9, 10)  # Reshape to 9x10 feature map
        frame = self.deconv_layers(h)
        return F.interpolate(frame, size=(144, 160), mode='bilinear', align_corners=False)


class DiscreteLatentDynamics(nn.Module):
    """
    Discrete latent dynamics model for DreamerV3 with categorical representations.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        action_dim: int = 9,
        num_categoricals: int = 32,
        num_classes: int = 32,
        hidden_dim: int = 1536,
        gru_layers: int = 2
    ):
        super().__init__()

        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Embedding(action_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64)
        )

        # GRU for dynamics
        self.gru = nn.GRU(
            input_size=embed_dim + 64,  # embedded state + action
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # Post-GRU projection to categorical logits
        self.post_gru = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes)
        )

        # Prior network (for imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes)
        )

        # Initial hidden state
        self.h0 = nn.Parameter(torch.zeros(gru_layers, 1, hidden_dim))

    def _to_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to categorical distribution."""
        logits = logits.view(logits.shape[0], self.num_categoricals, self.num_classes)
        return logits

    def _flatten_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Flatten categorical logits."""
        return logits.view(logits.shape[0], -1)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """Initialize the categorical state."""
        return torch.zeros(batch_size, self.num_categoricals, self.num_classes, device=self.h0.device)

    def forward(
        self,
        posterior: torch.Tensor,
        action: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the dynamics model.

        Args:
            posterior: Current posterior categorical logits (B, num_categoricals, num_classes)
            action: Action taken (B,)
            h: Hidden state for GRU (num_layers, B, hidden_dim)

        Returns:
            posterior: Next posterior logits
            prior: Prior logits (for KL loss)
            h: Next hidden state
        """
        batch_size = posterior.shape[0]

        if h is None:
            h = self.h0.expand(-1, batch_size, -1).contiguous()

        # Flatten posterior and get embedded representation
        flat_posterior = self._flatten_categorical(posterior)

        # Embed action
        a_emb = self.action_embed(action)

        # Concatenate posterior and action embedding
        x = torch.cat([flat_posterior, a_emb], dim=-1).unsqueeze(1)  # Add sequence dim

        # GRU forward pass
        gru_out, h_next = self.gru(x, h)

        # Get posterior (next state) and prior (for imagination)
        posterior_logits = self.post_gru(gru_out.squeeze(1))
        prior_logits = self.prior_net(gru_out.squeeze(1))

        # Reshape to categorical format
        posterior_next = self._to_categorical(posterior_logits)
        prior = self._to_categorical(prior_logits)

        return posterior_next, prior, h_next

    def initial(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial posterior and prior."""
        posterior = self.initial_state(batch_size)
        prior = self.initial_state(batch_size)
        return posterior, prior


class DreamerV3RewardPredictor(nn.Module):
    """
    Enhanced reward predictor for DreamerV3 with symlog transformation.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 1536,
        layers: int = 4
    ):
        super().__init__()

        layers_list = [nn.Linear(embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]

        for _ in range(layers - 1):
            layers_list.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ])

        layers_list.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers_list)

    def symlog(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric logarithm transformation."""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def symexp(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric exponential transformation."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for reward prediction."""
        raw_reward = self.mlp(z).squeeze(-1)
        return self.symexp(raw_reward)


class DreamerV3ContinuePredictor(nn.Module):
    """
    Continue predictor for DreamerV3 (discount factor prediction).
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 1536,
        layers: int = 2
    ):
        super().__init__()

        layers_list = [nn.Linear(embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]

        for _ in range(layers - 1):
            layers_list.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ])

        layers_list.extend([
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ])

        self.mlp = nn.Sequential(*layers_list)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for continue prediction."""
        return self.mlp(z).squeeze(-1)


class DreamerV3WorldModel(nn.Module):
    """
    Complete DreamerV3 World Model with discrete latent representations.
    """
    def __init__(
        self,
        input_channels: int = 4,
        embed_dim: int = 1024,
        action_dim: int = 9,
        num_categoricals: int = 32,
        num_classes: int = 32,
        hidden_dim: int = 1536
    ):
        super().__init__()

        self.encoder = DreamerV3WorldModelEncoder(input_channels, embed_dim)
        self.decoder = DreamerV3WorldModelDecoder(embed_dim, output_channels=1)
        self.dynamics = DiscreteLatentDynamics(
            embed_dim, action_dim, num_categoricals, num_classes, hidden_dim
        )
        self.reward_predictor = DreamerV3RewardPredictor(embed_dim, hidden_dim)
        self.continue_predictor = DreamerV3ContinuePredictor(embed_dim, hidden_dim)

        self.embed_dim = embed_dim
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes

    def _to_dense(self, categorical_logits: torch.Tensor) -> torch.Tensor:
        """Convert categorical logits to dense representation."""
        # Apply softmax to get probabilities
        probs = F.softmax(categorical_logits, dim=-1)
        # Sample from the distribution
        dist = OneHotCategorical(logits=categorical_logits.view(-1, self.num_classes))
        samples = dist.sample().view(-1, self.num_categoricals, self.num_classes)
        # Use straight-through estimator: probs + detach(samples - probs)
        samples = probs + (samples - probs).detach()
        # Flatten for dense representation
        return samples.view(samples.shape[0], -1)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to dense representation."""
        return self.encoder(frames)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode dense representation to frames."""
        return self.decoder(z)

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get initial states for dynamics."""
        posterior, prior = self.dynamics.initial(batch_size)
        h = self.dynamics.h0.expand(-1, batch_size, -1).contiguous()
        return posterior, prior, h

    def update_state(
        self,
        posterior: torch.Tensor,
        action: torch.Tensor,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update state with action."""
        posterior_next, prior, h_next = self.dynamics(posterior, action, h)
        return posterior_next, prior, h_next

    def predict_reward(self, posterior: torch.Tensor) -> torch.Tensor:
        """Predict reward from posterior state."""
        z_dense = self._to_dense(posterior)
        return self.reward_predictor(z_dense)

    def predict_continue(self, posterior: torch.Tensor) -> torch.Tensor:
        """Predict continue from posterior state."""
        z_dense = self._to_dense(posterior)
        return self.continue_predictor(z_dense)

    def imagine(
        self,
        posterior: torch.Tensor,
        actions: torch.Tensor,
        h: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Imagination rollout for planning."""
        B, T = actions.shape
        post_seq = []
        rew_seq = []
        cont_seq = []

        post = posterior
        h_state = h

        for t in range(T):
            post, prior, h_state = self.update_state(post, actions[:, t], h_state)
            reward = self.predict_reward(post)
            continue_prob = self.predict_continue(post)

            post_seq.append(post)
            rew_seq.append(reward)
            cont_seq.append(continue_prob)

        return {
            'posterior': torch.stack(post_seq, dim=1),
            'rewards': torch.stack(rew_seq, dim=1),
            'continues': torch.stack(cont_seq, dim=1)
        }

    def compute_loss(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_frames: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all world model losses."""
        B, T = actions.shape

        # Encode all frames
        all_frames = torch.cat([frames, next_frames.unsqueeze(1)], dim=1)  # (B, T+1, C, H, W)
        all_embeddings = []

        for t in range(T + 1):
            emb = self.encode(all_frames[:, t])
            all_embeddings.append(emb)

        embeddings = torch.stack(all_embeddings[:-1], dim=1)  # (B, T, embed_dim)
        next_embeddings = torch.stack(all_embeddings[1:], dim=1)  # (B, T, embed_dim)

        # Initialize states
        posterior, prior, h = self.initial_state(B)
        post_seq = []
        prior_seq = []
        embed_seq = []

        # Forward pass through dynamics
        for t in range(T):
            # Encode current frame
            current_embed = self.encode(frames[:, t])
            embed_seq.append(current_embed)

            # Update posterior with ground truth
            posterior_next, prior_next, h = self.dynamics(posterior, actions[:, t], h)
            post_seq.append(posterior_next)
            prior_seq.append(prior_next)
            posterior = posterior_next  # Update for next step

        # Stack sequences
        post_seq = torch.stack(post_seq, dim=1)
        prior_seq = torch.stack(prior_seq, dim=1)
        embed_seq = torch.stack(embed_seq, dim=1)

        # Reconstruction loss
        decoded_frames = self.decode(embed_seq)
        recon_loss = F.mse_loss(decoded_frames, frames[:, :, 0:1, :, :])  # Only first channel for simplicity

        # KL divergence loss between posterior and prior
        posterior_flat = post_seq.view(-1, self.num_categoricals * self.num_classes)
        prior_flat = prior_seq.view(-1, self.num_categoricals * self.num_classes)
        posterior_probs = F.softmax(posterior_flat, dim=-1)
        prior_probs = F.softmax(prior_flat, dim=-1)
        kl_loss = (posterior_probs * (torch.log(posterior_probs + 1e-8) - torch.log(prior_probs + 1e-8))).sum(-1).mean()

        # Reward prediction loss
        dense_post_seq = self._to_dense(post_seq)
        pred_rewards = self.reward_predictor(dense_post_seq)
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # Continue prediction loss
        pred_continues = self.continue_predictor(dense_post_seq)
        continue_loss = F.binary_cross_entropy_with_logits(
            pred_continues, 1.0 - dones.float()
        )

        total_loss = (
            0.1 * recon_loss +
            1.0 * kl_loss +
            1.0 * reward_loss +
            0.1 * continue_loss
        )

        return {
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'reward_loss': reward_loss,
            'continue_loss': continue_loss,
            'total': total_loss
        }


class WorldModelTrainer:
    def __init__(
        self,
        world_model: DreamerV3WorldModel,
        lr: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 16,  # Reduced batch size for memory efficiency
        device: torch.device = None
    ):
        self.world_model = world_model
        self.device = device or torch.device('cpu')
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(
            world_model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        self.buffer = {
            'frames': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'next_frames': []
        }
        self.buffer_size = buffer_size
        self.buffer_idx = 0

    def add_experience(
        self,
        frame: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        next_frame: torch.Tensor
    ):
        if len(self.buffer['frames']) < self.buffer_size:
            self.buffer['frames'].append(frame.cpu())
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['dones'].append(done)
            self.buffer['next_frames'].append(next_frame.cpu())
        else:
            idx = self.buffer_idx % self.buffer_size
            self.buffer['frames'][idx] = frame.cpu()
            self.buffer['actions'][idx] = action
            self.buffer['rewards'][idx] = reward
            self.buffer['dones'][idx] = done
            self.buffer['next_frames'][idx] = next_frame.cpu()
        self.buffer_idx += 1

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.buffer['frames']) < self.batch_size:
            return None

        indices = torch.randint(0, len(self.buffer['frames']), (self.batch_size,))
        frames = torch.stack([self.buffer['frames'][i] for i in indices]).to(self.device)
        actions = torch.tensor([self.buffer['actions'][i] for i in indices]).to(self.device)
        rewards = torch.tensor([self.buffer['rewards'][i] for i in indices], dtype=torch.float32).to(self.device)
        dones = torch.tensor([self.buffer['dones'][i] for i in indices], dtype=torch.float32).to(self.device)
        next_frames = torch.stack([self.buffer['next_frames'][i] for i in indices]).to(self.device)

        losses = self.world_model.compute_loss(
            frames, actions, rewards, dones, next_frames
        )
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}

    def save(self, path: str):
        torch.save({
            'world_model': self.world_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def count_world_model_parameters(model: DreamerV3WorldModel) -> Dict[str, int]:
    return {
        'encoder': sum(p.numel() for p in model.encoder.parameters()),
        'decoder': sum(p.numel() for p in model.decoder.parameters()),
        'dynamics': sum(p.numel() for p in model.dynamics.parameters()),
        'reward_predictor': sum(p.numel() for p in model.reward_predictor.parameters()),
        'continue_predictor': sum(p.numel() for p in model.continue_predictor.parameters()),
        'total': sum(p.numel() for p in model.parameters())
    }
