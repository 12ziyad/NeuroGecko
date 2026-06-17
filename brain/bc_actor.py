from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from brain.actor_critic import BrainV1Extractor


class BrainBCActor(nn.Module):
    """Feedforward behavior cloning actor using BrainV1Extractor."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_features_dim: int = 128,
        body_features_dim: int = 96,
        fused_features_dim: int = 256,
        use_privileged: bool = True,
        action_dim: int = 4,
    ):
        super().__init__()
        self.extractor = BrainV1Extractor(
            observation_space,
            image_features_dim=image_features_dim,
            body_features_dim=body_features_dim,
            features_dim=fused_features_dim,
            use_privileged=use_privileged,
        )
        self.action_head = nn.Sequential(
            nn.Linear(fused_features_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.extractor(obs)
        return self.action_head(features)

    def predict(self, obs_np: dict[str, np.ndarray], device: str = "cpu") -> np.ndarray:
        """Single-step inference: numpy obs dict -> numpy 4D action."""
        self.eval()
        with torch.no_grad():
            tensors = {
                key: torch.from_numpy(np.asarray(val)[None]).to(device)
                for key, val in obs_np.items()
            }
            action = self(tensors).squeeze(0)
        return action.cpu().numpy()


def build_obs_space(proprio_dim: int) -> spaces.Dict:
    """Reconstruct GeckoBrainEnv observation space from proprio dimension."""
    return spaces.Dict({
        "image": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
        "proprio": spaces.Box(-np.inf, np.inf, shape=(proprio_dim,), dtype=np.float32),
        "drives": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
        "prev_action": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        "privileged": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
    })
