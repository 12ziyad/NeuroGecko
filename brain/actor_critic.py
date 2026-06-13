from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from brain.anatomy_graph import AnatomyGraphEncoder
from brain.vision_encoder import VisionEncoder


def _flat_dim(space: spaces.Space) -> int:
    shape = getattr(space, "shape", None)
    if shape is None:
        raise ValueError(f"Expected flat Box space, got {space}")
    return int(np.prod(shape))


class BrainV1Extractor(BaseFeaturesExtractor):
    """Fuses vision, body state, drives, and previous high-level action."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_features_dim: int = 128,
        body_features_dim: int = 96,
        features_dim: int = 256,
        use_privileged: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("BrainV1Extractor requires a Dict observation space")

        self.use_privileged = bool(use_privileged)
        self.vision = VisionEncoder(out_dim=image_features_dim)

        body_keys = ["proprio", "drives", "prev_action"]
        if self.use_privileged:
            body_keys.append("privileged")
        self.body_keys = tuple(body_keys)
        body_input_dim = sum(_flat_dim(observation_space.spaces[key]) for key in self.body_keys)
        self.body = AnatomyGraphEncoder(body_input_dim, out_dim=body_features_dim)

        self.fuse = nn.Sequential(
            nn.Linear(image_features_dim + body_features_dim, features_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(features_dim),
        )
        self._features_dim = int(features_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image_features = self.vision(observations["image"])
        body_parts = [
            torch.flatten(observations[key].float(), start_dim=1)
            for key in self.body_keys
        ]
        body_features = self.body(torch.cat(body_parts, dim=1))
        return self.fuse(torch.cat([image_features, body_features], dim=1))


class BrainActorCriticPolicy(MultiInputActorCriticPolicy):
    """SB3 PPO policy constrained to the 4D brain action channel."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        if tuple(action_space.shape) != (4,):
            raise ValueError(
                "BrainActorCriticPolicy may only control the 4D brain action "
                "[target_dir_x, target_dir_y, target_distance, engage]"
            )
        kwargs.setdefault("features_extractor_class", BrainV1Extractor)
        kwargs.setdefault("features_extractor_kwargs", {})
        kwargs.setdefault("net_arch", {"pi": [128, 64], "vf": [128, 64]})
        kwargs.setdefault("activation_fn", nn.ReLU)
        kwargs.setdefault("normalize_images", True)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
