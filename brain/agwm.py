from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from brain.actor_critic import BrainActorCriticPolicy, BrainV1Extractor


@dataclass
class BrainV1Config:
    image_features_dim: int = 128
    body_features_dim: int = 96
    fused_features_dim: int = 256
    use_privileged: bool = False
    actor_layers: tuple[int, ...] = (128, 64)
    critic_layers: tuple[int, ...] = (128, 64)

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["actor_layers"] = list(self.actor_layers)
        data["critic_layers"] = list(self.critic_layers)
        return data


def recurrent_ppo_available() -> bool:
    try:
        import sb3_contrib  # noqa: F401
    except Exception:
        return False
    return True


def make_policy_kwargs(config: BrainV1Config | None = None) -> dict:
    cfg = config or BrainV1Config()
    return {
        "features_extractor_class": BrainV1Extractor,
        "features_extractor_kwargs": {
            "image_features_dim": cfg.image_features_dim,
            "body_features_dim": cfg.body_features_dim,
            "features_dim": cfg.fused_features_dim,
            "use_privileged": cfg.use_privileged,
        },
        "net_arch": {"pi": list(cfg.actor_layers), "vf": list(cfg.critic_layers)},
        "activation_fn": nn.ReLU,
        "normalize_images": True,
    }


def make_brain_ppo(env, config: BrainV1Config | None = None, **ppo_kwargs):
    from stable_baselines3 import PPO

    return PPO(
        BrainActorCriticPolicy,
        env,
        policy_kwargs=make_policy_kwargs(config),
        **ppo_kwargs,
    )
