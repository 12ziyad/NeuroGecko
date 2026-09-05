"""Verify a trusted off-host PPO bundle by resuming briefly on the local CPU.

This loads user-owned PPO/VecNormalize artifacts, which are pickle-based: hashes
detect corruption, not malicious pickle contents. No source files are modified,
no replacement production weights are saved, and no remote commands are used.
The XML must match the training configuration's exact SHA256. Current code is
used with the saved gait contract; this is not bit-exact simulator/RNG replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from common.checkpoints import atomic_json, sha256_file, verify_checkpoint_bundle
from train.train_walk_ppo import gait_profile_parameters, make_env, resume_gait_contract


def parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.policy.state_dict().items()):
        array = parameter.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class FiniteResumeCheck(BaseCallback):
    def __init__(self, started: float, max_wall_seconds: float):
        super().__init__()
        self.started = started
        self.max_wall_seconds = max_wall_seconds
        self.checked_steps = 0

    def _on_step(self):
        if time.monotonic() - self.started >= self.max_wall_seconds:
            raise TimeoutError("Local resume verification reached its wall-time limit")
        for key in ("new_obs", "actions", "rewards"):
            if not np.isfinite(np.asarray(self.locals[key])).all():
                raise FloatingPointError(f"Non-finite {key} during resume")
        self.checked_steps += 1
        return True


def verify_resume(bundle: Path, xml: Path, additional_steps: int = 64,
                  max_wall_seconds: float = 120.0) -> dict:
    if not 1 <= additional_steps <= 256:
        raise ValueError("Verification is bounded to 1..256 additional steps")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be finite and positive")
    started = time.monotonic()
    bundle, xml = bundle.resolve(strict=True), xml.resolve(strict=True)
    manifest = verify_checkpoint_bundle(bundle)
    config = json.loads((bundle / "train_config.json").read_text(encoding="utf-8"))
    xml_hash = sha256_file(xml)
    if config.get("xml_sha256") != xml_hash:
        raise ValueError("XML SHA256 does not match the training configuration; use its archived MJCF")
    if config.get("recurrent", False) or config.get("effective_training", {}).get("algorithm", "PPO") != "PPO":
        raise ValueError("This bounded verifier currently supports non-recurrent PPO only")
    gait = resume_gait_contract(bundle / "model.zip")
    current_parameters = gait_profile_parameters(gait["gait_profile"])
    if gait["parameters"] is not None and gait["parameters"] != current_parameters:
        raise ValueError("Current gait registry does not match the saved profile")
    if gait["gait_profile"] == "lab" and gait["parameters"] is None:
        raise ValueError("Lab resume cannot guess missing saved timing parameters")
    source_hashes = {name: sha256_file(bundle / name) for name in (*manifest["files"], "manifest.json")}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vector = None
    result = {
        "kind": "offhost_checkpoint_resume_smoke", "complete": False,
        "bundle": str(bundle), "source_manifest_sha256": source_hashes["manifest.json"],
        "source_file_sha256": {name: source_hashes[name] for name in manifest["files"]},
        "xml_path": str(xml), "xml_sha256": xml_hash,
        "gait_profile": gait["gait_profile"],
        "legacy_gait_inferred_from_missing_metadata": gait["legacy_inferred_from_missing_metadata"],
        "requested_additional_steps": additional_steps,
        "source_num_envs": config.get("effective_training", {}).get("num_envs", config.get("envs")),
        "verification_num_envs": 1, "verification_device": "cpu",
        "python": platform.python_version(), "torch": torch.__version__,
        "caveat": "Resume plumbing only; not gait quality, exact RNG continuation, or training convergence",
    }
    try:
        vector = DummyVecEnv([make_env(
            seed=int(config.get("seed", 0)),
            control_mode=config.get("control_mode", "raw"),
            residual_scale=float(config.get("residual_scale", .25)),
            contact_thresh=config.get("contact_thresh", .0564),
            front_stance_press=float(config.get("front_stance_press", .40)),
            front_swing_lift=float(config.get("front_swing_lift", .40)),
            reward_cfg=config.get("reward_cfg"), xml_path=str(xml),
            gait_profile=gait["gait_profile"],
        )])
        vector = VecNormalize.load(str(bundle / "vecnormalize.pkl"), vector)
        saved_mean, saved_variance = vector.obs_rms.mean.copy(), vector.obs_rms.var.copy()
        model = PPO.load(str(bundle / "model.zip"), env=vector, device="cpu")
        # Disable the source machine's absolute TensorBoard destination. This
        # smoke has no logger/checkpoint output besides the evidence JSON.
        model.tensorboard_log = None
        expected = config.get("effective_training", {})
        observation_shape = list(vector.observation_space.shape)
        action_shape = list(vector.action_space.shape)
        if observation_shape != list(model.observation_space.shape) or action_shape != list(model.action_space.shape):
            raise ValueError("Loaded model and environment spaces do not match")
        if expected.get("observation_shape", observation_shape) != observation_shape:
            raise ValueError("Observation shape differs from the saved training configuration")
        if expected.get("action_shape", action_shape) != action_shape:
            raise ValueError("Action shape differs from the saved training configuration")
        if list(saved_mean.shape) != observation_shape or list(saved_variance.shape) != observation_shape:
            raise ValueError("Normalizer shape does not match the observation")
        if not np.isfinite(saved_mean).all() or not np.isfinite(saved_variance).all():
            raise FloatingPointError("Loaded normalization statistics are non-finite")
        if model.num_timesteps != manifest["num_timesteps"]:
            raise ValueError("Model timestep counter differs from the manifest")
        if additional_steps % model.n_steps:
            raise ValueError(f"Choose an exact multiple of saved n_steps={model.n_steps}; PPO otherwise rounds up")
        vector.training = False
        vector.norm_reward = False
        vector.seed(int(config.get("seed", 0)))
        observation = vector.reset()
        action, _ = model.predict(observation, deterministic=True)
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise FloatingPointError("Initial normalized observation or prediction is non-finite")
        np.testing.assert_array_equal(vector.obs_rms.mean, saved_mean)
        np.testing.assert_array_equal(vector.obs_rms.var, saved_variance)
        before_step, before_updates = int(model.num_timesteps), int(model._n_updates)
        before_parameters = parameter_hash(model)
        vector.training = True
        vector.norm_reward = True
        callback = FiniteResumeCheck(started, max_wall_seconds)
        model.learn(total_timesteps=additional_steps, reset_num_timesteps=False, callback=callback)
        if int(model.num_timesteps) - before_step != additional_steps:
            raise RuntimeError("Resume completed a different number of steps than requested")
        if model._n_updates <= before_updates:
            raise RuntimeError("Resume did not execute any PPO optimizer updates")
        if not all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
            raise FloatingPointError("Model parameters are non-finite after learning")
        after_parameters = parameter_hash(model)
        if before_parameters == after_parameters:
            raise RuntimeError("PPO updates left every policy parameter unchanged")
        result.update({
            "complete": True, "observation_shape": observation_shape, "action_shape": action_shape,
            "normalizer_stats_loaded_unchanged_before_learning": True,
            "initial_prediction_finite": True, "all_rollout_values_finite": True,
            "parameters_finite_after_learning": True, "policy_parameters_changed": True,
            "timesteps_before": before_step, "timesteps_after": int(model.num_timesteps),
            "optimizer_updates_before": before_updates, "optimizer_updates_after": int(model._n_updates),
            "checked_vector_steps": callback.checked_steps,
            "policy_parameter_sha256_before": before_parameters,
            "policy_parameter_sha256_after": after_parameters,
            "effective_n_steps": model.n_steps, "effective_batch_size": model.batch_size,
            "effective_n_epochs": model.n_epochs,
        })
    finally:
        if vector is not None:
            vector.close()
        torch.set_num_threads(old_threads)
    verify_checkpoint_bundle(bundle)
    if any(sha256_file(bundle / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("Source checkpoint files changed during read-only verification")
    result["source_bundle_unchanged"] = True
    result["elapsed_wall_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int, default=64)
    parser.add_argument("--max-wall-seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_resume(args.bundle, args.xml, args.additional_steps, args.max_wall_seconds)
    except Exception as exc:
        atomic_json(args.output, {"kind": "offhost_checkpoint_resume_smoke", "complete": False,
                                  "error": type(exc).__name__ + ": " + str(exc)})
        raise
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
