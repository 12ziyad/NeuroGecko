from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from common.checkpoints import (
    CheckpointBundleCallback, atomic_json, export_legacy_final,
    mirror_checkpoint_bundle, save_checkpoint_bundle, sha256_file,
    verify_checkpoint_bundle, wait_for_backup_receipt,
)
from train.train_walk_ppo import reserve_run_directory, validate_training_args


class FakeModel:
    num_timesteps = 100_000

    def save(self, path):
        Path(path).write_bytes(b"test model bytes")

    def get_vec_normalize_env(self):
        return FakeNormalizer()


class FakeNormalizer:
    def save(self, path):
        Path(path).write_bytes(b"paired test normalization bytes")


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "models" / "test-run"

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self, final=False):
        return save_checkpoint_bundle(FakeModel(), FakeNormalizer(), self.run, {"fixture": True}, final=final)

    def test_manifest_pair_hashes_and_atomic_pointer(self):
        bundle = self.bundle()
        manifest = verify_checkpoint_bundle(bundle)
        self.assertEqual(manifest["num_timesteps"], 100_000)
        self.assertEqual(set(manifest["files"]), {"model.zip", "vecnormalize.pkl", "train_config.json"})
        self.assertEqual(manifest["files"]["model.zip"]["bytes"], len(b"test model bytes"))
        pointer = json.loads((self.run / "latest_checkpoint.json").read_text())
        self.assertEqual(pointer["bundle"], "checkpoints/step-100000")
        self.assertEqual(pointer["manifest_sha256"], sha256_file(bundle / "manifest.json"))
        self.assertFalse(list(bundle.parent.glob(".*incomplete*")))

    def test_existing_checkpoint_cannot_be_overwritten(self):
        bundle = self.bundle()
        original = sha256_file(bundle / "manifest.json")
        with self.assertRaises(FileExistsError):
            self.bundle()
        self.assertEqual(sha256_file(bundle / "manifest.json"), original)

    def test_missing_normalizer_fails_closed(self):
        with self.assertRaises(ValueError):
            save_checkpoint_bundle(FakeModel(), None, self.run, {})

    def test_corrupt_bundle_rejected(self):
        bundle = self.bundle()
        (bundle / "model.zip").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_checkpoint_bundle(bundle)

    def test_unexpected_manifest_filename_rejected(self):
        bundle = self.bundle()
        manifest = verify_checkpoint_bundle(bundle)
        manifest["files"]["../escape"] = manifest["files"].pop("model.zip")
        atomic_json(bundle / "manifest.json", manifest)
        with self.assertRaises(ValueError):
            verify_checkpoint_bundle(bundle)

    def test_failed_save_never_publishes_a_bundle(self):
        class FailingModel(FakeModel):
            def save(self, path):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            save_checkpoint_bundle(FailingModel(), FakeNormalizer(), self.run, {})
        self.assertFalse(list((self.run / "checkpoints").glob("step-*")))
        self.assertFalse((self.run / "latest_checkpoint.json").exists())

    def test_mirror_is_verified_and_idempotent(self):
        bundle = self.bundle()
        mirrored = mirror_checkpoint_bundle(bundle, self.root / "mirror")
        self.assertEqual(sha256_file(mirrored / "manifest.json"), sha256_file(bundle / "manifest.json"))
        self.assertEqual(mirror_checkpoint_bundle(bundle, self.root / "mirror"), mirrored)

    def test_acknowledgement_requires_exact_manifest_hash(self):
        bundle = self.bundle()
        atomic_json(bundle / "backup_receipt.json", {"verified": True, "manifest_sha256": "wrong", "destination": "test-laptop"})
        with self.assertRaises(TimeoutError):
            wait_for_backup_receipt(bundle, .01)
        atomic_json(bundle / "backup_receipt.json", {
            "verified": True, "manifest_sha256": sha256_file(bundle / "manifest.json"), "destination": "test-laptop"
        })
        self.assertEqual(wait_for_backup_receipt(bundle, .1)["destination"], "test-laptop")

    def test_final_bundle_and_legacy_aliases_are_verified(self):
        self.bundle()
        final = self.bundle(final=True)
        self.assertEqual(final.name, "step-100000-final")
        export_legacy_final(final, self.run)
        self.assertEqual(sha256_file(self.run / "final.zip"), sha256_file(final / "model.zip"))
        self.assertTrue((self.run / "final_manifest.json").exists())

    def test_run_directory_refuses_reuse(self):
        reserve_run_directory(self.root / "models", "new")
        with self.assertRaises(FileExistsError):
            reserve_run_directory(self.root / "models", "new")

    def test_training_argument_guards(self):
        defaults = dict(resume_from=None, resume_vec=None, reset_timesteps=False,
                        run="safe", envs=1, steps=32, n_steps=8, batch=8,
                        eval_freq=1000, checkpoint_freq=8, max_wall_seconds=None,
                        backup_ack_timeout=1.0, xml_path=None)
        validate_training_args(argparse.Namespace(**defaults))
        for change in ({"resume_from": "unpaired.zip"}, {"checkpoint_freq": 0},
                       {"max_wall_seconds": float("nan")}, {"run": "../escape"}):
            with self.assertRaises(ValueError):
                validate_training_args(argparse.Namespace(**(defaults | change)))

    def test_callback_counts_aggregate_steps_not_calls(self):
        model = FakeModel()
        model.num_timesteps = 0
        callback = CheckpointBundleCallback(self.run, {}, checkpoint_freq=10)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        # Synthetic four-env callbacks cross 10 aggregate steps at step12.
        for step in (4, 8):
            model.num_timesteps = step
            self.assertTrue(callback.on_step())
        self.assertFalse((self.run / "checkpoints/step-8").exists())
        model.num_timesteps = 12
        callback.on_step()
        self.assertTrue((self.run / "checkpoints/step-12/manifest.json").exists())

    def test_acknowledgement_is_a_preflight_barrier(self):
        model = FakeModel()
        model.num_timesteps = 0
        callback = CheckpointBundleCallback(self.run, {}, require_backup_ack=True, backup_ack_timeout=.01)
        callback.init_callback(model)
        with self.assertRaises(TimeoutError):
            callback.on_training_start({}, {})
        self.assertEqual(model.num_timesteps, 0)
        self.assertTrue((self.run / "checkpoints/step-0/manifest.json").exists())

    def test_wall_limit_includes_initial_backup_wait_budget(self):
        model = FakeModel()
        model.num_timesteps = 0
        callback = CheckpointBundleCallback(self.run, {}, max_wall_seconds=5.0,
                                            require_backup_ack=True, backup_ack_timeout=120.0)
        callback.init_callback(model)
        with patch("common.checkpoints.time.monotonic", side_effect=[10.0, 12.0]):
            with patch("common.checkpoints.wait_for_backup_receipt") as wait:
                callback.on_training_start({}, {})
                self.assertEqual(wait.call_args.args[1], 3.0)
        model.num_timesteps = 4
        with patch("common.checkpoints.time.monotonic", return_value=16.0):
            self.assertFalse(callback.on_step())
        self.assertTrue(callback.stopped_for_wall_limit)


class PPOResumeSmokeTests(unittest.TestCase):
    def test_tiny_ppo_pair_resumes_with_matching_actions_and_statistics(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        env = VecNormalize(DummyVecEnv([lambda: gym.make("CartPole-v1")]))
        restored = None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary) / "tiny"
                model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
                            policy_kwargs={"net_arch": [8]}, seed=17, device="cpu")
                callback = CheckpointBundleCallback(run, {"fixture": "CartPole smoke; not gecko training"}, checkpoint_freq=8)
                model.learn(total_timesteps=32, callback=callback)
                bundle = callback.last_bundle
                self.assertEqual(bundle.name, "step-32-final")
                self.assertEqual(verify_checkpoint_bundle(bundle)["num_timesteps"], 32)
                restored = VecNormalize.load(str(bundle / "vecnormalize.pkl"), DummyVecEnv([lambda: gym.make("CartPole-v1")]))
                loaded = PPO.load(str(bundle / "model.zip"), env=restored, device="cpu")
                np.testing.assert_array_equal(env.obs_rms.mean, restored.obs_rms.mean)
                np.testing.assert_array_equal(env.obs_rms.var, restored.obs_rms.var)
                probe = np.array([[.1, -.2, .03, .04]], dtype=np.float32)
                np.testing.assert_array_equal(model.predict(env.normalize_obs(probe.copy()), deterministic=True)[0],
                                              loaded.predict(restored.normalize_obs(probe.copy()), deterministic=True)[0])
                for name, parameter in model.policy.state_dict().items():
                    torch.testing.assert_close(parameter, loaded.policy.state_dict()[name], rtol=0, atol=0)
                loaded.learn(total_timesteps=16, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps, 48)
        finally:
            env.close()
            if restored is not None:
                restored.close()
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
