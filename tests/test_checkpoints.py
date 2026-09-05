from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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
from train.train_walk_ppo import (
    environment_calibration_snapshot, gait_profile_parameters, make_env,
    reserve_run_directory, resolve_training_contact_threshold,
    resume_gait_contract, validate_resume_calibration, validate_training_args,
)


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
                        backup_ack_timeout=1.0, xml_path=None, gait_profile="legacy")
        validate_training_args(argparse.Namespace(**defaults))
        for change in ({"resume_from": "unpaired.zip"}, {"checkpoint_freq": 0},
                       {"max_wall_seconds": float("nan")}, {"run": "../escape"}):
            with self.assertRaises(ValueError):
                validate_training_args(argparse.Namespace(**(defaults | change)))

    def resume_args(self, model_path, normalizer_path, gait_profile):
        return argparse.Namespace(
            resume_from=str(model_path), resume_vec=str(normalizer_path),
            reset_timesteps=False, run="resume-test", envs=1, steps=32,
            n_steps=8, batch=8, eval_freq=1000, checkpoint_freq=8,
            max_wall_seconds=None, backup_ack_timeout=1.0, xml_path=None,
            gait_profile=gait_profile,
        )

    def test_old_resume_explicitly_infers_legacy_and_rejects_lab(self):
        bundle = self.bundle()
        args = self.resume_args(bundle / "model.zip", bundle / "vecnormalize.pkl", "legacy")
        validate_training_args(args)
        contract = resume_gait_contract(args.resume_from)
        self.assertEqual(contract["gait_profile"], "legacy")
        self.assertTrue(contract["legacy_inferred_from_missing_metadata"])
        args.gait_profile = "lab"
        with self.assertRaisesRegex(ValueError, "Resume gait profile"):
            validate_training_args(args)

    def test_explicit_lab_resume_requires_matching_profile_and_parameters(self):
        bundle = save_checkpoint_bundle(FakeModel(), FakeNormalizer(), self.run, {
            "gait_profile": "lab", "gait_profile_parameters": gait_profile_parameters("lab")
        })
        args = self.resume_args(bundle / "model.zip", bundle / "vecnormalize.pkl", "lab")
        validate_training_args(args)
        self.assertFalse(resume_gait_contract(args.resume_from)["legacy_inferred_from_missing_metadata"])
        args.gait_profile = "legacy"
        with self.assertRaisesRegex(ValueError, "Resume gait profile"):
            validate_training_args(args)

    def test_lab_resume_rejects_changed_registry_parameters(self):
        previous = gait_profile_parameters("lab")
        previous["stance_ratios"][0] -= .01  # Synthetic incompatible fixture.
        bundle = save_checkpoint_bundle(FakeModel(), FakeNormalizer(), self.run, {
            "gait_profile": "lab", "gait_profile_parameters": previous
        })
        args = self.resume_args(bundle / "model.zip", bundle / "vecnormalize.pkl", "lab")
        with self.assertRaisesRegex(ValueError, "Saved gait parameters differ"):
            validate_training_args(args)

    def test_make_env_forwards_selected_profile(self):
        with patch("train.train_walk_ppo.GeckoWalkEnv") as constructor:
            make_env(17, gait_profile="lab", xml_path="candidate.xml")()
            self.assertEqual(constructor.call_args.kwargs["gait_profile"], "lab")
            self.assertEqual(constructor.call_args.kwargs["xml_path"], "candidate.xml")
            self.assertIsNone(constructor.call_args.kwargs["contact_thresh"])

    def test_contact_threshold_default_is_profile_aware(self):
        self.assertEqual(resolve_training_contact_threshold("legacy", None), .0564)
        self.assertIsNone(resolve_training_contact_threshold("lab", None))
        self.assertEqual(resolve_training_contact_threshold("lab", .02), .02)
        with self.assertRaises(ValueError):
            resolve_training_contact_threshold("legacy", float("nan"))

    def test_actual_worker_calibration_is_recorded(self):
        calibration = {"profile": "lab", "effective_contact_threshold_N": .03}
        data = {"reward_calibration": [calibration, dict(calibration)],
                "contact_threshold": [.03, .03],
                "reward_fn": [SimpleNamespace(w={"slow_penalty": 0.}), SimpleNamespace(w={"slow_penalty": 0.})]}
        snapshot = environment_calibration_snapshot(SimpleNamespace(get_attr=lambda name: data[name]))
        self.assertEqual(snapshot["effective_contact_threshold_N"], .03)
        self.assertEqual(snapshot["effective_reward_cfg"], {"slow_penalty": 0.})
        data["contact_threshold"][1] = .04
        with self.assertRaises(ValueError):
            environment_calibration_snapshot(SimpleNamespace(get_attr=lambda name: data[name]))

    def test_lab_resume_calibration_guard(self):
        calibration = {"model_inputs": {"mass_kg": .038}, "reward_overrides": {"slow_penalty": 0.}}
        source = {"gait_profile": "lab", "reward_calibration": calibration,
                  "effective_contact_threshold_N": .03, "xml_sha256": "test_xml_hash",
                  "effective_reward_cfg": {"slow_penalty": 0., "progress": 7.}}
        self.run.mkdir(parents=True)
        atomic_json(self.run / "train_config.json", source)
        args = argparse.Namespace(resume_from=str(self.run / "model.zip"), gait_profile="lab")
        current = json.loads(json.dumps(source))
        current["effective_reward_cfg"]["progress"] = 12.
        changes = validate_resume_calibration(args, current)
        self.assertEqual(changes["reward.progress"], {"saved": 7., "requested_effective": 12.})
        current["effective_contact_threshold_N"] = .04
        with self.assertRaisesRegex(ValueError, "contact threshold changed"):
            validate_resume_calibration(args, current)
        current["effective_contact_threshold_N"] = .03
        current["xml_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "XML differs"):
            validate_resume_calibration(args, current)

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
