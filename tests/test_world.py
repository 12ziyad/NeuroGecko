"""Contracts for the no-cheat world: the animal, the cheat, and the prey.

Three things must hold and none of them is obvious from reading the code:

1. Generating a world must not change the animal. The lab body's SHA256 is what
   the training evidence contract pins, so a world that quietly alters a mass or
   a friction would invalidate every committed measurement naming that hash.
2. Removing the privileged target must remove it everywhere, not just from the
   observation. The lab controller is separately steered by the same bearing.
3. Prey must be genuinely evasive. Prey that cannot escape is a moving target,
   not a hunt, and would make a capture rate meaningless.
"""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from envs.gecko_walk_env import GeckoWalkEnv
from envs.prey import FleeingPrey, PreyParameters
from utils.build_world import assert_body_unchanged, build

REPO = Path(__file__).resolve().parents[1]
BODY = REPO / "morphology" / "gecko_body_lab_v2.xml"
WORLD = REPO / "morphology" / "gecko_world_v1.xml"


def parameters(**overrides):
    base = dict(escape_speed_m_s=.118, flee_radius_m=.075, escape_latency_s=.085,
                capture_distance_m=.0407, radius_m=.009, arena_radius_m=.40)
    base.update(overrides)
    return PreyParameters(**base)


class WorldGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BODY.read_text(encoding="utf-8")

    def test_retexturing_and_narrowing_leave_the_animal_identical(self):
        world, edits = build(self.source, texrepeat=50, fovy=70)
        self.assertEqual(assert_body_unchanged(self.source, world), [])
        self.assertEqual({e["attribute"] for e in edits}, {"texrepeat", "fovy"})

    def test_adding_prey_leaves_the_animal_identical(self):
        world, _ = build(self.source, prey_radius=.009)
        self.assertEqual(assert_body_unchanged(self.source, world), ["prey"])

    def test_prey_is_mocap_so_it_never_enters_the_state_vector(self):
        world, _ = build(self.source, prey_radius=.009)
        path = REPO / ".test_world_mocap.xml"
        try:
            path.write_text(world, encoding="utf-8")
            body = mujoco.MjModel.from_xml_path(str(BODY))
            with_prey = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual((body.nq, body.nv, body.nu), (with_prey.nq, with_prey.nv, with_prey.nu))
            index = mujoco.mj_name2id(with_prey, mujoco.mjtObj.mjOBJ_BODY, "prey")
            self.assertGreaterEqual(with_prey.body_mocapid[index], 0)
        finally:
            path.unlink(missing_ok=True)

    def test_a_world_that_changed_the_animal_is_caught(self):
        world, _ = build(self.source, fovy=70)
        # Corrupt the generated world directly: a changed friction must be caught.
        broken = world.replace('friction="0.9 0.005 0.0001"', 'friction="0.4 0.005 0.0001"')
        self.assertNotEqual(broken, world, "fixture must actually change something")
        with self.assertRaises(ValueError):
            assert_body_unchanged(self.source, broken)

    def test_a_world_with_no_edits_is_refused(self):
        with self.assertRaisesRegex(ValueError, "No edits"):
            build(self.source)

    def test_the_committed_world_matches_its_manifest(self):
        if not WORLD.exists():
            self.skipTest("no generated world committed")
        manifest = json.loads(WORLD.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        import hashlib
        self.assertEqual(manifest["output_sha256"],
                         hashlib.sha256(WORLD.read_bytes()).hexdigest(),
                         "the committed world was hand-edited; regenerate it")
        self.assertEqual(manifest["source_sha256"],
                         hashlib.sha256(BODY.read_bytes()).hexdigest(),
                         "the world was generated from a different body than the one committed")


class PrivilegedTarget(unittest.TestCase):
    def env(self, privileged):
        return GeckoWalkEnv(xml_path=str(BODY), control_mode="cpg_residual", gait_profile="lab",
                            hind_stance_compensation="all", privileged_target=privileged)

    def test_the_cheat_is_five_observations_and_removing_it_shrinks_the_space(self):
        rich, plain = self.env(True), self.env(False)
        try:
            self.assertEqual(rich.observation_space.shape[0] - plain.observation_space.shape[0], 5)
            self.assertIn("privileged_target", dict(rich.observation_layout["blocks"]))
            self.assertNotIn("privileged_target", dict(plain.observation_layout["blocks"]))
        finally:
            rich.close(); plain.close()

    def test_the_layout_total_matches_the_real_observation(self):
        for privileged in (True, False):
            env = self.env(privileged)
            try:
                self.assertEqual(env.observation_layout["total"], env.observation_space.shape[0])
            finally:
                env.close()

    def test_removing_the_cheat_also_stops_steering_by_the_target(self):
        """The lab controller is fed the same privileged bearing; it must go too."""
        env = self.env(False)
        try:
            env.reset(seed=0)
            env.target = env.data.xpos[env._trunk][:2] + np.array([0., 10.])  # hard left
            seen = []
            original = env.cpg.compute
            env.cpg.compute = lambda action, t, **kw: seen.append(kw.get("heading_error")) or original(action, t, **kw)
            env.step(np.zeros(env.nu, dtype=np.float32))
            self.assertEqual(seen, [0.], "a target off to one side must not steer the controller")
        finally:
            env.close()

    def test_the_privileged_env_still_steers(self):
        env = self.env(True)
        try:
            env.reset(seed=0)
            env.target = env.data.xpos[env._trunk][:2] + np.array([0., 10.])
            seen = []
            original = env.cpg.compute
            env.cpg.compute = lambda action, t, **kw: seen.append(kw.get("heading_error")) or original(action, t, **kw)
            env.step(np.zeros(env.nu, dtype=np.float32))
            self.assertNotEqual(seen[0], 0.)
        finally:
            env.close()


class Prey(unittest.TestCase):
    def test_prey_sits_still_until_the_predator_is_close(self):
        prey = FleeingPrey(parameters(), height_m=.009)
        prey.reset(np.zeros(2), place_at=np.array([.30, 0.]))
        before = prey.position.copy()
        for _ in range(50):
            prey.step(.02, np.zeros(2))
        np.testing.assert_array_equal(prey.position, before)

    def test_prey_runs_directly_away_once_alarmed(self):
        prey = FleeingPrey(parameters(), height_m=.009)
        prey.reset(np.zeros(2), place_at=np.array([.05, 0.]))
        for _ in range(30):
            prey.step(.02, np.zeros(2))
        self.assertGreater(prey.position[0], .05, "prey must move away from the predator")
        self.assertAlmostEqual(prey.position[1], 0., places=9, msg="escape is directly away")

    def test_escape_is_delayed_by_the_published_latency(self):
        """A zero-latency prey would be uncatchable for a simulation reason."""
        prey = FleeingPrey(parameters(escape_latency_s=.085), height_m=.009)
        prey.reset(np.zeros(2), place_at=np.array([.05, 0.]))
        prey.step(.02, np.zeros(2))
        np.testing.assert_allclose(prey.position, [.05, 0.], atol=1e-12)
        for _ in range(4):
            prey.step(.02, np.zeros(2))
        self.assertGreater(prey.position[0], .05)

    def test_a_walking_gecko_cannot_run_prey_down(self):
        """0.055 m/s against 0.118 m/s. Capture needs a strike, and there isn't one."""
        prey = FleeingPrey(parameters(), height_m=.009)
        prey.reset(np.zeros(2), place_at=np.array([.06, 0.]))
        predator = np.zeros(2)
        for _ in range(500):
            predator = predator + np.array([.055 * .02, 0.])
            _, captured = prey.step(.02, predator)
            self.assertFalse(captured, "pursuit must not succeed at walking speed")

    def test_capture_respawns_prey_outside_the_flee_radius(self):
        prey = FleeingPrey(parameters(), height_m=.009)
        prey.reset(np.zeros(2), place_at=np.array([.30, 0.]))
        _, captured = prey.step(.02, np.array([.30, 0.]))
        self.assertTrue(captured)
        self.assertEqual(prey.captures, 1)
        self.assertGreater(np.linalg.norm(prey.position - np.array([.30, 0.])),
                           parameters().flee_radius_m)

    def test_an_arena_smaller_than_the_flee_radius_is_refused(self):
        with self.assertRaisesRegex(ValueError, "arena"):
            parameters(arena_radius_m=.05, flee_radius_m=.075)

    def test_registry_values_load_and_are_self_consistent(self):
        live = PreyParameters.from_registry()
        self.assertGreater(live.flee_radius_m, live.capture_distance_m,
                           "prey must react before the gecko is in strike range")
        self.assertGreater(live.escape_speed_m_s, .055,
                           "prey slower than the walker would be caught by walking, which the "
                           "published 82.9% capture rate does not describe")

    def test_prey_needs_a_world_that_has_somewhere_to_put_it(self):
        with self.assertRaisesRegex(ValueError, "no prey body"):
            GeckoWalkEnv(xml_path=str(BODY), control_mode="cpg_residual", gait_profile="lab",
                         prey_parameters=parameters())


class PreyInTheWorld(unittest.TestCase):
    def setUp(self):
        if not WORLD.exists():
            self.skipTest("no generated world committed")

    def test_prey_position_reaches_the_mocap_slot_the_camera_renders(self):
        env = GeckoWalkEnv(xml_path=str(WORLD), control_mode="cpg_residual", gait_profile="lab",
                           hind_stance_compensation="all", privileged_target=False,
                           prey_parameters=parameters())
        try:
            env.reset(seed=0)
            np.testing.assert_allclose(env.data.mocap_pos[0][:2], env.prey.position, atol=1e-12)
            env.step(np.zeros(env.nu, dtype=np.float32))
            np.testing.assert_allclose(env.data.mocap_pos[0][:2], env.prey.position, atol=1e-12)
        finally:
            env.close()

    def test_the_reward_still_measures_distance_to_the_moving_prey(self):
        env = GeckoWalkEnv(xml_path=str(WORLD), control_mode="cpg_residual", gait_profile="lab",
                           hind_stance_compensation="all", privileged_target=False,
                           prey_parameters=parameters())
        try:
            env.reset(seed=0)
            np.testing.assert_allclose(env.target, env.prey.position, atol=1e-12)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
