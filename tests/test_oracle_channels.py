"""Every privileged channel must be switchable, declared, and reported.

THE CLASS OF BUG. Three times now this project has recorded a privileged
channel as "removed everywhere" and been wrong -- FAILURE_MAP #26, #33, and
now #165. Each time the channel was not hidden deliberately; it was simply on
by default somewhere nobody looked, and no test asked.

These tests do not require the oracle to be OFF. It is measurably load-bearing
(tools/oracle_ablation.py: zeroing it costs 91 % of goal progress), so turning
it off without retraining would just break the walker. What they require is
that it cannot be SILENT: the flag exists, it reaches the walker, and every
step says out loud what it was.
"""

import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class TheOracleCannotBeSilent(unittest.TestCase):
    def test_the_flag_reaches_the_walker(self):
        """The actual defect: GeckoBrainEnv built the walker without naming
        privileged_target, so the walker's True default won and nothing here
        could override it."""
        from envs.gecko_brain_env import GeckoBrainEnv
        env = GeckoBrainEnv(walker_oracle=True)
        try:
            self.assertTrue(env.walk_env.privileged_target)
        finally:
            env.close()

    def test_turning_it_off_removes_the_slots_and_breaks_every_checkpoint(self):
        """Two facts in one, and the second is the uncomfortable one.

        Turning the channel off really does remove the five slots -- 92 -> 87.
        And that makes the frozen walker refuse to load, because EVERY
        checkpoint in this repository was trained with the oracle present.
        The failure is the evidence: there is no saved policy that can run
        without it. That is why the default stays True.
        """
        from envs.gecko_walk_env import GeckoWalkEnv
        from envs.gecko_brain_env import GeckoBrainEnv

        off = GeckoWalkEnv(privileged_target=False)
        try:
            self.assertEqual(off.observation_layout["total"], 87)
            names = [n for n, _ in off.observation_layout["blocks"]]
            self.assertNotIn("privileged_target", names)
        finally:
            off.close()

        with self.assertRaises(AssertionError) as caught:
            GeckoBrainEnv(walker_oracle=False)
        self.assertIn("92", str(caught.exception))
        self.assertIn("87", str(caught.exception))

    def test_every_step_reports_what_it_was(self):
        from envs.gecko_brain_env import GeckoBrainEnv
        env = GeckoBrainEnv()
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))
            self.assertIn("walker_oracle", info)
            self.assertIn("food_oracle_scale", info)
            self.assertIs(info["walker_oracle"], True,
                          "the default is ON and the record must say so")
        finally:
            env.close()

    def test_the_default_is_on_and_that_is_deliberate(self):
        """Pinned so nobody 'fixes' this by flipping the default and breaking
        every checkpoint. All twelve in the repository are 92-dimensional."""
        from envs.gecko_brain_env import GeckoBrainEnv
        env = GeckoBrainEnv()
        try:
            self.assertTrue(env.walker_oracle)
            self.assertEqual(env.walk_env.observation_layout["total"], 92)
        finally:
            env.close()


class TheAblationIsOnTheRecord(unittest.TestCase):
    def test_the_measurement_says_it_is_load_bearing(self):
        import json
        p = REPO / "artifacts/evidence/session9/oracle_ablation.json"
        if not p.is_file():
            self.skipTest("run tools/oracle_ablation.py first")
        payload = json.loads(p.read_text(encoding="utf-8"))
        self.assertTrue(payload["load_bearing"])
        # The shape of the result, not just its verdict: the animal keeps
        # walking and stops going anywhere. Distance is not the measurement.
        self.assertGreater(payload["oracle_zeroed"]["summary"]["mean_distance_m"],
                           payload["oracle_live"]["summary"]["mean_distance_m"],
                           "zeroing the oracle did not reduce how far it moved")
        self.assertLess(payload["fraction_of_progress_retained"], 0.5,
                        "and it did reduce how much of that was toward the goal")


if __name__ == "__main__":
    unittest.main()
