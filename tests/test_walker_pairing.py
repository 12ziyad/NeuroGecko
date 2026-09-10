"""The known-broken walker pairing must never assemble silently again.

Session 9 rendered every video with a trained residual bolted onto the `lab`
CPG -- the one combination of four that limps. Nothing in the code objected,
because the pairing was assembled out of two independent defaults and a
checkpoint path carrying no record of what it was trained against. The user
found it by watching the animal; six exchanges of mine blamed the camera, the
frame rate and the body first.

These tests do not forbid the configuration. It has to stay runnable or the
evidence condemning it becomes unreproducible. What they require is that it
cannot happen quietly.
"""

import pathlib
import sys
import unittest
import warnings

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.walker_pairing import (                            # noqa: E402
    LIMP_GAP, MEASURED, PairingWarning, check_pairing, describe_pairing)


class ExactlyOnePairingIsBroken(unittest.TestCase):
    def test_all_four_combinations_are_on_the_record(self):
        self.assertEqual(set(MEASURED), {("lab", True), ("lab", False),
                                         ("legacy", True), ("legacy", False)})

    def test_only_lab_plus_policy_limps(self):
        """The measurement that matters: three of four are fine. Blaming the
        policy in general, or the profile in general, is wrong -- it is the
        pairing (ledger #184, correcting #183)."""
        broken = [k for k, v in MEASURED.items() if not v["ok"]]
        self.assertEqual(broken, [("lab", True)])

    def test_the_broken_one_is_an_order_of_magnitude_worse(self):
        sound = [v["gap"] for k, v in MEASURED.items() if v["ok"]]
        broken = MEASURED[("lab", True)]["gap"]
        self.assertLess(max(sound), LIMP_GAP)
        self.assertGreater(broken, LIMP_GAP)
        self.assertGreater(broken, 10 * max(sound),
                           "the gap is unambiguous, not a judgement call")


class ItWarnsOnTheBrokenPairing(unittest.TestCase):
    def test_the_broken_pairing_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok = check_pairing("lab", True)
        self.assertFalse(ok)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, PairingWarning)
        text = str(caught[0].message)
        for expect in ("0.205", "0.581", "ZERO-RESIDUAL", "#71", "legacy"):
            self.assertIn(expect, text,
                          "the warning must say what was measured and what to "
                          "do instead, not merely that something is wrong")

    def test_the_three_sound_pairings_are_silent(self):
        for profile, policy in (("lab", False), ("legacy", True), ("legacy", False)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ok = check_pairing(profile, policy)
            self.assertTrue(ok, (profile, policy))
            self.assertEqual(caught, [], (profile, policy))

    def test_it_never_raises(self):
        """The broken configuration stays runnable. Refusing it would make the
        evidence against it unreproducible, and this project keeps its failures
        executable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertFalse(check_pairing("lab", True))

    def test_an_unmeasured_combination_is_not_condemned(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(check_pairing("some_future_profile", True))
        self.assertEqual(caught, [])
        self.assertIn("not measured", describe_pairing("some_future_profile", True))


class TheBrainEnvIsGuarded(unittest.TestCase):
    def test_the_default_configuration_is_silent(self):
        from envs.gecko_brain_env import GeckoBrainEnv
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PairingWarning)
            env = GeckoBrainEnv()
            env.close()
        self.assertEqual([w for w in caught if w.category is PairingWarning], [])

    def test_asking_for_lab_warns(self):
        from envs.gecko_brain_env import GeckoBrainEnv
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PairingWarning)
            env = GeckoBrainEnv(gait_profile="lab")
            env.close()
        hits = [w for w in caught if w.category is PairingWarning]
        self.assertEqual(len(hits), 1)
        self.assertIn("GeckoBrainEnv", str(hits[0].message))

    def test_it_can_be_silenced_deliberately(self):
        """Suppressing it must be an explicit act, not a default."""
        from envs.gecko_brain_env import GeckoBrainEnv
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PairingWarning)
            env = GeckoBrainEnv(gait_profile="lab", warn_pairing=False)
            env.close()
        self.assertEqual([w for w in caught if w.category is PairingWarning], [])


if __name__ == "__main__":
    unittest.main()
