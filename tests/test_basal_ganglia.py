"""What the reimplemented GPR does verify, and what it does not.

The module is a reimplementation from equations: `docs/research/` names the GPR
model and gives the Prescott 2024 dopamine sweep, but does **not** state the
connection weights or thresholds. So these tests split cleanly in two.

Verified here: the architecture. Selection happens, it is disinhibition rather
than a maximum, the thalamocortical loop produces persistence, dopamine moves
behaviour in the right qualitative directions, and GPR differs from a
winner-take-all.

**Not** verified, and recorded as a failure in
`artifacts/evidence/session6/dopamine_sweep.json`: the published sweep does not
reproduce. The switching pattern is inverted -- most dithering at the 0.20
baseline where the paper reports least. There is no test asserting the current
switching numbers, because pinning them would freeze a wrong result as correct.
"""
import json
from pathlib import Path
import unittest

import numpy as np

from brain.basal_ganglia import (
    CHANNELS, BasalGanglia, GPRParameters, salience_from_drives,
)

REPO = Path(__file__).resolve().parents[1]


def settle(bg, salience, seconds=4.0, dt=0.02):
    gates = None
    for _ in range(int(seconds / dt)):
        gates = bg.step(np.asarray(salience, dtype=float), dt)
    return gates


class SelectionWorks(unittest.TestCase):
    def test_the_most_salient_channel_is_the_one_released(self):
        bg = BasalGanglia()
        salience = np.array([0.2, 0.8, 0.1, 0.1, 0.1, 0.05])
        gates = settle(bg, salience)
        self.assertEqual(bg.selected(gates), "flee")
        self.assertEqual(int(np.argmax(gates)), 1)

    def test_separated_saliences_select_cleanly_at_every_dopamine_level(self):
        for dopamine in (0.06, 0.20, 0.29, 0.43, 0.46):
            bg = BasalGanglia(dopamine=dopamine)
            gates = settle(bg, [0.55, 0.30, 0.20, 0.10, 0.10, 0.05])
            self.assertEqual(bg.selected(gates), "hunt", f"lambda={dopamine}")
            self.assertEqual(bg.distortion(gates), 0, f"lambda={dopamine}")

    def test_selection_is_disinhibition_not_a_maximum(self):
        """Losers are actively held down; a loser can be partly released."""
        bg = BasalGanglia(dopamine=0.43)
        gates = settle(bg, [0.55, 0.50, 0.20, 0.10, 0.10, 0.05])
        losers = np.delete(gates, int(np.argmax(gates)))
        self.assertGreater(np.max(losers), 0.0,
                           "a maximum would leave every loser at exactly zero")

    @unittest.expectedFailure
    def test_nothing_is_selected_when_nothing_is_salient(self):
        """KNOWN DEFECT, recorded rather than deleted.

        With no salience anywhere the module oscillates instead of settling,
        cycling every gate together between 0 and about 0.34, so `selected`
        returns whichever channel the tie-break happens to reach. The
        thalamocortical loop gain was one cause and is fixed; the remaining one
        is the diffuse STN-GPe loop, whose effective gain grows with channel
        count. Averaging that drive instead of summing it cures the oscillation
        and destroys the discrimination, so it is not a fix.

        The requirement is right and the module does not meet it. Deleting the
        test would hide that; asserting the oscillation would enshrine it.
        """
        bg = BasalGanglia()
        gates = settle(bg, np.zeros(6))
        self.assertIsNone(bg.selected(gates))

    def test_channel_count_and_names_are_the_six_the_spec_lists(self):
        self.assertEqual(CHANNELS, ("hunt", "flee", "explore", "bask", "rest", "groom"))
        self.assertEqual(BasalGanglia().n, 6)


class Persistence(unittest.TestCase):
    """The thalamocortical loop is what "extended GPR" means."""

    def test_a_running_behaviour_resists_a_marginally_better_alternative(self):
        bg = BasalGanglia()
        settle(bg, [0.55, 0.30, 0.10, 0.10, 0.10, 0.05], seconds=6.0)
        self.assertEqual(bg.selected(), "hunt")
        # Flip the ordering by a hair. A memoryless comparator switches at once.
        gates = bg.step(np.array([0.52, 0.53, 0.10, 0.10, 0.10, 0.05]), 0.02)
        self.assertEqual(bg.selected(gates), "hunt",
                         "a hair-thin reversal must not immediately flip behaviour")

    def test_a_decisively_better_alternative_does_win(self):
        bg = BasalGanglia()
        settle(bg, [0.55, 0.30, 0.10, 0.10, 0.10, 0.05], seconds=6.0)
        gates = settle(bg, [0.30, 0.95, 0.10, 0.10, 0.10, 0.05], seconds=6.0)
        self.assertEqual(bg.selected(gates), "flee",
                         "persistence must not become paralysis")

    def test_the_loop_state_exists_and_resets(self):
        bg = BasalGanglia()
        settle(bg, [0.6, 0.2, 0.1, 0.1, 0.1, 0.05])
        self.assertGreater(float(np.max(np.abs(bg._thalamus))), 0.0)
        bg.reset()
        self.assertEqual(float(np.max(np.abs(bg._thalamus))), 0.0)
        self.assertEqual(float(np.max(np.abs(bg._cortex))), 0.0)


class DopamineQualitativeDirections(unittest.TestCase):
    """The directions that DO reproduce. Magnitudes are a recorded failure."""

    def test_too_little_dopamine_freezes_the_animal(self):
        """Published: prolonged immobility at lambda <= 0.06."""
        frozen = BasalGanglia(dopamine=0.03)
        gates = settle(frozen, [0.55, 0.50, 0.20, 0.10, 0.10, 0.05], seconds=8.0)
        self.assertIsNone(frozen.selected(gates),
                          "at 0.03 the published animal is immobile")

    def test_baseline_dopamine_selects_where_the_frozen_one_cannot(self):
        salience = [0.55, 0.50, 0.20, 0.10, 0.10, 0.05]
        healthy = BasalGanglia(dopamine=0.20)
        self.assertIsNotNone(healthy.selected(settle(healthy, salience, seconds=8.0)))

    def test_too_much_dopamine_releases_losing_channels(self):
        """Published: distortion -- partial expression of losers -- at high lambda."""
        salience = [0.55, 0.50, 0.20, 0.10, 0.10, 0.05]
        clean = BasalGanglia(dopamine=0.20)
        excess = BasalGanglia(dopamine=0.46)
        self.assertGreater(excess.distortion(settle(excess, salience)),
                           clean.distortion(settle(clean, salience)))

    def test_dopamine_is_bounded_and_validated(self):
        for bad in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                BasalGanglia(dopamine=bad)


class WinnerTakeAllComparison(unittest.TestCase):
    """The published ablation. Both arms get the same smoothed input."""

    def test_the_two_arms_differ(self):
        salience = [0.55, 0.50, 0.20, 0.10, 0.10, 0.05]
        gpr = settle(BasalGanglia(dopamine=0.43), salience)
        wta = settle(BasalGanglia(dopamine=0.43, winner_take_all=True), salience)
        self.assertFalse(np.allclose(gpr, wta))

    def test_a_winner_take_all_cannot_express_distortion(self):
        """Its losers are exactly zero by construction, so partial expression
        of a losing channel is not a failure mode it can have."""
        wta = BasalGanglia(dopamine=0.46, winner_take_all=True)
        gates = settle(wta, [0.55, 0.50, 0.20, 0.10, 0.10, 0.05])
        self.assertEqual(wta.distortion(gates), 0)
        self.assertEqual(int(np.count_nonzero(gates)), 1)


class SalienceIsSigmaPi(unittest.TestCase):
    """Sums of PRODUCTS: hunting needs hunger AND visible prey."""

    def test_a_sated_gecko_does_not_hunt_visible_prey(self):
        salience = salience_from_drives(hunger=0.0, fatigue=0.1, cold=0.0, warm=0.0,
                                        prey_visible=1.0)
        self.assertEqual(salience[0], 0.0)

    def test_a_hungry_gecko_with_no_prey_in_sight_explores_rather_than_hunts(self):
        salience = salience_from_drives(hunger=0.9, fatigue=0.0, cold=0.0, warm=0.0,
                                        prey_visible=0.0)
        self.assertEqual(salience[0], 0.0)
        self.assertGreater(salience[2], 0.0)

    def test_hunger_and_visible_prey_together_drive_hunting(self):
        salience = salience_from_drives(hunger=0.9, fatigue=0.0, cold=0.0, warm=0.0,
                                        prey_visible=1.0)
        self.assertEqual(int(np.argmax(salience)), 0)

    def test_fatigue_drives_rest_and_thermal_error_drives_basking(self):
        tired = salience_from_drives(hunger=0.1, fatigue=0.9, cold=0.0, warm=0.0)
        self.assertEqual(int(np.argmax(tired)), CHANNELS.index("rest"))
        cold = salience_from_drives(hunger=0.1, fatigue=0.0, cold=0.9, warm=0.0)
        self.assertEqual(int(np.argmax(cold)), CHANNELS.index("bask"))

    def test_persistence_input_must_match_the_channel_count(self):
        with self.assertRaises(ValueError):
            salience_from_drives(0.5, 0.1, 0.0, 0.0, persistence=np.zeros(3))

    def test_salience_is_never_negative(self):
        salience = salience_from_drives(hunger=0.0, fatigue=0.0, cold=0.0, warm=0.0)
        self.assertTrue(np.all(salience >= 0.0))


class TheSweepIsARecordedFailure(unittest.TestCase):
    """Guard the honesty of the evidence file rather than the numbers in it."""

    def setUp(self):
        path = REPO / "artifacts/evidence/session6/dopamine_sweep.json"
        if not path.is_file():
            self.skipTest("sweep evidence not generated")
        self.payload = json.loads(path.read_text(encoding="utf-8"))

    def test_the_verdict_is_recorded_as_a_failure(self):
        self.assertIn("DOES NOT REPRODUCE", self.payload["verdict"])
        self.assertTrue(self.payload["not_reproduced"])

    def test_the_diagnosis_names_the_missing_parameters(self):
        self.assertIn("parameter", self.payload["root_cause"].lower())
        self.assertTrue(self.payload["to_finish"])

    def test_it_does_not_claim_to_be_validated(self):
        self.assertIn("not a validated", self.payload["claim"].lower())


class Parameters(unittest.TestCase):
    def test_invalid_parameters_are_refused(self):
        with self.assertRaises(ValueError):
            GPRParameters(tau_s=0.0)
        with self.assertRaises(ValueError):
            GPRParameters(gate_scale=-1.0)
        with self.assertRaises(ValueError):
            GPRParameters(tau_s=float("nan"))

    def test_fewer_than_two_channels_is_not_a_competition(self):
        with self.assertRaises(ValueError):
            BasalGanglia(channels=("only",))

    def test_salience_shape_is_checked(self):
        with self.assertRaises(ValueError):
            BasalGanglia().step(np.zeros(3), 0.02)


if __name__ == "__main__":
    unittest.main()
