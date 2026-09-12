"""Tests for brain/gecko_selector.py -- the animal's action selection.

These are BEHAVIOURAL: they assert what the gecko does, not what the numbers
are. The numbers are guarded in tests/test_prescott_bg.py against the published
table; here the question is only whether a gecko carrying those numbers behaves
like an animal.
"""

import importlib.util
import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_gs = _load("_gecko_selector_test", "brain/gecko_selector.py")
GeckoSelector = _gs.GeckoSelector
salience_from_drives = _gs.salience_from_drives
CHANNELS = _gs.CHANNELS


def settle(sel, steps=20, **drives):
    for _ in range(steps):
        sel.step(**drives)
    return sel


class TheAnimalDecides(unittest.TestCase):
    def test_a_starving_gecko_that_can_see_prey_hunts(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        self.assertEqual(sel.selected(), "hunt")
        self.assertTrue(sel.committed(), "it should commit, not half-hunt")

    def test_a_sated_gecko_ignores_visible_prey(self):
        """Sigma-pi: hunting needs hunger AND prey, not either alone."""
        sel = settle(GeckoSelector(), hunger=0.05, prey_visible=1.0)
        self.assertNotEqual(sel.selected(), "hunt")

    def test_a_hungry_gecko_with_nothing_in_sight_explores(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=0.0)
        self.assertEqual(sel.selected(), "explore")

    def test_a_predator_outranks_a_meal(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0, threat=0.95)
        self.assertEqual(sel.selected(), "flee")

    def test_the_light_phase_produces_rest_and_cold_produces_basking(self):
        """Rest is the clock, not fatigue (#349, #352).

        This test asserted `fatigue=0.9` produces rest, and it passed on a
        regime the animal cannot reach: `Physiology.endurance_s` returns
        infinity at or below the 0.100 m/s aerobic ceiling, the accepted walker
        averages 0.031 m/s, and measured over five seeds x 1500 autonomous
        steps the animal's fatigue equilibrates at 0.002 -- correlating with
        the fraction of control steps whose INSTANTANEOUS displacement crosses
        the ceiling at r = +0.9998, and with its actual mean speed at +0.71.
        Averaged over one stride instead of one control step, the same trace
        gives fatigue exactly 0.0.
        """
        self.assertEqual(settle(GeckoSelector(), arousal=0.0).selected(), "rest")
        self.assertEqual(settle(GeckoSelector(), cold=0.7).selected(), "bask")

    def test_fatigue_no_longer_reaches_the_rest_channel(self):
        """The refuted wiring, pinned so it cannot come back by accident.

        Fatigue remains a real measured drive from brain 1 and stays in the
        hypothalamus vector; it simply selects nothing. If a future session
        wants it back it has to say so here first.
        """
        sel = settle(GeckoSelector(), fatigue=0.95)
        self.assertIsNone(sel.selected())
        self.assertEqual(sel.last_salience[CHANNELS.index("rest")], 0.0)

    def test_an_empty_salience_vector_selects_nothing(self):
        """The degenerate fixed point groom's invented tonic was hiding (#357).

        At exactly zero salience the extended model settles to a uniform
        0.000402 on all six channels, and a plain argmax returns whichever
        channel is declared first. The published classifier calls that "none",
        and `selected()` now agrees with it by using the authors' own
        PARTIAL = 0.05 rather than an invented epsilon.
        """
        sel = settle(GeckoSelector())
        self.assertTrue(np.allclose(sel.last_salience, 0.0))
        self.assertEqual(sel.bg.selection_class(), "none")
        self.assertIsNone(sel.selected())
        gates = sel.gates()
        self.assertGreater(float(np.max(gates)), 0.0,
                           "if this is ever exactly zero the fixed point moved")
        self.assertTrue(np.allclose(gates, gates[0]),
                        "the degenerate release is uniform across channels")

    def test_groom_carries_no_tonic_salience(self):
        """It was 0.05, untagged and uncited, and it raised the release floor
        for every OTHER channel from 0.19988 to 0.20109 (#353)."""
        sel = settle(GeckoSelector(), hunger=0.4)
        self.assertEqual(sel.last_salience[CHANNELS.index("groom")], 0.0)

    def test_a_contented_gecko_does_nothing_in_particular(self):
        """None is a real answer. There is no fallback behaviour, and adding
        one would hide a broken competition behind something that looked fine."""
        self.assertIsNone(settle(GeckoSelector()).selected())


class Persistence(unittest.TestCase):
    """A behaviour with a duration, from published structure rather than an
    invented stickiness constant."""

    def test_a_running_behaviour_resists_a_marginally_better_rival(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        self.assertEqual(sel.selected(), "hunt")
        for _ in range(5):
            sel.step(hunger=0.8, prey_visible=1.0, cold=0.62)
        self.assertEqual(sel.selected(), "hunt",
                         "a slightly better option should not cause a switch")

    def test_a_decisively_better_rival_does_win(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        for _ in range(20):
            sel.step(hunger=0.8, prey_visible=1.0, cold=0.99)
        self.assertEqual(sel.selected(), "bask",
                         "persistence must not become paralysis")

    def test_reset_forgets_the_running_behaviour(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        sel.reset()
        self.assertIsNone(sel.selected())

    def test_persistence_is_not_an_invented_constant_any_more(self):
        """The old salience fed its own output back through an INVENTED gain of
        0.15. Persistence now comes from the model's cortico-thalamic loop.
        Guarded so it cannot creep back in."""
        import inspect
        src = inspect.getsource(salience_from_drives)
        self.assertNotIn("persistence_gain", src)
        self.assertNotIn("persistence", inspect.signature(
            salience_from_drives).parameters)


class DopamineDoesWhatItDoesInAnimals(unittest.TestCase):
    def test_depletion_freezes_the_animal(self):
        """The published akinesia result: at zero tonic dopamine nothing is
        selected however loud the drives are."""
        sel = GeckoSelector(dopamine=0.0)
        settle(sel, hunger=0.9, prey_visible=1.0, threat=0.9, cold=0.9)
        self.assertIsNone(sel.selected())

    def test_excess_dopamine_releases_losing_behaviours(self):
        """Distortion: more than one behaviour partly expressed at once."""
        calm = settle(GeckoSelector(dopamine=0.3),
                      hunger=0.7, prey_visible=0.9, cold=0.6)
        wild = settle(GeckoSelector(dopamine=0.6),
                      hunger=0.7, prey_visible=0.9, cold=0.6)
        self.assertGreaterEqual(wild.bg.distortion(), calm.bg.distortion())

    def test_a_dopamine_outside_the_published_range_is_refused(self):
        with self.assertRaises(ValueError):
            GeckoSelector(dopamine=5.0)
        GeckoSelector(dopamine=1.0)                  # the published maximum


class TheSeamWithTheHypothalamus(unittest.TestCase):
    def test_it_accepts_the_hypothalamus_drive_vector_directly(self):
        sel = GeckoSelector()
        for _ in range(20):
            gates = sel.step_from_homeostasis([0.8, 0.0, 0.0, 0.0],
                                              prey_visible=1.0)
        self.assertEqual(len(gates), len(CHANNELS))
        self.assertEqual(sel.selected(), "hunt")

    def test_a_wrong_shaped_drive_vector_is_refused(self):
        with self.assertRaises(ValueError):
            GeckoSelector().step_from_homeostasis([0.1, 0.2])

    def test_non_finite_drives_are_refused(self):
        with self.assertRaises(ValueError):
            GeckoSelector().step(hunger=float("nan"))


class TheContract(unittest.TestCase):
    def test_the_six_behaviours_and_their_order_are_fixed(self):
        self.assertEqual(CHANNELS,
                         ("hunt", "flee", "explore", "bask", "rest", "groom"))
        self.assertEqual(len(GeckoSelector().gates()), 6)

    def test_gates_always_agree_with_what_was_selected(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        gates = sel.gates()
        self.assertEqual(sel.channels[int(np.argmax(gates))], sel.selected())

    def test_the_salience_weights_are_declared_invented(self):
        """They decide what the animal wants, and nothing published constrains
        them. The docstring must keep saying so."""
        self.assertIn("INVENTED", salience_from_drives.__doc__)

    def test_state_reports_whether_the_network_settled(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        self.assertIn("settled", sel.state())
        self.assertTrue(sel.state()["settled"])


class RunsInsideTheLiveEnvironment(unittest.TestCase):
    """Brain 1 -> brain 2, on simulated data, in the real loop.

    This is the seam that matters: the hypothalamus's drive vector reaching the
    selector every control step, with threat and prey visibility from the world
    rather than from a test fixture.
    """

    def _env(self, **kw):
        try:
            from envs.gecko_brain_env import GeckoBrainEnv
        except Exception as exc:                      # pragma: no cover
            self.skipTest(f"environment unavailable: {exc}")
        return GeckoBrainEnv(homeostasis=True, action_selection=True, **kw)

    def test_action_selection_requires_the_hypothalamus(self):
        """Its salience IS the drive vector, so the dependency is refused
        rather than silently satisfied with zeros."""
        from envs.gecko_brain_env import GeckoBrainEnv
        with self.assertRaises(ValueError):
            GeckoBrainEnv(homeostasis=False, action_selection=True)

    def test_the_choice_is_reported_and_steers_nothing(self):
        env = self._env()
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
            self.assertIn("behaviour", info)
            self.assertEqual(len(info["behaviour_gates"]), len(CHANNELS))
            # Only one behaviour has a controller. A six-way chooser wired to a
            # one-behaviour body would look like integration and mean nothing.
            self.assertTrue(info["behaviour_controls_nothing"])
        finally:
            env.close()

    def test_a_rested_unhungry_gecko_chooses_nothing(self):
        """No fallback behaviour. At the start of an episode every drive is
        near zero and the honest answer is that the animal wants nothing."""
        env = self._env()
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
            self.assertIsNone(info["behaviour"])
        finally:
            env.close()

    def test_drives_that_build_produce_a_behaviour(self):
        """With the clock compressed the animal gets tired and hungry, and then
        it does something. Without this the seam is never exercised."""
        env = self._env(homeostasis_time_compression=20000.0)
        try:
            env.reset(seed=0)
            chosen = set()
            for _ in range(120):
                _, _, term, trunc, info = env.step(env.action_space.sample())
                chosen.add(info["behaviour"])
                if term or trunc:
                    env.reset()
            self.assertTrue(chosen - {None},
                            "a hungry, tired gecko should want something")
        finally:
            env.close()


class Brain6ReachesBrain2(unittest.TestCase):
    """The day/night clock drives the rest channel (#352).

    Until now `brain/arousal.py` was built, tested and consumed by nothing
    (#262), and a `circadian` flag made it look otherwise: the object was
    constructed and reset, `clock.step` was never called, and `info["arousal"]`
    reported the constant 1.0 assigned at construction (#347).
    """

    def _env(self, **kw):
        try:
            from envs.gecko_brain_env import GeckoBrainEnv
        except Exception as exc:                      # pragma: no cover
            self.skipTest(f"environment unavailable: {exc}")
        return GeckoBrainEnv(homeostasis=True, action_selection=True,
                             circadian=True, **kw)

    def test_the_clock_is_actually_stepped(self):
        """The exact defect of #347: the number reported must be one the
        module produced, not a constant sitting next to it."""
        env = self._env(time_of_day_h=0.0, homeostasis_time_compression=20000.0)
        try:
            env.reset(seed=0)
            first = env.step(env.action_space.sample())[4]
            self.assertIn("clock", first, "the clock must report its own state")
            seen = {first["clock"]["time_of_day_h"]}
            for _ in range(20):
                seen.add(env.step(env.action_space.sample())[4]
                         ["clock"]["time_of_day_h"])
            self.assertGreater(len(seen), 1, "the clock never advanced")
        finally:
            env.close()

    def test_the_light_phase_releases_rest_and_the_night_does_not(self):
        """The behaviour, not the number. 2 h is inside the light phase, where
        the published activity data put this animal in a retreat; 16 h is past
        the evening peak window it was measured active in."""
        released = {}
        for tod in (2.0, 16.0):
            env = self._env(time_of_day_h=tod)
            try:
                env.reset(seed=0)
                got = set()
                for _ in range(20):
                    info = env.step(env.action_space.sample())[4]
                    got.add(info["behaviour"])
                released[tod] = got
            finally:
                env.close()
        self.assertIn("rest", released[2.0], "a gecko rests through the day")
        self.assertNotIn("rest", released[16.0],
                         "it should not be resting at its activity peak")

    def test_without_a_clock_nothing_moves(self):
        """Arousal defaults to 1.0, so rest's salience is 0.0 and every other
        channel is untouched. No gate, checkpoint or policy path changes."""
        from envs.gecko_brain_env import GeckoBrainEnv
        env = GeckoBrainEnv(homeostasis=True, action_selection=True)
        try:
            env.reset(seed=0)
            info = env.step(env.action_space.sample())[4]
            self.assertNotIn("arousal", info)
            self.assertEqual(info["behaviour_gates"][CHANNELS.index("rest")], 0.0)
        finally:
            env.close()

    def test_a_sleeping_gecko_shuts_its_eyes_and_a_walking_one_does_not(self):
        """PUBLISHED that it can and does: Bergel et al. 2026 recorded sleep in
        n = 2 E. macularius with EOG electrodes UNDER EACH EYELID, which is
        possible because eublepharids are the only geckos with movable lids.
        The ANGLE is the already-declared invented EYELID_CLOSED_DEG and the
        conjunction with locomotion is invented too (#356).

        `locomotor_drive` is what the motor programs set to nothing when the
        chosen behaviour does not move the legs; it defaults to 1.0, so the
        plain `step(action)` path -- where something outside is driving the
        body -- never sleeps, which is the intended reading and is asserted
        here rather than left to be discovered.
        """
        env = self._env(time_of_day_h=2.0)
        try:
            env.reset(seed=0)
            env.walk_env.locomotor_drive = 0.0
            env.step(env.action_space.sample())
            self.assertGreater(float(env.walk_env.eyelid_rad), 0.0,
                               "the lids never closed in the light phase")
            env.walk_env.locomotor_drive = 1.0
            env.step(env.action_space.sample())
            self.assertEqual(float(env.walk_env.eyelid_rad), 0.0,
                             "a gecko whose legs are running has its eyes open")
        finally:
            env.close()

    def test_the_night_leaves_the_eyes_open_however_still_the_animal_is(self):
        env = self._env(time_of_day_h=16.0)
        try:
            env.reset(seed=0)
            env.walk_env.locomotor_drive = 0.0
            env.step(env.action_space.sample())
            self.assertEqual(float(env.walk_env.eyelid_rad), 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
