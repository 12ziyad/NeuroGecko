"""Tests for brain/retina.py and brain/pretectum.py -- brain module 4.

The load-bearing test is the optokinetic asymmetry. Almost nothing about this
animal's eye has been measured, but its optokinetic reflex has, in the target
species, with a result sharp enough to falsify a whole class of front ends:
monocular naso-temporal stimulation elicits NO response at any velocity. A
symmetric optic-flow estimator predicts a non-zero gain there and is therefore
wrong, whatever else it gets right.
"""

import importlib.util
import math
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


_r = _load("_retina_test", "brain/retina.py")
_p = _load("_pretectum_test", "brain/pretectum.py")
_okr = _load("_okr_test", "tools/okr_sweep.py")
Retina, Pretectum = _r.Retina, _p.Pretectum


class TheRetinaKeepsPosition(unittest.TestCase):
    """The defect it exists to fix: the encoder average-pooled the whole field
    to 128 numbers, so the animal had 'how much' and never 'where'."""

    def test_the_output_is_a_map_not_a_summary(self):
        retina = Retina(pixels=64, cells=16)
        out = retina.step(np.zeros((64, 64, 3), np.uint8))
        for key in ("motion", "signed_motion", "contrast", "luminance"):
            self.assertEqual(out[key].shape, (16, 16), key)

    def test_a_moving_target_lights_up_where_it_is(self):
        retina = Retina(pixels=64, cells=16)
        blank = np.zeros((64, 64, 3), np.uint8)
        retina.step(blank)
        moved = blank.copy()
        moved[28:36, 44:52, 1] = 255          # a bright patch on the right
        motion = retina.step(moved)["motion"]
        hottest = np.unravel_index(int(np.argmax(motion)), motion.shape)
        self.assertGreater(hottest[1], motion.shape[1] // 2,
                           "the response must be on the side the target is")

    def test_motion_is_zero_on_the_first_frame_after_reset(self):
        """An episode must not begin with the world appearing to explode."""
        retina = Retina(pixels=64, cells=16)
        frame = np.full((64, 64, 3), 200, np.uint8)
        np.testing.assert_allclose(retina.step(frame)["motion"], 0.0)
        retina.reset()
        np.testing.assert_allclose(retina.step(frame)["motion"], 0.0)

    def test_the_red_channel_is_dropped(self):
        """This animal's longest pigment is about 521 nm and SWS2 is absent, so
        red is an axis with no receptor behind it."""
        self.assertEqual(_r.KEPT_CHANNELS, (1, 2))
        retina = Retina(pixels=64, cells=16)
        retina.step(np.zeros((64, 64, 3), np.uint8))
        red_only = np.zeros((64, 64, 3), np.uint8)
        red_only[:, :, 0] = 255
        np.testing.assert_allclose(retina.step(red_only)["motion"], 0.0,
                                   atol=1e-12)

    def test_sampling_is_uniform_in_cells_and_the_camera_is_not_in_angle(self):
        """The animal is afoveate and uniform. A rectilinear camera is uniform
        in PIXELS, which makes it an inverse fovea -- sharpest where the animal
        is not looking. Measured, not assumed."""
        on_axis, corner = _r.angular_sampling(120.0, 64)
        self.assertGreater(on_axis / corner, 3.5,
                           "fovy 120 samples the corner ~4x finer than the axis")
        on_axis70, corner70 = _r.angular_sampling(70.0, 64)
        self.assertLess(on_axis70 / corner70, on_axis / corner,
                        "a narrower field is less non-uniform")

    def test_it_cannot_resolve_what_the_animal_tracks_at_the_wide_field(self):
        """Published: E. macularius tracks a 1.6 degree random dot, n=4."""
        self.assertLess(_r.resolvable(120.0, 64, 1.6), 1.0)
        self.assertGreater(_r.resolvable(70.0, 64, 1.6), 1.0)

    def test_there_is_no_tapetum_and_no_low_light_gain(self):
        doc = _r.__doc__
        self.assertIn("NO TAPETUM", doc.upper())
        self.assertNotIn("gain_boost", dir(Retina))


class TheOptokineticAsymmetry(unittest.TestCase):
    """The published discriminating result, and the reason for rectification."""

    def _gain(self, direction, **eyes):
        gain, _ = _okr.measure_gain(20.0, direction=direction, **eyes)
        return gain

    def test_naso_temporal_stimulation_elicits_nothing(self):
        """Masseck, Roll & Hoffmann 2008, E. macularius, n=4: no optokinetic
        response at any velocity. This is what a symmetric estimator cannot do,
        and it is PREDICTED here by the rectification rather than written in."""
        for velocity in _p.PUBLISHED_VELOCITIES:
            gain, _ = _okr.measure_gain(velocity, left_eye=True,
                                        right_eye=False, direction=-1.0)
            self.assertAlmostEqual(gain, 0.0, places=9, msg=f"{velocity} deg/s")

    def test_the_same_stimulus_the_other_way_does_drive_it(self):
        """Otherwise the silence above would just be a broken detector."""
        self.assertGreater(self._gain(+1.0, left_eye=True, right_eye=False), 0.3)

    def test_the_gain_falls_with_stimulus_velocity(self):
        gains = [_okr.measure_gain(v)[0] for v in _p.PUBLISHED_VELOCITIES]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_the_gain_never_reaches_one(self):
        """A loop that stabilises perfectly has over-reproduced the animal: the
        published binocular gain is 0.9 and FALLS to 0.7-0.8 by 40 deg/s."""
        for velocity in _p.PUBLISHED_VELOCITIES:
            self.assertLess(_okr.measure_gain(velocity)[0], 1.0)

    def test_the_flow_estimate_is_unbiased(self):
        """It was 2.07x too large when differenced on the cell grid, and 1.15x
        when differenced against the pixel index. Against azimuth it is 1.02."""
        retina = Retina(fovy_deg=70.0, pixels=64, cells=16)
        pretectum = Pretectum(retina)
        dt = 1.0 / 50.0
        for velocity in (10.0, 20.0, 40.0):
            retina.reset()
            estimates = []
            for i in range(30):
                out = retina.step(_okr.drum_frame(retina, velocity * i * dt))
                if i >= 6:
                    estimates.append(pretectum._flow_deg_s(out, dt)[0])
            ratio = float(np.mean(estimates)) / velocity
            self.assertAlmostEqual(ratio, 1.0, delta=0.05,
                                   msg=f"{velocity} deg/s -> ratio {ratio:.3f}")

    def test_the_fitted_values_are_labelled_as_fitted(self):
        """The absolute gains were fitted to the published points, so they are
        not evidence, and the module must keep saying so."""
        self.assertIn("FITTED", _p.__doc__)
        self.assertIn("peak_gain", Pretectum(Retina()).state()["fitted"])

    def test_extrapolation_beyond_the_published_velocities_is_refused(self):
        self.assertEqual(_p.PUBLISHED_VELOCITIES, (20.0, 30.0, 40.0))


class WhatCannotBeTestedIsRecordedAsSuch(unittest.TestCase):
    def test_the_binocular_comparison_needs_two_eyes(self):
        """The body has ONE central camera, so covering an eye is not a thing
        that can be done to it. Recorded as blocked rather than passed."""
        import json
        evidence = REPO / "artifacts/evidence/session8/okr_sweep.json"
        if not evidence.is_file():
            self.skipTest("run tools/okr_sweep.py first")
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        check = payload["checks"]["binocular_beats_monocular"] \
            if "checks" in payload else payload["predicted_checks"]["binocular_beats_monocular"]
        self.assertIsNone(check["pass"])
        self.assertIn("ONE CAMERA", check["untestable"])

    def test_the_body_really_does_have_one_camera(self):
        world = REPO / "morphology/gecko_world_v1.xml"
        if not world.is_file():
            self.skipTest("no-cheat world not generated")
        text = world.read_text(encoding="utf-8")
        head_cams = text.count('name="head_cam"')
        self.assertEqual(head_cams, 1,
                         "if a second eye is ever added, the blocked "
                         "binocular test becomes runnable and must be run")


if __name__ == "__main__":
    unittest.main()
