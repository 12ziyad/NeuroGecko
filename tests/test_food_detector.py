"""The placeholder food detector must see the food that is actually there.

This exists because it did not. The detector was written against a painted
green marker and kept its green test after the marker was replaced by a brown
prey geom, so it returned exactly 0.0 for the real prey at every illumination.
Nothing caught it: the no-cheat world it broke has never been used by a
training or evidence run, so the animal that could not see its own food never
ran anywhere. That is the fifth time a vision or privileged-channel defect has
survived a "done" declaration in this project.

The detector is a PLACEHOLDER FOR A RETINA. Brain module 4 replaces it. What
these tests guard is not the algorithm but the property that broke: the
detector and the world must not be able to disagree about what the food is.
"""

import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from envs.gecko_brain_env import (                      # noqa: E402
    FOOD_CHROMA_TOLERANCE, LEGACY_MARKER_RGB, _food_visible_frac, _rendered_rgb)

#: Every material in morphology/gecko_world_v1.xml, so the confusers in these
#: tests are the real ones rather than invented colours.
PREY_RGB = (0.55, 0.35, 0.18)
WORLD_SURFACES = {
    "gecko spot": (0.45, 0.36, 0.22),
    "gecko skin": (0.83, 0.73, 0.42),
    "floor mat": (0.78, 0.72, 0.45),
    "belly": (0.93, 0.90, 0.80),
    "eye": (0.05, 0.05, 0.06),
}


def patch(rgb, gain=1.0, size=16):
    px = np.clip(np.asarray(rgb, dtype=float) * gain * 255.0, 0, 255)
    return np.tile(px.astype(np.uint8), (size, size, 1))


class ItSeesTheFoodThatIsActuallyThere(unittest.TestCase):
    def test_the_brown_prey_is_visible(self):
        """The regression. This returned 0.0 at every illumination."""
        for gain in (0.4, 0.7, 1.0, 1.4, 1.8):
            self.assertGreater(_food_visible_frac(patch(PREY_RGB, gain), PREY_RGB),
                               0.9, f"gain {gain}")

    def test_the_old_green_test_would_have_failed_this(self):
        """Pinned so the failure mode cannot come back wearing a new colour:
        a detector keyed to the legacy marker sees nothing of the real prey."""
        seen = _food_visible_frac(patch(PREY_RGB), LEGACY_MARKER_RGB)
        self.assertEqual(seen, 0.0)

    def test_nothing_else_in_the_world_is_mistaken_for_food(self):
        for name, rgb in WORLD_SURFACES.items():
            self.assertLess(_food_visible_frac(patch(rgb), PREY_RGB), 0.01, name)

    def test_darkness_is_not_food(self):
        """Normalising a pixel by its own brightness makes black match
        everything unless unlit pixels are excluded first."""
        self.assertEqual(_food_visible_frac(np.zeros((16, 16, 3), np.uint8),
                                            PREY_RGB), 0.0)

    def test_it_survives_illumination_change(self):
        """The old absolute-channel test failed even for its OWN marker once
        the light dropped: at gain 0.4 the painted sphere also scored 0.0."""
        bright = _food_visible_frac(patch(PREY_RGB, 1.8), PREY_RGB)
        dim = _food_visible_frac(patch(PREY_RGB, 0.4), PREY_RGB)
        self.assertAlmostEqual(bright, dim, places=6)


class TheDetectorCannotDisagreeWithTheWorld(unittest.TestCase):
    """The class of bug, not the instance."""

    def test_the_target_colour_is_read_from_the_model(self):
        """THIS TEST USED TO PASS WHILE THE BUG WAS LIVE, and that is the more
        important finding than the bug.

        It read `geom_rgba`, then asked whether the detector could see a patch
        of `geom_rgba`. Of course it could -- it was comparing the detector
        against itself. Meanwhile the prey geom carries `material="prey"` and
        no rgba, so `geom_rgba` was MuJoCo's default grey and the world was
        drawing brown. Self-consistency is not correctness: the assertion has
        to reach the colour the RENDERER uses, or it tests nothing.
        """
        import mujoco
        world = REPO / "morphology/gecko_world_v1.xml"
        if not world.is_file():
            self.skipTest("no-cheat world not generated")
        model = mujoco.MjModel.from_xml_path(str(world))
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
        self.assertGreaterEqual(gid, 0, "the prey geom must be findable by name")

        drawn = _rendered_rgb(model, gid)
        self.assertGreater(_food_visible_frac(patch(drawn), drawn), 0.9,
                           "the detector must see the colour that is DRAWN")
        # Whatever colour the world is regenerated in, the detector follows.
        self.assertAlmostEqual(
            float(np.abs(np.asarray(drawn) - np.asarray(PREY_RGB)).max()), 0.0,
            places=6, msg="the world's prey is no longer the colour these "
                          "tests were written against -- update PREY_RGB")

    def test_the_regression_itself_the_material_not_the_geom(self):
        """The exact shape of the second occurrence, pinned.

        The prey states its colour on a MATERIAL. `geom_rgba` therefore holds
        MuJoCo's default grey, and a detector keyed to that scores 0.0 on the
        prey the renderer actually draws.
        """
        import mujoco
        world = REPO / "morphology/gecko_world_v1.xml"
        if not world.is_file():
            self.skipTest("no-cheat world not generated")
        model = mujoco.MjModel.from_xml_path(str(world))
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")

        raw = tuple(float(x) for x in model.geom_rgba[gid][:3])
        drawn = _rendered_rgb(model, gid)
        self.assertNotAlmostEqual(
            float(np.abs(np.asarray(raw) - np.asarray(drawn)).max()), 0.0,
            places=3, msg="if the world ever sets rgba on the geom directly "
                          "this test stops guarding anything -- delete it and "
                          "say so, do not weaken it")
        self.assertEqual(_food_visible_frac(patch(drawn), raw), 0.0,
                         "reading geom_rgba scores exactly zero on the real prey")

    def test_the_env_reads_it_at_construction(self):
        from envs.gecko_brain_env import GeckoBrainEnv
        env = GeckoBrainEnv()
        try:
            # No prey geom in the default walker, so there is nothing to read
            # and the legacy marker stands in -- explicitly, not by accident.
            self.assertIsNone(env._food_rgb)
        finally:
            env.close()


class TheToleranceIsDerivedNotChosen(unittest.TestCase):
    def test_it_is_half_the_margin_to_the_nearest_confuser(self):
        """A brown cricket on sand really is camouflaged, so the margin is thin
        and the number that sets it must be justified rather than picked."""
        def chroma(rgb):
            a = np.asarray(rgb, dtype=float)
            return a / a.sum()

        prey = chroma(PREY_RGB)
        nearest = min(float(np.abs(chroma(rgb) - prey).sum())
                      for rgb in WORLD_SURFACES.values())
        self.assertAlmostEqual(nearest, 0.1447, places=3,
                               msg="the gecko's own spots are the confuser")
        self.assertLess(FOOD_CHROMA_TOLERANCE, nearest,
                        "the tolerance must not reach the nearest confuser")
        self.assertAlmostEqual(FOOD_CHROMA_TOLERANCE, nearest / 2.0, places=2)

    def test_it_is_documented_as_a_placeholder_for_a_retina(self):
        self.assertIn("PLACEHOLDER FOR A RETINA",
                      _food_visible_frac.__doc__.upper())


if __name__ == "__main__":
    unittest.main()
