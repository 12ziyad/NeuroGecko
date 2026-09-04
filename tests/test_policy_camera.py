"""Camera contracts: graphics isolation is narrower than physical invariance."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from envs.gecko_brain_env import GeckoBrainEnv, _camera_scene_options


class CameraOptionTests(unittest.TestCase):
    def test_legacy_matches_mujoco_default_options(self):
        policy, human = _camera_scene_options("legacy")
        default = mujoco.MjvOption()
        for field in ("geomgroup", "sitegroup", "skingroup", "flags"):
            np.testing.assert_array_equal(getattr(policy, field), getattr(default, field))
            np.testing.assert_array_equal(getattr(human, field), getattr(default, field))

    def test_sealed_options_are_independent_of_human_options(self):
        policy, human = _camera_scene_options("sealed")
        self.assertEqual(policy.geomgroup[1], 0)
        self.assertEqual(policy.geomgroup[3], 0)
        self.assertFalse(policy.sitegroup.any())
        self.assertFalse(policy.skingroup.any())
        self.assertEqual(human.geomgroup[1], 1)
        self.assertEqual(human.geomgroup[3], 0)
        self.assertEqual(human.sitegroup[4], 0)
        policy.geomgroup[:] = 0
        self.assertEqual(human.geomgroup[0], 1)

    def test_invalid_mode_fails_before_loading_a_walker(self):
        with self.assertRaisesRegex(ValueError, "policy_camera_mode"):
            GeckoBrainEnv(policy_camera_mode="silent-fallback")

    def test_scene_excludes_body_and_all_sites(self):
        model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "morphology/gecko_body_r.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        # Head poses are synthetic boundary tests, not biological target values.
        for name in ("head_yaw", "head_pitch", "neck_yaw", "neck_pitch"):
            data.qpos[model.joint(name).qposadr[0]] = model.joint(name).range[1]
        mujoco.mj_forward(model, data)
        policy, _ = _camera_scene_options("sealed")
        scene, camera = mujoco.MjvScene(model, maxgeom=1000), mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = model.camera("head_cam").id
        mujoco.mjv_updateScene(model, data, policy, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        for geom in scene.geoms[:scene.ngeom]:
            self.assertNotEqual(geom.objtype, mujoco.mjtObj.mjOBJ_SITE)
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
                self.assertNotIn(model.geom_group[geom.objid], (1, 3))


class CameraPixelTests(unittest.TestCase):
    def setUp(self):
        # Test fixture coordinates/colors are engineering test inputs, not
        # measured gecko parameters or simulator defaults.
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <visual><global offwidth="64" offheight="64"/></visual>
          <worldbody><light pos="0 0 3"/>
            <camera name="head_cam" pos="0 0 2" fovy="60"/>
            <geom name="floor" type="plane" size="2 2 .1" pos="0 0 -.2" rgba=".4 .4 .4 1"/>
            <body><geom name="visual_body" type="sphere" size=".3" group="1" rgba=".8 .2 .1 1"/>
              <site name="site_zero" pos="-.5 0 0" size=".1" rgba="0 1 0 1"/>
              <site name="site_four" pos=".5 0 0" size=".1" group="4" rgba="0 1 0 1"/>
            </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.env = GeckoBrainEnv.__new__(GeckoBrainEnv)
        self.env.walk_env = SimpleNamespace(model=model, data=data)
        self.env.camera_height = self.env.camera_width = 64
        self.env.food_xy, self.env.food_radius = np.array([.7, .4]), .08
        self.env.policy_camera_mode = "sealed"
        self.env._policy_scene_option, self.env._render_scene_option = _camera_scene_options("sealed")
        self.env._head_renderer = None
        try:
            self.human = mujoco.Renderer(model, 64, 64)
        except Exception as exc:
            self.skipTest(f"Offscreen GL unavailable (option/scene tests still run): {exc}")

    def tearDown(self):
        if hasattr(self, "human"):
            self.human.close()
        if getattr(self.env, "_head_renderer", None) is not None:
            self.env._head_renderer.close()

    def human_image(self):
        self.human.update_scene(self.env.walk_env.data, camera="head_cam", scene_option=self.env._render_scene_option)
        self.human.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        return self.human.render().copy()

    def test_hidden_visual_edits_do_not_change_policy_pixels(self):
        before = self.env._head_cam_image().copy()
        human_before = self.human_image()
        model = self.env.walk_env.model
        model.geom("visual_body").rgba[:] = [0, 1, 0, 1]
        model.site_rgba[:] = [1, 0, 1, 1]
        np.testing.assert_array_equal(self.env._head_cam_image(), before)
        self.assertTrue(np.any(self.human_image() != human_before))
        self.assertEqual(self.env._head_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW], 0)
        self.assertEqual(self.human.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW], 1)
        np.testing.assert_array_equal(model.light_castshadow, [1])

    def test_legacy_and_sealed_are_explicitly_different_contracts(self):
        sealed = self.env._head_cam_image().copy()
        self.env.policy_camera_mode = "legacy"
        self.env._policy_scene_option, self.env._render_scene_option = _camera_scene_options("legacy")
        legacy = self.env._head_cam_image()
        self.assertTrue(np.any(sealed != legacy))
        self.assertEqual(self.env._head_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW], 1)


if __name__ == "__main__":
    unittest.main()
