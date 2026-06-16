from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from brain.drives import DriveState
from envs.gecko_walk_env import GeckoWalkEnv

REPO = Path(__file__).resolve().parent.parent


def _scene_obj(renderer):
    return getattr(renderer, "scene", getattr(renderer, "_scene", None))


def _add_scene_sphere(renderer, pos, radius=0.025, rgba=(1.0, 0.1, 0.05, 1.0)):
    scene = _scene_obj(renderer)
    if scene is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_scene_capsule(
    renderer,
    start,
    end,
    radius=0.006,
    rgba=(1.0, 0.7, 0.05, 0.9),
):
    scene = _scene_obj(renderer)
    if scene is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    try:
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
        scene.ngeom += 1
    except Exception:
        return


def _food_visible_frac(image: np.ndarray) -> float:
    """Fraction of pixels matching the green food marker in a uint8 HxWx3 image."""
    r = image[:, :, 0].astype(np.int32)
    g = image[:, :, 1].astype(np.int32)
    b = image[:, :, 2].astype(np.int32)
    mask = (g > 120) & (g > r + 30) & (g > b + 30)
    return float(mask.sum()) / float(mask.size)


class GeckoBrainEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        walker_run: str = "v4_5b_speed_polish_1m",
        frame_skip: int = 25,
        max_steps: int = 1000,
        brain_steps_per_action: int = 1,
        food_radius: float = 0.035,
        food_spawn_radius: tuple[float, float] = (0.25, 0.70),
        food_spawn_angle_deg: float = 180.0,
        eat_radius: float = 0.10,
        camera_width: int = 64,
        camera_height: int = 64,
        render_mode: str | None = None,
        seed: int | None = None,
        privileged_target: float = 1.0,
        privileged_food_dropout_prob: float = 0.0,
        control_mode: str = "cpg_residual",
        residual_scale: float = 0.25,
        front_stance_press: float = 0.40,
        front_swing_lift: float = 0.40,
        contact_thresh: float = 0.0564,
        show_debug_markers: bool = False,
        view_mode: str = "close",
        camera_smoothing: float = 0.0,
    ):
        super().__init__()
        self.walker_run = str(walker_run)
        self.show_debug_markers = bool(show_debug_markers)
        _valid_views = ("fixed", "chase", "close")
        if str(view_mode).lower() not in _valid_views:
            raise ValueError(f"view_mode must be one of {_valid_views}, got '{view_mode}'")
        self._view_mode = str(view_mode).lower()
        self._camera_smoothing = float(max(0.0, min(1.0, camera_smoothing)))
        self._smooth_lookat: np.ndarray | None = None
        self._smooth_azimuth: float | None = None
        self.max_steps = int(max_steps)
        self.brain_steps_per_action = int(brain_steps_per_action)
        if self.brain_steps_per_action < 1:
            raise ValueError("brain_steps_per_action must be >= 1")
        self.food_radius = float(food_radius)
        self.food_spawn_radius = (
            float(food_spawn_radius[0]),
            float(food_spawn_radius[1]),
        )
        self.food_spawn_angle_deg = float(food_spawn_angle_deg)
        self.eat_radius = float(eat_radius)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.render_mode = render_mode
        self.privileged_target = float(privileged_target)
        self.privileged_food_dropout_prob = float(np.clip(privileged_food_dropout_prob, 0.0, 1.0))
        self._rng = np.random.default_rng(seed)

        walker_max_steps = max(1, self.max_steps * self.brain_steps_per_action)
        self.walk_env = GeckoWalkEnv(
            frame_skip=frame_skip,
            max_steps=walker_max_steps,
            control_mode=control_mode,
            residual_scale=residual_scale,
            contact_thresh=contact_thresh,
            front_stance_press=front_stance_press,
            front_swing_lift=front_swing_lift,
            render_mode=render_mode,
            seed=seed,
        )
        self._nose_sid = mujoco.mj_name2id(
            self.walk_env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "nose_tip",
        )
        if self._nose_sid < 0:
            raise RuntimeError("XML site 'nose_tip' not found; required for mouth-based eating.")

        self.walk_model, self.walk_norm = self._load_frozen_walker(self.walker_run)
        self.walk_obs_dim = int(np.prod(self.walk_env.observation_space.shape))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    0,
                    255,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.walk_obs_dim,),
                    dtype=np.float32,
                ),
                "drives": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
                "prev_action": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "privileged": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
            }
        )

        self.drives = DriveState()
        self.food_xy = np.zeros(2, dtype=np.float64)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._brain_target_xy = np.zeros(2, dtype=np.float64)
        self._step = 0
        self._head_renderer = None
        self._render_renderer = None
        self._last_info = {}

    def set_privileged_food_scale(self, scale: float) -> None:
        self.privileged_target = float(scale)

    def set_privileged_food_dropout_prob(self, prob: float) -> None:
        self.privileged_food_dropout_prob = float(np.clip(prob, 0.0, 1.0))

    def set_food_radius(self, radius: float) -> None:
        self.food_radius = float(max(radius, 1e-3))

    def _load_frozen_walker(self, walker_run: str):
        run_dir = REPO / "models" / walker_run
        model_path = run_dir / "final.zip"
        vec_path = run_dir / "vecnormalize.pkl"
        missing = [p for p in (model_path, vec_path) if not p.exists()]
        if missing:
            paths = "\n".join(f"  - {p}" for p in missing)
            raise FileNotFoundError(
                "Missing frozen walker artifact(s):\n"
                f"{paths}\n"
                "Brain Phase Patch 1 expects a frozen walker with final.zip and "
                "vecnormalize.pkl. No locomotion training is started here."
            )

        vec_env = DummyVecEnv([lambda: self.walk_env])
        norm = VecNormalize.load(str(vec_path), vec_env)
        norm.training = False
        norm.norm_reward = False
        model = PPO.load(str(model_path), device="cpu")
        return model, norm

    def _trunk_xy(self) -> np.ndarray:
        return self.walk_env.data.xpos[self.walk_env._trunk][:2].copy()

    def _trunk_rot(self) -> np.ndarray:
        return self.walk_env.data.xmat[self.walk_env._trunk].reshape(3, 3).copy()

    def _spawn_food(self) -> None:
        lo, hi = self.food_spawn_radius
        if hi < lo:
            lo, hi = hi, lo
        if self.food_spawn_angle_deg >= 180.0:
            angle = self._rng.uniform(-np.pi, np.pi)
        else:
            a = math.radians(max(self.food_spawn_angle_deg, 0.0))
            forward_world = self._trunk_rot() @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            heading = math.atan2(forward_world[1], forward_world[0])
            angle = heading + self._rng.uniform(-a, a)
        radius = self._rng.uniform(max(lo, 0.0), max(hi, 0.0))
        offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        self.food_xy = self._trunk_xy() + offset

    def _food_delta_body(self):
        trunk_xy = self._trunk_xy()
        d_world = np.array(
            [self.food_xy[0] - trunk_xy[0], self.food_xy[1] - trunk_xy[1], 0.0],
            dtype=np.float64,
        )
        dist = float(np.linalg.norm(d_world[:2])) + 1e-9
        d_body = self._trunk_rot().T @ d_world
        ego = d_body[:2] / dist
        heading = float(np.arctan2(ego[1], ego[0]))
        return ego.astype(np.float32), dist, heading

    def food_egocentric(self):
        ego, dist, heading = self._food_delta_body()
        return ego.copy(), float(dist), float(heading)

    def oracle_action(self, engage: float = 1.0) -> np.ndarray:
        ego, dist, _ = self._food_delta_body()
        dist_cmd = 2.0 * (np.clip(dist, 0.05, 0.80) - 0.05) / 0.75 - 1.0
        engage_cmd = 2.0 * np.clip(float(engage), 0.0, 1.0) - 1.0
        return np.array([ego[0], ego[1], dist_cmd, engage_cmd], dtype=np.float32)

    def _food_distance(self) -> float:
        return float(np.linalg.norm(self.food_xy - self._trunk_xy()))

    def _mouth_food_distance(self) -> float:
        nose_xy = self.walk_env.data.site_xpos[self._nose_sid][:2]
        return float(np.linalg.norm(self.food_xy - nose_xy))

    def _set_walk_target(self, target: np.ndarray) -> None:
        self.walk_env.target = np.asarray(target, dtype=np.float64).copy()
        _, self.walk_env._prev_dist, _ = self.walk_env._target_egocentric()

    def _brain_action_to_target(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dir_body = np.asarray(action[:2], dtype=np.float64)
        dir_norm = float(np.linalg.norm(dir_body))
        if dir_norm < 1e-6:
            dir_body = np.array([1.0, 0.0], dtype=np.float64)
        else:
            dir_body = dir_body / dir_norm

        distance = 0.05 + (float(action[2]) + 1.0) * 0.5 * (0.80 - 0.05)
        engage = (float(action[3]) + 1.0) * 0.5
        engage = float(np.clip(engage, 0.0, 1.0))
        stop_dist = max(float(self.walk_env.reach_dist) + 0.01, 0.05)
        target_dist = stop_dist + engage * (distance - stop_dist)

        body_vec = np.array([dir_body[0], dir_body[1], 0.0], dtype=np.float64)
        world_vec = self._trunk_rot() @ body_vec
        world_dir = world_vec[:2]
        world_norm = float(np.linalg.norm(world_dir))
        if world_norm < 1e-6:
            world_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            world_dir = world_dir / world_norm

        target = self._trunk_xy() + target_dist * world_dir
        self._set_walk_target(target)
        self._brain_target_xy = self.walk_env.target.copy()
        return self._brain_target_xy.copy(), engage

    def _walker_obs_raw(self) -> np.ndarray:
        return self.walk_env._obs().astype(np.float32)

    def _walker_obs_normalized(self) -> np.ndarray:
        raw = self._walker_obs_raw().reshape(1, -1)
        return self.walk_norm.normalize_obs(raw.copy())

    def _head_cam_image(self) -> np.ndarray:
        if self._head_renderer is None:
            self._head_renderer = mujoco.Renderer(
                self.walk_env.model,
                self.camera_height,
                self.camera_width,
            )
        try:
            self._head_renderer.update_scene(self.walk_env.data, camera="head_cam")
            food_xyz = np.array(
                [self.food_xy[0], self.food_xy[1], self.food_radius], dtype=np.float64
            )
            _add_scene_sphere(
                self._head_renderer, food_xyz, radius=self.food_radius,
                rgba=(0.1, 0.95, 0.25, 1.0),
            )
            image = self._head_renderer.render()
        except Exception as exc:
            raise RuntimeError(
                "Failed to render 64x64 RGB image from camera 'head_cam'. "
                "Check that MuJoCo can create a renderer in this process."
            ) from exc

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(
                f"head_cam render returned invalid image shape {image.shape}; "
                "expected HxWx3 RGB."
            )
        image = image[:, :, :3]
        if image.shape[:2] != (self.camera_height, self.camera_width):
            raise RuntimeError(
                f"head_cam render returned {image.shape[:2]}, expected "
                f"{(self.camera_height, self.camera_width)}."
            )
        return image.astype(np.uint8, copy=False)

    def _privileged_vector(self) -> np.ndarray:
        ego, dist, heading = self._food_delta_body()
        privileged = np.array(
            [
                ego[0],
                ego[1],
                np.clip(dist, 0.0, 2.0),
                np.cos(heading),
                np.sin(heading),
            ],
            dtype=np.float32,
        )
        privileged = privileged * np.float32(self.privileged_target)
        if self.privileged_food_dropout_prob > 0.0 and self._rng.random() < self.privileged_food_dropout_prob:
            return np.zeros(5, dtype=np.float32)
        return privileged

    def _obs(self):
        return {
            "image": self._head_cam_image(),
            "proprio": self._walker_obs_raw(),
            "drives": self.drives.vector(),
            "prev_action": self._prev_action.copy(),
            "privileged": self._privileged_vector(),
        }

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.walk_env.reset(seed=seed)
        self.drives.reset()
        self._step = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._smooth_lookat = None
        self._smooth_azimuth = None
        self._spawn_food()
        self._brain_action_to_target(np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32))
        self._last_info = {
            "food_dist": self._food_distance(),
            "mouth_food_dist": self._mouth_food_distance(),
            "ate": False,
            "hunger": self.drives.hunger,
            "energy": self.drives.energy,
            "fear": self.drives.fear,
            "danger": self.drives.danger,
            "engage": 0.0,
            "walker_forward_speed": 0.0,
            "belly_contact": 0.0,
            "fallen": False,
            "brain_target_xy": self._brain_target_xy.copy(),
            "food_xy": self.food_xy.copy(),
        }
        return self._obs(), dict(self._last_info)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        food_dist_before = self._food_distance()
        mouth_dist_before = self._mouth_food_distance()
        trunk_before = self._trunk_xy()
        _, engage = self._brain_action_to_target(action)

        walker_terminated = False
        walker_truncated = False
        last_walker_info = {}
        steps_run = 0
        for _ in range(self.brain_steps_per_action):
            self._set_walk_target(self._brain_target_xy)
            norm_obs = self._walker_obs_normalized()
            walker_action, _ = self.walk_model.predict(norm_obs, deterministic=True)
            walker_action = np.asarray(walker_action, dtype=np.float32).reshape(-1)
            _, _, terminated, truncated, info = self.walk_env.step(walker_action)
            last_walker_info = info
            steps_run += 1
            walker_terminated = walker_terminated or bool(terminated)
            walker_truncated = walker_truncated or bool(truncated)
            if walker_terminated or walker_truncated:
                break

        self._set_walk_target(self._brain_target_xy)
        food_dist_after = self._food_distance()
        mouth_dist_after = self._mouth_food_distance()
        ate = bool(mouth_dist_after <= self.eat_radius)
        fallen = bool(walker_terminated or last_walker_info.get("fallen", 0.0) > 0.5)
        belly_contact = float(last_walker_info.get("belly_contact", 0.0))
        danger = float(np.clip(0.65 * belly_contact + (1.0 if fallen else 0.0), 0.0, 1.0))
        total_dt = max(steps_run, 1) * float(self.walk_env.dt)
        moving_speed = float(np.linalg.norm(self._trunk_xy() - trunk_before) / total_dt)
        moving_drive = float(np.clip(moving_speed / 0.25, 0.0, 1.0))

        self.drives.update(total_dt, ate=ate, danger=danger, moving=moving_drive)

        progress = mouth_dist_before - mouth_dist_after
        r_progress = 12.0 * progress
        r_eat = 10.0 if ate else 0.0
        r_close = 0.5 * max(0.0, 1.0 - mouth_dist_after / 0.20) if ate else 0.0
        r_time = -0.01 * max(steps_run, 1)
        r_danger = -0.35 * danger + (-2.0 if fallen else 0.0)

        if ate:
            self._spawn_food()
            food_dist_after = self._food_distance()
            mouth_dist_after = self._mouth_food_distance()

        self._step += 1
        self._prev_action = action.copy()
        terminated = bool(fallen)
        truncated = bool((self._step >= self.max_steps or walker_truncated) and not terminated)

        obs = self._obs()
        food_visible_frac = _food_visible_frac(obs["image"])
        food_visible_signal = min(food_visible_frac / 0.012, 1.0)
        reward = r_progress + r_eat + r_close + r_time + r_danger

        info = {
            "food_dist": float(food_dist_after),
            "mouth_food_dist": float(mouth_dist_after),
            "ate": ate,
            "hunger": float(self.drives.hunger),
            "energy": float(self.drives.energy),
            "fear": float(self.drives.fear),
            "danger": float(danger),
            "engage": float(engage),
            "walker_forward_speed": float(last_walker_info.get("forward_speed", 0.0)),
            "belly_contact": float(belly_contact),
            "fallen": bool(fallen),
            "brain_target_xy": self._brain_target_xy.copy(),
            "food_xy": self.food_xy.copy(),
            "progress": float(progress),
            "moving_speed": float(moving_speed),
            "food_visible_frac": float(food_visible_frac),
            "food_visible_signal": float(food_visible_signal),
            "food_radius": float(self.food_radius),
            "reward_progress": float(r_progress),
            "reward_eat_bonus": float(r_eat),
            "reward_close_bonus": float(r_close),
            "reward_time_penalty": float(r_time),
            "reward_danger_penalty": float(r_danger),
        }
        self._last_info = dict(info)
        return obs, float(reward), terminated, truncated, info

    def _wide_camera_lookat(self, lookahead: float = 0.8) -> np.ndarray:
        trunk = self.walk_env.data.xpos[self.walk_env._trunk].copy()
        lookat = trunk.copy()
        delta = self._brain_target_xy - trunk[:2]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            lookat[:2] += delta / norm * float(lookahead)
            return lookat
        forward = self._trunk_rot()[:, 0]
        lookat[:2] += forward[:2] * float(lookahead)
        return lookat

    def _add_render_markers(self, renderer) -> None:
        food_xyz = np.array(
            [self.food_xy[0], self.food_xy[1], self.food_radius], dtype=np.float64
        )
        _add_scene_sphere(renderer, food_xyz, radius=self.food_radius,
                          rgba=(0.1, 0.95, 0.25, 1.0))
        if not self.show_debug_markers:
            return

        trunk = self.walk_env.data.xpos[self.walk_env._trunk].copy()
        trunk_ground = np.array([trunk[0], trunk[1], 0.045], dtype=np.float64)
        target_xyz = np.array(
            [self._brain_target_xy[0], self._brain_target_xy[1], 0.045], dtype=np.float64
        )
        _add_scene_sphere(renderer, target_xyz, radius=0.022,
                          rgba=(1.0, 0.15, 0.05, 0.9))
        _add_scene_capsule(renderer, trunk_ground, target_xyz,
                           rgba=(1.0, 0.65, 0.05, 0.9))
        _add_scene_capsule(renderer, target_xyz, food_xyz, radius=0.003,
                           rgba=(0.1, 0.8, 1.0, 0.55))

    def _render_camera_params(self):
        """Return (lookat, azimuth, distance, elevation) for the current view mode."""
        trunk = self.walk_env.data.xpos[self.walk_env._trunk].copy()
        rot = self._trunk_rot()
        forward = rot[:, 0]
        heading_deg = math.degrees(math.atan2(forward[1], forward[0]))

        if self._view_mode == "fixed":
            distance = 2.0
            height = 0.70
            target_azimuth = 130.0
            target_lookat = np.array([trunk[0], trunk[1], 0.05], dtype=np.float64)
        elif self._view_mode == "chase":
            distance = 2.2
            height = 0.70
            # camera sits directly behind: heading_deg + 180 in MuJoCo azimuth space
            target_azimuth = heading_deg + 180.0
            target_lookat = np.array([trunk[0], trunk[1], 0.05], dtype=np.float64)
        else:  # "close" — closer 3/4-rear view
            distance = 1.5
            height = 0.50
            target_azimuth = heading_deg + 180.0
            # slight forward bias so the gecko stays in lower-half of frame
            lookahead = forward[:2] * 0.15
            target_lookat = np.array(
                [trunk[0] + lookahead[0], trunk[1] + lookahead[1], 0.05],
                dtype=np.float64,
            )

        alpha = self._camera_smoothing
        if alpha > 0.0 and self._smooth_lookat is not None and self._smooth_azimuth is not None:
            lookat = alpha * self._smooth_lookat + (1.0 - alpha) * target_lookat
            # wrap azimuth delta to [-180, 180] to avoid spinning through 360
            delta_az = (target_azimuth - self._smooth_azimuth + 180.0) % 360.0 - 180.0
            azimuth = self._smooth_azimuth + (1.0 - alpha) * delta_az
        else:
            lookat = target_lookat.copy()
            azimuth = target_azimuth

        self._smooth_lookat = lookat.copy()
        self._smooth_azimuth = float(azimuth)

        elevation = -math.degrees(math.asin(min(height / distance, 0.9999)))
        return lookat, float(azimuth), float(distance), float(elevation)

    def render(self):
        if self._render_renderer is None:
            self._render_renderer = mujoco.Renderer(self.walk_env.model, 480, 640)
        lookat, azimuth, distance, elevation = self._render_camera_params()
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        self._render_renderer.update_scene(self.walk_env.data, camera=cam)
        self._add_render_markers(self._render_renderer)
        return self._render_renderer.render()

    def close(self):
        if self._head_renderer is not None:
            self._head_renderer.close()
            self._head_renderer = None
        if self._render_renderer is not None:
            self._render_renderer.close()
            self._render_renderer = None
        if getattr(self, "walk_norm", None) is not None:
            self.walk_norm.close()
        elif getattr(self, "walk_env", None) is not None:
            self.walk_env.close()
