"""
gecko_walk_env.py  --  Phase-1 locomotion environment for GeckoBody-R.

State-based (NO camera yet): proprioception + foot/belly contacts + an
egocentric target vector. Camera is added in Phase 2 via a subclass.

Key design points
------------------
* Control decimation: the physics runs at 1/dt = 1250 Hz (dt=0.0008 s), but the
  POLICY must not. We step the simulator `frame_skip` times per action so the
  control rate is ~50 Hz (frame_skip=25). This is essential for both learnability
  and speed -- without it, RL is ~20x slower and barely trains.
* Action = 25 actuator targets in [-1, 1], affine-mapped to each actuator's
  ctrlrange (radians / tendon units). Optional residual scaling for safer
  exploration. An action low-pass (EMA) is available; jerk is also penalised.
* Observations are read from the model's named sensors (already in the XML).
* Termination on flip / belly-slam; truncation at max episode length.

Gymnasium API.  Reward logic lives in rewards/walk_reward.py.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from envs.cpg_residual_controller import CPGResidualController
from rewards.gait_prior import LateralSequenceCPG
from common.energetics import actuator_work

REPO = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO / "morphology" / "gecko_body_r.xml"

# proprio sensors pulled from the model (names must exist in the XML)
_HINGES = ["spine_lat_1", "spine_pitch", "spine_lat_2", "spine_lat_3",
           "neck_yaw", "neck_pitch", "head_yaw", "head_pitch",
           "tail_yaw_1", "tail_lift", "tail_yaw_2", "tail_yaw_3", "tail_yaw_4", "tail_yaw_5",
           "shoulder_sprawl_L", "shoulder_proret_L", "elbow_L", "wrist_L",
           "hip_sprawl_L", "hip_proret_L", "hip_rot_L", "knee_L", "ankle_L",
           "shoulder_sprawl_R", "shoulder_proret_R", "elbow_R", "wrist_R",
           "hip_sprawl_R", "hip_proret_R", "hip_rot_R", "knee_R", "ankle_R"]
_QP = [f"qp_{j}" for j in _HINGES]
_QV = [f"qv_{j}" for j in _HINGES]
_TENDON_P = ["tp_spine_bend", "tp_tail_bend_L", "tp_tail_bend_R"]
_TENDON_V = ["tv_spine_bend", "tv_tail_bend_L", "tv_tail_bend_R"]
_FEET = ["touch_fore_L", "touch_fore_R", "touch_hind_L", "touch_hind_R"]
_BELLY = ["touch_belly_mid", "touch_belly_post"]
_GAIT_FEET = ("HL", "FL", "HR", "FR")
_FOOT_SENSOR_BY_LABEL = {
    "FL": "touch_fore_L",
    "FR": "touch_fore_R",
    "HL": "touch_hind_L",
    "HR": "touch_hind_R",
}
_FOOT_SITE_BY_LABEL = {
    "FL": "footzone_fore_L",
    "FR": "footzone_fore_R",
    "HL": "footzone_hind_L",
    "HR": "footzone_hind_R",
}


class GeckoWalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, xml_path=None, frame_skip=25, max_steps=1000,
                 target_radius=0.25, reach_dist=0.04, action_scale=1.0,
                 action_ema=0.0, reset_noise=0.02, reward_cfg=None,
                 control_mode="raw", residual_scale=0.2, contact_thresh=None,
                 front_stance_press=0.40, front_swing_lift=0.40,
                 render_mode=None, seed=None, gait_profile="legacy", lab_parameters=None,
                 hind_stance_compensation=False, front_lift_residual_scale=None,
                 stance_target_clearance_m=None, stance_sprawl_nodes=None,
                 stance_sprawl_limit_rad=None, privileged_target=True,
                 prey_parameters=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or DEFAULT_XML))
        self.data = mujoco.MjData(self.model)
        # Optional read-only evaluation observer; never changes control cadence.
        self.physics_observer = None
        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip          # control dt (~0.02 s)
        self.max_steps = int(max_steps)
        self.target_radius = float(target_radius)
        self.reach_dist = float(reach_dist)
        # The five task observations hand the policy the target's bearing and
        # range directly: the animal is told where the food is instead of
        # looking for it, and the lab controller is additionally steered by the
        # same privileged bearing. With this False the policy must find the
        # target through the camera, and the controller steers straight.
        #
        # The REWARD still uses target distance. That is a separate shortcut,
        # named in docs/HANDOFF.md alongside oracle supervision, and it is not
        # in the animal's senses -- a reward is external by construction. It is
        # left in place and left recorded rather than quietly conflated with
        # this one.
        self.privileged_target = bool(privileged_target)
        self.action_scale = float(action_scale)
        self.action_ema = float(action_ema)
        self.reset_noise = float(reset_noise)
        if control_mode not in ("raw", "cpg_residual"):
            raise ValueError("control_mode must be 'raw' or 'cpg_residual'")
        self.control_mode = control_mode
        self.residual_scale = float(residual_scale)
        self.front_stance_press = float(front_stance_press)
        self.front_swing_lift = float(front_swing_lift)
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        # Prey is opt-in and needs a world that has somewhere to put it. The
        # mocap body renders to the camera and never enters qpos, so an episode
        # with prey has exactly the same physics state as one without.
        self.prey = None
        self._prey_mocap = -1
        if prey_parameters is not None:
            from envs.prey import FleeingPrey
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")
            if body < 0:
                raise ValueError("This model has no prey body. Generate a world with "
                                 "utils/build_world.py --prey-radius before asking for prey.")
            self._prey_mocap = int(self.model.body_mocapid[body])
            if self._prey_mocap < 0:
                raise ValueError("The prey body must be mocap so it stays out of qpos.")
            self.prey = FleeingPrey(prey_parameters, height_m=prey_parameters.radius_m,
                                    rng=self._rng)
        self.gait = LateralSequenceCPG(gait_profile=gait_profile)
        self.gait_profile = self.gait.gait_profile
        from rewards.walk_reward import WalkReward, lab_reward_calibration
        if self.gait_profile == "lab":
            self.reward_calibration = lab_reward_calibration(self.model, self.gait.profile)
            self.contact_threshold = float(self.reward_calibration["contact_threshold_N"]
                                           if contact_thresh is None else contact_thresh)
            if not np.isfinite(self.contact_threshold) or self.contact_threshold < 0:
                raise ValueError("Lab contact threshold must be finite and nonnegative.")
            # Old phase presets contain old-body dimensional targets. Keep their
            # weights but resolve these lab targets from this body's landmarks.
            overrides = self.reward_calibration["reward_overrides"]
            self.reward_calibration["superseded_reward_cfg"] = {
                k: {"requested": reward_cfg[k], "resolved": v}
                for k, v in overrides.items() if reward_cfg and k in reward_cfg and reward_cfg[k] != v
            }
            resolved_reward_cfg = dict(reward_cfg or {})
            resolved_reward_cfg.update(overrides)
            self.reward_calibration["effective_contact_threshold_N"] = self.contact_threshold
            self.reward_calibration["contact_threshold_source"] = (
                "derived bodyweight-scaled engineering default" if contact_thresh is None else "explicit caller override")
        else:
            # None preserves the original environment default; training/eval
            # CLIs separately retain their historical .0564 N legacy default.
            self.contact_threshold = float(1e-6 if contact_thresh is None else contact_thresh)
            self.reward_calibration = {"profile": "legacy", "status": "historical reward/contact behavior preserved",
                                       "effective_contact_threshold_N": self.contact_threshold}
            resolved_reward_cfg = reward_cfg

        M, O = self.model, mujoco.mjtObj
        self._kf_stand = mujoco.mj_name2id(M, O.mjOBJ_KEY, "stand")
        self._trunk = mujoco.mj_name2id(M, O.mjOBJ_BODY, "trunk_middle")
        sid = lambda n: mujoco.mj_name2id(M, O.mjOBJ_SENSOR, n)
        self._sid = {n: sid(n) for n in (_QP + _QV + _TENDON_P + _TENDON_V
                                         + _FEET + _BELLY + ["up_trunk", "gyro_trunk", "vel_trunk"])}
        self._foot_site_id = {
            foot: mujoco.mj_name2id(M, O.mjOBJ_SITE, site)
            for foot, site in _FOOT_SITE_BY_LABEL.items()
        }
        # action mapping (radians / tendon units), neutral = 0 for every actuator
        self.act_low = M.actuator_ctrlrange[:, 0].copy()
        self.act_high = M.actuator_ctrlrange[:, 1].copy()
        self.nu = M.nu
        self.control_dt = self.dt
        self._cpg_t = 0.0
        self._last_cpg_command_time_s = 0.0
        self.cpg = None
        # Retained for the training checkpoint contract: a resumed lab policy must
        # be able to prove it meets the same base controller it was trained on.
        self.hind_stance_compensation = hind_stance_compensation
        if self.control_mode == "cpg_residual":
            cpg_kwargs = {}
            if stance_sprawl_nodes is not None:
                cpg_kwargs["stance_sprawl_nodes"] = int(stance_sprawl_nodes)
            if stance_sprawl_limit_rad is not None:
                cpg_kwargs["stance_sprawl_limit_rad"] = float(stance_sprawl_limit_rad)
            if stance_target_clearance_m is not None:
                cpg_kwargs["stance_target_clearance_m"] = (
                    dict(stance_target_clearance_m) if isinstance(stance_target_clearance_m, dict)
                    else float(stance_target_clearance_m))
            if front_lift_residual_scale is not None:
                # The lock itself stays on; only its scale is caller-selected, so
                # the front-lift channels remain explicitly enumerated and capped.
                cpg_kwargs["front_lift_residual_scale"] = float(front_lift_residual_scale)
            self.cpg = CPGResidualController(
                self.model,
                residual_scale=self.residual_scale,
                front_stance_press=self.front_stance_press,
                front_swing_lift=self.front_swing_lift,
                verbose=False,
                gait_profile=self.gait.profile,
                lab_parameters=lab_parameters,
                hind_stance_compensation=hind_stance_compensation,
                **cpg_kwargs,
            )

        # build one obs to size the space
        mujoco.mj_resetDataKeyframe(M, self.data, self._kf_stand)
        mujoco.mj_forward(M, self.data)
        self._prev_action = np.zeros(self.nu)
        self._ctrl = np.zeros(self.nu)
        self._step = 0
        self._cpg_t = 0.0
        self.target = np.array([1.0, 0.0])
        _, self._prev_dist, _ = self._target_egocentric()
        self._prev_foot_xy = self._foot_xy().copy()
        self._last_step_metrics = {}
        self._step_work = np.zeros(3, dtype=np.float64)
        self._episode_work = np.zeros(3, dtype=np.float64)
        self._episode_path_m = 0.0
        self._work_prev_xy = self.data.xpos[self._trunk, :2].copy()
        obs = self._obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)

        self.reward_fn = WalkReward(resolved_reward_cfg)
        self._renderer = None

    @property
    def observation_layout(self):
        """Named observation blocks, so a shape change is legible in a config."""
        blocks = [("joint_position", 32), ("joint_velocity", 32), ("tendon_position", 3),
                  ("tendon_velocity", 3), ("foot_force", 4), ("belly_force", 2),
                  ("gravity_up", 3), ("gyro", 3), ("linear_velocity", 3)]
        if self.privileged_target:
            blocks.append(("privileged_target", 5))
        blocks.append(("gait_phase", 2))
        return {"blocks": blocks, "total": sum(n for _, n in blocks),
                "privileged_target": self.privileged_target}

    @property
    def lab_controller_snapshot(self):
        """Plain picklable record of the effective lab base controller.

        BLOCKED.md requires the exact effective `cpg.lab_parameters` to be
        persisted in checkpoint contracts before lab training or resume; the
        older timing-only contract does not cover the mechanical controls.
        This is that record, read from the live controller rather than from
        caller intent, so a vectorised worker cannot silently disagree with the
        config. Returns None for legacy, leaving historical runs unchanged.
        """
        if self.gait_profile != "lab" or self.cpg is None:
            return None
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(self.model.nu)]
        compensation = self.hind_stance_compensation
        return {
            "effective_lab_parameters": dict(self.cpg.lab_parameters),
            "hind_stance_compensation": (compensation if isinstance(compensation, str)
                                         else bool(compensation)),
            "phase_offsets_cycle": dict(self.cpg.phase),
            "commanded_stance_by_foot": {foot: self.cpg.stance_for(foot)
                                         for foot in _GAIT_FEET},
            "frequency_hz": float(self.cpg.freq),
            "stance_target_clearance_m": (None if self.cpg._hind_comp is None
                                          else self.cpg._hind_comp.target_clearance_m),
            "residual_scale_by_actuator": {name: float(scale) for name, scale
                                           in zip(names, self.cpg.res_scale_vec)},
        }

    # ---- sensor access ----------------------------------------------------
    def _s(self, name):
        s = self._sid[name]; a = self.model.sensor_adr[s]; d = self.model.sensor_dim[s]
        return self.data.sensordata[a:a + d]

    def _gait_time(self):
        return self._step * self.dt

    def _foot_contact_forces(self):
        return np.array([
            float(self._s(_FOOT_SENSOR_BY_LABEL[foot])[0])
            for foot in _GAIT_FEET
        ], dtype=np.float32)

    def _foot_contacts(self):
        return (self._foot_contact_forces() > self.contact_threshold).astype(np.float32)

    def _belly_contact_forces(self):
        return np.array([float(self._s(n)[0]) for n in _BELLY], dtype=np.float32)

    def _foot_xy(self):
        return np.array([
            self.data.site_xpos[self._foot_site_id[foot]][:2]
            for foot in _GAIT_FEET
        ], dtype=np.float64)

    def _write_prey(self):
        """Publish prey position into the mocap slot the camera renders."""
        self.data.mocap_pos[self._prey_mocap] = self.prey.mocap_position
        mujoco.mj_forward(self.model, self.data)

    def _target_egocentric(self):
        """direction & distance to target expressed in the trunk frame (yaw)."""
        root = self.data.xpos[self._trunk][:2]
        R = self.data.xmat[self._trunk].reshape(3, 3)
        d_world = np.array([self.target[0] - root[0], self.target[1] - root[1], 0.0])
        dist = float(np.linalg.norm(d_world[:2])) + 1e-9
        d_body = R.T @ d_world
        ego = d_body[:2] / dist                      # unit direction in body frame
        heading_err = float(np.arctan2(ego[1], ego[0]))
        return ego, dist, heading_err

    def _obs(self):
        qp = np.concatenate([self._s(n) for n in _QP])
        qv = np.concatenate([self._s(n) for n in _QV])
        tp = np.concatenate([self._s(n) for n in _TENDON_P])
        tv = np.concatenate([self._s(n) for n in _TENDON_V])
        feet = np.concatenate([np.tanh(self._s(n) * 20.0) for n in _FEET])   # squashed force
        belly = np.concatenate([np.tanh(self._s(n) * 20.0) for n in _BELLY])
        up = self._s("up_trunk")            # gravity/up vector in trunk frame (3)
        gyro = self._s("gyro_trunk")        # ang vel (3)
        vel = self._s("vel_trunk")          # lin vel in trunk frame (3)
        phase = self.gait.phase_observation(self._gait_time())
        parts = [qp, qv, tp, tv, feet, belly, up, gyro, vel]
        if self.privileged_target:
            ego, dist, head = self._target_egocentric()
            parts.append(np.array([ego[0], ego[1], np.clip(dist, 0, 2.0),
                                   np.cos(head), np.sin(head)]))
        parts.append(phase)
        return np.concatenate(parts).astype(np.float32)

    def _step_metrics(self, dist, head, up_z, reached, fallen):
        # Lab scores the stance schedule that generated this held control
        # interval, not the next interval's schedule. Keep historical end-time
        # reward timing untouched for legacy checkpoints. The policy's phase
        # observation still uses the current clock in _obs().
        time_s = (self._last_cpg_command_time_s
                  if self.gait_profile == "lab" and self.cpg is not None
                  else self._gait_time())
        target_contacts = self.gait.target_contact_array(time_s, _GAIT_FEET)
        foot_forces = self._foot_contact_forces()
        foot_contacts = (foot_forces > self.contact_threshold).astype(np.float32)
        gait_match = float(np.mean(foot_contacts == target_contacts))

        foot_xy = self._foot_xy()
        prev_foot_xy = getattr(self, "_prev_foot_xy", foot_xy)
        foot_speed = np.linalg.norm((foot_xy - prev_foot_xy) / max(self.dt, 1e-9), axis=1)
        stance_mask = (target_contacts > 0.5) & (foot_contacts > 0.5)
        swing_mask = target_contacts < 0.5
        slip = float(np.mean(foot_speed[stance_mask])) if np.any(stance_mask) else 0.0
        swing_contact = float(np.mean(foot_contacts[swing_mask])) if np.any(swing_mask) else 0.0

        belly_forces = self._belly_contact_forces()
        belly_contact = float(np.any(belly_forces > self.contact_threshold))
        belly_force = float(np.clip(np.sum(belly_forces), 0.0, 1.0))
        progress = float((self._prev_dist - dist) / max(self.dt, 1e-9))
        forward_speed = float(self._s("vel_trunk")[0])
        yaw_rate = abs(float(self._s("gyro_trunk")[2]))

        metrics = dict(
            reached=float(reached),
            fallen=float(fallen),
            gait_match=gait_match,
            progress=progress,
            belly_contact=belly_contact,
            belly_force=belly_force,
            forward_speed=forward_speed,
            slip=slip,
            swing_contact=swing_contact,
            distance=float(dist),
            dist=float(dist),
            heading_error=float(head),
            up_z=float(up_z),
            yaw_rate=yaw_rate,
            foot_contacts=foot_contacts.copy(),
            target_contacts=target_contacts.copy(),
            foot_contact_forces=foot_forces.copy(),
            foot_speed=foot_speed.astype(np.float32),
            gait_phase=self.gait.phase(time_s),
            gait_target_time_s=float(time_s),
            mechanical_work_positive_J=float(self._step_work[0]),
            mechanical_work_negative_J=float(self._step_work[1]),
            mechanical_work_abs_J=float(self._step_work[2]),
            mechanical_work_episode_abs_J=float(self._episode_work[2]),
            mechanical_path_length_m=float(self._episode_path_m),
            mechanical_work_J_per_m=(float(self._episode_work[2] / self._episode_path_m)
                                     if self._episode_path_m > 1e-9 else None),
        )
        return metrics, foot_xy

    # ---- gym API ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        M, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(M, d, self._kf_stand)
        d.qpos[7:] += self._rng.uniform(-self.reset_noise, self.reset_noise, M.nq - 7)
        d.qvel[:] += self._rng.uniform(-self.reset_noise, self.reset_noise, M.nv)
        mujoco.mj_forward(M, d)
        ang = self._rng.uniform(-np.pi, np.pi)
        self.target = self.target_radius * np.array([np.cos(ang), np.sin(ang)]) \
            + d.xpos[self._trunk][:2]
        self._prev_action = np.zeros(self.nu)
        self._ctrl = np.zeros(self.nu)
        self._step = 0
        self._cpg_t = 0.0
        self._last_cpg_command_time_s = 0.0
        if self.prey is not None:
            # Prey owns the target once it exists: the reward keeps measuring
            # distance to food, but the food now moves and runs away.
            self.target = self.prey.reset(d.xpos[self._trunk][:2], rng=self._rng)
            self._write_prey()
        _, self._prev_dist, _ = self._target_egocentric()
        self._prev_foot_xy = self._foot_xy().copy()
        self._last_step_metrics = {}
        self._step_work[:] = 0.0
        self._episode_work[:] = 0.0
        self._episode_path_m = 0.0
        self._work_prev_xy = d.xpos[self._trunk, :2].copy()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0) * self.action_scale
        if self.action_ema > 0:
            action = self.action_ema * self._prev_action + (1 - self.action_ema) * action
        if self.control_mode == "cpg_residual":
            fc = self._foot_contacts()  # [HL, FL, HR, FR] in _GAIT_FEET order
            front_contact = {"FL": bool(fc[1] > 0.5), "FR": bool(fc[3] > 0.5)}
            self._last_cpg_command_time_s = self._cpg_t
            if self.gait_profile == "lab":
                # Steering off the target bearing is the same privileged channel
                # as the task observation, so it goes with it.
                _, _, heading_error = (self._target_egocentric() if self.privileged_target
                                       else (None, None, 0.))
                self._ctrl = self.cpg.compute(action, self._cpg_t, front_contact=front_contact,
                                              heading_error=heading_error)
            else:
                self._ctrl = self.cpg.compute(action, self._cpg_t, front_contact=front_contact)
            self._cpg_t += self.control_dt
        else:
            # affine map [-1,1] -> [low, high]
            self._ctrl = self.act_low + (action + 1.0) * 0.5 * (self.act_high - self.act_low)
        self.data.ctrl[:] = self._ctrl
        self._step_work[:] = 0.0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            work = actuator_work(self.data.actuator_force,
                                 self.data.actuator_velocity,
                                 self.model.opt.timestep)
            self._step_work += work
            if self.physics_observer is not None:
                self.physics_observer(work)
        if self.prey is not None:
            position, captured = self.prey.step(self.dt, self.data.xpos[self._trunk][:2])
            self.target = position
            self._write_prey()
            self._prey_captured_this_step = captured
        self._episode_work += self._step_work
        work_xy = self.data.xpos[self._trunk, :2].copy()
        self._episode_path_m += float(np.linalg.norm(work_xy - self._work_prev_xy))
        self._work_prev_xy = work_xy
        self._step += 1

        ego, dist, head = self._target_egocentric()
        up_z = float(self._s("up_trunk")[2])
        reached = dist < self.reach_dist
        flipped = up_z < 0.3                              # tipped past ~70 deg
        metrics, foot_xy = self._step_metrics(dist, head, up_z, reached, flipped)
        reward, reward_info = self.reward_fn(self, action, metrics)
        info = dict(reward_info)
        info.update(metrics)
        self._prev_action = action.copy()
        self._prev_dist = dist
        self._prev_foot_xy = foot_xy.copy()
        self._last_step_metrics = metrics
        if self.prey is not None:
            # Prey decides when it has been caught, by its own capture distance,
            # and respawns itself. The env's reach_dist goal-resampling is the
            # static-target behaviour and does not apply.
            info["prey_captured"] = float(getattr(self, "_prey_captured_this_step", False))
            info["prey_captures"] = float(self.prey.captures)
            if info["prey_captured"]:
                _, self._prev_dist, _ = self._target_egocentric()
        elif reached:                                      # resample a new goal, keep going
            ang = self._rng.uniform(-np.pi, np.pi)
            self.target = self.target_radius * np.array([np.cos(ang), np.sin(ang)]) \
                + self.data.xpos[self._trunk][:2]
            _, self._prev_dist, _ = self._target_egocentric()
        terminated = bool(flipped)
        truncated = self._step >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = self.data.xpos[self._trunk]
        cam.distance, cam.azimuth, cam.elevation = 0.34, 130, -18
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
