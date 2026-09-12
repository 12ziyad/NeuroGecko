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

#: Hind-leg push-off amplitude during a strike, as a fraction of each joint's
#: travel. INVENTED -- no lunge kinematics have been published for this species,
#: and the same-family peak speed (0.57-0.85 m/s, Vollin & Higham 2021) is the
#: thing this is measured AGAINST, never the thing it is fitted to.
LUNGE_EXTEND = 0.55
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
                 approach_site="trunk",
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
        # WHICH PART OF THE ANIMAL HAS TO ARRIVE.
        #
        # "trunk" is the historical behaviour and stays the default: distance is
        # measured from the body centre, and reach_dist 0.04 means the episode
        # succeeds when the TRUNK is 4 cm from the goal. Every checkpoint in the
        # repository was trained that way and changing the default would
        # silently move the ground under all of them.
        #
        # It is also why the animal can never eat. A strike launches from
        # 0.0203 m (Vollin & Higham 2021), the nose sits ~5 cm ahead of the
        # trunk, and the goal test never mentions the mouth -- so a policy that
        # has "arrived" is still two strike-lengths short with its head pointing
        # somewhere else. Measured across 700 steps of a scripted approach with
        # prey motion, a working stalk and a target aimed past the animal, the
        # best mouth-to-prey range reached was 34.5 mm against the 20.3 mm
        # needed (FAILURE_MAP #174).
        #
        # "mouth" measures from the nose_tip site instead. It is the goal a
        # hunting policy actually has to satisfy, and it needs its own training
        # run -- a policy trained on one cannot be graded on the other.
        if approach_site not in ("trunk", "mouth"):
            raise ValueError("approach_site must be 'trunk' or 'mouth'")
        self.approach_site = approach_site
        self._approach_sid = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "nose_tip")
            if approach_site == "mouth" else -1)
        if approach_site == "mouth" and self._approach_sid < 0:
            raise ValueError("This body has no nose_tip site, so the mouth cannot "
                             "be the thing that arrives.")
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
        # The head site the camera sits on. Used for head_velocity() below.
        self._head_site_id = mujoco.mj_name2id(M, O.mjOBJ_SITE, "head_site")
        self._foot_site_id = {
            foot: mujoco.mj_name2id(M, O.mjOBJ_SITE, site)
            for foot, site in _FOOT_SITE_BY_LABEL.items()
        }
        # action mapping (radians / tendon units), neutral = 0 for every actuator
        self.act_low = M.actuator_ctrlrange[:, 0].copy()
        self.act_high = M.actuator_ctrlrange[:, 1].copy()
        self.nu = M.nu
        # THE POLICY'S ACTION IS NOT THE ACTUATOR COUNT (#286). Actuators in
        # MuJoCo group 0 belong to the walker and are what every checkpoint was
        # trained on; group 2 is the brain's -- the jaw -- and is driven by
        # `_apply_jaw`, never by the policy. On a body with no group-2
        # actuators `nu_policy == nu` and nothing here changes.
        self._policy_act = np.flatnonzero(M.actuator_group == 0)
        self.nu_policy = int(len(self._policy_act))
        import mujoco as _mj
        self._jaw_act = _mj.mj_name2id(M, _mj.mjtObj.mjOBJ_ACTUATOR, "jaw")
        #: Commanded gape, radians. 0.0 = closed. Set by the brain's strike.
        self.jaw_rad = 0.0
        #: Commanded eyelid closure, radians. 0.0 = open. Set by the brain.
        self.eyelid_rad = 0.0
        #: Commanded throat (gular) angle, radians. Set by the breathing pump.
        self.gular_rad = 0.0
        self.tongue_m = 0.0   # tongue protrusion, metres (#343)
        #: Hindlimb push-off, 0..1, during a strike. Set by the brain's strike.
        self.lunge = 0.0
        # One actuator per lid since #343: each rolls about the head's long
        # axis, the left in the negative sense and the right in the positive,
        # so one commanded angle is applied with opposite signs. The old single
        # "eyelids" actuator is looked up too so an unsplit body still works.
        self._eyelid_act = _mj.mj_name2id(M, _mj.mjtObj.mjOBJ_ACTUATOR, "eyelids")
        self._eyelid_acts = [
            (_mj.mj_name2id(M, _mj.mjtObj.mjOBJ_ACTUATOR, "eyelid_L"), -1.0),
            (_mj.mj_name2id(M, _mj.mjtObj.mjOBJ_ACTUATOR, "eyelid_R"), +1.0),
        ]
        self._tongue_act = _mj.mj_name2id(M, _mj.mjtObj.mjOBJ_ACTUATOR, "tongue")
        self._gular_act = _mj.mj_name2id(M, _mj.mjtObj.mjOBJ_ACTUATOR, "gular")
        self._lunge_ids = {}
        for _n in ("hip_proret_L", "hip_proret_R", "knee_L", "knee_R",
                   "ankle_L", "ankle_R"):
            _i = _mj.mj_name2id(M, _mj.mjtObj.mjOBJ_ACTUATOR, _n)
            if _i >= 0:
                self._lunge_ids[_n] = _i
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
        #: Commanded downward gaze, RADIANS, positive = look down. 0.0 is
        #: exactly the shipped behaviour: the CPG leaves neck and head at
        #: neutral and this writes nothing. Set by the brain from the EYE's
        #: own reported elevation -- never from the prey's true position.
        self.gaze_pitch_rad = 0.0
        self.gaze_yaw_rad = 0.0
        #: 1.0 walks, 0.0 stands. See CPGResidualController.compute.
        self.locomotor_drive = 1.0
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
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu_policy,), np.float32)

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
                "privileged_target": self.privileged_target,
                "approach_site": self.approach_site,
                "reach_dist": self.reach_dist}

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
    #: Actuator indices for the two pitch joints, resolved once. Both are
    #: position servos already present in the body and driven by nothing: the
    #: CPG writes the whole control vector and leaves them at neutral.
    def _pitch_actuators(self):
        if getattr(self, "_pitch_idx", None) is None:
            import mujoco
            idx = {}
            for name in ("neck_pitch", "head_pitch"):
                i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if i >= 0:
                    idx[name] = i
            self._pitch_idx = idx
        return self._pitch_idx

    def _yaw_actuators(self):
        if getattr(self, "_yaw_idx", None) is None:
            import mujoco
            idx = {}
            for name in ("neck_yaw", "head_yaw"):
                i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if i >= 0:
                    idx[name] = i
            self._yaw_idx = idx
        return self._yaw_idx

    def _apply_gaze_yaw(self):
        """Turn the head left or right by the commanded angle.

        Exactly the pitch routine with a different pair of joints, and for the
        same reason: the morphology carries neck_yaw at +-40 deg and head_yaw at
        +-30 deg, seventy degrees of look-around that nothing in this project
        has ever commanded. tools/oracle_free_diagnosis.py measured the cost of
        that: with the prey-position oracle off, the prey sat BEHIND the camera
        on 581 of 600 frames. The animal had no way to look anywhere but where
        its body pointed.

        UNPUBLISHED. No head-yaw scan amplitude, rate or dwell has been measured
        in *Eublepharis macularius*. What is published is that the head moves
        during a capture at all -- Delheusy, Brillet & Bels 1995 measured about
        8 mm of horizontal head translation -- and that is a capture, not a
        search. The controller is INVENTED and says so.
        """
        if not self.gaze_yaw_rad:
            return
        idx = self._yaw_actuators()
        if not idx:
            return
        spans = {n: float(self.act_high[i] - self.act_low[i]) for n, i in idx.items()}
        total = sum(spans.values())
        if total <= 0.0:
            return
        for n, i in idx.items():
            want = float(self.gaze_yaw_rad) * (spans[n] / total)
            self._ctrl[i] = float(np.clip(want, self.act_low[i], self.act_high[i]))

    def _apply_eyelids(self):
        """Close the eyes by the commanded angle. No-op on a lidless body.

        PUBLISHED, and it is the reason this joint exists at all: Eublepharidae
        are the EYELID geckos -- alone among Gekkota they have movable eyelids
        instead of a fused transparent spectacle. What is NOT published, for
        this species or for any lizard, is a blink rate, a blink duration or a
        lid excursion, and the session-12 literature sweep closed that search
        rather than filling it. Whoever commands this angle owns that gap.
        """
        if self._eyelid_act >= 0:
            i = self._eyelid_act
            self._ctrl[i] = float(np.clip(self.eyelid_rad, self.act_low[i], self.act_high[i]))
        for i, sign in self._eyelid_acts:
            if i >= 0:
                self._ctrl[i] = float(np.clip(sign * self.eyelid_rad,
                                              self.act_low[i], self.act_high[i]))

    def _apply_tongue(self):
        """Slide the tongue out by the commanded distance. No-op without one."""
        if self._tongue_act < 0:
            return
        i = self._tongue_act
        self._ctrl[i] = float(np.clip(self.tongue_m, self.act_low[i], self.act_high[i]))

    def _apply_gular(self):
        """Move the throat floor. No-op on a body without one."""
        if self._gular_act < 0:
            return
        i = self._gular_act
        self._ctrl[i] = float(np.clip(self.gular_rad, self.act_low[i], self.act_high[i]))

    def _apply_lunge(self):
        """Push off with the hind legs. The strike's forward motion, from muscle.

        WHY THIS EXISTS, measured (#297/#301). With a physical mouth the animal
        strikes from wherever its walk stopped -- 13-19 mm short -- and lands 1
        catch in 60, because nothing carries the mouth the last centimetre. The
        published capture is not a walk: *Coleonyx variegatus* (same family)
        lunges from 1.5-2.1 cm at a peak 0.57-0.85 m/s (Vollin & Higham 2021),
        fifteen times this animal's 0.035 m/s walk, and Delheusy et al. 1995
        record the head of *E. macularius* translating ~8 mm forward and ~27 mm
        DOWN through the 80 ms capture.

        The push comes from the HIND LEGS through the existing joints -- hip
        retraction plus knee and ankle extension -- and not from a force applied
        to the trunk. A lunge that is an external impulse is a teleport with a
        physics-shaped name; this one has to be produced by the body, and what
        it achieves is measured rather than asserted.

        `LUNGE_EXTEND` is INVENTED. It is a fraction of each joint's travel, and
        the peak speed it actually produces is the thing to compare against the
        published 0.57-0.85 m/s.
        """
        if not self.lunge or not self._lunge_ids:
            return
        f = float(np.clip(self.lunge, 0.0, 1.0)) * LUNGE_EXTEND
        for n, i in self._lunge_ids.items():
            span = float(self.act_high[i] - self.act_low[i])
            # Retract the hip and extend knee and ankle: the direction that
            # drives the body forward over a planted foot.
            want = self._ctrl[i] + (-f if "proret" in n else f) * span
            self._ctrl[i] = float(np.clip(want, self.act_low[i], self.act_high[i]))

    def _apply_jaw(self):
        """Open the mouth by the commanded gape. A no-op on a jawless body.

        PUBLISHED, TARGET SPECIES, for the range: Delheusy, Brillet & Bels 1995
        (Amphibia-Reptilia 16:185-201) -- peak gape about 37 deg at 47 ms in an
        80 ms open-close cycle, n = 6 adult *E. macularius*, 64 fps. The jaw
        actuator's ctrlrange admits that peak. The TIMING of a strike's gape
        cycle is owned by brain/strike.py, which reads the same numbers.
        """
        if self._jaw_act < 0:
            return
        # ALWAYS written, even at 0.0. The residual controller fills every
        # limited actuator with the midpoint of its ctrlrange as "neutral", and
        # the jaw's midpoint is 20 deg -- measured: with no command the mouth
        # hung 20.1 deg open. A closed mouth is 0, and it is commanded, not
        # assumed (#287).
        i = self._jaw_act
        self._ctrl[i] = float(np.clip(self.jaw_rad, self.act_low[i], self.act_high[i]))

    def _apply_gaze_pitch(self):
        """Point the head down by the commanded angle, sharing it across the
        two joints in proportion to the travel each one has.

        A no-op at 0.0, which is the default, so a walker that never commands
        gaze is bit-identical to the one the gates accepted. The split is
        proportional rather than all-on-one because the neck carries 30 deg and
        the head 20 deg, and loading either to its stop first would make the
        command non-linear exactly where it is needed most.

        PUBLISHED, TARGET SPECIES, for the behaviour but not for the value:
        Delheusy, Brillet & Bels 1995 (Amphibia-Reptilia 16(2):185-201,
        doi:10.1163/156853895X00361) measured the head of *Eublepharis
        macularius* translating about 8 mm horizontally and 28 mm VERTICALLY
        through a capture. A gecko drops its head onto its prey. How far, as a
        function of range, is not published and the controller gain here is
        INVENTED.
        """
        if not self.gaze_pitch_rad:
            return
        idx = self._pitch_actuators()
        if not idx:
            return
        spans = {n: float(self.act_high[i] - self.act_low[i]) for n, i in idx.items()}
        total = sum(spans.values())
        if total <= 0.0:
            return
        for n, i in idx.items():
            want = float(self.gaze_pitch_rad) * (spans[n] / total)
            self._ctrl[i] = float(np.clip(want, self.act_low[i], self.act_high[i]))

    def head_velocity(self):
        """Angular and linear velocity of the head, in the head's own frame.

        THE VESTIBULAR ORGAN IS IN THE SKULL (#247). The only inertial sensors
        on this animal are `gyro_trunk` / `vel_trunk`, mounted above the hips,
        and until session 12 the efference copy that removes self-motion from
        the visual field was computed from them -- while the eye rides on a
        head that yaws +-70 deg relative to that trunk and bobs with every step
        of the gait. Measured walking straight, the head's own yaw rate is 3-4x
        the trunk's: median 0.37 deg/s against 0.13, 95th percentile 2.75
        against 0.73. The prediction was systematically too small and the
        residual was read as the world moving.
        
        The semicircular canals of every vertebrate sit beside the eye. This is
        not privileged information: knowing that your own head is turning is
        exactly what a vestibular system is for.

        NO MORPHOLOGY CHANGE. A `<gyro site="head_site">` would have been the
        obvious way to get this and it is the wrong way -- the world XML is
        generated and hash-checked against a manifest that asserts `nsensor`
        unchanged (`tests/test_world.py`), so adding one alters the body hash
        that the session 3-4 gate evidence references. `mj_objectVelocity`
        returns the same six numbers a sensor on that site would report,
        computed from state that already exists.

        Returns (angular_rad_s, linear_m_s), each a 3-vector.
        """
        import mujoco
        if getattr(self, "_head_site_id", -1) < 0:
            return self._s("gyro_trunk"), self._s("vel_trunk")
        res = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_SITE,
                                 self._head_site_id, res, 1)
        return res[:3], res[3:]

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
        """direction & distance to target expressed in the trunk frame (yaw).

        The DIRECTION is always expressed in the trunk frame -- that is the
        animal's own heading and the controller steers by it. What
        `approach_site` changes is the point the distance is measured FROM:
        the body centre, or the mouth that has to reach the prey.
        """
        root = (self.data.site_xpos[self._approach_sid][:2]
                if self._approach_sid >= 0 else self.data.xpos[self._trunk][:2])
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
        self.jaw_rad = 0.0
        self.eyelid_rad = 0.0
        self.gular_rad = 0.0
        self.tongue_m = 0.0   # tongue protrusion, metres (#343)
        self.lunge = 0.0
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
        if action.shape[0] < self.nu:
            # A policy-sized action on a body with brain-driven actuators: the
            # extra slots are zero residual and are written by `_apply_jaw`.
            padded = np.zeros(self.nu, dtype=np.float32)
            padded[self._policy_act] = action
            action = padded
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
                self.cpg.locomotor_drive = float(self.locomotor_drive)
                self._ctrl = self.cpg.compute(action, self._cpg_t, front_contact=front_contact,
                                              heading_error=heading_error)
            else:
                self.cpg.locomotor_drive = float(self.locomotor_drive)
                self._ctrl = self.cpg.compute(action, self._cpg_t, front_contact=front_contact)
            self._cpg_t += self.control_dt
        else:
            # affine map [-1,1] -> [low, high]
            self._ctrl = self.act_low + (action + 1.0) * 0.5 * (self.act_high - self.act_low)
        self._apply_gaze_pitch()
        self._apply_gaze_yaw()
        self._apply_jaw()
        self._apply_eyelids()
        self._apply_gular()
        self._apply_tongue()
        self._apply_lunge()
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
