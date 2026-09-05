"""
walk_reward.py -- CPG-guided walking reward for GeckoWalkEnv.

This V4.2.1 objective keeps target progress always-on while a CPG-residual
controller supplies the gait scaffold. Reward shaping favors hind propulsion,
front stance loading, and a higher supported trunk.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np

from common.energetics import mean_absolute_actuator_power

_FOOT_INDEX = {"HL": 0, "FL": 1, "HR": 2, "FR": 3}

DEFAULTS = dict(
    alive=0.01,
    progress=12.0,
    forward=1.5,
    gait=0.40,
    gait_match_target=0.65,
    front_phase_match=0.0,
    front_phase_target=0.70,
    freeze=0.22,
    slip=0.15,
    swing_contact=0.10,
    belly=0.30,
    belly_force=0.15,
    spin=0.30,
    smooth=0.025,
    reach=10.0,
    fall=6.0,
    progress_cap=0.50,
    reverse_cap=0.30,
    forward_cap=0.35,
    freeze_progress=0.004,
    freeze_speed=0.012,
    yaw_rate_cap=2.0,
    v4_metric_window=50,
    v4_speed_threshold=0.025,
    trunk_height_min=0.028,
    trunk_height_target=0.040,
    trunk_support=0.70,
    low_trunk=0.10,
    front_load=1.20,
    front_load_force_scale=0.0564,
    # V4.2.7: direct front contact-DURATION (duty) objective.
    # front_duty       -> additive reward weight for windowed front stance fraction
    # *_target         -> duty at which each front foot's score saturates at 1.0.
    #                     Set ~0.05 ABOVE the gate (gate: FL>=0.40, FR>=0.30) so PPO
    #                     overshoots and clears the gate with margin, not at the edge.
    front_duty=0.80,
    front_duty_fl_target=0.45,
    front_duty_fr_target=0.35,
    belly_rate=0.05,
    hind_participation=0.0,
    hind_push=0.90,
    hind_push_target=0.05,
    foot_participation=0.02,
    front_pair_sync=0.20,
    front_pair_sync_free=0.40,
    front_pair_hop=0.18,
    front_pair_hop_free=0.05,
    body_bounce=0.10,
    body_bounce_free=0.008,
    # V4.2.8: ungated CPG front contact-reference tracking.
    # These terms are deliberately NOT multiplied by speed_gate, so they supply
    # a gradient even at low/zero speed. V4.2.7's front terms were all gated and
    # self-throttled (FL moved only 0.327 -> 0.342). Design intent:
    #   commanded front stance & contact   -> reward  (front_track)
    #   commanded front stance & airborne   -> penalty (front_miss, asymmetric > hit)
    #   commanded front swing  & contact     -> small penalty (front_swing_penalty)
    front_track=1.10,
    front_miss=1.60,
    front_swing_penalty=0.25,

    # V4.3: target-speed objective. Current good gait is safe but stuck at ~0.09 m/s.
    # This rewards moving into the 0.12 m/s band while preserving front-support gates.
    target_speed=0.12,
    speed_floor=0.09,
    speed_track_sigma=0.035,
    speed_track=0.0,
    slow_speed=0.105,
    slow_penalty=0.0,
)


def lab_reward_calibration(model, gait_profile="lab"):
    """Resolve engineered lab guards from this model, without changing its state.

    This is reward calibration, not a successful biological validation. In
    particular, the scalar touch-force stance threshold is NOT skin tactile
    sensitivity. The height guard is a one-sided root-height proxy and cannot
    verify both girdle heights, trunk pitch, or excessive body elevation.
    """
    import mujoco
    from common.gait_config import get_gait_profile
    from common.provenance import parameter_value

    profile = get_gait_profile(gait_profile)
    if profile.name != "lab":
        raise ValueError("Body-aware reward calibration is opt-in for lab only.")
    gravity = np.asarray(model.opt.gravity, dtype=float)
    if not (np.isfinite(gravity).all() and gravity[2] < 0 and np.allclose(gravity[:2], 0, atol=1e-12)):
        raise ValueError("Lab root-height calibration requires downward vertical gravity and a level-floor model.")

    def required(kind, name):
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError("Lab reward calibration requires model landmark/key: " + name)
        return index

    site = {name: required(mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("nose_tip", "vent", "hip_L", "hip_R", "shoulder_L", "shoulder_R")}
    trunk = required(mujoco.mjtObj.mjOBJ_BODY, "trunk_middle")
    stand = required(mujoco.mjtObj.mjOBJ_KEY, "stand")
    reference = mujoco.MjData(model)
    mujoco.mj_forward(model, reference)
    svl = float(np.linalg.norm(reference.site_xpos[site["nose_tip"]] - reference.site_xpos[site["vent"]]))
    mujoco.mj_resetDataKeyframe(model, reference, stand)
    mujoco.mj_forward(model, reference)
    root_height = float(reference.xpos[trunk, 2])
    offsets = {girdle: float(np.mean([reference.site_xpos[site[girdle+"_"+side], 2]
                                    for side in ("L", "R")]) - root_height)
               for girdle in ("hip", "shoulder")}
    mass = float(np.sum(model.body_mass[1:]))
    if not (np.isfinite(svl) and svl > 0 and np.isfinite(mass) and mass > 0):
        raise ValueError("Lab calibration requires positive finite neutral SVL and total moving-body mass.")
    bodyweight = mass * float(-gravity[2])
    force_fraction = float(parameter_value("lab_contact_bodyweight_fraction"))
    force_scale = force_fraction * bodyweight
    root_targets = {g: float(parameter_value(g+"_height_svl"))*svl-offsets[g]
                    for g in ("hip", "shoulder")}
    nominal_root = float(np.mean(list(root_targets.values())))
    accepted_height_margin = float(parameter_value("girdle_height_svl_tolerance"))*svl
    transition = float(parameter_value("lab_height_guard_transition_svl"))*svl
    height_target = nominal_root - accepted_height_margin
    height_min = height_target - transition
    stride_range = np.asarray(parameter_value("stride_length_svl_range"), dtype=float)
    stride_choice = float(parameter_value("lab_speed_stride_fraction"))
    if not (force_fraction > 0 and transition > 0 and accepted_height_margin >= 0 and
            height_min > 0 and stride_range.shape == (2,) and
            0 < stride_range[0] < stride_choice < stride_range[1]):
        raise ValueError("Invalid lab calibration registry values or incompatible model geometry.")
    metres_per_stride_fraction_second = svl * profile.frequency_hz
    speed = stride_choice * metres_per_stride_fraction_second
    speed_floor = float(stride_range[0]) * metres_per_stride_fraction_second
    speed_sigma = float((stride_range[1]-stride_range[0])/2) * metres_per_stride_fraction_second
    overrides = dict(
        trunk_height_min=height_min, trunk_height_target=height_target,
        front_load_force_scale=force_scale,
        front_duty_fl_target=profile.stance_for("FL"), front_duty_fr_target=profile.stance_for("FR"),
        target_speed=speed, speed_floor=speed_floor, speed_track_sigma=speed_sigma,
        slow_speed=speed_floor, slow_penalty=0.0,
        total_power_effort_alpha=float(parameter_value("lab_total_power_effort_alpha")),
    )
    return {
        "profile": "lab", "schema_version": 1,
        "status": "ENGINEERED reward guards; not biological validation or a cutaneous-sensitivity model",
        "contact_threshold_N": force_scale, "reward_overrides": overrides,
        "model_inputs": {"mass_kg": mass, "gravity_m_s2": float(-gravity[2]),
                         "bodyweight_N": bodyweight, "neutral_svl_m": svl,
                         "stand_root_height_m": root_height, "stand_girdle_offsets_from_root_m": offsets},
        "height_guard": {"girdle_implied_root_targets_m": root_targets,
                         "nominal_root_target_m": nominal_root,
                         "saturation_margin_m": accepted_height_margin, "linear_transition_m": transition,
                         "method": "mean(target girdle height minus stand-site/root offset), saturating one tolerance below that mean; linear guard one engineered width lower",
                         "limitation": "Root-height support only; high posture is not penalized here and both girdle heights still require independent measurement."},
        "contact_classifier": {"bodyweight_fraction": force_fraction,
                               "method": "scalar foot touch force > engineered fraction * model mass * vertical gravity; front-load tanh scale uses same derived force",
                               "limitation": "No equivalence to skin tactile sensitivity; not a measured optimal contact detector; endpoint force can chatter."},
        "speed_reference": {"chosen_stride_svl": stride_choice, "frequency_hz": profile.frequency_hz,
                            "method": "chosen stride * neutral SVL * locked frequency; lower supplied stride bound sets ramp floor, half supplied interval sets engineered Gaussian width",
                            "limitation": "Single operating point; supplied speed/stride ranges are context, not a joint biological acceptance region."},
        "effort_reference": {
            "measure": "mean total absolute actuator mechanical power over the control interval",
            "calculation": "sum of per-actuator absolute force*actuator_velocity work at physics substeps / control interval seconds",
            "coefficient_units": "reward units per watt per control sample",
            "limitation": "Engineered coefficient, not animal metabolic cost; alpha ramps alone cannot demonstrate learned residual/effort adaptation.",
        },
        "registry_keys": ["lab_contact_bodyweight_fraction", "hip_height_svl", "shoulder_height_svl",
                          "girdle_height_svl_tolerance", "lab_height_guard_transition_svl", "lab_speed_stride_fraction",
                          "stride_length_svl_range", "lab_gait_frequency_hz", "fore_duty_factor",
                          "lab_total_power_effort_alpha"],
    }


class WalkReward:
    def __init__(self, cfg=None):
        self.w = dict(DEFAULTS)
        if cfg:
            self.w.update(cfg)
        self._validate_effort_alpha(self.w.get("total_power_effort_alpha", 0.0))
        window = int(self.w["v4_metric_window"])
        self._contact_history = deque(maxlen=window)
        self._belly_history = deque(maxlen=window)
        self._height_history = deque(maxlen=window)
        self._hind_push_history = deque(maxlen=window)
        self._prev_hind_body_x = None
        self._fls_ema = 0.0
        self._fds_ema = 0.0    # V4.2.7: EMA of front_duty_score (gate input)

    @staticmethod
    def _validate_effort_alpha(alpha):
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Total-power effort alpha must be finite and nonnegative")
        return alpha

    def set_total_power_effort_alpha(self, alpha):
        """Explicit caller-controlled coefficient; no automatic adaptive ratchet.

        This can replay an offline coefficient ramp against frozen controls.
        Changing a reward coefficient does not itself train or change a policy.
        """
        self.w["total_power_effort_alpha"] = self._validate_effort_alpha(alpha)

    def total_power_effort_terms(self, env, metrics):
        """Charge total plant work, never residual magnitude or net signed work."""
        alpha = self._validate_effort_alpha(self.w.get("total_power_effort_alpha", 0.0))
        if getattr(env, "gait_profile", "legacy") != "lab":
            if alpha != 0.0:
                raise ValueError("Total-power effort is opt-in for the lab profile only")
            # Preserve legacy reward arithmetic and info dictionary exactly.
            return None
        if "mechanical_work_abs_J" not in metrics:
            raise ValueError("Total-power effort requires integrated absolute actuator work")
        power = mean_absolute_actuator_power(float(metrics["mechanical_work_abs_J"]), float(env.dt))
        return {
            "mechanical_power_abs_mean_W": power,
            "total_power_effort_alpha": alpha,
            "r_total_power_effort": -alpha * power,
        }

    def _reset_v4_history(self):
        self._contact_history.clear()
        self._belly_history.clear()
        self._height_history.clear()
        self._hind_push_history.clear()
        self._prev_hind_body_x = None
        self._fls_ema = 0.0
        self._fds_ema = 0.0

    def _participation_fraction(self, arr, feet):
        if arr.size == 0:
            return 0.0
        participating = []
        for foot in feet:
            series = arr[:, _FOOT_INDEX[foot]]
            participating.append(float(np.any(series) and np.any(~series)))
        return float(np.mean(participating)) if participating else 0.0

    def _hind_push_score(self, env, contacts):
        trunk_pos = env.data.xpos[env._trunk]
        trunk_mat = env.data.xmat[env._trunk].reshape(3, 3)
        hind_x = np.array([
            (trunk_mat.T @ (env.data.site_xpos[env._foot_site_id[foot]] - trunk_pos))[0]
            for foot in ("HL", "HR")
        ], dtype=np.float32)

        if self._prev_hind_body_x is None:
            self._prev_hind_body_x = hind_x
            return 0.0

        hind_contacts = np.array([
            contacts[_FOOT_INDEX["HL"]],
            contacts[_FOOT_INDEX["HR"]],
        ], dtype=bool)
        hind_vx = (hind_x - self._prev_hind_body_x) / max(env.dt, 1e-9)
        self._prev_hind_body_x = hind_x
        backward_stance_v = np.maximum(0.0, -hind_vx[hind_contacts])
        if backward_stance_v.size == 0:
            return 0.0
        return float(np.clip(np.mean(backward_stance_v) / self.w["hind_push_target"], 0.0, 1.0))

    def _v4_metrics(self, env, metrics):
        if getattr(env, "_step", 0) <= 1:
            self._reset_v4_history()

        contacts = np.asarray(metrics.get("foot_contacts", np.zeros(4)), dtype=np.float32) > 0.5
        self._contact_history.append(contacts.copy())
        self._belly_history.append(float(metrics.get("belly_contact", 0.0) > 0.5))
        self._hind_push_history.append(self._hind_push_score(env, contacts))
        trunk_height = float(env.data.xpos[env._trunk][2])
        self._height_history.append(trunk_height)

        contact_arr = np.asarray(list(self._contact_history), dtype=bool)
        if contact_arr.size == 0:
            front_pair_sync_rate = 0.0
            front_pair_hop_rate = 0.0
            fl_duty = 0.0
            fr_duty = 0.0
        else:
            fl = contact_arr[:, _FOOT_INDEX["FL"]]
            fr = contact_arr[:, _FOOT_INDEX["FR"]]
            front_pair_sync_rate = float(np.mean(fl == fr))
            front_pair_hop_rate = float(np.mean((~fl) & (~fr)))
            # V4.2.7: windowed front DUTY = fraction of recent steps each front
            # foot is load-bearing. contact_history is built from foot_contacts,
            # which use the SAME contact_thresh (0.0564) the eval gate scores at,
            # so this duty matches the reported foot_duty.
            fl_duty = float(np.mean(fl))
            fr_duty = float(np.mean(fr))

        # V4.2.7: per-foot duty scores, each saturating at its own target.
        # GATE score = min  -> progress reward stays gated until BOTH fronts clear
        #   their targets (FL can never be masked by a high FR).
        # REWARD score = mean -> FL keeps a positive gradient up to its own 0.45
        #   target regardless of FR, so it overshoots the 0.40 gate instead of
        #   stalling exactly at the edge.
        fl_score = float(np.clip(fl_duty / max(self.w["front_duty_fl_target"], 1e-9), 0.0, 1.0))
        fr_score = float(np.clip(fr_duty / max(self.w["front_duty_fr_target"], 1e-9), 0.0, 1.0))
        front_duty_score = float(min(fl_score, fr_score))     # gate input
        front_duty_reward = float(0.5 * (fl_score + fr_score))  # additive-term input

        height_arr = np.asarray(self._height_history, dtype=np.float32)
        return dict(
            front_pair_sync_rate=front_pair_sync_rate,
            front_pair_hop_rate=front_pair_hop_rate,
            hind_participation=self._participation_fraction(contact_arr, ("HL", "HR")),
            foot_participation=self._participation_fraction(contact_arr, ("HL", "FL", "HR", "FR")),
            hind_push=float(np.mean(self._hind_push_history)) if self._hind_push_history else 0.0,
            body_bounce=float(np.std(height_arr)) if height_arr.size > 1 else 0.0,
            belly_contact_rate=float(np.mean(self._belly_history)) if self._belly_history else 0.0,
            trunk_height=trunk_height,
            forward_speed=float(metrics.get("forward_speed", 0.0)),
            fl_duty=fl_duty,
            fr_duty=fr_duty,
            front_duty_score=front_duty_score,
            front_duty_reward=front_duty_reward,
        )

    def __call__(self, env, action, metrics):
        w = self.w
        effort_terms = self.total_power_effort_terms(env, metrics)
        v4 = self._v4_metrics(env, metrics)

        progress = float(np.clip(
            metrics["progress"], -w["reverse_cap"], w["progress_cap"]
        ))
        forward_speed = float(np.clip(
            metrics["forward_speed"], -w["reverse_cap"], w["forward_cap"]
        ))
        gait_match = float(metrics["gait_match"])
        contacts = np.asarray(metrics.get("foot_contacts", np.zeros(4)), dtype=np.float32) > 0.5
        targets = np.asarray(metrics.get("target_contacts", np.zeros(4)), dtype=np.float32) > 0.5
        foot_forces = np.asarray(metrics.get("foot_contact_forces", np.zeros(4)), dtype=np.float32)
        front_phase_match = float(np.mean([
            contacts[_FOOT_INDEX["FL"]] == targets[_FOOT_INDEX["FL"]],
            contacts[_FOOT_INDEX["FR"]] == targets[_FOOT_INDEX["FR"]],
        ]))
        no_progress = (
            metrics["progress"] < w["freeze_progress"]
            and abs(metrics["forward_speed"]) < w["freeze_speed"]
        )

        # Action-delta regularization is retained for checkpoint compatibility.
        # It is NOT a mechanical-effort proxy or a residual-magnitude penalty.
        smooth = float(np.mean((action - env._prev_action) ** 2))
        spin_excess = max(0.0, float(metrics["yaw_rate"]) - w["yaw_rate_cap"])
        speed_gate = float(np.clip(
            (v4["forward_speed"] - w["v4_speed_threshold"]) / max(w["v4_speed_threshold"], 1e-9),
            0.0,
            1.0,
        ))

        # V4.3: explicit target-speed shaping.
        # At the old slow gait (~0.09), speed_ramp is near 0, so slow crawling no longer
        # earns this bonus. It ramps up toward target_speed, but remains front-gated
        # through front_factor below so front-support cannot collapse for speed.
        v_fwd = max(0.0, float(v4["forward_speed"]))
        speed_ramp = float(np.clip(
            (v_fwd - w["speed_floor"]) / max(w["target_speed"] - w["speed_floor"], 1e-9),
            0.0,
            1.0,
        ))
        speed_band_score = float(np.exp(
            -0.5 * ((v_fwd - w["target_speed"]) / max(w["speed_track_sigma"], 1e-9)) ** 2
        ))
        slow_excess = float(np.clip(
            (w["slow_speed"] - v_fwd) / max(w["slow_speed"], 1e-9),
            0.0,
            1.0,
        ))
        trunk_span = max(w["trunk_height_target"] - w["trunk_height_min"], 1e-9)
        trunk_support_score = float(np.clip(
            (v4["trunk_height"] - w["trunk_height_min"]) / trunk_span,
            0.0,
            1.0,
        ))
        front_stance = np.array([
            targets[_FOOT_INDEX["FL"]],
            targets[_FOOT_INDEX["FR"]],
        ], dtype=np.float32)
        front_forces = np.array([
            foot_forces[_FOOT_INDEX["FL"]],
            foot_forces[_FOOT_INDEX["FR"]],
        ], dtype=np.float32)
        if np.any(front_stance > 0.5):
            front_load_score = float(np.sum(
                np.tanh(front_forces / max(w["front_load_force_scale"], 1e-9)) * front_stance
            ) / np.sum(front_stance))
        else:
            front_load_score = 0.0
        self._fls_ema = 0.95 * self._fls_ema + 0.05 * front_load_score
        # V4.2.7: track front contact-duration engagement alongside force.
        self._fds_ema = 0.95 * self._fds_ema + 0.05 * v4["front_duty_score"]
        # Gate progress/forward/reach on the WORSE of {force engagement, duty
        # engagement}. Brief hard taps keep _fls_ema high but leave _fds_ema low,
        # so tap-farming no longer opens the gate -- the policy must hold the
        # fronts down (raise duty) to earn the large locomotion reward.
        front_engage = min(self._fls_ema, self._fds_ema)
        front_factor = 0.10 + 0.90 * front_engage
        front_sync_excess = float(np.clip(
            (v4["front_pair_sync_rate"] - w["front_pair_sync_free"]) / (1.0 - w["front_pair_sync_free"]),
            0.0,
            1.0,
        ))
        front_hop_excess = float(np.clip(
            (v4["front_pair_hop_rate"] - w["front_pair_hop_free"]) / (1.0 - w["front_pair_hop_free"]),
            0.0,
            1.0,
        ))
        bounce_excess = max(0.0, v4["body_bounce"] - w["body_bounce_free"])

        # ---- V4.2.8 ungated front contact-reference tracking ----------------
        # CPG-commanded stance/swing for the FRONT pair this step, vs. actual
        # load-bearing contact. NOT gated by speed_gate (that is the whole point:
        # V4.2.7's gated front terms could not move the operating point).
        fl_t = bool(targets[_FOOT_INDEX["FL"]])
        fr_t = bool(targets[_FOOT_INDEX["FR"]])
        fl_c = bool(contacts[_FOOT_INDEX["FL"]])
        fr_c = bool(contacts[_FOOT_INDEX["FR"]])

        front_stance_cmds = float(fl_t) + float(fr_t)
        front_hits = float(fl_t and fl_c) + float(fr_t and fr_c)
        front_miss_cnt = float(fl_t and not fl_c) + float(fr_t and not fr_c)
        front_swing_touch = float((not fl_t) and fl_c) + float((not fr_t) and fr_c)

        track_hit = (front_hits / front_stance_cmds) if front_stance_cmds > 0 else 0.0
        track_miss = (front_miss_cnt / front_stance_cmds) if front_stance_cmds > 0 else 0.0
        r_front_track = w["front_track"] * track_hit
        r_front_miss = -w["front_miss"] * track_miss
        r_front_swing_touch = -w["front_swing_penalty"] * 0.5 * front_swing_touch
        # ---------------------------------------------------------------------

        r_alive = w["alive"]
        r_progress = w["progress"] * progress
        r_forward = w["forward"] * forward_speed
        r_gait = w["gait"] * speed_gate * (gait_match - w["gait_match_target"])
        r_front_phase_match = w["front_phase_match"] * speed_gate * (
            front_phase_match - w["front_phase_target"]
        )
        r_trunk_support = w["trunk_support"] * speed_gate * trunk_support_score
        r_front_load = w["front_load"] * speed_gate * front_load_score
        # V4.2.7: explicit, bounded reward for front contact DURATION (duty).
        # Provides a smooth local gradient even when progress/speed_gate dips; the
        # gate above provides the strong implicit pressure via the progress reward.
        r_front_duty = w["front_duty"] * speed_gate * v4["front_duty_reward"]
        r_low_trunk = -w["low_trunk"] * speed_gate * (1.0 - trunk_support_score)
        r_hind_participation = w["hind_participation"] * speed_gate * v4["hind_participation"]
        r_hind_push = w["hind_push"] * speed_gate * v4["hind_push"]
        r_foot_participation = w["foot_participation"] * speed_gate * v4["foot_participation"]
        r_freeze = -w["freeze"] if no_progress else 0.0
        r_slip = -w["slip"] * float(metrics["slip"])
        r_swing = -w["swing_contact"] * float(metrics["swing_contact"])
        r_belly = -w["belly"] * float(metrics["belly_contact"])
        r_belly_force = -w["belly_force"] * float(metrics["belly_force"])
        r_belly_rate = -w["belly_rate"] * speed_gate * v4["belly_contact_rate"]
        r_spin = -w["spin"] * spin_excess
        r_smooth = -w["smooth"] * smooth
        r_reach = w["reach"] if metrics["reached"] else 0.0
        r_fall = -w["fall"] if metrics["fallen"] else 0.0
        r_progress *= front_factor
        r_forward *= front_factor
        r_reach *= front_factor
        r_front_pair_sync = -w["front_pair_sync"] * speed_gate * (front_sync_excess ** 2)
        r_front_pair_hop = -w["front_pair_hop"] * speed_gate * (front_hop_excess ** 2)
        r_body_bounce = -w["body_bounce"] * speed_gate * bounce_excess
        r_speed_track = w["speed_track"] * front_factor * speed_ramp * speed_band_score
        r_slow = -w["slow_penalty"] * slow_excess

        total = (
            r_alive
            + r_progress
            + r_forward
            + r_gait
            + r_front_phase_match
            + r_trunk_support
            + r_front_load
            + r_front_duty
            + r_low_trunk
            + r_hind_participation
            + r_hind_push
            + r_foot_participation
            + r_freeze
            + r_slip
            + r_swing
            + r_belly
            + r_belly_force
            + r_belly_rate
            + r_spin
            + r_smooth
            + r_reach
            + r_fall
            + r_front_pair_sync
            + r_front_pair_hop
            + r_body_bounce
            + r_speed_track
            + r_slow
            + r_front_track
            + r_front_miss
            + r_front_swing_touch
        )

        reward_info = dict(
            r_alive=r_alive,
            r_progress=r_progress,
            r_forward=r_forward,
            r_gait=r_gait,
            r_front_phase_match=r_front_phase_match,
            r_trunk_support=r_trunk_support,
            r_front_load=r_front_load,
            r_front_duty=r_front_duty,
            r_low_trunk=r_low_trunk,
            r_hind_participation=r_hind_participation,
            r_hind_push=r_hind_push,
            r_foot_participation=r_foot_participation,
            r_freeze=r_freeze,
            r_slip=r_slip,
            r_swing_contact=r_swing,
            r_belly=r_belly,
            r_belly_force=r_belly_force,
            r_belly_rate=r_belly_rate,
            r_spin=r_spin,
            r_smooth=r_smooth,
            r_reach=r_reach,
            r_fall=r_fall,
            r_front_pair_sync=r_front_pair_sync,
            r_front_pair_hop=r_front_pair_hop,
            r_body_bounce=r_body_bounce,
            r_speed_track=r_speed_track,
            r_slow=r_slow,
            speed_ramp=speed_ramp,
            speed_band_score=speed_band_score,
            clean_gait_speed_gate=speed_gate,
            trunk_support_score=trunk_support_score,
            front_load=front_load_score,
            front_phase_match=front_phase_match,
            front_pair_sync_excess=front_sync_excess,
            front_pair_hop_excess=front_hop_excess,
            front_pair_sync_rate=v4["front_pair_sync_rate"],
            front_pair_hop_rate=v4["front_pair_hop_rate"],
            hind_participation=v4["hind_participation"],
            hind_push=v4["hind_push"],
            foot_participation=v4["foot_participation"],
            body_bounce=v4["body_bounce"],
            belly_contact_rate=v4["belly_contact_rate"],
            trunk_height=v4["trunk_height"],
            forward_speed=v4["forward_speed"],
            fl_duty=v4["fl_duty"],
            fr_duty=v4["fr_duty"],
            front_duty_score=v4["front_duty_score"],
            front_factor=front_factor,
            r_front_track=r_front_track,
            r_front_miss=r_front_miss,
            r_front_swing_touch=r_front_swing_touch,
            front_track_hit=track_hit,
            front_track_miss=track_miss,
        )
        if effort_terms is not None:
            # Do not add even a zero term to the legacy expression above: exact
            # historical reward arithmetic is part of checkpoint reproducibility.
            total += effort_terms["r_total_power_effort"]
            reward_info.update(effort_terms)
        return total, reward_info

