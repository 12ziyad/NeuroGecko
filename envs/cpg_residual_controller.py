#!/usr/bin/env python3
"""
GeckoBrain V4.2.4 - escape-proof front loading.

DIAGNOSIS (measured, open-loop, gecko_body_r.xml):
  A ZERO policy on the V4.2.3 base gives FL/FR = 0.49/0.27, hop 0.28 (correct).
  The TRAINED policy gives FL/FR = 0.243/0.165, hop 0.659 -- WORSE than no
  policy. PPO learned to unload the fronts because progress (always-on) rewards
  the faster front-light crawl. The unloading mechanism, reproduced exactly:
      hind-knee extend (pitch body up)  +  shoulder_sprawl residual (cancel tuck)
      -> FL/FR 0.21/0.21, hop 0.59  == the trained result.

V4.2.4 closes the escape STRUCTURALLY by capping the residual on exactly those
levers (everything else unchanged from V4.2.3). Pair this with the reward gate
(progress *= 0.25 + 0.75*front_load_score) and a FRESH policy -- a fresh policy
starts at the loaded base (FL 0.47, gate open) and is never rewarded for
unloading, so it cannot learn the crawl.

  residual caps (per actuator), via RESIDUAL_OVERRIDES below:
    elbow_L/R            0.00   (front lift lock, as before)
    shoulder_sprawl_L/R  0.05   (tuck-cancel lever -> near-locked)
    shoulder_proret_L/R  0.08   (foreleg-lift lever)
    knee_L/R (hind)      0.08   (body-pitch lever -- the dominant unloader)
    spine_pitch          0.10
    spine_bend,tail_*    0.15   (undulation, kept)
    hips/hip_rot/ankle/neck/head  0.25  (steering & balance -- untouched)

NO XML change. obs/action shape unchanged. CPG base identical to V4.2.3.
"""

import math
from types import MappingProxyType
import numpy as np
import mujoco

from common.gait_config import FOOT_ORDER, get_gait_profile, require_lab_value
from common.provenance import parameter_value


FREQ_HZ = 1.1888
STANCE = 0.62
# V4.4D: longer front stance for support.
FRONT_STANCE = {"FL": 0.68, "FR": 0.70}
PHASE = {"HL": 0.00, "FL": 0.25, "HR": 0.50, "FR": 0.75}
AMP = {"fa": 0.68, "lift": 0.6, "other": 0.15}

DEFAULT_MAP = {
    "hip_sprawl_L": ("HL", "other"), "hip_proret_L": ("HL", "fa"),
    "hip_rot_L": ("HL", "other"), "knee_L": ("HL", "lift"), "ankle_L": ("HL", "other"),
    "hip_sprawl_R": ("HR", "other"), "hip_proret_R": ("HR", "fa"),
    "hip_rot_R": ("HR", "other"), "knee_R": ("HR", "lift"), "ankle_R": ("HR", "other"),
    "shoulder_sprawl_L": ("FL", "other"), "shoulder_proret_L": ("FL", "fa"),
    "elbow_L": ("FL", "lift"),
    "shoulder_sprawl_R": ("FR", "other"), "shoulder_proret_R": ("FR", "fa"),
    "elbow_R": ("FR", "lift"),
}
SIGN = {}

# V4.2.4: per-actuator residual-scale caps on the unloading levers.
# Any actuator not listed keeps the global residual_scale (0.25).
RESIDUAL_OVERRIDES = {
    "shoulder_sprawl_L": 0.05, "shoulder_sprawl_R": 0.05,
    "shoulder_proret_L": 0.08, "shoulder_proret_R": 0.08,
    "knee_L": 0.05, "knee_R": 0.05,
    "spine_pitch": 0.10,
    "spine_bend": 0.15, "tail_bend_L": 0.15, "tail_bend_R": 0.15,
}


def _limb_signals(phi, stance):
    if phi < stance:
        s = phi / stance
        return (1.0 - 2.0 * s), 0.0
    s = (phi - stance) / (1.0 - stance)
    return (-1.0 + 2.0 * s), math.sin(math.pi * s)


class CPGResidualController:

    def __init__(self, model, mapping=None, freq=None, stance=None,
                 phase=None, amp=None, sign=None,
                 residual_scale=0.25, lock_front_lift=True,
                 front_lift_residual_scale=0.0,
                 front_stance_press=0.40, front_stance_press_fr=0.50,
                 front_swing_lift=0.40,
                 front_stance_seek=0.12,
                 front_seek_relax=0.35,
                 shoulder_sprawl_tuck=0.30,
                 spine_amp=0.30, spine_phase=0.0,
                 tail_amp=0.15, tail_phase_lag=0.15,
                 residual_overrides=None,            # V4.2.4: per-joint caps
                 verbose=True, gait_profile="legacy", lab_parameters=None,
                 hind_stance_compensation=False):
        self.model = model
        self.profile = get_gait_profile(gait_profile)
        self.gait_profile = self.profile.name
        if self.gait_profile == "legacy":
            self.freq = float(FREQ_HZ if freq is None else freq)
            self.stance = float(STANCE if stance is None else stance)
            self.phase = dict(PHASE if phase is None else phase)
        else:
            self.freq = require_lab_value("frequency", freq, self.profile.frequency_hz)
            self.stance = require_lab_value("hind stance", stance, self.profile.stance_for("HL"))
            if phase is not None and dict(phase) != self.profile.touchdown_delays:
                raise ValueError("Lab phase is a fixed touchdown-delay map, not legacy additive offsets.")
            self.phase = MappingProxyType(self.profile.touchdown_delays)
        self.amp = dict(AMP if amp is None else amp)
        self.sign = dict(SIGN if sign is None else sign)
        self.residual_scale = float(residual_scale)
        self.lock_front_lift = bool(lock_front_lift)
        self.front_lift_residual_scale = float(front_lift_residual_scale)
        self.front_stance_press = float(front_stance_press)
        self.front_stance_press_fr = float(front_stance_press_fr)
        self.front_swing_lift = float(front_swing_lift)
        self.front_stance_seek = float(front_stance_seek)
        self.front_seek_relax = float(front_seek_relax)
        self.shoulder_sprawl_tuck = float(shoulder_sprawl_tuck)
        self.spine_amp = float(spine_amp)
        self.spine_phase = float(spine_phase)
        self.tail_amp = float(tail_amp)
        self.tail_phase_lag = float(tail_phase_lag)
        self.residual_overrides = dict(RESIDUAL_OVERRIDES if residual_overrides is None
                                       else residual_overrides)
        self.lab_parameters = None
        if self.gait_profile == "legacy":
            if lab_parameters is not None:
                raise ValueError("lab_parameters are opt-in lab controls; legacy must remain unchanged.")
        else:
            values = dict(parameter_value("lab_base_parameters"))
            supplied = {} if lab_parameters is None else dict(lab_parameters)
            unknown = set(supplied) - set(values)
            if unknown:
                raise ValueError(f"Unknown lab parameters: {sorted(unknown)}")
            values.update(supplied)
            for name, value in values.items():
                if name in ("mirror_left_fa", "continuous_front_lift"):
                    if type(value) is not bool:
                        raise ValueError(f"{name} must be boolean.")
                elif isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite scalar.")
            for name in ("hind_fa_amplitude", "fore_hind_amplitude_ratio", "heading_gain"):
                if values[name] < 0:
                    raise ValueError(f"{name} must be nonnegative.")
            self.lab_parameters = MappingProxyType(values)
            # Lab dictionaries are authoritative for these channels. Legacy
            # constructor defaults/overrides continue through their original path.
            for name in ("front_stance_press", "front_stance_press_fr", "front_swing_lift",
                         "front_stance_seek", "front_seek_relax", "shoulder_sprawl_tuck"):
                setattr(self, name, float(values[name]))
            # Session 3f: the axial channels are part of the fitted lab set, so
            # they come from the registry too rather than from constructor
            # defaults. Legacy still uses its constructor values untouched.
            for name in ("spine_amp", "tail_amp", "tail_phase_lag"):
                if name in values:
                    setattr(self, name, float(values[name]))

        # Opt-in hind knee/ankle stance compensation (Session 3). The frozen-pose
        # table holds the hind collision foot at a constant commanded height
        # through stance instead of letting the hip sweep trace a 2.06 mm arc
        # that leaves the foot 0.863 mm airborne at commanded touchdown. Legacy
        # is untouched, and the default is off, so no existing run changes.
        # The table's reference assumes zero base knee/ankle/sprawl/rotation
        # targets; that holds for lab only while other_amplitude == 0, which is
        # asserted here rather than assumed.
        self._hind_comp = None
        self._hind_comp_ids = None
        self._swing_blend = None
        if hind_stance_compensation not in (False, True, "all"):
            raise ValueError("hind_stance_compensation must be False, True (hind only) or 'all'.")
        if hind_stance_compensation:
            if self.gait_profile == "legacy":
                raise ValueError("hind_stance_compensation is an opt-in lab control; legacy must remain unchanged.")
            if float(self.lab_parameters["other_amplitude"]) != 0.:
                raise ValueError(
                    "hind_stance_compensation requires other_amplitude == 0; its frozen-pose table is "
                    "solved with zero ankle/sprawl/rotation base targets and would be invalid otherwise.")
            from common.hind_stance_geometry import StanceCompensator, swing_blend
            feet = ("HL", "HR", "FL", "FR") if hind_stance_compensation == "all" else ("HL", "HR")
            self._hind_comp = StanceCompensator(model, feet=feet)
            self._swing_blend = swing_blend
            limb_actuators = {
                "HL": ("hip_proret_L", ("knee_L", "ankle_L")),
                "HR": ("hip_proret_R", ("knee_R", "ankle_R")),
                "FL": ("shoulder_proret_L", ("elbow_L",)),
                "FR": ("shoulder_proret_R", ("elbow_R",)),
            }
            ids = {}
            for foot in feet:
                drive_name, free_names = limb_actuators[foot]
                drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, drive_name)
                free = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in free_names)
                if drive < 0 or min(free) < 0:
                    raise ValueError(f"Missing drive/free actuators for {foot}.")
                ids[foot] = (drive, free)
            self._hind_comp_ids = ids

        nu = model.nu
        lo = model.actuator_ctrlrange[:, 0].copy()
        hi = model.actuator_ctrlrange[:, 1].copy()
        self.lim = model.actuator_ctrllimited.astype(bool)
        self.lo, self.hi = lo, hi
        self.neutral = np.where(self.lim, 0.5 * (lo + hi), 0.0)
        self.half = np.where(self.lim, 0.5 * (hi - lo), 1.0)

        mp = DEFAULT_MAP if mapping is None else mapping
        self.entries = []
        missing = []
        for name, (limb, role) in mp.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                missing.append(name); continue
            self.entries.append((aid, limb, role))
        self.mapped_ids = {e[0] for e in self.entries}
        self._lab_fa_amplitude = {}
        if self.gait_profile == "lab":
            requested_ratio = self.lab_parameters["fore_hind_amplitude_ratio"]
            hind_amplitude = self.lab_parameters["hind_fa_amplitude"]
            for side in ("L", "R"):
                hind_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_proret_" + side)
                fore_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_proret_" + side)
                if min(hind_id, fore_id) < 0 or min(self.half[hind_id], self.half[fore_id]) <= 0:
                    raise ValueError("Lab angular ratio requires hip/shoulder protraction actuator ranges.")
                natural_ratio = self.half[fore_id] / self.half[hind_id]
                # 8/9 is the versioned equal-normalized compatibility setting.
                # V2's inherited shoulder ctrlrange is rounded to 0.698132 rad,
                # so its actual original angular ratio is approximately 0.88888927.
                # Recognize that declared baseline explicitly, not via a loose
                # tolerance. Other requested ratios use the real model ranges.
                fore_amplitude = (hind_amplitude if requested_ratio == 8/9 or math.isclose(requested_ratio, natural_ratio, rel_tol=0, abs_tol=1e-12)
                                  else hind_amplitude * requested_ratio / natural_ratio)
                self._lab_fa_amplitude["H" + side] = hind_amplitude
                self._lab_fa_amplitude["F" + side] = fore_amplitude

        def _id(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self._ssl = _id("shoulder_sprawl_L"); self._ssr = _id("shoulder_sprawl_R")
        self._spine = _id("spine_bend"); self._tail_l = _id("tail_bend_L"); self._tail_r = _id("tail_bend_R")

        self.front_lift_ids = [aid for (aid, limb, role) in self.entries
                               if role == "lift" and limb in ("FL", "FR")]

        # residual scale vector: global default, then front-lift lock, then per-joint caps
        self.res_scale_vec = np.full(nu, self.residual_scale, dtype=float)
        if self.lock_front_lift:
            for aid in self.front_lift_ids:
                self.res_scale_vec[aid] = self.front_lift_residual_scale
        for name, sc in self.residual_overrides.items():
            aid = _id(name)
            if aid >= 0:
                self.res_scale_vec[aid] = float(sc)

        if missing:
            print(f"[cpg] WARNING: actuators not found and will be ignored: {missing}")
        if verbose:
            self.report()

    # ------------------------------------------------------------------ API
    def foot_phase_fraction(self, foot, time_s):
        if self.gait_profile == "lab":
            return self.profile.phase_fraction(foot, time_s)
        # Deliberately preserve the old ADDITIVE controller convention.
        return (time_s*self.freq+self.phase[foot]) % 1.0

    def stance_for(self, foot):
        if self.gait_profile == "lab":
            return self.profile.stance_for(foot)
        return FRONT_STANCE.get(foot, self.stance)

    def commanded_contacts(self, time_s):
        """Controller schedule, not measured physical foot contacts."""
        return {foot: float(self.foot_phase_fraction(foot, time_s) < self.stance_for(foot))
                for foot in FOOT_ORDER}

    def commanded_contact_array(self, time_s, order=FOOT_ORDER):
        contact = self.commanded_contacts(time_s)
        return np.array([contact[foot] for foot in order], dtype=np.float32)

    def base_ctrl(self, t, front_contact=None, heading_error=0.0):
        # front_contact: optional dict {"FL": bool, "FR": bool} of CURRENT
        # load-bearing contact. When provided, the controller seeks the ground
        # (slightly stronger press) if a commanded-stance front foot is airborne,
        # and relaxes the press once contact exists (avoid dragging/slip). This
        # is a small reflex assist, NOT a hard override. front_contact=None keeps
        # the exact V4.2.7 open-loop behavior.
        ctrl = self.neutral.copy()
        steer = 0.0
        if self.gait_profile == "lab":
            if not math.isfinite(heading_error):
                raise ValueError("heading_error must be a finite angle in radians.")
            # Positive bearing error means turn left: more right-side rearward
            # stance travel, less left-side travel. Clip bounds are engineering
            # definitions recorded in lab_base_parameters, not animal gains.
            steer = float(np.clip(self.lab_parameters["heading_gain"] * heading_error, -1., 1.))
        sig = {}
        for limb, off in self.phase.items():
            phi = self.foot_phase_fraction(limb, t)
            limb_stance = self.stance_for(limb)
            sig[limb] = _limb_signals(phi, limb_stance)
        for aid, limb, role in self.entries:
            fa, lift = sig[limb]
            if self.gait_profile == "lab":
                s = self._lab_limb_signal(limb, role, fa, lift, front_contact, steer)
            elif role == "fa":
                s = self.amp["fa"] * fa
            elif role == "lift":
                if limb in ("FL", "FR"):
                    press = self.front_stance_press if limb == "FL" else self.front_stance_press_fr
                    if lift > 0.0:
                        s = self.front_swing_lift * lift
                    else:
                        in_contact = (None if front_contact is None
                                      else bool(front_contact.get(limb, False)))
                        if in_contact is False:
                            s = +(press + self.front_stance_seek)
                        elif in_contact is True:
                            s = +(press * self.front_seek_relax)
                        else:
                            s = +press
                else:
                    s = self.amp["lift"] * lift
            else:
                s = self.amp["other"] * fa
            ctrl[aid] = self.neutral[aid] + self._sign(aid) * s * self.half[aid]

        if self._hind_comp is not None:
            # Applied after the limb loop so the driving sweep command is final,
            # and before the shoulder-tuck/spine/tail terms, which do not move a
            # foot. The compensated value is an ABSOLUTE target blended against
            # the base command by the same swing weight: full authority through
            # stance, the untouched base command at mid-swing. Absolute rather
            # than additive matters for the forelimb, whose elbow already carries
            # front_stance_press in stance -- adding an offset there would stack
            # two independent height commands. For the hind limb the two forms
            # coincide, because knee ("lift") and ankle ("other", amplitude 0)
            # are both zero-offset from neutral during stance.
            for foot, (drive_aid, free_aids) in self._hind_comp_ids.items():
                phi = self.foot_phase_fraction(foot, t)
                stance = self.stance_for(foot)
                weight = self._swing_blend(phi, stance)
                raw = self._hind_comp.offsets(foot, float(ctrl[drive_aid]), 0., stance)
                for aid, value in zip(free_aids, raw):
                    target = self.neutral[aid] + value
                    ctrl[aid] = weight * target + (1. - weight) * ctrl[aid]

        if self._ssl >= 0: ctrl[self._ssl] -= self.shoulder_sprawl_tuck
        if self._ssr >= 0: ctrl[self._ssr] += self.shoulder_sprawl_tuck

        if self._spine >= 0 and self.spine_amp > 0.0:
            w = math.sin(2.0 * math.pi * (t * self.freq + self.spine_phase))
            ctrl[self._spine] = self.spine_amp * self.half[self._spine] * w
            if self.tail_amp > 0.0 and self._tail_l >= 0 and self._tail_r >= 0:
                wt = math.sin(2.0 * math.pi * (t * self.freq + self.spine_phase - self.tail_phase_lag))
                ctrl[self._tail_l] = +self.tail_amp * self.half[self._tail_l] * wt
                ctrl[self._tail_r] = -self.tail_amp * self.half[self._tail_r] * wt
        return ctrl

    def compute(self, action, t, front_contact=None, heading_error=0.0):
        action = np.asarray(action, dtype=float).reshape(-1)
        base = self.base_ctrl(t, front_contact=front_contact, heading_error=heading_error)
        residual = action * self.res_scale_vec * self.half
        ctrl = base + residual
        ctrl = np.where(self.lim, np.clip(ctrl, self.lo, self.hi), ctrl)
        return ctrl

    __call__ = compute

    # ----------------------------------------------------------- internals
    def _lab_limb_signal(self, limb, role, fa, lift, front_contact, steer):
        params = self.lab_parameters
        if role == "fa":
            signal = self._lab_fa_amplitude[limb] * fa
            if params["mirror_left_fa"] and limb.endswith("L"):
                signal = -signal
            if steer != 0.:
                signal *= 1. - steer if limb.endswith("L") else 1. + steer
            return signal
        if role != "lift":
            return params["other_amplitude"] * fa
        if limb.startswith("H"):
            signal = self.amp["lift"] * lift
            return signal if params["hind_lift_multiplier"] == 1. else signal * params["hind_lift_multiplier"]
        press = self.front_stance_press if limb == "FL" else self.front_stance_press_fr
        if params["continuous_front_lift"]:
            # A single closed, continuous command: stance press plus a signed
            # swing excursion. With this body's +Y elbows a negative excursion
            # lifts the collision foot. The optional contact reflex is disabled
            # in this mode, avoiding a stance-to-swing target discontinuity.
            if lift > 0.:
                return press + self.front_swing_lift * lift
            # No phase-dependent stance reflex here: keeping a contact-dependent
            # jump would defeat this option's continuity contract. For open-loop
            # base trials, press remains the explicit stance target.
            return press
        # Exactly the original front branch when the continuity option is off.
        if lift > 0.:
            return self.front_swing_lift * lift
        in_contact = None if front_contact is None else bool(front_contact.get(limb, False))
        if in_contact is False:
            return +(press + self.front_stance_seek)
        if in_contact is True:
            return +(press * self.front_seek_relax)
        return +press

    def _sign(self, aid):
        nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        return float(self.sign.get(nm, 1.0))

    def report(self):
        name = lambda i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        print("[cpg] CPG-residual controller  (V4.2.4)")
        print(f"      freq={self.freq}Hz stance={self.stance} residual_scale={self.residual_scale}")
        print(f"      front press FL=+{self.front_stance_press} FR=+{self.front_stance_press_fr}  tuck={self.shoulder_sprawl_tuck}")
        print(f"      spine_amp={self.spine_amp} tail_amp={self.tail_amp}")
        print("      residual caps:")
        for nm in self.residual_overrides:
            print(f"        {nm:<20} {self.res_scale_vec[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,nm)]:.2f}")
        locked = ", ".join(name(i) for i in self.front_lift_ids) or "(none!)"
        print(f"      FL/FR lift LOCKED (0.0): {locked}")

    def anti_phase_ok(self, n=400):
        both_off = 0
        for k in range(n):
            t = k / (n * self.freq)
            if self.gait_profile == "lab":
                contact = self.commanded_contacts(t)
                both_off += int(not contact["FL"] and not contact["FR"])
                continue
            # Historical helper also uses hind stance for the fore pair; retained
            # in legacy mode to avoid silently rewriting compatibility behavior.
            phiL = (t * self.freq + self.phase["FL"]) % 1.0
            phiR = (t * self.freq + self.phase["FR"]) % 1.0
            both_off += int((phiL >= self.stance) and (phiR >= self.stance))
        frac = both_off / n
        return frac, (frac < 1e-9)

