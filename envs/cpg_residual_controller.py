#!/usr/bin/env python3
"""
GeckoBrain V4.2.3 - front-loading + lateral undulation.

WHY (measured in gecko_body_r.xml, open-loop, threshold 0.0564, stand reset):
  V4.2.2 fixed posture but left the fronts LIGHT (FL/FR base 0.32/0.21, trained
  0.247/0.186) and the body RIGID (no lateral undulation -> "crawl" look).
  Press/lift/per-side tuning could NOT raise the fronts past ~0.32/0.21: the
  ceiling is at the SHOULDERS, not the elbows.

  V4.2.3 adds three things (NO XML change, obs/action shape unchanged):

  1. shoulder_sprawl_tuck (0.30): static tuck that pulls the front feet UNDER
     the body. This is the front-loading lever.
        FL 0.32 -> 0.49,  front_hop 0.50 -> 0.28,  trunk 0.0280 -> 0.0318,
        weight redistributes onto the fronts (HL 0.81 -> 0.67), belly stays 0.
  2. per-side front press (FL 0.40, FR 0.50): FR is the chronically-light foot;
     pressing it a touch harder nudges it (0.25 -> 0.27). FR stays the laggard
     (structural asymmetry) but now bears real weight and clears the 0.23 bar.
  3. lateral spine (+tail) undulation: the S-wave a real gecko uses. Verified to
     raise lateral COM swing 0.0197 -> 0.0271 with belly still 0. This is the
     NATURALNESS lever; it does not change front duty.

  Spine/tail residual is partially locked (0.15) so PPO fine-tunes the wave but
  cannot flatten it (the same idea as the FL/FR lift lock).

UNCHANGED: freq/stance/phase, hind amps, residual_scale 0.25 on the other
joints, FL/FR lift residual LOCKED at 0.0, front_swing_lift 0.40.
"""

import math
import numpy as np
import mujoco


FREQ_HZ = 1.1
STANCE = 0.62
PHASE = {"HL": 0.00, "FL": 0.25, "HR": 0.50, "FR": 0.75}
AMP = {"fa": 0.6, "lift": 0.6, "other": 0.15}

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


def _limb_signals(phi, stance):
    if phi < stance:
        s = phi / stance
        return (1.0 - 2.0 * s), 0.0
    s = (phi - stance) / (1.0 - stance)
    return (-1.0 + 2.0 * s), math.sin(math.pi * s)


class CPGResidualController:

    def __init__(self, model, mapping=None, freq=FREQ_HZ, stance=STANCE,
                 phase=None, amp=None, sign=None,
                 residual_scale=0.25, lock_front_lift=True,
                 front_lift_residual_scale=0.0,
                 front_stance_press=0.40,          # FL stance press
                 front_stance_press_fr=0.50,       # V4.2.3: FR pressed harder
                 front_swing_lift=0.40,
                 shoulder_sprawl_tuck=0.30,        # V4.2.3: pull front feet under body
                 spine_amp=0.30, spine_phase=0.0,  # V4.2.3: lateral undulation
                 tail_amp=0.15, tail_phase_lag=0.15,
                 undulation_residual_scale=0.15,   # partial lock so PPO can't flatten the wave
                 verbose=True):
        self.model = model
        self.freq = float(freq)
        self.stance = float(stance)
        self.phase = dict(PHASE if phase is None else phase)
        self.amp = dict(AMP if amp is None else amp)
        self.sign = dict(SIGN if sign is None else sign)
        self.residual_scale = float(residual_scale)
        self.lock_front_lift = bool(lock_front_lift)
        self.front_lift_residual_scale = float(front_lift_residual_scale)
        self.front_stance_press = float(front_stance_press)
        self.front_stance_press_fr = float(front_stance_press_fr)
        self.front_swing_lift = float(front_swing_lift)
        self.shoulder_sprawl_tuck = float(shoulder_sprawl_tuck)
        self.spine_amp = float(spine_amp)
        self.spine_phase = float(spine_phase)
        self.tail_amp = float(tail_amp)
        self.tail_phase_lag = float(tail_phase_lag)

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
                missing.append(name)
                continue
            self.entries.append((aid, limb, role))
        self.mapped_ids = {e[0] for e in self.entries}

        def _id(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self._ssl = _id("shoulder_sprawl_L")
        self._ssr = _id("shoulder_sprawl_R")
        self._spine = _id("spine_bend")
        self._tail_l = _id("tail_bend_L")
        self._tail_r = _id("tail_bend_R")

        self.front_lift_ids = [aid for (aid, limb, role) in self.entries
                               if role == "lift" and limb in ("FL", "FR")]

        self.res_scale_vec = np.full(nu, self.residual_scale, dtype=float)
        if self.lock_front_lift:
            for aid in self.front_lift_ids:
                self.res_scale_vec[aid] = self.front_lift_residual_scale
        # partial-lock the undulation joints so PPO refines but cannot erase the wave
        for aid in (self._spine, self._tail_l, self._tail_r):
            if aid >= 0:
                self.res_scale_vec[aid] = self.undulation_residual_scale = float(undulation_residual_scale)

        if missing:
            print(f"[cpg] WARNING: actuators not found and will be ignored: {missing}")
        if verbose:
            self.report()

    # ------------------------------------------------------------------ API
    def base_ctrl(self, t):
        ctrl = self.neutral.copy()
        sig = {}
        for limb, off in self.phase.items():
            phi = (t * self.freq + off) % 1.0
            sig[limb] = _limb_signals(phi, self.stance)
        for aid, limb, role in self.entries:
            fa, lift = sig[limb]
            if role == "fa":
                s = self.amp["fa"] * fa
            elif role == "lift":
                if limb in ("FL", "FR"):
                    press = self.front_stance_press if limb == "FL" else self.front_stance_press_fr
                    s = (self.front_swing_lift * lift) if lift > 0.0 else +press
                else:
                    s = self.amp["lift"] * lift
            else:  # other
                s = self.amp["other"] * fa
            ctrl[aid] = self.neutral[aid] + self._sign(aid) * s * self.half[aid]

        # V4.2.3: shoulder-sprawl tuck (front feet under body). L -= tuck, R += tuck.
        if self._ssl >= 0:
            ctrl[self._ssl] -= self.shoulder_sprawl_tuck
        if self._ssr >= 0:
            ctrl[self._ssr] += self.shoulder_sprawl_tuck

        # V4.2.3: lateral spine + tail undulation at stride frequency
        if self._spine >= 0 and self.spine_amp > 0.0:
            w = math.sin(2.0 * math.pi * (t * self.freq + self.spine_phase))
            ctrl[self._spine] = self.spine_amp * self.half[self._spine] * w
            if self.tail_amp > 0.0 and self._tail_l >= 0 and self._tail_r >= 0:
                wt = math.sin(2.0 * math.pi * (t * self.freq + self.spine_phase - self.tail_phase_lag))
                ctrl[self._tail_l] = +self.tail_amp * self.half[self._tail_l] * wt
                ctrl[self._tail_r] = -self.tail_amp * self.half[self._tail_r] * wt

        return ctrl

    def compute(self, action, t):
        action = np.asarray(action, dtype=float).reshape(-1)
        base = self.base_ctrl(t)
        residual = action * self.res_scale_vec * self.half
        ctrl = base + residual
        ctrl = np.where(self.lim, np.clip(ctrl, self.lo, self.hi), ctrl)
        return ctrl

    __call__ = compute

    # ----------------------------------------------------------- internals
    def _sign(self, aid):
        nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        return float(self.sign.get(nm, 1.0))

    def report(self):
        name = lambda i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        print("[cpg] CPG-residual controller  (V4.2.3)")
        print(f"      freq={self.freq} Hz  stance={self.stance}  residual_scale={self.residual_scale}")
        print(f"      front press FL=+{self.front_stance_press} FR=+{self.front_stance_press_fr}  front_swing_lift={self.front_swing_lift}")
        print(f"      shoulder_sprawl_tuck={self.shoulder_sprawl_tuck} (front feet tucked under body)")
        print(f"      spine_amp={self.spine_amp} phase={self.spine_phase}  tail_amp={self.tail_amp} (lateral undulation)")
        locked = ", ".join(name(i) for i in self.front_lift_ids) or "(none!)"
        print(f"      FL/FR lift LOCKED (residual={self.front_lift_residual_scale}): {locked}")
        print(f"      spine/tail residual partial-locked to {getattr(self,'undulation_residual_scale','?')}")

    def anti_phase_ok(self, n=400):
        both_off = 0
        for k in range(n):
            t = k / (n * self.freq)
            phiL = (t * self.freq + self.phase["FL"]) % 1.0
            phiR = (t * self.freq + self.phase["FR"]) % 1.0
            both_off += int((phiL >= self.stance) and (phiR >= self.stance))
        frac = both_off / n
        return frac, (frac < 1e-9)
