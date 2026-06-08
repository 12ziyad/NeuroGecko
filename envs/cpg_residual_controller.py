"""
CPG-driven residual controller for GeckoBrain V4.2.1.

The action shape stays 25. A policy action is interpreted as a residual around
an open-loop lateral-sequence CPG:

    final_ctrl = clip(CPG_base(t) + action * residual_scale * ctrl_half_range)

The front lift actuators, `elbow_L` and `elbow_R`, are locked to the CPG with
zero residual scale so the policy cannot merge the front swing windows. V4.2.1
also presses the front lifts downward during stance so the front feet carry
load instead of floating through their stance windows.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


FREQ_HZ = 1.1
STANCE = 0.62
PHASE = {"HL": 0.00, "FL": 0.25, "HR": 0.50, "FR": 0.75}
AMP = {"fa": 0.6, "lift": 0.6, "other": 0.15}

DEFAULT_MAP = {
    "hip_sprawl_L": ("HL", "other"),
    "hip_proret_L": ("HL", "fa"),
    "hip_rot_L": ("HL", "other"),
    "knee_L": ("HL", "lift"),
    "ankle_L": ("HL", "other"),
    "hip_sprawl_R": ("HR", "other"),
    "hip_proret_R": ("HR", "fa"),
    "hip_rot_R": ("HR", "other"),
    "knee_R": ("HR", "lift"),
    "ankle_R": ("HR", "other"),
    "shoulder_sprawl_L": ("FL", "other"),
    "shoulder_proret_L": ("FL", "fa"),
    "elbow_L": ("FL", "lift"),
    "shoulder_sprawl_R": ("FR", "other"),
    "shoulder_proret_R": ("FR", "fa"),
    "elbow_R": ("FR", "lift"),
}

SIGN = {}


def _limb_signals(phi, stance):
    """Return fore-aft signal in [-1, 1] and swing lift in [0, 1]."""
    if phi < stance:
        s = phi / stance
        return (1.0 - 2.0 * s), 0.0
    s = (phi - stance) / (1.0 - stance)
    return (-1.0 + 2.0 * s), math.sin(math.pi * s)


class CPGResidualController:
    """CPG base actuator targets plus clipped PPO residual."""

    def __init__(
        self,
        model,
        mapping=None,
        freq=FREQ_HZ,
        stance=STANCE,
        phase=None,
        amp=None,
        sign=None,
        residual_scale=0.2,
        lock_front_lift=True,
        front_lift_residual_scale=0.0,
        front_stance_press=0.35,
        verbose=False,
    ):
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

        nu = model.nu
        lo = model.actuator_ctrlrange[:, 0].copy()
        hi = model.actuator_ctrlrange[:, 1].copy()
        self.lim = model.actuator_ctrllimited.astype(bool)
        self.lo = lo
        self.hi = hi
        self.neutral = np.where(self.lim, 0.5 * (lo + hi), 0.0)
        self.half = np.where(self.lim, 0.5 * (hi - lo), 1.0)

        mapping = DEFAULT_MAP if mapping is None else mapping
        self.entries = []
        missing = []
        for name, (limb, role) in mapping.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                missing.append(name)
            else:
                self.entries.append((aid, limb, role))
        self.mapped_ids = {aid for aid, _, _ in self.entries}
        self.front_lift_ids = [
            aid for aid, limb, role in self.entries
            if role == "lift" and limb in ("FL", "FR")
        ]

        self.res_scale_vec = np.full(nu, self.residual_scale, dtype=float)
        if self.lock_front_lift:
            for aid in self.front_lift_ids:
                self.res_scale_vec[aid] = self.front_lift_residual_scale

        if missing:
            print(f"[cpg] WARNING: missing actuator names ignored: {missing}")
        if verbose:
            self.report()

    def base_ctrl(self, t):
        ctrl = self.neutral.copy()
        signals = {}
        for limb, offset in self.phase.items():
            phi = (t * self.freq + offset) % 1.0
            signals[limb] = _limb_signals(phi, self.stance)

        for aid, limb, role in self.entries:
            fa, lift = signals[limb]
            if role == "fa":
                signal = self.amp["fa"] * fa
            elif role == "lift":
                if limb in ("FL", "FR"):
                    signal = self.amp["lift"] * lift if lift > 0.0 else -self.front_stance_press
                else:
                    signal = self.amp["lift"] * lift
            else:
                signal = self.amp["other"] * fa
            ctrl[aid] = self.neutral[aid] + self._sign(aid) * signal * self.half[aid]
        return ctrl

    def compute(self, action, t):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.model.nu,):
            raise ValueError(f"Expected action shape {(self.model.nu,)}, got {action.shape}")
        base = self.base_ctrl(t)
        residual = action * self.res_scale_vec * self.half
        ctrl = base + residual
        return np.where(self.lim, np.clip(ctrl, self.lo, self.hi), ctrl)

    __call__ = compute

    def _sign(self, aid):
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        return float(self.sign.get(name, 1.0))

    def report(self):
        name = lambda i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        print("[cpg] CPG-residual controller")
        print(f"      freq={self.freq} Hz  stance={self.stance}  residual_scale={self.residual_scale}")
        print(f"      front_stance_press={self.front_stance_press} (FL/FR press down during stance)")
        by_limb = {"HL": [], "FL": [], "HR": [], "FR": []}
        for aid, limb, role in self.entries:
            by_limb.setdefault(limb, []).append(f"{name(aid)}({role})")
        for limb in ("HL", "FL", "HR", "FR"):
            mapped = ", ".join(by_limb.get(limb, [])) or "--"
            print(f"      {limb}: {mapped}")
        locked = ", ".join(name(i) for i in self.front_lift_ids) or "(none)"
        print(f"      FL/FR lift locked to CPG residual={self.front_lift_residual_scale}: {locked}")
        unmapped = [name(i) for i in range(self.model.nu) if i not in self.mapped_ids]
        print(f"      held neutral base, residual still allowed: {unmapped}")

    def anti_phase_ok(self, n=400):
        both_off = 0
        for k in range(n):
            t = k / (n * self.freq)
            phi_l = (t * self.freq + self.phase["FL"]) % 1.0
            phi_r = (t * self.freq + self.phase["FR"]) % 1.0
            both_off += int(phi_l >= self.stance and phi_r >= self.stance)
        frac = both_off / n
        return frac, frac < 1e-9
