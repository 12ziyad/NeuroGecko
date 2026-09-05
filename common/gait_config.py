"""Explicit gait profiles with a shared touchdown-delay convention.

The legacy controller ADDs its phase offsets, while its legacy reward SUBTRACTs
them. That historical mismatch is retained only for checkpoint reproducibility.
The opt-in lab profile uses positive touchdown delays in BOTH consumers: local
phase = (global cycle - touchdown delay) modulo one. Timing targets are design
constraints informed by literature, not proof that simulated feet follow them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

FOOT_ORDER = ("HL", "FL", "HR", "FR")
LOCKED_FREQUENCY_HZ = 1.1888


@dataclass(frozen=True)
class GaitProfile:
    name: str
    frequency_hz: float
    touchdown_delays_cycle: tuple[float, ...]
    stance_ratios: tuple[float, ...]

    def __post_init__(self):
        if self.name not in ("legacy", "lab"):
            raise ValueError("gait_profile must be 'legacy' or 'lab'.")
        if len(self.touchdown_delays_cycle) != 4 or len(self.stance_ratios) != 4:
            raise ValueError("Profiles require one phase and stance ratio for each foot.")
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("Gait frequency must be finite and positive.")
        if any(not math.isfinite(x) or not 0 <= x < 1 for x in self.touchdown_delays_cycle):
            raise ValueError("Touchdown delays must be finite fractions in [0, 1).")
        if any(not math.isfinite(x) or not 0 < x < 1 for x in self.stance_ratios):
            raise ValueError("Stance ratios must be finite fractions in (0, 1).")
        if self.name == "lab" and self.frequency_hz != LOCKED_FREQUENCY_HZ:
            raise ValueError("Lab frequency is locked at 1.1888 Hz; no frequency channel is added.")

    @property
    def touchdown_delays(self):
        return dict(zip(FOOT_ORDER, self.touchdown_delays_cycle))

    @property
    def stance_by_foot(self):
        return dict(zip(FOOT_ORDER, self.stance_ratios))

    def stance_for(self, foot):
        return self.stance_ratios[FOOT_ORDER.index(foot)]

    def phase_fraction(self, foot, time_s):
        """Canonical lab phase, including exact-boundary floating-point hygiene."""
        index = FOOT_ORDER.index(foot)
        cycle = (float(time_s)*self.frequency_hz) % 1.0
        phase = (cycle-self.touchdown_delays_cycle[index]) % 1.0
        # Machine-precision cleanup, not a biological/contact detection threshold.
        if min(phase, 1-phase) < 1e-12:
            return 0.0
        if abs(phase-self.stance_ratios[index]) < 1e-12:
            return self.stance_ratios[index]
        return phase

    def contact(self, foot, time_s):
        return float(self.phase_fraction(foot, time_s) < self.stance_for(foot))


def get_gait_profile(profile="legacy"):
    if isinstance(profile, GaitProfile):
        return profile
    if profile == "legacy":
        # Versioned compatibility constants, NOT fresh claims about animal data.
        return GaitProfile("legacy", LOCKED_FREQUENCY_HZ,
                           (0., .25, .50, .75), (.62, .68, .62, .70))
    if profile != "lab":
        raise ValueError("gait_profile must be 'legacy' or 'lab'.")
    from common.provenance import load_registry
    entries = load_registry()["entries"]
    delay = entries["lab_gait_touchdown_delays"]["value"]
    if set(delay) != set(FOOT_ORDER):
        raise ValueError("Lab registry must specify all four touchdown delays exactly once.")
    hind = float(entries["hind_duty_factor"]["value"])
    fore = float(entries["fore_duty_factor"]["value"])
    return GaitProfile("lab", float(entries["lab_gait_frequency_hz"]["value"]),
                       tuple(float(delay[f]) for f in FOOT_ORDER), (hind, fore, hind, fore))


def require_lab_value(name, requested, expected):
    """Reject contradictory overrides instead of silently changing a lab profile."""
    if requested is not None and not math.isclose(float(requested), expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Lab {name} is fixed at {expected}; received {requested}.")
    return expected
