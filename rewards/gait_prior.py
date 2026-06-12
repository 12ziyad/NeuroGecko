"""
gait_prior.py -- lateral-sequence CPG prior for GeckoBrain walking.

The prior is deliberately small: it supplies the stride clock, expected
foot-contact schedule, and phase helpers that reward/controller code can use.
It does not change the policy action dimension.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


TAU = 2.0 * math.pi
LATERAL_SEQUENCE_ORDER = ("HL", "FL", "HR", "FR")
# V4.4D: longer commanded front stance; must match controller.
FRONT_STANCE = {"FL": 0.68, "FR": 0.70}


@dataclass(frozen=True)
class GaitPriorConfig:
    gait_type: str = "lateral_sequence"
    phase_order: tuple[str, ...] = LATERAL_SEQUENCE_ORDER
    frequency_hz: float = 1.1888
    frequency_range_hz: tuple[float, float] = (1.0, 1.2)
    stance_ratio: float = 0.62
    swing_ratio: float = 0.38
    belly_clearance_target_m: tuple[float, float] = (0.005, 0.015)
    spine_wave: str = "medium-small"
    tail_wave: str = "small counter-wave"


class LateralSequenceCPG:
    """CPG clock and contact targets for HL -> FL -> HR -> FR walking."""

    def __init__(
        self,
        frequency_hz: float = 1.1888,
        stance_ratio: float = 0.62,
        swing_ratio: float = 0.38,
        phase_order: Iterable[str] = LATERAL_SEQUENCE_ORDER,
    ):
        self.config = GaitPriorConfig(
            phase_order=tuple(phase_order),
            frequency_hz=float(frequency_hz),
            stance_ratio=float(stance_ratio),
            swing_ratio=float(swing_ratio),
        )
        if self.config.gait_type != "lateral_sequence":
            raise ValueError("Only lateral_sequence gait is supported by this prior.")
        if self.config.phase_order != LATERAL_SEQUENCE_ORDER:
            raise ValueError("Locked phase order is HL -> FL -> HR -> FR.")
        if not np.isclose(self.config.stance_ratio + self.config.swing_ratio, 1.0):
            raise ValueError("stance_ratio + swing_ratio must equal 1.0.")
        if self.config.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive.")
        if not self.config.frequency_range_hz[0] <= self.config.frequency_hz <= self.config.frequency_range_hz[1]:
            raise ValueError("frequency_hz must stay in the real-video 1.0-1.2 Hz range.")
        if not 0.0 < self.config.stance_ratio < 1.0:
            raise ValueError("stance_ratio must be in (0, 1).")

        step = 1.0 / len(self.config.phase_order)
        self.phase_offsets = {
            foot: i * step for i, foot in enumerate(self.config.phase_order)
        }
        self.front_stance = dict(FRONT_STANCE)

    @property
    def foot_order(self) -> tuple[str, ...]:
        return self.config.phase_order

    @property
    def frequency_hz(self) -> float:
        return self.config.frequency_hz

    @property
    def stance_ratio(self) -> float:
        return self.config.stance_ratio

    @property
    def swing_ratio(self) -> float:
        return self.config.swing_ratio

    def cycle_fraction(self, time_s: float) -> float:
        return float((time_s * self.frequency_hz) % 1.0)

    def phase(self, time_s: float) -> float:
        return self.cycle_fraction(time_s) * TAU

    def phase_observation(self, time_s: float) -> np.ndarray:
        phi = self.phase(time_s)
        return np.array([math.sin(phi), math.cos(phi)], dtype=np.float32)

    def foot_phase_fraction(self, foot: str, time_s: float) -> float:
        if foot not in self.phase_offsets:
            raise KeyError(f"Unknown foot label {foot!r}; expected {self.foot_order}.")
        return float((self.cycle_fraction(time_s) - self.phase_offsets[foot]) % 1.0)

    def _stance_for(self, foot: str) -> float:
        return self.front_stance.get(foot, self.stance_ratio)

    def target_contacts(self, time_s: float) -> dict[str, float]:
        return {
            foot: float(self.foot_phase_fraction(foot, time_s) < self._stance_for(foot))
            for foot in self.foot_order
        }

    def target_contact_array(
        self, time_s: float, order: Iterable[str] | None = None
    ) -> np.ndarray:
        contacts = self.target_contacts(time_s)
        foot_order = tuple(order) if order is not None else self.foot_order
        return np.array([contacts[foot] for foot in foot_order], dtype=np.float32)

    def contact_match(self, actual_contacts, time_s: float) -> float:
        actual = np.asarray(actual_contacts, dtype=np.float32)
        target = self.target_contact_array(time_s)
        if actual.shape != target.shape:
            raise ValueError(f"actual_contacts shape {actual.shape} != {target.shape}")
        return float(np.mean(actual == target))

    def helper_values(self, time_s: float) -> dict[str, float]:
        phi = self.phase(time_s)
        spine_phi = phi
        tail_phi = (phi + math.pi) % TAU
        return {
            "phase": phi,
            "cycle_fraction": self.cycle_fraction(time_s),
            "spine_phase": spine_phi,
            "spine_sin": math.sin(spine_phi),
            "spine_cos": math.cos(spine_phi),
            "tail_phase": tail_phi,
            "tail_sin": math.sin(tail_phi),
            "tail_cos": math.cos(tail_phi),
        }
