from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class DriveState:
    hunger: float = 0.35
    energy: float = 1.0
    fear: float = 0.0
    curiosity: float = 0.25
    danger: float = 0.0
    target_interest: float = 0.45

    def reset(self) -> "DriveState":
        self.hunger = 0.35
        self.energy = 1.0
        self.fear = 0.0
        self.curiosity = 0.25
        self.danger = 0.0
        self.target_interest = 0.45
        return self

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.hunger,
                self.energy,
                self.fear,
                self.curiosity,
                self.danger,
                self.target_interest,
            ],
            dtype=np.float32,
        )

    def update(
        self,
        dt: float,
        ate: bool = False,
        danger: float = 0.0,
        moving: float = 0.0,
    ) -> "DriveState":
        dt = max(float(dt), 0.0)
        danger_signal = _unit(danger)
        moving_signal = _unit(moving)

        self.hunger += 0.015 * dt
        self.curiosity += 0.010 * dt

        if ate:
            self.hunger -= 0.55
            self.curiosity -= 0.20
            self.energy += 0.10
        else:
            self.curiosity += 0.006 * dt

        self.energy += 0.020 * dt * (1.0 - moving_signal)
        self.energy -= 0.045 * dt * moving_signal

        self.danger += (danger_signal - self.danger) * min(4.0 * dt, 1.0)
        if danger_signal <= 1e-6:
            self.danger -= 0.45 * dt

        self.fear += 0.75 * self.danger * dt
        if danger_signal <= 1e-6:
            self.fear -= 0.35 * dt

        self.hunger = _unit(self.hunger)
        self.energy = _unit(self.energy)
        self.curiosity = _unit(self.curiosity)
        self.danger = _unit(self.danger)
        self.fear = _unit(self.fear)
        self.target_interest = _unit(
            0.70 * self.hunger + 0.25 * self.curiosity - 0.35 * self.fear
        )
        return self
