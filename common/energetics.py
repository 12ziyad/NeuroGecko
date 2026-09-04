"""Mechanical actuator work, not metabolic energy or electrical consumption."""
from __future__ import annotations

import numpy as np


def actuator_work(force, velocity, dt: float) -> tuple[float, float, float]:
    """Rectangle quadrature of force * transmission velocity for one physics step.

    MuJoCo actuator force and actuator velocity are conjugate transmission
    quantities. Sum positive and negative work per actuator before combining;
    do not cancel an actuator doing positive work against another absorbing work.
    Negative work is signed (<=0); absolute work is positive - negative.
    """
    force, velocity = np.asarray(force), np.asarray(velocity)
    if force.shape != velocity.shape or dt <= 0:
        raise ValueError('Matching force/velocity shapes and positive dt required')
    power = force * velocity
    if not np.isfinite(power).all():
        raise ValueError('Non-finite actuator power')
    positive = float(np.maximum(power, 0).sum() * dt)
    negative = float(np.minimum(power, 0).sum() * dt)
    return positive, negative, positive - negative
