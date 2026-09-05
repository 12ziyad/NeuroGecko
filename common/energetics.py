"""Mechanical actuator work, not metabolic energy or electrical consumption."""
from __future__ import annotations

import math
import numpy as np


def _actuator_power_values(force, velocity):
    """One power per actuator, using its conjugate transmission velocity.

    ``MjData.actuator_velocity`` has the actuator dimension and incorporates
    tendon/joint transmissions. Generalized ``qvel`` is not interchangeable.
    """
    force, velocity = np.asarray(force), np.asarray(velocity)
    if force.shape != velocity.shape:
        raise ValueError('Matching actuator force/transmission-velocity shapes required')
    power = force * velocity
    if not np.isfinite(power).all():
        raise ValueError('Non-finite actuator power')
    return power


def actuator_power(force, velocity) -> tuple[float, float, float]:
    """Positive, signed negative, and summed absolute mechanical power in W.

    Sum absolute power PER ACTUATOR: two actuators producing and absorbing
    equal power must not cancel. This includes whatever total actuator force
    the plant generated, regardless of a base/residual control decomposition.
    It is not metabolic effort, electrical consumption, or an isometric cost.
    """
    power = _actuator_power_values(force, velocity)
    positive = float(np.maximum(power, 0).sum())
    negative = float(np.minimum(power, 0).sum())
    return positive, negative, positive - negative


def mean_absolute_actuator_power(absolute_work_J: float, interval_s: float) -> float:
    """Mean sum_i |force_i * actuator_velocity_i| over a measured interval.

    The input work must already sum absolute work per actuator AND physics
    substep. Using absolute net work here would hide opposing/regenerative work.
    """
    if not math.isfinite(absolute_work_J) or absolute_work_J < 0:
        raise ValueError('Absolute mechanical work must be finite and nonnegative')
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError('Power averaging interval must be finite and positive')
    power = float(absolute_work_J / interval_s)
    if not math.isfinite(power):
        raise ValueError('Mean actuator power overflowed')
    return power


def actuator_work(force, velocity, dt: float) -> tuple[float, float, float]:
    """Rectangle quadrature of force * transmission velocity for one physics step.

    MuJoCo actuator force and actuator velocity are conjugate transmission
    quantities. Sum positive and negative work per actuator before combining;
    do not cancel an actuator doing positive work against another absorbing work.
    Negative work is signed (<=0); absolute work is positive - negative.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Finite positive physics timestep required')
    power = _actuator_power_values(force, velocity)
    positive = float(np.maximum(power, 0).sum() * dt)
    negative = float(np.minimum(power, 0).sum() * dt)
    return positive, negative, positive - negative
