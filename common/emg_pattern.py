"""Measured muscle-burst timing with an explicitly invented interpolation shape.

Jagnandan & Higham (2018), J Exp Biol 221:jeb179564, Table 3,
https://doi.org/10.1242/jeb.179564. Timing is relative to same-foot touchdown.

These are EMG activation priors, NOT muscle forces or desired joint angles.
The paper supplies burst onset, duration and peak, not the full waveform. The
piecewise cubic smoothstep below is an engineering interpolation. Amplitudes
are normalized within each muscle; comparing them between muscles is invalid.
No recovered policy or controller is silently changed by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from common.provenance import load_registry


@dataclass(frozen=True)
class Burst:
    onset: float
    duration: float
    peak: float
    amplitude: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.onset, self.duration, self.peak, self.amplitude)):
            raise ValueError('Burst values must be finite')
        if not 0 < self.duration < 1 or not 0 <= self.amplitude <= 1:
            raise ValueError('Duration must be in (0,1), normalized amplitude in [0,1]')
        if not 0 < (self.peak-self.onset) % 1 < self.duration:
            raise ValueError('The peak must lie strictly inside the burst window')

    def sample(self, phase):
        """Periodic, nonnegative C1 envelope; input can be scalar or an array."""
        phase = np.asarray(phase, dtype=float)
        if not np.all(np.isfinite(phase)):
            raise ValueError('Phase must be finite')
        age = (phase-self.onset) % 1.0
        peak_age = (self.peak-self.onset) % 1.0
        rising = np.clip(age/peak_age, 0., 1.)
        falling = np.clip((self.duration-age)/(self.duration-peak_age), 0., 1.)
        ramp = np.where(age <= peak_age, rising, falling)
        envelope = self.amplitude*ramp*ramp*(3.-2.*ramp)
        return np.where(age < self.duration, envelope, 0.)


def hindlimb_patterns():
    """Return immutable bursts, loaded from the single provenance registry."""
    values = load_registry()['entries']['emg_hindlimb_preautotomy']['value']
    return {muscle: tuple(Burst(**burst) for burst in bursts)
            for muscle, bursts in values.items()}


def activation_prior(muscle, local_phase):
    """Combine each explicitly represented burst without inventing force units."""
    patterns = hindlimb_patterns()
    if muscle not in patterns:
        raise ValueError(f'No verified timing entry for muscle {muscle!r}')
    return sum(burst.sample(local_phase) for burst in patterns[muscle])


def sample_hindlimbs(gait_profile, time_s):
    """Use the shared touchdown clock, never a free-running muscle oscillator."""
    from common.gait_config import get_gait_profile
    profile = get_gait_profile(gait_profile)
    if profile.name != 'lab':
        raise ValueError('EMG timing uses the lab touchdown convention only')
    return {foot: {muscle: float(activation_prior(muscle, profile.phase_fraction(foot, time_s)))
                   for muscle in hindlimb_patterns()}
            for foot in ('HL', 'HR')}
