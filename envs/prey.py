#!/usr/bin/env python3
"""Prey that runs away, so the gecko has something to actually hunt.

The environment's original target is a point the policy is handed the bearing and
range of. It does not move, it cannot be lost, and nothing about finding it is
hard. Replacing it with prey that flees is what makes the camera load-bearing.

Every parameter here is supplied by the caller and comes from `config/proxies.yaml`
with its own species, source and confidence. Nothing is defaulted, because a
default in this file would be an invented number wearing a published one's
clothes. `PreyParameters.from_registry` is the only way to build a live set.

What this deliberately does NOT model: prey that turns, evades sideways, uses
cover, tires, or has any behaviour beyond "move directly away when the predator
is close". Real crickets do all of those. The escape is a straight line because
that is the one thing the literature actually constrains, and inventing the rest
would make the gecko's hunting look better without any of it being true.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: Mean length of one walk-or-pause bout, seconds. INVENTED. Bout structure is
#: unmeasured in any cricket; this sets the timescale on which the prey's motion
#: switches on and off, which is exactly what a temporal-contrast detector sees.
_MEAN_BOUT_S = 1.6

#: Mean length of one walk-or-pause bout, seconds. INVENTED. Bout structure is
#: unmeasured in any cricket; this sets the timescale on which the prey's motion
#: switches on and off, which is exactly what a temporal-contrast detector sees.
_MEAN_BOUT_S = 1.6

REGISTRY_KEYS = (
    "prey_escape_speed_m_s",
    "prey_flee_radius_m",
    "prey_escape_latency_s",
    "prey_capture_distance_m",
    "prey_radius_m",
    "prey_arena_radius_m",
    "prey_ambient_speed_m_s",
    "prey_ambient_move_fraction",
    "prey_ambient_turn_rate_rad_s",
    "prey_flee_speed_reference_m_s",
    "prey_flee_speed_exponent",
    "prey_flee_radius_floor_m",
)


@dataclass(frozen=True)
class PreyParameters:
    """Values only; provenance lives in the registry entry each one came from."""

    escape_speed_m_s: float
    flee_radius_m: float
    escape_latency_s: float
    capture_distance_m: float
    radius_m: float
    arena_radius_m: float
    #: Undisturbed locomotion. All three are INVENTED -- see the registry.
    ambient_speed_m_s: float
    ambient_move_fraction: float
    ambient_turn_rate_rad_s: float
    #: How the flee radius responds to how fast the predator is closing. The
    #: reason a stalk can work at all. All three INVENTED -- see the registry.
    flee_speed_reference_m_s: float
    flee_speed_exponent: float
    flee_radius_floor_m: float
    #: Undisturbed locomotion. All three are INVENTED -- see the registry.
    ambient_speed_m_s: float
    ambient_move_fraction: float
    ambient_turn_rate_rad_s: float
    #: How the flee radius responds to how fast the predator is closing. The
    #: reason a stalk can work at all. All three INVENTED -- see the registry.
    flee_speed_reference_m_s: float
    flee_speed_exponent: float
    flee_radius_floor_m: float

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not math.isfinite(value):
                raise ValueError(f"Prey parameter {name} must be finite.")
            if value < 0:
                raise ValueError(f"Prey parameter {name} must be nonnegative.")
        if self.arena_radius_m <= self.flee_radius_m:
            raise ValueError("The arena must be larger than the flee radius, or prey has "
                             "nowhere to escape to and the chase is trivial.")
        if self.capture_distance_m <= 0:
            raise ValueError("Capture distance must be positive.")
        if self.ambient_speed_m_s >= self.escape_speed_m_s:
            raise ValueError("An undisturbed cricket must not walk faster than a fleeing "
                             "one. That ordering is the only thing justifying the ambient "
                             "speed at all, since no undisturbed value is published.")
        if not 0.0 <= self.ambient_move_fraction <= 1.0:
            raise ValueError("ambient_move_fraction is a fraction of time.")
        if self.flee_speed_reference_m_s <= 0.0:
            raise ValueError("The flee-speed reference must be positive.")
        if self.flee_radius_floor_m >= self.flee_radius_m:
            raise ValueError("The floor must sit inside the nominal flee radius, or "
                             "approach speed cannot matter.")
        if self.ambient_speed_m_s >= self.escape_speed_m_s:
            raise ValueError("An undisturbed cricket must not walk faster than a fleeing "
                             "one. That ordering is the only thing justifying the ambient "
                             "speed at all, since no undisturbed value is published.")
        if not 0.0 <= self.ambient_move_fraction <= 1.0:
            raise ValueError("ambient_move_fraction is a fraction of time.")
        if self.flee_speed_reference_m_s <= 0.0:
            raise ValueError("The flee-speed reference must be positive.")
        if self.flee_radius_floor_m >= self.flee_radius_m:
            raise ValueError("The floor must sit inside the nominal flee radius, or "
                             "approach speed cannot matter.")

    @classmethod
    def from_registry(cls, overrides=None):
        from common.provenance import parameter_value
        resolved = {
            "escape_speed_m_s": float(parameter_value("prey_escape_speed_m_s")),
            "flee_radius_m": float(parameter_value("prey_flee_radius_m")),
            "escape_latency_s": float(parameter_value("prey_escape_latency_s")),
            "capture_distance_m": float(parameter_value("prey_capture_distance_m")),
            "radius_m": float(parameter_value("prey_radius_m")),
            "arena_radius_m": float(parameter_value("prey_arena_radius_m")),
            "ambient_speed_m_s": float(parameter_value("prey_ambient_speed_m_s")),
            "ambient_move_fraction": float(parameter_value("prey_ambient_move_fraction")),
            "ambient_turn_rate_rad_s": float(parameter_value("prey_ambient_turn_rate_rad_s")),
            "flee_speed_reference_m_s": float(parameter_value("prey_flee_speed_reference_m_s")),
            "flee_speed_exponent": float(parameter_value("prey_flee_speed_exponent")),
            "flee_radius_floor_m": float(parameter_value("prey_flee_radius_floor_m")),
        }
        for key, value in (overrides or {}).items():
            if key not in resolved:
                raise ValueError(f"Unknown prey parameter: {key}")
            resolved[key] = float(value)
        return cls(**resolved)


class FleeingPrey:
    """A cricket that walks about, and runs directly away when a predator closes.

    IT USED TO SIT PERFECTLY STILL, AND THAT WAS THE BUG UNDER THE TECTUM.
    `step` moved the prey only inside the 0.075 m flee radius, while the eye is
    asked to find it at 0.30-0.50 m. Measured over 1800 frames,
    `prey_total_travel_mm` was 0.0. Three sessions of vision work were spent
    pointing a motion detector at a stationary object (ledger #135), and no eye
    could have succeeded.

    So the prey now has undisturbed locomotion: walk bouts separated by pauses,
    with the heading drifting during a bout. Every parameter of that is
    INVENTED and says so in the registry -- no walking speed, bout structure or
    turn rate has ever been published for an undisturbed cricket. The ambient
    speed is pinned BELOW the published escape speed, which is an ordering
    claim rather than a measurement and is the only defensible thing available.

    Bouts rather than constant gliding is deliberate and is not conservatism: a
    target moving at constant velocity is the EASIEST possible stimulus for a
    motion detector, so a gliding cricket would flatter the tectum.

    Position is world XY at a fixed height; the caller writes it into a mocap
    body so it renders to the camera without entering the physics state.
    """

    def __init__(self, parameters, height_m, rng=None, proximity_capture=True):
        if not isinstance(parameters, PreyParameters):
            raise TypeError("parameters must be a PreyParameters")
        self.p = parameters
        self.height_m = float(height_m)
        # WHO OWNS CAPTURE. With no strike in the model, walking within the
        # capture distance had to count as eating or nothing ever could. Once a
        # strike exists that is wrong twice over: it feeds the animal for
        # proximity alone, and it respawns the prey the instant it comes into
        # striking range, so the strike can never fire. Set False and capture
        # becomes the strike's business entirely.
        self.proximity_capture = bool(proximity_capture)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self.position = np.zeros(2)
        self.captures = 0
        self._alarm_countdown = None
        self._fled_distance = 0.0
        self._heading = 0.0
        self._walking = False
        self._bout_left = 0.0
        self._ambient_distance = 0.0
        self._last_predator_xy = None
        self._last_distance = None
        self._approach_speed = 0.0

    def reset(self, predator_xy, rng=None, place_at=None):
        """Place prey somewhere the predator has to travel to reach."""
        if rng is not None:
            self._rng = rng
        self.captures = 0
        self._alarm_countdown = None
        self._fled_distance = 0.0
        self._ambient_distance = 0.0
        self._heading = float(self._rng.uniform(-math.pi, math.pi))
        self._walking = False
        self._bout_left = 0.0
        self._last_predator_xy = None
        self._last_distance = None
        self._approach_speed = 0.0
        self.position = (np.asarray(place_at, dtype=float) if place_at is not None
                         else self._respawn(predator_xy))
        return self.position.copy()

    def _respawn(self, predator_xy):
        """Uniform on the arena disc, never inside the flee radius."""
        predator_xy = np.asarray(predator_xy, dtype=float)[:2]
        for _ in range(64):
            angle = self._rng.uniform(-math.pi, math.pi)
            # sqrt keeps the sample uniform over area rather than clustered inward.
            radius = self.p.arena_radius_m * math.sqrt(self._rng.uniform(0., 1.))
            candidate = predator_xy + radius * np.array([math.cos(angle), math.sin(angle)])
            if np.linalg.norm(candidate - predator_xy) > self.p.flee_radius_m:
                return candidate
        # Deterministic fallback so a pathological rng cannot hang a rollout.
        return predator_xy + np.array([self.p.arena_radius_m, 0.])

    def flee_radius_for(self, approach_speed_m_s):
        """How close the predator gets before this prey bolts.

        THE STALK LIVES HERE. With a fixed radius the model was unwinnable: the
        prey bolted at 0.075 m, the strike must be launched from 0.0203 m, and
        the prey outruns the walker 2:1. Measured over 60 s of pursuit the
        closest approach was 51.1 mm and no strike was ever possible. Adding a
        strike was necessary and not sufficient.

        A real gecko closes that gap by creeping. The published ethogram for
        this species names the behaviour -- `walk slow motion`, "walking with a
        strongly reduced speed, mostly in context of prey capture" -- and it is
        meaningless against a prey that cannot tell a creep from a charge.

        The DIRECTION is well supported: flight initiation distance rises with
        approach speed throughout the antipredator literature, and crickets in
        particular sense an approaching predator through cercal air-current
        receptors, where a faster approach makes a larger signal. The MAGNITUDE
        is INVENTED. Any capture rate this model produces rests on it.
        """
        speed = max(float(approach_speed_m_s), 0.0)
        scale = (speed / self.p.flee_speed_reference_m_s) ** self.p.flee_speed_exponent
        return float(np.clip(self.p.flee_radius_m * scale,
                             self.p.flee_radius_floor_m, self.p.flee_radius_m))

    def step(self, dt, predator_xy):
        """Advance prey by dt. Returns (position, captured_this_step)."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        predator_xy = np.asarray(predator_xy, dtype=float)[:2]
        offset = self.position - predator_xy
        distance = float(np.linalg.norm(offset))
        # CLOSING SPEED, NOT PREDATOR SPEED, and the difference mattered. The
        # predator point is the gecko's nose, which sways several centimetres
        # per second with the gait even when the animal is barely advancing --
        # measured 0.129 m/s of apparent approach during a creep that netted
        # almost nothing. Reading that as a charge made stalking impossible.
        # What a prey animal actually has to go on is how fast the threat is
        # getting closer, so that is what is measured: the rate at which the
        # distance shrinks, floored at zero because retreating is not
        # approaching.
        if self._last_distance is not None:
            closing = max((self._last_distance - distance) / dt, 0.0)
            self._approach_speed = 0.8 * self._approach_speed + 0.2 * closing
        self._last_distance = distance
        self._last_predator_xy = predator_xy.copy()

        captured = self.proximity_capture and distance <= self.p.capture_distance_m
        if captured:
            self.captures += 1
            self.position = self._respawn(predator_xy)
            self._alarm_countdown = None
            return self.position.copy(), True

        if distance <= self.flee_radius_for(self._approach_speed):
            # Latency: a startled animal does not accelerate instantaneously, and
            # a zero-latency prey is uncatchable for reasons that are an artefact
            # of the simulation rather than of the animal.
            if self._alarm_countdown is None:
                self._alarm_countdown = self.p.escape_latency_s
            self._alarm_countdown -= dt
            if self._alarm_countdown <= 0.:
                direction = offset / distance if distance > 1e-9 else self._random_unit()
                self.position = self.position + direction * self.p.escape_speed_m_s * dt
                self._fled_distance += self.p.escape_speed_m_s * dt
        else:
            self._alarm_countdown = None
            # Undisturbed: walk in bouts. A cricket beyond the flee radius has
            # not noticed the gecko and is going about its business, which is
            # the whole reason the eye has anything to detect at all.
            self._ambient_step(dt)

        # Keep prey inside the arena, measured from where it was placed, so a
        # long chase cannot walk it over the horizon and out of the camera.
        radius = float(np.linalg.norm(self.position))
        if radius > self.p.arena_radius_m * 2.:
            self.position *= (self.p.arena_radius_m * 2.) / radius
        return self.position.copy(), False

    def _ambient_step(self, dt):
        """One step of undisturbed walking. Bouts, not a glide."""
        if self.p.ambient_speed_m_s <= 0.0:
            return
        self._bout_left -= dt
        if self._bout_left <= 0.0:
            # Flip between walking and pausing. Mean bout length is scaled so
            # the long-run duty cycle is ambient_move_fraction. Both INVENTED.
            self._walking = not self._walking
            mean = (_MEAN_BOUT_S * self.p.ambient_move_fraction if self._walking
                    else _MEAN_BOUT_S * (1.0 - self.p.ambient_move_fraction))
            self._bout_left = float(self._rng.exponential(max(mean, 1e-3)))
            if self._walking:
                self._heading = float(self._rng.uniform(-math.pi, math.pi))
        if not self._walking:
            return
        # Heading drifts within a bout so the path curves rather than tracking
        # a straight line -- again, a straight line is the easy case.
        self._heading += float(self._rng.normal(
            0.0, self.p.ambient_turn_rate_rad_s * math.sqrt(dt)))
        travelled = self.p.ambient_speed_m_s * dt
        self.position = self.position + travelled * np.array(
            [math.cos(self._heading), math.sin(self._heading)])
        self._ambient_distance += travelled

    def _ambient_step(self, dt):
        """One step of undisturbed walking. Bouts, not a glide."""
        if self.p.ambient_speed_m_s <= 0.0:
            return
        self._bout_left -= dt
        if self._bout_left <= 0.0:
            # Flip between walking and pausing. Mean bout length is scaled so
            # the long-run duty cycle is ambient_move_fraction. Both INVENTED.
            self._walking = not self._walking
            mean = (_MEAN_BOUT_S * self.p.ambient_move_fraction if self._walking
                    else _MEAN_BOUT_S * (1.0 - self.p.ambient_move_fraction))
            self._bout_left = float(self._rng.exponential(max(mean, 1e-3)))
            if self._walking:
                self._heading = float(self._rng.uniform(-math.pi, math.pi))
        if not self._walking:
            return
        # Heading drifts within a bout so the path curves rather than tracking
        # a straight line -- again, a straight line is the easy case.
        self._heading += float(self._rng.normal(
            0.0, self.p.ambient_turn_rate_rad_s * math.sqrt(dt)))
        travelled = self.p.ambient_speed_m_s * dt
        self.position = self.position + travelled * np.array(
            [math.cos(self._heading), math.sin(self._heading)])
        self._ambient_distance += travelled

    def _random_unit(self):
        angle = self._rng.uniform(-math.pi, math.pi)
        return np.array([math.cos(angle), math.sin(angle)])

    @property
    def mocap_position(self):
        return np.array([self.position[0], self.position[1], self.height_m])

    def state(self):
        return {"position_xy": self.position.tolist(), "captures": self.captures,
                "fled_distance_m": self._fled_distance,
                "ambient_distance_m": self._ambient_distance,
                "total_travel_m": self._fled_distance + self._ambient_distance,
                "walking": self._walking,
                "approach_speed_m_s": self._approach_speed,
                "flee_radius_m": self.flee_radius_for(self._approach_speed),
                "alarmed": self._alarm_countdown is not None}
