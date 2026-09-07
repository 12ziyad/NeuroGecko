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

REGISTRY_KEYS = (
    "prey_escape_speed_m_s",
    "prey_flee_radius_m",
    "prey_escape_latency_s",
    "prey_capture_distance_m",
    "prey_radius_m",
    "prey_arena_radius_m",
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
        }
        for key, value in (overrides or {}).items():
            if key not in resolved:
                raise ValueError(f"Unknown prey parameter: {key}")
            resolved[key] = float(value)
        return cls(**resolved)


class FleeingPrey:
    """A point that sits still until a predator is close, then runs directly away.

    Position is world XY at a fixed height; the caller writes it into a mocap
    body so it renders to the camera without entering the physics state.
    """

    def __init__(self, parameters, height_m, rng=None):
        if not isinstance(parameters, PreyParameters):
            raise TypeError("parameters must be a PreyParameters")
        self.p = parameters
        self.height_m = float(height_m)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self.position = np.zeros(2)
        self.captures = 0
        self._alarm_countdown = None
        self._fled_distance = 0.0

    def reset(self, predator_xy, rng=None, place_at=None):
        """Place prey somewhere the predator has to travel to reach."""
        if rng is not None:
            self._rng = rng
        self.captures = 0
        self._alarm_countdown = None
        self._fled_distance = 0.0
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

    def step(self, dt, predator_xy):
        """Advance prey by dt. Returns (position, captured_this_step)."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        predator_xy = np.asarray(predator_xy, dtype=float)[:2]
        offset = self.position - predator_xy
        distance = float(np.linalg.norm(offset))

        captured = distance <= self.p.capture_distance_m
        if captured:
            self.captures += 1
            self.position = self._respawn(predator_xy)
            self._alarm_countdown = None
            return self.position.copy(), True

        if distance <= self.p.flee_radius_m:
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

        # Keep prey inside the arena, measured from where it was placed, so a
        # long chase cannot walk it over the horizon and out of the camera.
        radius = float(np.linalg.norm(self.position))
        if radius > self.p.arena_radius_m * 2.:
            self.position *= (self.p.arena_radius_m * 2.) / radius
        return self.position.copy(), False

    def _random_unit(self):
        angle = self._rng.uniform(-math.pi, math.pi)
        return np.array([math.cos(angle), math.sin(angle)])

    @property
    def mocap_position(self):
        return np.array([self.position[0], self.position[1], self.height_m])

    def state(self):
        return {"position_xy": self.position.tolist(), "captures": self.captures,
                "fled_distance_m": self._fled_distance,
                "alarmed": self._alarm_countdown is not None}
