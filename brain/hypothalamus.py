#!/usr/bin/env python3
"""Homeostatic drives built from published physiology instead of invented rates.

`brain/drives.py` has hunger rising at 0.015 per second, so it saturates in 67
seconds, and a meal subtracting a flat 0.55. Nobody measured those. This module
replaces them with the animal's own energy budget.

The mechanism is Homeostatically Regulated Reinforcement Learning (Keramati &
Gutkin, eLife 2014), which `docs/research/best_achievable_brain.md` section 3.4
specifies: an internal state `h` in a homeostatic space with setpoints `h*`, a
convex drive `D(h)`, and reward defined as the **reduction in drive**,
`r = D(h_t) - D(h_t+1)`. Under that definition reward maximisation is provably
homeostatic regulation, so the animal's goals are not a hand-written scalar --
they fall out of its physiology.

The numbers, and where they come from
-------------------------------------
Resting metabolism is **0.075 +/- 0.011 mL O2/g/h at 30 C**, n=6, and the
scorecard calls it the only direct metabolic measurement of *E. macularius*. That
single value sets how fast the animal gets hungry. Everything else follows:

* the oxycalorific equivalent 20.12 J/mL is recovered from the scorecard's own
  arithmetic (51.1 J/h / 2.54 mL/h) and lands on the standard 20.1 J/mL, which is
  the check that the published conversion was read correctly;
* a meal is `wet mass x dry fraction x 17.7 kJ/g x 0.71 digestibility`;
* the reserve is the tail, 22 % of body mass at an **assumed** 50 % lipid.

What that arithmetic says, and it is the point of this module
-------------------------------------------------------------
A 38 g gecko burns **49.5 J/h**. Its tail holds about **163 kJ**, or **137 days**
of resting metabolism -- the scorecard independently says ~140. So:

    one 20 s episode   costs 0.27 J = 0.00017 % of the reserve
    one 300 s episode  costs 4.12 J = 0.0025 %

**Hunger is invisible at episode timescales.** A real leopard gecko cannot get
hungry inside an episode, and a model whose hunger saturates in 67 seconds is not
modelling this animal at all. Energy therefore **persists across episodes** by
default, which is what `docs/research/` asks for, and any compression of that
timescale must be declared through `time_compression` rather than hidden in a
rate constant.

An independent consistency check that nobody arranged: one meal is ~79 h of
resting metabolism, and the husbandry schedule feeds these animals every ~56 h.
Those two numbers were measured by different people for different reasons and
they agree, with the small surplus a captive animal maintaining condition should
have. The corpus separately derives ~81 h for the same meal, which this module
reaches at 79 h from the raw measurement alone.

Two timescales, and conflating them was this module's first mistake
-------------------------------------------------------------------
The tail holds ~140 days. The published mean inter-meal interval is **2.33 days**.
Between meals the animal burns **1.7 % of its reserve** but **71 % of one meal**.
So hunger normalised against the reserve would never fire, and hunger normalised
against a meal fires exactly on the schedule the animal is fed on. Behavioural
hunger is therefore measured in meals (`energy_deficit`) and physical starvation
is reported separately (`starvation_fraction`). Both are real; only one drives
behaviour.

What is deliberately absent
---------------------------
* **The thermostat is inert.** Section 3.4 calls a temperature field the module's
  one real dependency, and no such field exists anywhere in this repository. The
  Hammel-style warm/cold rectified error is implemented and wired, but with body
  temperature pinned to the calibration constant it reads exactly zero forever.
  It is here so that adding a field switches it on, not so it can be claimed.
* **Digestion costs nothing.** Postprandial metabolism peaks at 3.7-7.3x resting
  for 62-170 h in another lizard genus. Adding meal energy without that cost
  overstates the benefit of eating. Recorded in the registry, not implemented.
* **`curiosity` and `target_interest` are gone.** They were in `drives.py` with no
  published basis of any kind. `fear` is also absent: section 3.4 is explicit that
  it should be *emergent* from the tectal escape integrator rather than
  hand-written, and the published finding is that fear here is chemically gated.

`brain/drives.py` is left untouched, because recovered checkpoints were trained
against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def _registry():
    from common.provenance import parameter_value
    return parameter_value


@dataclass(frozen=True)
class Physiology:
    """Published constants. Build with `from_registry`; nothing is defaulted."""

    body_mass_kg: float
    rmr_mL_O2_per_g_per_h: float
    rmr_mass_exponent: float
    rmr_reference_mass_g: float
    rmr_reference_temperature_C: float
    oxycalorific_J_per_mL: float
    digestibility: float
    prey_energy_kJ_per_g_dry: float
    prey_dry_fraction: float
    tail_mass_fraction: float
    tail_lipid_fraction: float
    lipid_J_per_g: float
    preferred_temperature_C: tuple
    metabolic_q10: float
    cost_of_transport_mL_O2_per_kg_per_m: float

    @classmethod
    def from_registry(cls, body_mass_kg, metabolic_q10=None):
        value = _registry()
        band = value("preferred_body_temperature_C")
        return cls(
            body_mass_kg=float(body_mass_kg),
            rmr_mL_O2_per_g_per_h=float(value("resting_metabolic_rate_mL_O2_per_g_per_h")),
            rmr_mass_exponent=float(value("resting_metabolic_rate_mass_exponent")),
            # A mass-specific rate is meaningless without the mass it was
            # measured on, and the scorecard does not state it. It is recovered
            # in the registry from the scorecard's own two figures: 19.5 g.
            # Anchoring at 40 g -- the obvious mistake, and one made here first --
            # inflates resting metabolism by 17%.
            rmr_reference_mass_g=float(value("resting_metabolic_rate_reference_mass_g")),
            rmr_reference_temperature_C=float(value("calibration_temperature_C")),
            oxycalorific_J_per_mL=float(value("oxycalorific_J_per_mL_O2")),
            digestibility=float(value("meal_apparent_dry_matter_digestibility")),
            prey_energy_kJ_per_g_dry=float(value("cricket_energy_kJ_per_g_dry")),
            prey_dry_fraction=float(value("cricket_dry_matter_fraction")),
            tail_mass_fraction=float(value("tail_mass_fraction")),
            tail_lipid_fraction=float(value("tail_lipid_fraction")),
            lipid_J_per_g=float(value("lipid_energy_J_per_g")),
            preferred_temperature_C=(float(band[0]), float(band[1])),
            # NOT IN CORPUS for E. macularius metabolism. 1.0 means "no thermal
            # dependence", which is wrong but is at least visibly wrong; supplying
            # a plausible Q10 would be inventing physiology.
            metabolic_q10=1.0 if metabolic_q10 is None else float(metabolic_q10),
            cost_of_transport_mL_O2_per_kg_per_m=float(
                value("cost_of_transport_mL_O2_per_kg_per_m")),
        )

    def resting_power_W(self, temperature_C=None):
        """Resting metabolic power in watts, at this mass and temperature."""
        grams = self.body_mass_kg * 1000.0
        specific = self.rmr_mL_O2_per_g_per_h * (grams / self.rmr_reference_mass_g) ** self.rmr_mass_exponent
        joules_per_hour = specific * grams * self.oxycalorific_J_per_mL
        if temperature_C is not None and self.metabolic_q10 != 1.0:
            exponent = (float(temperature_C) - self.rmr_reference_temperature_C) / 10.0
            joules_per_hour *= self.metabolic_q10 ** exponent
        return joules_per_hour / 3600.0

    def locomotion_power_W(self, speed_m_s):
        """Cost of moving, from the published cost of transport.

        0.73 mL O2 per kg per metre, and the published finding is that it is
        temperature-INDEPENDENT: the animal pays the same per metre warm or cool,
        even though how fast it can go is strongly temperature dependent. So this
        is one of the few terms here that needs no thermal correction.
        """
        speed = max(0.0, float(speed_m_s))
        mL_per_second = self.cost_of_transport_mL_O2_per_kg_per_m * self.body_mass_kg * speed
        return mL_per_second * self.oxycalorific_J_per_mL

    @property
    def reserve_J(self):
        """Tail lipid store. The 50 % lipid fraction is assumed, not measured."""
        tail_g = self.body_mass_kg * 1000.0 * self.tail_mass_fraction
        return tail_g * self.tail_lipid_fraction * self.lipid_J_per_g

    def meal_energy_J(self, wet_mass_kg):
        dry_g = wet_mass_kg * 1000.0 * self.prey_dry_fraction
        return dry_g * self.prey_energy_kJ_per_g_dry * 1000.0 * self.digestibility

    @property
    def preferred_temperature_midpoint_C(self):
        return 0.5 * sum(self.preferred_temperature_C)

    def summary(self):
        return {
            "body_mass_kg": self.body_mass_kg,
            "resting_power_W": self.resting_power_W(),
            "resting_J_per_hour": self.resting_power_W() * 3600.0,
            "reserve_J": self.reserve_J,
            "reserve_days_at_rest": self.reserve_J / (self.resting_power_W() * 86400.0),
            "preferred_temperature_C": list(self.preferred_temperature_C),
            "metabolic_q10": self.metabolic_q10,
            "locomotion_power_W_at_0.055_m_s": self.locomotion_power_W(0.055),
            "not_modelled": ["postprandial digestion cost (3.7-7.3x resting for 62-170 h)",
                             "thermal dependence of metabolism (no metabolic Q10 published "
                             "for this species; the corpus's 2.3 is a sleep-period Q10)",
                             "cold side of the thermostat (CTmin is NOT IN CORPUS)",
                             "hydration", "nutrient composition"],
        }


@dataclass
class Homeostasis:
    """HRRL internal state: energy against a setpoint, temperature against a band.

    `time_compression` multiplies elapsed time. It exists because the real
    timescale is days and an episode is seconds; it defaults to 1.0 so nothing is
    silently sped up, and whatever value is used is reported in `state()` so a
    result can never quietly depend on an undeclared compression.
    """

    physiology: Physiology
    energy_J: float = field(default=None)
    body_temperature_C: float = field(default=None)
    time_compression: float = 1.0
    energy_setpoint_fraction: float = 1.0
    meal_scale_J: float = field(default=None)
    # Keramati & Gutkin's D(h) = (sum_i |h*_i - h_i|^n)^(1/m). n=4, m=2 is their
    # canonical choice and it makes D genuinely CONVEX. That convexity is not
    # decoration: with n=m=2 the drive collapses to a linear distance and a meal
    # is worth exactly the same however hungry the animal is, which contradicts
    # the anticipatory and satiation effects HRRL is cited here for. At n=4, m=2
    # the same meal is worth more to a hungrier animal, which is the published
    # phenomenon. The exponents are a modelling choice, not a measurement.
    drive_exponent_n: float = 4.0
    drive_exponent_m: float = 2.0

    def __post_init__(self):
        if not math.isfinite(self.time_compression) or self.time_compression <= 0:
            raise ValueError("time_compression must be finite and positive.")
        if not 0 < self.energy_setpoint_fraction <= 1.0:
            raise ValueError("energy_setpoint_fraction must lie in (0, 1].")
        if self.meal_scale_J is None:
            from common.provenance import parameter_value
            self.meal_scale_J = self.physiology.meal_energy_J(
                float(parameter_value("prey_item_wet_mass_kg")) * 3.0)
        if self.meal_scale_J <= 0 or not math.isfinite(self.meal_scale_J):
            raise ValueError("meal_scale_J must be finite and positive.")
        if not (self.drive_exponent_n > 0 and self.drive_exponent_m > 0):
            raise ValueError("Drive exponents must be positive.")
        if self.drive_exponent_n <= self.drive_exponent_m:
            raise ValueError("n must exceed m or the drive is not convex, and a meal "
                             "would be worth the same however hungry the animal is.")
        if self.energy_J is None:
            self.energy_J = self.physiology.reserve_J * self.energy_setpoint_fraction
        if self.body_temperature_C is None:
            self.body_temperature_C = self.physiology.preferred_temperature_midpoint_C
        self.meals = 0
        self.elapsed_s = 0.0
        self._last_drive = self.drive()

    # ---- homeostatic space --------------------------------------------------

    @property
    def energy_setpoint_J(self):
        return self.physiology.reserve_J * self.energy_setpoint_fraction

    @property
    def energy_debt_J(self):
        """Joules below setpoint. The raw quantity both timescales read from."""
        return max(0.0, self.energy_setpoint_J - self.energy_J)

    @property
    def energy_deficit(self):
        """Behavioural hunger, in [0, 1], measured in MEALS not in reserve.

        There are two energy timescales here and conflating them was the first
        version's mistake. The tail holds ~140 days of resting metabolism, so a
        deficit measured against the reserve moves 1.7 % between meals and the
        animal would never become hungry. But the published mean inter-meal
        interval is 2.33 days, over which the animal burns about 71 % of one
        meal's energy -- so a deficit measured in meals reaches hunger exactly on
        the schedule the animal is actually fed on.

        Hunger is therefore normalised by meal size. The reserve is not the
        setpoint scale; it is the starvation buffer, and it is reported
        separately as `starvation_fraction`.
        """
        return float(min(self.deficit_in_meals, 1.0))

    @property
    def deficit_in_meals(self):
        """The same deficit, UNCLIPPED. This is what the drive reads.

        Clipping the drive would flatten it: an animal several meals in debt
        would sit at the ceiling, so eating would reduce nothing and earn no
        reward, and the HRRL reward would be exactly zero at precisely the moment
        food matters most. The observation saturates because a policy needs a
        bounded input; the drive does not, because a gradient has to survive out
        there.
        """
        meal = self.meal_scale_J
        if meal <= 0:
            return 0.0
        return float(self.energy_debt_J / meal)

    @property
    def starvation_fraction(self):
        """How far through the tail reserve the animal is, in [0, 1].

        A different question from hunger. This is the one that kills.
        """
        reserve = self.physiology.reserve_J
        if reserve <= 0:
            return 0.0
        return float(np.clip(1.0 - self.energy_J / reserve, 0.0, 1.0))

    @property
    def thermal_error_C(self):
        """Hammel-style rectified warm/cold error, zero inside the band.

        Inert while body temperature is pinned to the calibration constant: the
        world has no temperature field, so this reads exactly zero forever. It is
        wired so that adding a field switches it on.
        """
        low, high = self.physiology.preferred_temperature_C
        cold = max(0.0, low - self.body_temperature_C)
        warm = max(0.0, self.body_temperature_C - high)
        return cold, warm

    def drive(self):
        """Convex drive D(h): distance from setpoint in homeostatic space.

        Keramati & Gutkin use `D(h) = (sum_i |h*_i - h_i|^n)^(1/m)`. With n=m=2
        this is Euclidean distance, which is the form the corpus states and the
        simplest convex choice; the exponents are a modelling choice, not a
        measurement, and changing them changes what the animal wants.
        """
        cold, warm = self.thermal_error_C
        # Thermal error is divided by the band's own half-width so it enters the
        # same space as a deficit measured in meals rather than as raw degrees.
        low, high = self.physiology.preferred_temperature_C
        half_width = max(0.5 * (high - low), 1e-9)
        thermal = (cold + warm) / half_width
        n, m = self.drive_exponent_n, self.drive_exponent_m
        total = abs(self.deficit_in_meals) ** n + abs(thermal) ** n
        return float(total ** (1.0 / m))

    # ---- dynamics -----------------------------------------------------------

    def step(self, dt_s, meal_wet_mass_kg=0.0, activity_power_W=0.0):
        """Advance by dt_s. Returns the HRRL reward: the reduction in drive."""
        if not math.isfinite(dt_s) or dt_s < 0:
            raise ValueError("dt_s must be finite and nonnegative.")
        effective = dt_s * self.time_compression
        power = self.physiology.resting_power_W(self.body_temperature_C) + max(0.0, activity_power_W)
        self.energy_J -= power * effective
        if meal_wet_mass_kg > 0:
            self.energy_J += self.physiology.meal_energy_J(meal_wet_mass_kg)
            self.meals += 1
        # A reserve cannot be negative, and it cannot exceed what the tail holds.
        self.energy_J = float(np.clip(self.energy_J, 0.0, self.physiology.reserve_J))
        self.elapsed_s += effective

        drive_now = self.drive()
        reward = self._last_drive - drive_now
        self._last_drive = drive_now
        return float(reward)

    def reset(self, energy_J=None, body_temperature_C=None):
        """Reset the EPISODE, not the animal.

        Energy deliberately persists unless explicitly given, because a real
        gecko does not become full because an episode ended. Passing energy_J
        resets the body too, and that is a different experiment.
        """
        if energy_J is not None:
            self.energy_J = float(np.clip(energy_J, 0.0, self.physiology.reserve_J))
        if body_temperature_C is not None:
            self.body_temperature_C = float(body_temperature_C)
        self.elapsed_s = 0.0
        self._last_drive = self.drive()
        return self

    # ---- outputs ------------------------------------------------------------

    def vector(self):
        """Interoception for the policy: [hunger, energy, cold, warm].

        Four channels, not the six of `drives.py`. `curiosity` and
        `target_interest` had no published basis and are gone; `fear` belongs to
        the tectal escape integrator, not here; `danger` was an input wearing a
        drive's name.
        """
        cold, warm = self.thermal_error_C
        low, high = self.physiology.preferred_temperature_C
        half_width = max(0.5 * (high - low), 1e-9)
        return np.array([
            self.energy_deficit,
            1.0 - self.energy_deficit,
            min(cold / half_width, 1.0),
            min(warm / half_width, 1.0),
        ], dtype=np.float32)

    def state(self):
        return {
            "energy_J": self.energy_J,
            "energy_setpoint_J": self.energy_setpoint_J,
            "energy_debt_J": self.energy_debt_J,
            "meal_scale_J": self.meal_scale_J,
            "energy_deficit": self.energy_deficit,
            "deficit_in_meals": self.deficit_in_meals,
            "starvation_fraction": self.starvation_fraction,
            "hours_of_reserve_remaining": self.energy_J / (self.physiology.resting_power_W() * 3600.0),
            "body_temperature_C": self.body_temperature_C,
            "thermal_error_C": list(self.thermal_error_C),
            "thermostat_inert": self.thermal_error_C == (0.0, 0.0),
            "drive": self.drive(),
            "drive_exponents": [self.drive_exponent_n, self.drive_exponent_m],
            "meals": self.meals,
            "elapsed_s": self.elapsed_s,
            "time_compression": self.time_compression,
        }
