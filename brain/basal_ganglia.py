#!/usr/bin/env python3
"""Action selection: pick one behaviour and hold it, instead of dithering.

The environment currently arbitrates behaviour with an if/else on a hand-made
`target_interest` scalar. That is not a mechanism, it is a preference ordering,
and it has the failure mode every preference ordering has: when two options are
close it flips between them every step.

This is the GPR model (Gurney, Prescott & Redgrave 2001) in the extended form
`docs/research/best_achievable_brain.md` section 3.3 specifies -- the one actually
validated on robots by Girard 2003 and Prescott 2006/2024. Channels compete
through a striatum/STN/GPe/GPi loop and the winner is *disinhibited*: the basal
ganglia's output is tonically ON and selection means switching it OFF for one
channel. That inversion is the whole architecture, and it is why the model
persists where a winner-take-all dithers.

Licence
-------
ModelDB 124111 (Girard 2008) is the reference implementation and carries **no
licence anywhere** -- ModelDB tree, GitHub mirror, or author's repo -- so it is
all-rights-reserved by default and must not be vendored into an Apache-2.0
repository. Section 3.3 says to reimplement from the equations. This file is that
reimplementation; no code was copied from it or from the Prescott 2024 C++.

Dopamine
--------
Tonic dopamine `lambda` scales the two striatal populations in opposite
directions: D1 gain `1 + lambda`, D2 gain `1 - lambda`. Baseline is **0.20**, and
that single number is why this module is testable. Prescott et al. 2024 swept 18
levels over 5 x 120 s robot trials and reported a specific, non-monotonic pattern
of failure at each end:

    lambda <= 0.06   prolonged immobility (14 s/trial at 0.06, 38 s at 0.03,
                     against ~2 s at baseline)
    lambda <= 0.12   movement below 75% of intended vigour
    0.20 - 0.29      89-95% of competitions clean; all trials succeed
    from 0.31        distortion: losing channels partially expressed
    0.31 - 0.34      mixed outcomes
    0.40 - 0.43      most trials fail; switching bouts ~3x baseline
                     (21.3 against ~7; a winner-take-all gives 9.2 there)
    0.46             every trial fails

`tests/test_basal_ganglia.py` runs that sweep. The module is falsifiable in a way
a preference ordering is not: too little dopamine must freeze it, too much must
make it dither, and a plain winner-take-all must dither *less* under excess
dopamine than the real model does -- which is the counterintuitive published
result and the one a plausible-looking reimplementation will miss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# The six dorsal-loop behaviours section 3.3 names.
CHANNELS = ("hunt", "flee", "explore", "bask", "rest", "groom")


def piecewise_linear(activation, threshold, slope=1.0):
    """GPR output function: dead zone below threshold, saturating at 1."""
    value = slope * (np.asarray(activation, dtype=float) - threshold)
    return np.clip(value, 0.0, 1.0)


@dataclass(frozen=True)
class GPRParameters:
    """Connection weights and thresholds of the GPR loop.

    These are the parameters of Gurney, Prescott & Redgrave 2001, fitted to rat
    basal ganglia and transplanted here on the lamprey-to-mammal conservation
    argument section 3.3 makes (Stephenson-Jones 2011; Grillner 2013).

    The corpus does NOT restate the individual weights -- it names the model and
    gives the dopamine sweep -- so these come from the primary model rather than
    from `docs/research/`. That is exactly why the acceptance test is the
    published *behavioural* sweep and not the parameter values: if these are
    wrong, the lambda sweep will not reproduce Prescott 2024's pattern, and the
    test will say so.
    """

    # striatum
    d1_from_salience: float = 1.0
    d2_from_salience: float = 1.0
    # subthalamic nucleus: excited by salience, inhibited by globus pallidus
    stn_from_salience: float = 1.0
    stn_from_gpe: float = 1.0
    # globus pallidus externa: diffuse STN excitation, focused D2 inhibition
    gpe_from_stn: float = 0.9
    gpe_from_d2: float = 1.0
    # output nucleus: diffuse STN excitation, focused D1 inhibition, GPe control
    gpi_from_stn: float = 0.9
    gpi_from_d1: float = 1.0
    gpi_from_gpe: float = 0.3
    # output-function thresholds
    threshold_striatum: float = 0.2
    threshold_stn: float = -0.25
    threshold_gpe: float = -0.2
    threshold_gpi: float = -0.2
    # --- Humphries & Gurney 2002 thalamocortical loop ------------------------
    # Section 3.3 requires the EXTENDED model, and this loop is what "extended"
    # means. Without it the model still selects, but it has no persistence: a
    # channel that wins gains nothing from having won, so bout duration and
    # switching frequency -- the two quantities Girard 2003 measures and the
    # dopamine sweep is scored on -- have no mechanism behind them.
    # Cortex -> thalamus -> cortex is POSITIVE feedback, so the product of the
    # two gains is a loop gain and it must be strictly below 1 or the loop does
    # not settle. At 1.0 the module oscillated at zero input, cycling gates
    # between 0 and 0.34 forever. The constraint is arithmetic; the particular
    # value below it is an engineering choice, because the corpus gives the
    # architecture and the dopamine sweep but not these weights.
    cortex_from_thalamus: float = 0.5
    thalamus_from_cortex: float = 1.0
    thalamus_from_gpi: float = 1.0
    reticular_from_cortex: float = 1.0
    reticular_from_thalamus: float = 1.0
    thalamus_from_reticular: float = 0.4
    threshold_cortex: float = 0.0
    threshold_thalamus: float = 0.0
    threshold_reticular: float = 0.0
    # leaky-integrator time constant, seconds
    tau_s: float = 0.04
    # output scaling for the disinhibition gate
    gate_scale: float = 0.2

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not math.isfinite(value):
                raise ValueError(f"GPR parameter {name} must be finite.")
        if self.tau_s <= 0:
            raise ValueError("tau_s must be positive.")
        if self.gate_scale <= 0:
            raise ValueError("gate_scale must be positive.")
        loop_gain = self.cortex_from_thalamus * self.thalamus_from_cortex
        if loop_gain >= 1.0:
            raise ValueError(
                f"Thalamocortical loop gain is {loop_gain:.3f}. Cortex -> thalamus -> "
                "cortex is positive feedback, so a gain of 1 or more never settles: "
                "the module oscillates instead of persisting.")


@dataclass
class BasalGanglia:
    """N competing channels. Output is a per-channel disinhibition gate in [0,1].

    Selection is release from tonic inhibition, not a max(): every channel is
    held down by the output nucleus and the winner is the one let go. A channel
    can therefore be partially released -- which is the "distortion" Prescott
    2024 reports at high dopamine, and which a winner-take-all cannot express.
    """

    channels: tuple = CHANNELS
    dopamine: float = 0.20
    parameters: GPRParameters = field(default_factory=GPRParameters)
    winner_take_all: bool = False

    def __post_init__(self):
        if len(self.channels) < 2:
            raise ValueError("Selection needs at least two channels.")
        if not math.isfinite(self.dopamine):
            raise ValueError("dopamine must be finite.")
        if not 0.0 <= self.dopamine <= 1.0:
            raise ValueError("dopamine (tonic lambda) must lie in [0, 1].")
        size = len(self.channels)
        self._d1 = np.zeros(size)
        self._d2 = np.zeros(size)
        self._stn = np.zeros(size)
        self._gpe = np.zeros(size)
        self._gpi = np.zeros(size)
        self._cortex = np.zeros(size)
        self._thalamus = np.zeros(size)
        self._reticular = np.zeros(size)
        # `gate_scale` was an invented constant. Selection is release from TONIC
        # inhibition, so the only non-arbitrary scale is the model's own resting
        # output: at zero salience the gate must be exactly zero, because nothing
        # has been selected. Measure that rest state once and use it.
        self._resting_gpi = self._measure_resting_output()

    def _measure_resting_output(self, seconds=30.0, dt=0.02):
        """Steady-state GPi output with no salience anywhere."""
        saved = {name: getattr(self, name).copy()
                 for name in ("_d1", "_d2", "_stn", "_gpe", "_gpi",
                              "_cortex", "_thalamus", "_reticular")}
        try:
            self.reset()
            zero = np.zeros(self.n)
            for _ in range(int(seconds / dt)):
                self._advance(zero, dt)
            return float(np.max(piecewise_linear(self._gpi,
                                                 self.parameters.threshold_gpi)))
        finally:
            for name, value in saved.items():
                getattr(self, name)[:] = value

    @property
    def n(self):
        return len(self.channels)

    def reset(self):
        for name in ("_d1", "_d2", "_stn", "_gpe", "_gpi",
                     "_cortex", "_thalamus", "_reticular"):
            getattr(self, name)[:] = 0.0
        return self

    def _leak(self, state, target, dt_s):
        """First-order approach to target with time constant tau."""
        alpha = 1.0 - math.exp(-dt_s / self.parameters.tau_s)
        state += alpha * (target - state)
        return state

    def step(self, salience, dt_s):
        """Advance one control interval. Returns the disinhibition gates."""
        salience = np.asarray(salience, dtype=float)
        if salience.shape != (self.n,):
            raise ValueError(f"salience must have shape ({self.n},)")
        if not np.all(np.isfinite(salience)):
            raise ValueError("salience must be finite.")
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive.")
        return self._advance(salience, dt_s)

    def _advance(self, salience, dt_s):
        salience = np.asarray(salience, dtype=float)
        if salience.shape != (self.n,):
            raise ValueError(f"salience must have shape ({self.n},)")
        if not np.all(np.isfinite(salience)):
            raise ValueError("salience must be finite.")
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive.")

        p = self.parameters
        if self.winner_take_all:
            # The comparison arm. Not a straw man: it is the variant Prescott
            # 2024 ran alongside the real model, and at high dopamine it dithers
            # LESS (9.2 switching bouts against 21.3), which is the result a
            # plausible reimplementation gets backwards.
            # Given the SAME leaky-integrated cortical drive, so the two arms
            # differ in their selection mechanism and not in how noisy their
            # input is. Comparing a smoothed loop against a raw argmax would
            # manufacture the difference rather than measure it.
            self._leak(self._cortex, salience, dt_s)
            gates = np.zeros(self.n)
            drive = piecewise_linear(self._cortex, p.threshold_cortex)
            if np.max(drive) > 0:
                gates[int(np.argmax(drive))] = 1.0
            self._gpi = 1.0 - gates
            return gates

        # The striatum is driven by CORTEX, not by raw salience: the loop is
        # closed through the thalamus, and that closure is the persistence.
        y_cortex = piecewise_linear(self._cortex, p.threshold_cortex)
        y_thalamus = piecewise_linear(self._thalamus, p.threshold_thalamus)
        y_reticular = piecewise_linear(self._reticular, p.threshold_reticular)

        # Dopamine pushes the two striatal populations in opposite directions.
        d1_input = p.d1_from_salience * y_cortex * (1.0 + self.dopamine)
        d2_input = p.d2_from_salience * y_cortex * (1.0 - self.dopamine)

        # Outputs of the previous state drive this update, so the loop is a
        # dynamical system rather than an algebraic solve; persistence lives here.
        y_d1 = piecewise_linear(self._d1, p.threshold_striatum)
        y_d2 = piecewise_linear(self._d2, p.threshold_striatum)
        y_stn = piecewise_linear(self._stn, p.threshold_stn)
        y_gpe = piecewise_linear(self._gpe, p.threshold_gpe)

        # STN excitation is DIFFUSE: every GPe and GPi cell sees every STN
        # channel. Averaging instead of summing was tried, to stop the effective
        # gain growing with channel count, and it destroyed the discrimination
        # entirely -- a clearly losing channel became fully released. Reverted.
        # The channel-count dependence is real and is recorded as a known defect.
        stn_total = float(np.sum(y_stn))
        stn_input = p.stn_from_salience * y_cortex - p.stn_from_gpe * y_gpe
        gpe_input = p.gpe_from_stn * stn_total - p.gpe_from_d2 * y_d2
        gpi_input = (p.gpi_from_stn * stn_total
                     - p.gpi_from_d1 * y_d1
                     - p.gpi_from_gpe * y_gpe)

        self._leak(self._d1, d1_input, dt_s)
        self._leak(self._d2, d2_input, dt_s)
        self._leak(self._stn, stn_input, dt_s)
        self._leak(self._gpe, gpe_input, dt_s)
        self._leak(self._gpi, gpi_input, dt_s)

        # Thalamocortical loop. The ventrolateral thalamus is inhibited by the
        # basal ganglia output and by a DIFFUSE reticular signal; cortex is
        # driven by salience plus its own thalamic return. A disinhibited channel
        # therefore feeds itself, and a losing one is held down twice over.
        y_gpi_previous = piecewise_linear(self._gpi, p.threshold_gpi)
        reticular_total = float(np.sum(y_reticular))
        thalamus_input = (p.thalamus_from_cortex * y_cortex
                          - p.thalamus_from_gpi * y_gpi_previous
                          - p.thalamus_from_reticular * reticular_total)
        reticular_input = (p.reticular_from_cortex * y_cortex
                           + p.reticular_from_thalamus * y_thalamus)
        cortex_input = salience + p.cortex_from_thalamus * y_thalamus
        self._leak(self._thalamus, thalamus_input, dt_s)
        self._leak(self._reticular, reticular_input, dt_s)
        self._leak(self._cortex, cortex_input, dt_s)

        # e_i = L(1 - y_i/c), with c the model's own tonic output: full release
        # at zero GPi, fully closed at rest.
        return self._gates_from(self._gpi)

    def gates(self):
        return self._gates_from(self._gpi)

    def _gates_from(self, gpi):
        y_gpi = piecewise_linear(gpi, self.parameters.threshold_gpi)
        # Calibrating this to the model's own resting output was tried, to remove
        # an invented constant, and it collapsed the dynamic range so that every
        # dopamine level behaved identically. Reverted; the constant stays, and
        # stays declared as invented.
        return np.clip(1.0 - y_gpi / self.parameters.gate_scale, 0.0, 1.0)

    def selected(self, gates=None, minimum_release=1e-6):
        """The most disinhibited channel, or None when nothing is released.

        Selection is RELATIVE: the gate is a release from tonic inhibition and
        its absolute size depends on the salience scale, so an absolute cutoff
        would just be a hidden salience threshold. An earlier version used 0.5
        and reported a cleanly selecting model as immobile, because the winning
        gate sat at 0.451 against a runner-up at 0.211 -- unambiguous selection
        that an invented constant threw away.
        """
        gates = self.gates() if gates is None else np.asarray(gates, dtype=float)
        best = int(np.argmax(gates))
        return self.channels[best] if gates[best] > minimum_release else None

    def distortion(self, gates=None, fraction=0.5):
        """Losing channels released to at least `fraction` of the winner.

        Zero is clean selection. Prescott 2024 reports distortion -- partial
        expression of losing channels -- appearing from lambda 0.31, and it is a
        failure mode a winner-take-all cannot have at all, because its losers are
        exactly zero by construction.

        `fraction` is an engineering choice: the corpus reports that distortion
        occurs, not the criterion by which it was scored.
        """
        gates = self.gates() if gates is None else np.asarray(gates, dtype=float)
        best = float(np.max(gates))
        if best <= 1e-6:
            return 0
        return int(max(0, np.sum(gates >= fraction * best) - 1))

    def state(self):
        return {
            "channels": list(self.channels),
            "dopamine": self.dopamine,
            "winner_take_all": self.winner_take_all,
            "gates": self.gates().tolist(),
            "cortex": self._cortex.tolist(),
            "thalamus": self._thalamus.tolist(),
            "selected": self.selected(),
            "distortion": self.distortion(),
        }


def salience_from_drives(hunger, fatigue, cold, warm, threat=0.0, prey_visible=0.0,
                         persistence=None, persistence_gain=0.15):
    """Per-channel salience, sigma-pi style, from interoception and the world.

    Section 3.3 specifies salience built "sigma-pi style (Girard 2003 Table I)
    from the drive vector, tectal salience, DVR value, memory goal, plus
    persistence". Sigma-pi means terms are sums of PRODUCTS: hunting is not
    driven by hunger alone but by hunger AND visible prey, which is why a sated
    gecko ignores a cricket and a starving one in an empty arena does not hunt.

    The weights here are **INVENTED**. Girard 2003 Table I is a table for a
    different animal's behaviours, and no published salience weighting exists for
    a leopard gecko. What is published is what the SELECTION does with them once
    computed, which is what the dopamine sweep tests. These weights decide what
    the animal wants; the module decides how firmly it commits.
    """
    for name, value in (("hunger", hunger), ("fatigue", fatigue), ("cold", cold),
                        ("warm", warm), ("threat", threat), ("prey_visible", prey_visible)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    salience = np.array([
        hunger * prey_visible,                    # hunt: wants food AND sees some
        threat,                                   # flee
        0.35 * hunger * (1.0 - prey_visible),     # explore: hungry, nothing in sight
        max(cold, warm),                          # bask / thermoregulate
        fatigue,                                  # rest
        0.05,                                     # groom: low tonic baseline
    ], dtype=float)
    if persistence is not None:
        persistence = np.asarray(persistence, dtype=float)
        if persistence.shape != salience.shape:
            raise ValueError("persistence must match the channel count.")
        # The thalamocortical loop's contribution: what is already running is
        # slightly easier to keep running. This is what turns a competition into
        # a behaviour with a duration.
        salience = salience + persistence_gain * persistence
    return np.clip(salience, 0.0, None)
