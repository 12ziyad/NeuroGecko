"""A faithful reimplementation of the Prescott et al. 2024 basal ganglia.

WHY THIS FILE EXISTS. `brain/basal_ganglia.py` was built from the Gurney,
Prescott & Redgrave 2001 equations as restated by Fox et al. 2009, with the
thalamocortical extension of Girard et al. 2005, and then scored against
Prescott et al. 2024's published selection statistics. It is a different model
from the one that produced those statistics, so the comparison could not
succeed however well either model worked. That mismatch -- not any defect in
the arithmetic -- is why our winner failed to saturate.

This module is the published model itself, so that the published numbers have
something to be compared against.

SOURCE. Prescott, Montes Gonzalez, Gurney, Humphries & Redgrave, "Simulated
dopamine modulation of a neurorobotic model of the basal ganglia", Biomimetics
2024, 9(3):139, CC BY 4.0 -- the paper, its Supplementary Methods, and the
authors' own C++ in the supplementary Model Code archive. Structure, weights,
thresholds, the dopamine transfer, the integrator, the convergence criterion,
the selection classifier and the two summary statistics are all taken from that
code. Nothing is vendored: this is a clean-room reimplementation in Python from
the published equations, which CC BY would permit anyway.

WHAT DIFFERS FROM THE 2001 MODEL WE HAD, and each of these is load-bearing:

  1. DOPAMINE MODULATES THE OUTPUT SLOPE, NOT THE INPUT GAIN. The 2001 model
     scales the cortical drive into the striatum by (1 +/- lambda). This model
     leaves the drive alone and changes the SLOPE of the striatal output
     function, with the D1 ramp pivoting about a fixed point. The paper calls
     this "new DA" in its own source. It is the paper's mechanism and the whole
     subject of the study, so using the 2001 form makes the dopamine sweep
     meaningless.
  2. THE STRIATAL THRESHOLD IS 0.1 (basic) or 0.15 (extended), not 0.2.
  3. SALIENCE REACHES THE STRIATUM AND STN THROUGH A LEAKY CORTICAL RELAY,
     not directly.
  4. The extended variant inserts a motor-cortex population between cortex and
     the striatum and splits the striatal and subthalamic drive 0.5/0.5 between
     salience and the thalamocortical return -- which is where persistence
     comes from in this lineage. That is NOT the Girard arrangement.

The five-nucleus weights (0.9 diffuse subthalamic, 0.3 pallidal) are the same
in both lineages and are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def rectify(x):
    """Piecewise-linear output with unit slope, clipped to [0, 1]."""
    return np.clip(x, 0.0, 1.0)


@dataclass(frozen=True)
class PrescottParameters:
    """Published parameters. `basic()` and `extended()` build the two variants.

    The authors ship both. Figure 5 -- the dopamine sweep this module is
    validated against -- is generated with EXTENDED set to 0, so the basic
    variant is the one the acceptance test uses. The extended variant is what a
    behaving animal needs, because it is the one with persistence.
    """

    # --- structure -------------------------------------------------------
    extended: bool = False
    #: Salience -> striatum / STN. 1.0 in the basic model; the extended model
    #: splits it 0.5 from cortex and 0.5 from the thalamocortical return.
    salience_share: float = 1.0
    motor_share: float = 0.0
    #: Thalamus -> motor cortex. Zero in the basic model: no persistence.
    thalamus_to_motor: float = 0.0

    # --- five-nucleus core, identical in both variants --------------------
    stn_out: float = 0.9          # STN -> GPe and STN -> GPi, within AND between
    gpe_gpi: float = 0.3          # GPe -> GPi, within channel
    gpe_stn: float = 1.0          # GPe -> STN, within channel
    sd1_gpi: float = 1.0
    sd2_gpe: float = 1.0

    # --- thresholds -------------------------------------------------------
    threshold_striatum: float = 0.1     # 0.15 in the extended variant
    threshold_stn: float = -0.25
    threshold_gpe: float = -0.2
    threshold_gpi: float = -0.2

    # --- the "new DA" transfer -------------------------------------------
    #: D1 slope is 1 + gain*lambda and its ramp pivots about `d1_pivot`.
    d1_gain: float = 1.0
    d1_pivot: float = 0.5               # 0.2 in the extended variant
    #: D2 slope is 1 - gain*lambda, no pivot.
    d2_gain: float = 0.8

    # --- thalamic reticular, extended variant only ------------------------
    trn_within: float = 0.1
    trn_between: float = 0.4
    #: Output nucleus -> reticular. The reticular population is inhibited by
    #: the basal ganglia as well as excited by cortex and thalamus.
    trn_from_gpi: float = 0.2
    #: True in the basic model, False in the extended one. See step().
    thalamus_threshold_is_tonic: bool = True

    # --- integration ------------------------------------------------------
    #: The leaky integrator is Euler with a fixed fractional step: the source
    #: sets k = 1/tau and step = 0.3*tau, so each update is a += 0.3*(u - a),
    #: independent of tau. Recorded as the published 0.3 rather than as a time
    #: constant, because that is what the number actually is.
    relax: float = 0.3
    tolerance: float = 1e-4             # max |da| for convergence
    max_steps: int = 5000

    @classmethod
    def basic(cls):
        """"Basic (new DA)" -- the variant that generates the published sweep."""
        return cls(extended=False, salience_share=1.0, motor_share=0.0,
                   thalamus_to_motor=0.0, threshold_striatum=0.1, d1_pivot=0.5)

    @classmethod
    def extended(cls):
        """"Extended (new DA)" -- adds motor cortex, thalamus and reticular."""
        return cls(extended=True, salience_share=0.5, motor_share=0.5,
                   thalamus_to_motor=1.0, threshold_striatum=0.15, d1_pivot=0.2,
                   thalamus_threshold_is_tonic=False)


class PrescottBasalGanglia:
    """The published model. Run it to convergence and read `gates()`.

    Selection is disinhibition: the output nucleus is tonically active and
    releasing a channel means switching it off. The gate is the published
    e_i = L(1 - y_gpi / c), where c is the model's OWN resting output -- not a
    free constant. It is computed here rather than hard-coded, so the two can
    never silently disagree.
    """

    def __init__(self, n_channels=5, n_primary=2, dopamine=0.2, parameters=None,
                 batch=None):
        if n_channels < 2:
            raise ValueError("Selection needs at least two channels.")
        if not 1 <= n_primary <= n_channels:
            raise ValueError("n_primary must lie within the channel count.")
        if not math.isfinite(dopamine) or dopamine < 0.0:
            raise ValueError("dopamine (tonic lambda) must be finite and >= 0.")
        self.p = parameters or PrescottParameters.basic()
        # The D2 slope is 1 - gain*lambda and the published transfer is
        # undefined once it goes negative, so refuse it here rather than
        # halfway through a sweep. The source warns and returns zero; refusing
        # is better, because a silent zero looks like a working model.
        if self.p.d2_gain * dopamine > 1.0:
            raise ValueError(
                f"dopamine {dopamine} drives the D2 slope negative "
                f"(gain {self.p.d2_gain}); the published transfer is undefined "
                f"above lambda = {1.0 / self.p.d2_gain:.4g}.")
        self.n = int(n_channels)
        #: The classifier scores only the PRIMARY channels. The published test
        #: runs five channels and reports on the two that carry salience.
        self.n_primary = int(n_primary)
        self.dopamine = float(dopamine)
        #: Competitions are independent, so a whole salience grid can be run at
        #: once as a leading batch dimension. The published sweep is 10201
        #: competitions per dopamine level and 61 levels; one at a time in
        #: Python that is hours, batched it is seconds. The equations are
        #: identical either way -- nothing couples one competition to another.
        self.batch = None if batch is None else int(batch)
        self._alloc()
        self.gpi_tonic = self._resting_output()

    # ------------------------------------------------------------------ state
    def _alloc(self):
        shape = (self.n,) if self.batch is None else (self.batch, self.n)
        z = lambda: np.zeros(shape)
        self.ctx, self.mot = z(), z()
        self.sd1, self.sd2 = z(), z()
        self.stn, self.gpe, self.gpi = z(), z(), z()
        self.thl, self.trn = z(), z()

    def reset(self):
        self._alloc()
        return self

    # ------------------------------------------------- the published DA ramps
    def _y_d1(self, a):
        """D1 output. Slope rises with dopamine; the ramp pivots about d1_pivot.

        y = m*(a - theta) + (1 - m)*pivot,  m = 1 + gain*lambda
        """
        m = 1.0 + self.p.d1_gain * self.dopamine
        return rectify(m * (a - self.p.threshold_striatum)
                       + (1.0 - m) * self.p.d1_pivot)

    def _y_d2(self, a):
        """D2 output. Slope FALLS with dopamine. No pivot term."""
        m = 1.0 - self.p.d2_gain * self.dopamine
        if m < 0.0:
            raise ValueError(
                f"dopamine {self.dopamine} drives the D2 slope negative; the "
                "published transfer is undefined there.")
        return rectify(m * (a - self.p.threshold_striatum))

    # ------------------------------------------------------------------ step
    def step(self, salience):
        """One Euler update. Returns the largest absolute activation change."""
        p = self.p
        s = np.asarray(salience, dtype=float)
        if s.shape[-1] != self.n:
            raise ValueError(f"salience must have {self.n} channels")

        y_ctx = rectify(self.ctx)
        y_mot = rectify(self.mot)
        y_d1, y_d2 = self._y_d1(self.sd1), self._y_d2(self.sd2)
        y_stn = rectify(self.stn - p.threshold_stn)
        y_gpe = rectify(self.gpe - p.threshold_gpe)
        y_gpi = rectify(self.gpi - p.threshold_gpi)
        # The BASIC model gives the thalamus a threshold of minus its own
        # tonic output, so it reads out as a released fraction. The EXTENDED
        # model sets that threshold to zero, because there the thalamus is a
        # real population inside a loop rather than a readout.
        thl_threshold = (-getattr(self, 'gpi_tonic', 0.0)
                         if p.thalamus_threshold_is_tonic else 0.0)
        y_thl = rectify(self.thl - thl_threshold)
        y_trn = rectify(self.trn)

        # The diffuse subthalamic projection is summed over ALL channels,
        # including the target's own. That is deliberate in the source and is
        # what makes stability depend on channel count.
        stn_total = np.sum(y_stn, axis=-1, keepdims=True)

        # Cortex relays salience. In the extended variant a motor-cortex
        # population carries the thalamocortical return alongside it, and the
        # striatum and STN see a 0.5/0.5 mix of the two.
        u_ctx = s
        u_mot = y_ctx + p.thalamus_to_motor * y_thl
        drive = p.salience_share * y_ctx + p.motor_share * y_mot

        u_sd1 = drive
        u_sd2 = drive
        u_stn = drive - p.gpe_stn * y_gpe
        u_gpe = p.stn_out * stn_total - p.sd2_gpe * y_d2
        u_gpi = p.stn_out * stn_total - p.sd1_gpi * y_d1 - p.gpe_gpi * y_gpe
        u_thl = -y_gpi + (y_mot if p.extended else 0.0)
        if p.extended:
            trn_total = np.sum(y_trn, axis=-1, keepdims=True)
            u_thl = u_thl - (p.trn_within * y_trn
                             + p.trn_between * (trn_total - y_trn))
        # The reticular nucleus is INHIBITED by the basal ganglia as well as
        # excited by cortex and thalamus. Easy to miss and load-bearing.
        u_trn = ((y_mot + y_thl - p.trn_from_gpi * y_gpi) if p.extended
                 else np.zeros_like(self.trn))

        biggest = 0.0
        for state, target in ((self.ctx, u_ctx), (self.mot, u_mot),
                              (self.sd1, u_sd1), (self.sd2, u_sd2),
                              (self.stn, u_stn), (self.gpe, u_gpe),
                              (self.gpi, u_gpi), (self.thl, u_thl),
                              (self.trn, u_trn)):
            delta = p.relax * (target - state)
            state += delta
            biggest = max(biggest, float(np.max(np.abs(delta))))
        return biggest

    def converge(self, salience, strict=False):
        """Run to the published tolerance. Returns (steps, converged).

        NOT every competition settles. The extended variant closes a positive
        feedback loop between motor cortex and thalamus at unit gain, damped
        only by the reticular nucleus, and for some salience pairs that loop
        limit-cycles instead of settling: the outputs clip while the underlying
        activations keep moving. That is a property of the published wiring,
        not of this transcription, and the honest thing is to MEASURE how often
        it happens rather than to raise, silently cap, or damp it away.

        `strict=True` raises instead, for callers that need a settled state.
        """
        for i in range(self.p.max_steps):
            if self.step(salience) < self.p.tolerance:
                return i + 1, True
        if strict:
            raise RuntimeError("model did not converge within max_steps")
        return self.p.max_steps, False

    def unsettled(self, salience, window=200):
        """Per-competition mask of what is still moving after max_steps.

        Run AFTER converge(). A competition counts as unsettled if any
        activation moves by more than the tolerance at any point in a short
        window, which catches limit cycles that a single-step check misses.
        """
        moving = np.zeros(self.gpi.shape[:-1] if self.batch else (), dtype=bool)
        for _ in range(window):
            before = self.gpi.copy()
            self.step(salience)
            moving |= np.max(np.abs(self.gpi - before), axis=-1) > self.p.tolerance
        return moving

    def _resting_output(self):
        """The output nucleus's own tonic level -- the gate's denominator.

        With no salience the striatum is silent at every dopamine level (both
        ramps sit below threshold at zero input), so this is a constant of the
        model and not of the experiment. It is computed rather than asserted.
        """
        probe = PrescottBasalGanglia.__new__(PrescottBasalGanglia)
        probe.p = self.p
        probe.n = self.n
        probe.n_primary = self.n_primary
        probe.dopamine = self.dopamine
        probe.batch = None
        probe.gpi_tonic = 0.0
        probe._alloc()
        probe.converge(np.zeros(self.n), strict=True)
        return float(rectify(probe.gpi - self.p.threshold_gpi)[0])

    # ------------------------------------------------------------- readout
    def gates(self):
        """Published disinhibition gates, e_i = L(1 - y_gpi/c)."""
        y_gpi = rectify(self.gpi - self.p.threshold_gpi)
        return np.clip(1.0 - y_gpi / self.gpi_tonic, 0.0, 1.0)

    #: Published classifier thresholds, verbatim from the source:
    #: a channel is fully selected at or above 0.95 and partially selected at
    #: or above 0.05.
    FULL = 0.95
    PARTIAL = 0.05

    def selection_class(self, gates=None):
        """The published six-way outcome label, scored over PRIMARY channels.

        none / part / cln / intf / dual / mult, exactly as the source names
        them. The published table collapses the last two into one "multiple"
        column; `sweep` does that collapsing, not this method.
        """
        g = self.gates() if gates is None else np.asarray(gates, dtype=float)
        g = g[:self.n_primary]
        full = int(np.sum(g >= self.FULL))
        part = int(np.sum((g >= self.PARTIAL) & (g < self.FULL)))
        if full == 0 and part == 0:
            return "none"
        if full == 0:
            return "part"
        if full == 1 and part == 0:
            return "cln"
        if full == 1:
            return "intf"
        if full == 2:
            return "dual"
        return "mult"

    def efficiency(self, gates=None):
        """The winner's gate value. Published `eff_mn` is the mean of this."""
        g = self.gates() if gates is None else np.asarray(gates, dtype=float)
        return float(np.max(g[:self.n_primary]))

    def distortion(self, gates=None):
        """(sum - max) / sum over the primary channels.

        NOTE, and this matters. The paper's Equation 3 carries a factor of two;
        the authors' own code, which produced the published table, does not:

            aDistortion = (sum_g==0.0)? 0.0: (sum_g - max_g)/sum_g;

        The published means top out at 0.217, which is inside the [0, 0.5]
        range of the unfactored form and would be halfway up the factored one.
        The table is what we compare against, so the code's version is what is
        implemented. The discrepancy is recorded rather than resolved silently.
        """
        g = self.gates() if gates is None else np.asarray(gates, dtype=float)
        g = g[:self.n_primary]
        total = float(np.sum(g))
        if total == 0.0:
            return 0.0
        return float((total - float(np.max(g))) / total)
