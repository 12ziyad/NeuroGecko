"""Tests for brain/prescott_bg.py -- the published Prescott 2024 model.

These guard the things that must not drift: the published constants, the
dopamine transfer, the classifier thresholds, the self-consistency of the gate,
and the reproduction of the published Figure 5 behaviour.

The Figure 5 test gates on ORDERINGS and MAGNITUDES, not on exact percentages.
The exact sampling of the salience plane behind the published figure is not
recoverable from the archive, and there is a known constant offset along the
dopamine axis that is recorded rather than tuned away.
"""

import importlib.util
import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    """Import by path: brain/__init__ pulls in torch, which need not be present."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pbg = _load("_pbg_test", "brain/prescott_bg.py")
PrescottBasalGanglia = _pbg.PrescottBasalGanglia
PrescottParameters = _pbg.PrescottParameters


class PublishedConstants(unittest.TestCase):
    """Every one of these was read out of the authors' own source."""

    def test_the_five_nucleus_core(self):
        p = PrescottParameters.basic()
        self.assertEqual(p.stn_out, 0.9)
        self.assertEqual(p.gpe_gpi, 0.3)
        self.assertEqual(p.gpe_stn, 1.0)
        self.assertEqual(p.threshold_stn, -0.25)
        self.assertEqual(p.threshold_gpe, -0.2)
        self.assertEqual(p.threshold_gpi, -0.2)

    def test_the_two_variants_differ_exactly_where_the_source_says(self):
        b, e = PrescottParameters.basic(), PrescottParameters.extended()
        # Supplementary Methods section 5 gives the striatal threshold as 0.2
        # in the published model, quoted: y_i^d1 = L(a_i^d1, 0.2).
        self.assertEqual(b.threshold_striatum, 0.2)
        self.assertEqual(e.threshold_striatum, 0.2)
        self.assertEqual((b.da_mode, e.da_mode), ("afferent", "afferent"))
        # The shipped-but-unused "new DA" builders carry the other numbers.
        bn, en = PrescottParameters.basic_new_da(), PrescottParameters.extended_new_da()
        self.assertEqual((bn.threshold_striatum, bn.d1_pivot), (0.1, 0.5))
        self.assertEqual((en.threshold_striatum, en.d1_pivot), (0.15, 0.2))
        self.assertEqual((bn.da_mode, en.da_mode), ("slope", "slope"))
        # The basic model has NO thalamocortical persistence; the extended one
        # splits striatal and subthalamic drive 0.5/0.5 with the thalamic return.
        self.assertEqual((b.salience_share, b.motor_share, b.thalamus_to_motor),
                         (1.0, 0.0, 0.0))
        self.assertEqual((e.salience_share, e.motor_share, e.thalamus_to_motor),
                         (0.5, 0.5, 1.0))
        # The five-nucleus core is IDENTICAL between them.
        for field in ("stn_out", "gpe_gpi", "gpe_stn", "threshold_stn",
                      "threshold_gpe", "threshold_gpi", "d1_gain", "d2_gain"):
            self.assertEqual(getattr(b, field), getattr(e, field), field)

    def test_the_integrator_is_the_published_fixed_fraction(self):
        # The source sets k = 1/tau and step = 0.3*tau, so the update is
        # a += 0.3*(u - a) whatever tau is. It is a relaxation fraction, not a
        # time constant, and is recorded as one.
        self.assertEqual(PrescottParameters.basic().relax, 0.3)
        self.assertEqual(PrescottParameters.basic().tolerance, 1e-4)


class TheDopamineTransfer(unittest.TestCase):
    """Dopamine changes the striatal output SLOPE, not the input gain.

    This is the whole subject of the 2024 paper and it is NOT the 2001
    mechanism. Getting it wrong makes the dopamine sweep meaningless while
    leaving every other test green, which is why it is pinned here.
    """

    def test_dopamine_multiplies_the_input_not_the_output_function(self):
        """The published mechanism. Dopamine is "a multiplicative factor in the
        equations, specifying afferent input to the striatum" -- so the OUTPUT
        function is the same plain rectifier at every dopamine level, and the
        difference is upstream."""
        a = np.array([0.8])
        lo = PrescottBasalGanglia(dopamine=0.1)
        hi = PrescottBasalGanglia(dopamine=0.5)
        self.assertEqual(float(hi._y_d1(a)[0]), float(lo._y_d1(a)[0]))
        self.assertEqual(float(hi._y_d2(a)[0]), float(lo._y_d2(a)[0]))
        # ...and it separates the pathways where it actually acts.
        strong = PrescottBasalGanglia(dopamine=0.5)
        strong.converge(np.array([0.6, 0.0, 0.0, 0.0, 0.0]))
        weak = PrescottBasalGanglia(dopamine=0.05)
        weak.converge(np.array([0.6, 0.0, 0.0, 0.0, 0.0]))
        self.assertGreater(strong.gates()[0], weak.gates()[0])

    def test_the_slope_variant_still_works_and_is_not_the_default(self):
        """The "new DA" builders are real code in the archive. Every shipped
        program disables them, so they are kept and labelled rather than
        deleted -- a variant that exists is worth recording."""
        a = np.array([0.8])
        p = PrescottParameters.extended_new_da()
        lo = PrescottBasalGanglia(dopamine=0.1, parameters=p)
        hi = PrescottBasalGanglia(dopamine=0.5, parameters=p)
        self.assertGreater(hi._y_d1(a)[0], lo._y_d1(a)[0])
        self.assertLess(hi._y_d2(a)[0], lo._y_d2(a)[0])

    def test_the_d1_ramp_pivots_rather_than_shifting(self):
        """Above the pivot dopamine RAISES D1; below it, dopamine LOWERS it.

        A ramp that merely shifted would move the same way everywhere. This is
        what distinguishes the published transfer from a gain change.
        """
        p = PrescottParameters.extended_new_da()
        bg_lo = PrescottBasalGanglia(dopamine=0.1, parameters=p)
        bg_hi = PrescottBasalGanglia(dopamine=0.5, parameters=p)
        pivot_input = np.array([bg_lo.p.threshold_striatum + bg_lo.p.d1_pivot])
        above = pivot_input + 0.3
        # Far below the pivot both ramps rectify to zero and the comparison is
        # vacuous, so probe inside the ramp rather than under it.
        below = pivot_input - 0.05
        self.assertGreater(bg_hi._y_d1(above)[0], bg_lo._y_d1(above)[0])
        self.assertLess(bg_hi._y_d1(below)[0], bg_lo._y_d1(below)[0])
        # and the ramps cross AT the pivot
        self.assertAlmostEqual(float(bg_hi._y_d1(pivot_input)[0]),
                               float(bg_lo._y_d1(pivot_input)[0]), places=9)

    def test_dopamine_outside_the_published_range_is_refused(self):
        """The paper's range is 0 <= lambda <= 1; above 1 the D2 afferent gain
        goes negative. Refused at construction, not halfway through a sweep."""
        with self.assertRaises(ValueError):
            PrescottBasalGanglia(dopamine=1.5)
        PrescottBasalGanglia(dopamine=1.0)              # exactly at the limit
        # the slope variant has its own, different limit
        p = PrescottParameters.extended_new_da()
        with self.assertRaises(ValueError):
            PrescottBasalGanglia(dopamine=2.0, parameters=p)

    def test_the_striatum_is_silent_at_rest_at_every_dopamine_level(self):
        """Which is why the resting output, and so the gate constant, does not
        depend on dopamine. If this broke, the gate's denominator would move
        under it."""
        for da in (0.0, 0.2, 0.4, 0.6):
            bg = PrescottBasalGanglia(dopamine=da)
            self.assertEqual(float(bg._y_d1(np.zeros(1))[0]), 0.0)
            self.assertEqual(float(bg._y_d2(np.zeros(1))[0]), 0.0)


class TheGate(unittest.TestCase):
    def test_the_gate_closes_exactly_at_rest(self):
        """c is the model's OWN resting output, so this is definitional and
        must hold to machine precision -- not approximately."""
        bg = PrescottBasalGanglia(dopamine=0.2)
        bg.converge(np.zeros(bg.n))
        np.testing.assert_allclose(bg.gates(), 0.0, atol=1e-9)

    def test_the_resting_output_is_computed_not_asserted(self):
        # Close to the published 0.169, and derived rather than hard-coded.
        bg = PrescottBasalGanglia(dopamine=0.2)
        self.assertAlmostEqual(bg.gpi_tonic, 0.169, places=2)

    def test_batching_is_identical_to_one_at_a_time(self):
        saliences = [[0.2, 0.0], [0.6, 0.3], [1.0, 0.9]]
        one = []
        for s in saliences:
            bg = PrescottBasalGanglia(dopamine=0.25)
            sal = np.zeros(bg.n)
            sal[:2] = s
            bg.converge(sal)
            one.append(bg.gates())
        many = PrescottBasalGanglia(dopamine=0.25, batch=len(saliences))
        sal = np.zeros((len(saliences), many.n))
        sal[:, :2] = saliences
        many.converge(sal)
        np.testing.assert_allclose(np.array(one), many.gates(), atol=1e-9)


class TheClassifier(unittest.TestCase):
    def test_the_published_thresholds_are_verbatim(self):
        self.assertEqual(PrescottBasalGanglia.FULL, 0.95)
        self.assertEqual(PrescottBasalGanglia.PARTIAL, 0.05)

    def test_every_class_is_reachable_and_correctly_labelled(self):
        bg = PrescottBasalGanglia()
        cases = {
            (0.00, 0.00): "none",
            (0.50, 0.00): "part",
            (1.00, 0.00): "cln",
            (1.00, 0.50): "intf",
            (1.00, 1.00): "dual",
        }
        for gates, expected in cases.items():
            g = np.array(gates + (0.0, 0.0, 0.0))
            self.assertEqual(bg.selection_class(g), expected, gates)

    def test_only_the_primary_channels_are_scored(self):
        """The published test runs five channels and reports on two."""
        bg = PrescottBasalGanglia(n_channels=5, n_primary=2)
        g = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(bg.selection_class(g), "cln")

    def test_distortion_follows_the_code_not_the_paper_text(self):
        """The paper's Equation 3 carries a factor of two; the source that
        produced the published table does not. The table is what we compare
        against, so the unfactored form is implemented -- and the discrepancy
        is recorded rather than resolved silently."""
        bg = PrescottBasalGanglia()
        g = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(bg.distortion(g), 0.5, places=9)
        self.assertAlmostEqual(bg.efficiency(g), 1.0, places=9)
        self.assertEqual(bg.distortion(np.zeros(5)), 0.0)


class ReproducesFigure5(unittest.TestCase):
    """The acceptance test. Coarse grid so it runs with the rest of the suite;
    tools/selection_sweep.py runs it properly and writes the evidence."""

    GRID = 21

    @classmethod
    def setUpClass(cls):
        axis = np.linspace(0.0, 1.0, cls.GRID)
        cls.grid = np.stack(np.meshgrid(axis, axis, indexing="ij"),
                            axis=-1).reshape(-1, 2)

    def _classes(self, dopamine):
        sal = np.zeros((self.grid.shape[0], 5))
        sal[:, :2] = self.grid
        bg = PrescottBasalGanglia(dopamine=dopamine,
                                  parameters=PrescottParameters.extended(),
                                  batch=self.grid.shape[0])
        bg.converge(sal)
        g = bg.gates()[:, :2]
        full = np.sum(g >= bg.FULL, axis=1)
        part = np.sum((g >= bg.PARTIAL) & (g < bg.FULL), axis=1)
        n = float(len(full))
        return {
            "none": 100.0 * np.sum((full == 0) & (part == 0)) / n,
            "part": 100.0 * np.sum((full == 0) & (part > 0)) / n,
            "cln": 100.0 * np.sum((full == 1) & (part == 0)) / n,
            "multi": 100.0 * np.sum(full >= 2) / n,
            "eff": float(np.mean(np.max(g, axis=1))),
        }

    def test_no_dopamine_means_no_selection_at_all(self):
        """Published: 100% no selection at lambda 0. The akinesia result."""
        self.assertAlmostEqual(self._classes(0.0)["none"], 100.0, places=6)

    def test_selection_appears_and_then_multiplies_as_dopamine_rises(self):
        low, mid, high = (self._classes(d) for d in (0.10, 0.35, 0.60))
        self.assertEqual(low["cln"], 0.0)          # published: none until 0.12
        self.assertGreater(mid["cln"], 60.0)       # published peak ~78.6
        self.assertGreater(high["multi"], mid["multi"])

    def test_the_winner_saturates(self):
        """The fault this module was built to diagnose. Our previous model
        peaked near 0.40; the published mean winner efficiency reaches 1.0."""
        self.assertGreater(self._classes(0.40)["eff"], 0.90)

    def test_the_peak_clean_selection_matches_the_published_magnitude(self):
        peak = max(self._classes(d)["cln"] for d in np.arange(0.2, 0.65, 0.05))
        self.assertGreater(peak, 70.0)             # published 78.61
        self.assertLess(peak, 90.0)


class TheKnownOffsetIsRecorded(unittest.TestCase):
    """The one thing that does NOT reproduce, pinned so it cannot be quietly
    forgotten or quietly tuned away."""

    def test_the_module_documents_the_offset(self):
        text = (REPO / "tools/selection_sweep.py").read_text(encoding="utf-8")
        self.assertIn("OFFSET", text.upper())

    def test_the_published_table_is_vendored_with_attribution(self):
        csv = REPO / "artifacts/published/prescott2024_figure5.csv"
        self.assertTrue(csv.is_file(), "the published comparison must be in-repo")
        header = csv.read_text(encoding="utf-8").splitlines()[0]
        for column in ("DA", "none", "part", "cln", "dist", "mltp"):
            self.assertIn(column, header)


if __name__ == "__main__":
    unittest.main()
