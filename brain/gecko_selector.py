"""The gecko's action selector, on the validated published model.

This is what decides what the animal does next. It is the gecko-facing wrapper
around `brain/prescott_bg.py`, which is the Prescott et al. 2024 basal ganglia
reimplemented from the authors' own source and checked against their published
selection statistics.

WHY THIS EXISTS RATHER THAN `brain/basal_ganglia.py`. That module implements
Gurney, Prescott & Redgrave 2001 with the Girard 2005 thalamocortical
extension. It is correct for that lineage and reproduces the Gurney 2001b
selection sequence to four decimal places, and it keeps its tests. But it was
being scored against a 2024 paper that uses a different dopamine mechanism, and
the animal should run on the model whose behaviour has actually been checked
against published statistics. Both are kept; the 2001 one is no longer on the
animal's path.

WHAT MIGRATING BOUGHT, concretely:

  * Persistence stops being invented. The old salience carried an INVENTED
    `persistence_gain = 0.15` fed back into its own input to make a chosen
    behaviour sticky. The extended published model supplies persistence
    STRUCTURALLY, through the motor-cortex/thalamus loop, so the invented
    constant is deleted rather than re-tuned. One fewer number in the registry
    that nothing could check.
  * The winner commits. The old model's winner topped out near 0.40 of full
    release; this one reaches 0.996 against a published 0.999. A gecko that
    half-selects three behaviours at once is not deciding.
  * Dopamine depletion produces akinesia, exactly as published: at zero tonic
    dopamine, nothing is ever selected.

HOW IT IS STEPPED. The published model has no time constant -- its integrator
is a fixed relaxation fraction -- and the authors run it TO CONVERGENCE on
every step of the behaving robot. This does the same: each control step,
salience is computed from the drives, the network is run to its fixed point,
and the gates are read. State is NOT reset between steps, which is what lets
the cortico-thalamic loop carry a running behaviour forward.

WHAT IS STILL INVENTED HERE, and stays flagged: the salience weights. Nothing
published says how a leopard gecko trades hunger against cold. What IS
published is what the selection mechanism does with those numbers once it has
them, which is what the acceptance test covers. These weights decide what the
animal wants; the model decides how firmly it commits.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np


def _load_prescott():
    """Import the model by path: `brain/__init__` pulls in torch."""
    name = "_prescott_bg_for_selector"
    if name in sys.modules:
        return sys.modules[name]
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(name, here / "prescott_bg.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pbg = _load_prescott()
PrescottBasalGanglia = _pbg.PrescottBasalGanglia
PrescottParameters = _pbg.PrescottParameters

#: The six behaviours, in order. Fixed: the gate vector is positional.
CHANNELS = ("hunt", "flee", "explore", "bask", "rest", "groom")

#: Tonic dopamine. 0.3 is the extended model's OWN default in the authors'
#: source; the basic variant defaults to 0.2 and the paper describes 0.2 as the
#: normal level. Our reimplementation sits about +0.10 along the dopamine axis
#: relative to the published figure -- an open, recorded discrepancy -- so this
#: is cited from the source for this variant rather than chosen to make
#: anything line up. Changing it to make behaviour look better would be tuning.
BASELINE_DOPAMINE = 0.3


def salience_from_drives(hunger, fatigue, cold, warm, threat=0.0,
                         prey_visible=0.0):
    """Per-channel salience, sigma-pi style, from interoception and the world.

    Sigma-pi means terms are sums of PRODUCTS: hunting is not driven by hunger
    alone but by hunger AND visible prey, which is why a sated gecko ignores a
    cricket and a starving one in an empty arena explores instead.

    The weights are **INVENTED** and flagged as such. No published salience
    weighting exists for this species.

    There is no `persistence` argument any more. The old version fed the
    previous gate vector back into its own salience through an invented gain;
    persistence now comes from the model's own cortico-thalamic loop, which is
    published structure rather than a number someone chose.
    """
    for name, value in (("hunger", hunger), ("fatigue", fatigue),
                        ("cold", cold), ("warm", warm), ("threat", threat),
                        ("prey_visible", prey_visible)):
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
    return np.clip(salience, 0.0, None)


class GeckoSelector:
    """Six behaviours competing for release. Selection is disinhibition.

    Nothing is "picked". Every behaviour is held down by default and choosing
    means letting one go, which is why a losing behaviour can be partly
    released -- something a plain take-the-largest cannot represent, and the
    reason real animals dither.
    """

    def __init__(self, dopamine=BASELINE_DOPAMINE, channels=CHANNELS):
        self.channels = tuple(channels)
        if len(self.channels) < 2:
            raise ValueError("Selection needs at least two behaviours.")
        # Every channel is scored: unlike the published disembodied test, which
        # runs five channels and reports on two, all six of these are real
        # behaviours the animal can be doing.
        self.bg = PrescottBasalGanglia(
            n_channels=len(self.channels), n_primary=len(self.channels),
            dopamine=dopamine, parameters=PrescottParameters.extended())
        self.last_salience = np.zeros(len(self.channels))
        self.settled = True

    # ---------------------------------------------------------------- state
    @property
    def dopamine(self):
        return self.bg.dopamine

    def reset(self):
        """Forget the running behaviour. Use between episodes, not between steps."""
        self.bg.reset()
        self.last_salience = np.zeros(len(self.channels))
        self.settled = True
        return self

    # ----------------------------------------------------------------- step
    def step(self, hunger=0.0, fatigue=0.0, cold=0.0, warm=0.0, threat=0.0,
             prey_visible=0.0):
        """One control step. Returns the gate vector, one per behaviour.

        The network is run to its fixed point, as the authors run it on every
        step of their behaving robot. State persists across calls, so a running
        behaviour carries forward through the cortico-thalamic loop.
        """
        salience = salience_from_drives(hunger, fatigue, cold, warm,
                                        threat, prey_visible)
        self.last_salience = salience
        _, self.settled = self.bg.converge(salience)
        return self.gates()

    def step_from_homeostasis(self, drives, threat=0.0, prey_visible=0.0):
        """Step from the hypothalamus's own 4-channel output vector.

        The hypothalamus emits [hunger, fatigue, cold, warm]; this is the seam
        between brain module 1 and brain module 2.
        """
        drives = np.asarray(drives, dtype=float)
        if drives.shape != (4,):
            raise ValueError(
                "the hypothalamus drive vector is [hunger, fatigue, cold, warm]")
        return self.step(hunger=float(drives[0]), fatigue=float(drives[1]),
                         cold=float(drives[2]), warm=float(drives[3]),
                         threat=threat, prey_visible=prey_visible)

    # -------------------------------------------------------------- readout
    def gates(self):
        return self.bg.gates()

    def selected(self, minimum_release=1e-6):
        """The behaviour that is released, or None when nothing is.

        None is a real answer, not a failure: with no salience anywhere, or
        with dopamine depleted, the animal does nothing. That is the published
        akinesia result and it must not be papered over with a fallback.
        """
        gates = self.gates()
        best = int(np.argmax(gates))
        if gates[best] <= minimum_release:
            return None
        return self.channels[best]

    def committed(self):
        """True when one behaviour is fully released and no other is partly.

        This is the published "clean selection": the animal is doing one thing.
        """
        return self.bg.selection_class() == "cln"

    def state(self):
        gates = self.gates()
        return {
            "channels": list(self.channels),
            "salience": [round(float(x), 5) for x in self.last_salience],
            "gates": [round(float(x), 5) for x in gates],
            "selected": self.selected(),
            "selection_class": self.bg.selection_class(),
            "efficiency": round(self.bg.efficiency(), 5),
            "distortion": round(self.bg.distortion(), 5),
            "dopamine": self.bg.dopamine,
            "settled": bool(self.settled),
        }
