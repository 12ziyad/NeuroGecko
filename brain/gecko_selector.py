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

#: Tonic dopamine at rest. The paper describes 0.2 as the normal level, and
#: with the published dopamine mechanism our sweep now peaks at exactly the
#: published 0.22 with the axis fit at slope 1.000, so 0.2 here means what it
#: means in the paper. It was 0.3 while the reimplementation still carried a
#: dopamine-axis offset; that offset is resolved and the value moves back.
BASELINE_DOPAMINE = 0.2

#: Groom's tonic salience. **Now 0.0, and it was 0.05** (#353).
#:
#: The 0.05 was a bare literal here and in `brain/basal_ganglia.py`, with no
#: entry in `config/proxies.yaml` at any confidence, no citation, and no
#: species. It is the only number in the salience vector that was never even
#: collectively declared invented, because the docstring above declares the
#: WEIGHTS and 0.05 is not a weight on anything -- it is a floor under a
#: channel with no drive behind it.
#:
#: It was not inert. Measured by bisection on this model: the release threshold
#: on any channel with every other channel at zero is 0.19988, and with groom's
#: 0.05 sitting in the vector it is 0.20109. So an untagged invented constant
#: was raising the bar for every OTHER behaviour, including `flee`, whose
#: salience is a published probability. Removing it widens flee's margin over
#: the floor from 0.0089 to 0.0101 without touching a published number.
#:
#: WHAT THIS DOES NOT DO. It does not make groom reachable -- it makes its
#: unreachability honest. Grooming in this species has a real published
#: trigger, post-feeding labial licking at 0.165 licks/s (Cooper, DePerno &
#: Steele 1996, n = 16, `post_feeding_labial_lick_rate_per_s`), and that
#: behaviour ALREADY RUNS in `envs/gecko_brain_env.py` as a reflex on a window
#: after eating, entirely outside this selector. Whether the channel should
#: carry it is open, and is blocked on a contradiction inside this repository
#: about whether eye-licking is documented for any eublepharid (#354).
GROOM_TONIC = 0.0


def salience_from_drives(hunger, fatigue, cold, warm, threat=0.0,
                         prey_visible=0.0, arousal=1.0):
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

    REST IS THE CLOCK, NOT FATIGUE (#349, #350, #352). It used to be `fatigue`,
    and that was the wrong quantity twice over. Measured here, five seeds x
    1500 autonomous steps on the accepted walker: fatigue's resting value
    tracks the FRACTION OF CONTROL STEPS whose instantaneous trunk displacement
    crosses the 0.100 m/s aerobic ceiling at r = +0.9998, and tracks the
    animal's actual mean speed at only +0.71. `Physiology.endurance_s` returns
    infinity at or below that ceiling and a finite time above it, so it is a
    cliff and not a curve: at the walking speed fatigue decays to exactly zero,
    and the small non-zero number the record carried was the residue of
    per-step velocity noise, whose size depends on the seed and the run length
    rather than on the animal. A strolling gecko does not tire -- that part of
    #284 was right -- so a channel driven by its tiredness can never open.

    What a leopard gecko actually rests FOR is the light phase. That is the one
    thing about this animal's arousal anybody has published: activity onset
    81 min after lights-out (Digirolamo 2024, n = 1) and an evening peak window
    (Kronke & Xu 2023, n = 18, captive). Both are thin and both say so in
    `config/proxies.yaml`. So rest reads `brain/arousal.py` -- brain 6, which
    until now was built, tested and connected to nothing (#262, #347).

    The MAPPING `1 - arousal` is **INVENTED** and introduces no constant of its
    own, which is the only reason to prefer it: resting is not-being-active,
    and the clock already owns the shape of being active. The default of 1.0
    means fully awake, so a caller that does not run a clock gets a rest
    salience of zero and nothing else in the vector moves.

    `fatigue` stays in the signature and in brain 1's drive vector. It is a
    real measured drive and it is now consumed by nothing in the selector,
    which is recorded rather than hidden -- see #351.
    """
    for name, value in (("hunger", hunger), ("fatigue", fatigue),
                        ("cold", cold), ("warm", warm), ("threat", threat),
                        ("prey_visible", prey_visible), ("arousal", arousal)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    salience = np.array([
        hunger * prey_visible,                    # hunt: wants food AND sees some
        threat,                                   # flee
        0.35 * hunger * (1.0 - prey_visible),     # explore: hungry, nothing in sight
        max(cold, warm),                          # bask / thermoregulate
        1.0 - float(np.clip(arousal, 0.0, 1.0)),  # rest: the day/night clock
        GROOM_TONIC,                              # groom: 0.0, and it was 0.05
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
             prey_visible=0.0, arousal=1.0):
        """One control step. Returns the gate vector, one per behaviour.

        The network is run to its fixed point, as the authors run it on every
        step of their behaving robot. State persists across calls, so a running
        behaviour carries forward through the cortico-thalamic loop.

        `arousal` is brain 6's output and defaults to 1.0 -- fully awake -- so
        a caller that runs no clock gets exactly the vector it got before,
        except that rest now reads 0.0 instead of a fatigue residue that was
        never above 0.003 on any measured run.
        """
        salience = salience_from_drives(hunger, fatigue, cold, warm,
                                        threat, prey_visible, arousal)
        self.last_salience = salience
        _, self.settled = self.bg.converge(salience)
        return self.gates()

    def step_from_homeostasis(self, drives, threat=0.0, prey_visible=0.0,
                              arousal=1.0):
        """Step from the hypothalamus's own 4-channel output vector.

        The hypothalamus emits [hunger, fatigue, cold, warm]; this is the seam
        between brain module 1 and brain module 2. `arousal` is the separate
        seam from brain 6, which is not part of the drive vector because the
        clock is not a homeostatic drive -- it does not have a set point.
        """
        drives = np.asarray(drives, dtype=float)
        if drives.shape != (4,):
            raise ValueError(
                "the hypothalamus drive vector is [hunger, fatigue, cold, warm]")
        return self.step(hunger=float(drives[0]), fatigue=float(drives[1]),
                         cold=float(drives[2]), warm=float(drives[3]),
                         threat=threat, prey_visible=prey_visible,
                         arousal=arousal)

    # -------------------------------------------------------------- readout
    def gates(self):
        return self.bg.gates()

    def selected(self, minimum_release=None):
        """The behaviour that is released, or None when nothing is.

        None is a real answer, not a failure: with no salience anywhere, or
        with dopamine depleted, the animal does nothing. That is the published
        akinesia result and it must not be papered over with a fallback.

        THE THRESHOLD IS THE AUTHORS' OWN, AND IT USED TO BE 1e-6 (#357).
        `PrescottBasalGanglia.PARTIAL` is 0.05 verbatim from the source: the
        gate at which the published classifier counts a channel as selected at
        all. This method used an invented epsilon instead, and on an EXACTLY
        zero salience vector that was wrong in a way nothing could see.

        Measured: with every salience at zero the extended model does not
        settle to zero gates. It settles to a UNIFORM 0.000402 on all six
        channels -- a trace of every behaviour released at once -- and
        `argmax` on six identical numbers returns index 0, so the animal
        "chose" `hunt` because hunt is declared first. The model itself was
        never fooled: `selection_class()` returned "none" throughout. Only this
        readout disagreed with it.

        That defect was invisible until now because groom's invented 0.05 tonic
        (see GROOM_TONIC) kept the vector from ever being exactly zero -- any
        single channel at or above about 0.01 drives the network to the all-zero
        fixed point. So the test asserting that a contented gecko does nothing
        was passing because of an untagged constant, which is precisely what its
        own docstring warns against.

        `minimum_release` is kept as an override for callers that want a
        different cut, and defaults to the published value.
        """
        threshold = (self.bg.PARTIAL if minimum_release is None
                     else float(minimum_release))
        gates = self.gates()
        best = int(np.argmax(gates))
        if gates[best] < threshold:
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
