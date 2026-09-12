"""The seam between a chosen behaviour and a body that can do more than one thing.

WHY THIS FILE EXISTS. Until now `envs/gecko_brain_env.py` ran the basal ganglia,
read its choice, and put it in the info dict next to a field that said so:

    info["behaviour_controls_nothing"] = True

That was the right call when it was written, and the env says why: "Hooking a
six-way chooser to a one-behaviour body would look like integration and mean
nothing." There was one behaviour -- walk at the target -- so a selector could
not select.

THAT REASON HAS EXPIRED. The body can now do several distinguishable things:

    search      brain/search.py: walk a leg, stop, saccade the head, look
    chase       steer on a bearing the eye committed to
    hold still  envs/gecko_walk_env.py locomotor_drive, which did not exist
                before session 12 -- the animal had no off switch (#253)

Three motor programs is enough for selection to mean something, so the wire gets
built. This module is that wire and nothing else: a behaviour NAME in, a motor
command out.

WHAT IT DOES NOT DO. It does not decide. The choosing is `brain/gecko_selector.py`
running the Prescott 2024 basal ganglia, and the drive-to-effort map is
`brain/brainstem.py`. This file owns no competition and no scoring; if it ever
starts preferring one behaviour to another, the architecture has been broken.

PROVENANCE. Every number in the mapping below comes from a module that already
declares its own provenance -- `brainstem.LOCOMOTOR`, `brainstem.DRIVE_THRESHOLD`,
`search.SearchPattern` -- and this file adds exactly one of its own, declared
here. It is deliberately thin.
"""

from __future__ import annotations

import math

__all__ = ["MotorPrograms", "PROGRAMS"]

#: Which motor program each of the selector's six channels runs. The channel
#: names are `brain/gecko_selector.py`'s CHANNELS and are not chosen here.
PROGRAMS = {
    "hunt": "chase",
    "explore": "search",
    "flee": "shelter",
    "bask": "warm",
    "rest": "still",
    "groom": "still",
}

#: The only number this file introduces. When the eye's evidence for a target
#: expires mid-chase the animal does not stand there: it resumes searching from
#: where it was looking. INVENTED -- this is a controller-continuity choice, not
#: a claim about an animal.
RESUME_SEARCH_ON_LOST = True

#: STALKING: how long the animal walks toward a committed bearing before it
#: stops and re-checks, in control steps, and how long it then looks.
#:
#: WHY A CHASE HAS TO PAUSE AT ALL, measured (#270). Committing sets the
#: selector's `prey_visible` to 1, and the explore channel's salience is
#: 0.35 * hunger * (1 - prey_visible) -- so a committed animal's drive to look
#: around is EXACTLY ZERO. It cannot choose to check. Without a pause built into
#: the chase itself it walks on a bearing it can no longer verify: measured at
#: 1331 of 1500 steps chasing, prey on screen 0 of 1500.
#:
#: The RATIO has same-family support, and that support is UNVERIFIED and says
#: so. Werner et al. on wild *Goniurosaurus kuroiwae* (Eublepharidae, nocturnal,
#: terrestrial -- the closest foraging budget that exists for this family) is
#: reported as mean move 12.3 s against mean pause 301 s, roughly 15-23 % of
#: time in active locomotion. A literature sweep could reach only a
#: machine-reconstructed abstract, never the tabulated values, so these numbers
#: are recorded as a LEAD and not as a citation this project has opened. The
#: paper's own reachable abstract gives a range of 0-84 % of time moving, which
#: is wide enough that the specific 15 % must not be leaned on.
#:
#: WHAT IS SAFE TO SAY: a wild eyelid gecko moves in short bursts and pauses for
#: considerably longer than it moves. That is the shape used here. The numbers
#: below are INVENTED.
#:
#: The ABSOLUTE durations below are COMPRESSED and that is declared, exactly as
#: `homeostasis_time_compression` is declared: a 301 s pause is 15050 control
#: steps and no rollout in this project is that long. What is preserved is the
#: shape -- move briefly, stop, look longer than you moved.
STALK_MOVE_STEPS = 40
STALK_LOOK_STEPS = 120

#: How long the head stays on a candidate after the post-saccade blind window
#: closes, before the scan is allowed to take the head back. DERIVED, not
#: chosen: it is `FixationEvidence`'s own decision window, because the point of
#: holding is to give the accumulator a full look at the thing that was
#: salient. Anything shorter throws away the look the saccade paid for.
ORIENT_HOLD_STEPS = 50


class MotorPrograms:
    """Behaviour name in; heading, head yaw and locomotor drive out.

    Consumes only things the animal has: the behaviour its own basal ganglia
    released, the gate value it was released with, its own trunk angle from
    proprioception, and whatever its eye reported. It never reads the prey's
    position.
    """

    def __init__(self, search, evidence, brainstem=None, orienting=None):
        from brain.search import OrientingReflex
        self._orient = orienting if orienting is not None else OrientingReflex()
        self.search = search
        self.evidence = evidence
        self.brainstem = brainstem
        self.reset()

    def reset(self):
        self.last = {}
        self._program = None
        self._stalk_t = 0
        self._orient.reset()
        if self.search is not None:
            self.search.reset()
        if self.evidence is not None:
            self.evidence.reset()
        return self

    # ------------------------------------------------------------------ drive
    def _drive_for(self, behaviour, urgency):
        """Effort in [0, 1]. Delegated to the brainstem when one is attached.

        `brain/brainstem.py` already owns this map -- LOCOMOTOR gives hunt and
        flee 1.0, explore 0.55, bask 0.35, and everything else 0.0, with a
        DRIVE_THRESHOLD below which the animal does not step at all. Those
        numbers are INVENTED and that module says so; they are not restated
        here, because a second copy is a second thing to get out of step.
        """
        from brain.brainstem import DRIVE_THRESHOLD, LOCOMOTOR
        if self.brainstem is not None:
            drive = self.brainstem.drive_for(behaviour, urgency)
        else:
            drive = float(LOCOMOTOR.get(behaviour, 0.0)
                          * min(max(float(urgency), 0.0), 1.0))
        # Below threshold the animal stands. This is the same threshold the
        # brainstem uses to decide there is no stride frequency; here it
        # decides there is no locomotion, which is the same statement made to a
        # body that has an off switch.
        return 0.0 if drive <= DRIVE_THRESHOLD else float(drive)

    # ------------------------------------------------------------------- step
    def step(self, behaviour, urgency=1.0, *, bearing_deg=None,
             believed=None, trunk_yaw_deg=0.0, shelter_bearing_deg=None,
             warm_bearing_deg=None, on_warm_ground=False):
        """One control step.

        Returns heading_deg (body-relative), head_yaw_deg, locomotor_drive and
        the program that produced them. Consumes only what the animal has: its
        own released behaviour, its own trunk angle, its own eye's bearing.
        """
        program = PROGRAMS.get(behaviour, "still")
        drive = self._drive_for(behaviour, urgency)
        heading_deg = 0.0
        scan_head_deg = self._orient.head_deg
        ran_search = False
        if program != "chase":
            self._stalk_t = 0

        # ---- 1. WHERE THE BODY IS GOING ------------------------------------
        if program in ("shelter", "warm"):
            known = (shelter_bearing_deg if program == "shelter"
                     else warm_bearing_deg)
            if known is None:
                program = "search"          # nothing remembered; brain 7 absent
            else:
                heading_deg = float(known)
            # A BASKING ANIMAL LIES DOWN. Walking toward warm ground and walking
            # ONCE YOU ARE ON IT are not the same act: the second one walks you
            # off the other side. Measured before this was added -- the animal
            # closed from 34.0 cm to 3.6 cm, warmed from 25.2 to 28.9 C, then
            # carried on over the far edge and cooled again, cycling forever
            # without ever reaching its preferred band. This species takes heat
            # by LYING on the ground (thigmothermy, r2 = 0.97), so stopping is
            # not a convenience, it is the behaviour.
            if program == "warm" and on_warm_ground:
                drive = 0.0

        committed_prior = (self.evidence.last.get("committed_bearing_deg")
                           if self.evidence is not None else None)
        if program == "chase" and committed_prior is None:
            program = "search" if RESUME_SEARCH_ON_LOST else "still"

        if program == "chase":
            # The committed bearing is HEAD-relative; the walker's heading is
            # TRUNK-relative. The head is no longer snapped back to centre, so
            # the two differ by wherever the head now points, and the sum is the
            # trunk-relative direction. Same frame discipline as #242.
            heading_deg = self._orient.head_deg + float(committed_prior)
            cycle = STALK_MOVE_STEPS + STALK_LOOK_STEPS
            if (self._stalk_t % cycle) >= STALK_MOVE_STEPS:
                drive = 0.0                 # the looking half of the stalk
            self._stalk_t += 1
        elif program == "search" and self.search is not None:
            heading_deg, scan_head_deg, moving = self.search.step(trunk_yaw_deg)
            ran_search = True
            if not moving:
                drive = 0.0
        elif program == "still":
            drive = 0.0

        body_still = drive <= 0.0

        # ---- 2. WHERE THE HEAD IS LOOKING ----------------------------------
        # THE EYE NOW MOVES THE HEAD (#280). Until this, the head swept a fixed
        # pattern that ignored the eye entirely and was snapped to centre during
        # a chase, so the animal walked at a target while looking away from it.
        # A detector with no orienting reflex attached is half a tectum.
        #
        # Ownership: the reflex owns the head while it is examining a candidate;
        # when idle the scan owns it and the reflex is re-seeded to wherever the
        # scan put it, so its internal angle never drifts from the real one.
        orient_busy = (self._orient.last.get("steps_since_saccade", 10 ** 6)
                       <= self._orient.move_steps + self._orient.blind_steps
                       + ORIENT_HOLD_STEPS)
        if ran_search and not orient_busy:
            self._orient.head_deg = float(scan_head_deg)

        # ORIENT ONLY TO WHAT HAS ALREADY BEEN DECIDED REAL (#282).
        #
        # The first two versions let the reflex fire on any single-frame report
        # and it made the animal much worse -- 3 seeds x 3000 steps went from
        # 144/158/96 decisions down to 1/19/0, because the visual field is
        # +-35 deg, the false-alarm bearings are spread across it, and every
        # noise report bought a saccade plus the measured 78-step blind window.
        # It is the lesson of #245 and #271 again in a new place: a single frame
        # is not evidence, and anything that ACTS on one inherits its noise.
        #
        # A tectum orients to what it has committed to. During a search the
        # scripted scan is a COVERAGE pattern and is right to ignore the eye --
        # there is nothing yet worth looking at. During a chase there is, and
        # keeping it on the axis while the body walks is the whole point.
        track = (float(committed_prior)
                 if (program == "chase" and committed_prior is not None
                     and body_still)
                 else None)
        head_yaw_deg, orient_ok = self._orient.step(track, allow=body_still)

        # ---- 3. IS THE EYE WORTH BELIEVING THIS STEP? ----------------------
        # Three measured conditions, all of which must hold:
        #   body still      standing more than doubles the signal (#257, #259)
        #   head settled    past its measured post-saccade blind window (#279)
        #   scan agrees     when the scan is what is driving the head
        if believed is not None:
            looking = bool(believed)
        else:
            looking = bool(body_still and orient_ok
                           and (not ran_search
                                or self.search.last.get("looking")))

        committed = None
        if self.evidence is not None:
            committed = self.evidence.step(
                float(bearing_deg) if (bearing_deg is not None and looking)
                else None, looking)
            if program == "chase" and committed is not None:
                heading_deg = self._orient.head_deg + float(committed)

        self._program = program
        self.last = {
            "behaviour": behaviour, "program": program,
            "urgency": float(urgency),
            "heading_deg": float(heading_deg),
            "head_yaw_deg": float(head_yaw_deg),
            "locomotor_drive": float(drive),
            "committed_bearing_deg": committed,
            "believed_eye": bool(looking),
            "saccades": self._orient.saccades,
            "orienting": bool(orient_busy),
        }
        return dict(self.last)

    def provenance(self):
        return {"module": "MotorPrograms",
                "invented": ["RESUME_SEARCH_ON_LOST", "STALK_MOVE_STEPS",
                             "STALK_LOOK_STEPS (durations compressed; the "
                             "move<pause SHAPE is published, same-family, "
                             "Werner et al. on Goniurosaurus kuroiwae)"],
                "delegated": ["brainstem.LOCOMOTOR", "brainstem.DRIVE_THRESHOLD",
                              "search.SearchPattern.*", "search.FixationEvidence.*"],
                "published": [],
                "note": "This file is a wire, not a model. It introduces one "
                        "controller-continuity choice and delegates every "
                        "number to a module that declares its own provenance. "
                        "It deliberately does no scoring: the choosing belongs "
                        "to brain/gecko_selector.py and nowhere else."}
