"""Brain module 4d: looking for prey, and deciding there is prey there.

WHY THIS EXISTS, MEASURED. With the prey-position oracle disconnected
(tools/oracle_free_diagnosis.py, 600 steps, seed 2):

    prey on the image                 0 / 600
    prey BEHIND the camera          581 / 600
    prey outside the horizontal FOV  19 / 600
    prey outside the vertical FOV     0 / 600
    median bearing to prey           82 deg off the optical axis
    median range                    361 mm

The animal spends 97 % of its life facing away from its food. Not mis-seeing
it -- facing away from it. Every hour this project spent on the retina was
spent on the 3 % of frames where seeing was even possible in principle. The
binding constraint was never sharpness. It was that NOTHING WAS LOOKING.

THE FIRST VERSION OF THIS FILE MADE IT WORSE, and the reason is recorded here
rather than quietly fixed (#244). The walker's heading command is BODY
RELATIVE -- `envs/gecko_brain_env.py::_brain_action_to_target` rotates it by
the trunk matrix. So a constant heading makes the animal circle, and a heading
that alternates +35/-35 makes it wiggle down a straight line forever. It can
never turn around. That is the 581.

WHAT CHANGED. Two things the animal already had and had never used:

  HEAD YAW. The morphology carries neck_yaw at +-40 deg and head_yaw at +-30.
  Seventy degrees of look-around, actuated in the XML since the first session,
  commanded by nothing until now. A stationary scan sweeps the head; the body
  does not have to turn to inspect a sector.

  TRUNK YAW AS FEEDBACK. A turn ends when the animal HAS TURNED, read from its
  own trunk angle, instead of after a step count that was never calibrated. A
  gecko knows which way its body points; that is not privileged information.

THE SHAPE IS PUBLISHED, THE NUMBERS ARE NOT. #214 measured this eye firing on
9.3 % of frames standing still against 5.8 % moving -- walking is what blinds
it, so the scan is stationary. Minton 1966 (Bull AMNH 134(2):73) on wild
*Eublepharis macularius*: prey is "slowly stalked or ambushed". An animal that
stops to look is what the field record describes and what this eye measures as
better. No gecko scan amplitude, rate, dwell or turn angle has ever been
published. NOT ONE NUMBER IN THIS FILE COMES FROM A REAL ANIMAL. Most are
INVENTED; SETTLE_STEPS is DERIVED from a measurement of the simulated body;
SCAN_HALF_WIDTH_DEG is set by the morphology's joint ranges, which are
themselves this project's own untagged modelling choices -- no range-of-motion
study exists for any joint of this species except the hip. Each constant's own
tag says which (audited, #264).
"""

from __future__ import annotations

import math

__all__ = ["FixationEvidence", "OrientingReflex", "PreyEvidence",
           "SearchPattern"]


def _wrap(deg):
    """Fold an angle into (-180, 180]."""
    return (float(deg) + 180.0) % 360.0 - 180.0


class SearchPattern:
    """Stop and sweep the head, turn the body, walk, repeat.

    Emits a command each control step. Reads only the animal's own trunk angle,
    which it has from proprioception, and nothing about the prey.

    The cycle is SCAN -> TURN -> TRAVEL -> SCAN. Three scans separated by two
    120 deg turns inspect the whole circle, because a scan covers the head's
    +-65 deg sweep plus the camera's own 35 deg half-angle -- 200 deg of world
    per scan, overlapping on purpose.
    """

    #: Degrees the head jumps between fixations. INVENTED, but bounded above
    #: by the camera's 70 deg field: a step wider than the field leaves a gap
    #: the animal never looks at.
    SACCADE_DEG = 22.0
    #: Control steps before the dwell timer starts after commanding a new head
    #: position. NOT the duration of the movement, and conflating the two made
    #: the animal much worse (#282).
    #:
    #: The neck's real travel time IS measured and IS 17-18 steps (#278) -- the
    #: servo reaches 90 % of any commanded step in that time whatever its size.
    #: Setting this to 18 on that basis was wrong, because `SETTLE_STEPS` = 70
    #: already waits out the travel AND the post-saccade blind window before the
    #: eye is believed. Raising this simply spent 15 extra steps per fixation
    #: and bought fewer looks per rollout. Measured, 3 seeds x 3000 autonomous
    #: steps, everything else identical:
    #:
    #:     value 3    closest 34.9 / 147.3 / 156.2 mm, decisions right
    #:                177 of 224, 0 of 80, 95 of 95
    #:     value 18   closest 263.4 / 174.6 / 150.4 mm, decisions right
    #:                0 of 0, 0 of 93, 96 of 96
    #:
    #: A correct number applied to the wrong quantity is still wrong, and this
    #: one is recorded rather than quietly reverted.
    SACCADE_STEPS = 3
    #: Control steps the head holds still at each fixation. INVENTED, but the
    #: VALUE is derived rather than picked (#250). Evidence for a real target
    #: accumulates linearly with dwell while the quorum needed to reject noise
    #: grows only as a Poisson tail, so a longer look is strictly better until
    #: the cost of looking elsewhere bites. At the rates measured here -- 21.4 %
    #: recall, 19 % false alarms per frame, +-6 deg agreement:
    #:
    #:     dwell  quorum  expected true reports  P(detect a look with prey)
    #:        18       5                    3.9                      34.2 %
    #:        30       6                    6.4                      61.9 %
    #:        50       7                   10.7                      90.8 %
    #:       100      11                   21.4                      99.5 %
    #:
    #: The look itself is 50 steps, 1.0 s at 50 Hz, the knee of that curve; the
    #: constant is 120 because the first SETTLE_STEPS of every fixation are not
    #: believed. A stop-and-stare of about a second is what "slowly stalked"
    #: (Minton 1966) looks like. NO PREY-SEARCH FIXATION DURATION HAS BEEN
    #: PUBLISHED FOR THIS SPECIES. One fixation statistic HAS: binocular
    #: fixation bouts in *E. macularius* facing a snake predator, recorded in the
    #: project's scorecard. It is antipredator staring, not search, and it is
    #: deliberately not used here as if it were.
    FIXATE_STEPS = 120
    #: Steps at the start of a fixation during which the eye is NOT believed,
    #: because the body is still swaying from having been walking. DERIVED
    #: (#258): after the gait is switched off, the head's angular speed decays
    #: 27.9 -> 17.5 -> 7.8 -> 3.5 -> 0.86 -> 0.13 -> 0.04 deg/s over the first
    #: seventy steps, and the eye's firing rate only settles to its floor of
    #: 14-19 % once it gets there. Before that the animal is reading its own
    #: wobble. An ambush predator that stops and waits is not being patient for
    #: its own sake.
    SETTLE_STEPS = 70
    #: Head sweep half-width, degrees. Set by the MORPHOLOGY: neck_yaw 40 +
    #: head_yaw 30 is the travel the joints have, less a margin off the stops.
    #: That is one step removed from invented, not zero: the joint ranges in
    #: morphology/gecko_body_lab_v2.xml carry no provenance tag, and the
    #: project's scorecard records that no range-of-motion study exists for
    #: this species on any joint but the hip (#264). "Seventy degrees of head
    #: yaw" is the sum of two modelling choices, not a measurement.
    SCAN_HALF_WIDTH_DEG = 65.0
    #: How far the body turns between scans, degrees. INVENTED, but chosen so
    #: that three scans overlap rather than leave a blind sector.
    TURN_DEG = 120.0
    #: Body-relative heading commanded while turning. INVENTED.
    TURN_COMMAND_DEG = 80.0
    #: Give up on a turn after this many steps, so a wedged animal still moves
    #: on instead of spinning in place for the whole episode. INVENTED.
    TURN_MAX_STEPS = 220
    #: Control steps spent walking forward between scan sites. INVENTED.
    TRAVEL_STEPS = 120
    #: Scans completed before the animal relocates instead of turning again.
    SCANS_PER_SITE = 3

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "scan"
        self._t = 0
        self._head = 0.0            # head yaw command, degrees
        self._sign = 1.0
        self._scans = 0
        self._fixating = True       # False while a saccade is in flight
        self._fix_target = 0.0
        self._fix_index = 0
        self._turn_from = None      # trunk yaw when the turn began, degrees
        self.last = {}
        return self

    def step(self, trunk_yaw_deg=0.0):
        """One control step.

        Returns (heading_deg, head_yaw_deg, moving):
          heading_deg   body-relative direction for the walker
          head_yaw_deg  where to point the head, relative to the trunk
          moving        False while scanning, so the walker stays put
        """
        self._t += 1
        trunk = float(trunk_yaw_deg)

        if self.state == "scan":
            # SACCADE AND FIXATE, not sweep. Measured, and this is the whole
            # reason the file changed shape (#246): a head sweeping smoothly at
            # 80 deg/s put 27.2 % of frames into false alarm, because the
            # camera rides on the head and everything in the world slides.
            # Subtracting that motion in the efference copy fixed the false
            # alarms -- 3.8 % -- and destroyed the detector with them, dropping
            # recall from 52.6 % to 3.4 %. Over-subtraction, the same failure
            # as #231.
            #
            # A head that is either MOVING FAST or COMPLETELY STILL has neither
            # problem. During a fixation the head yaw rate is exactly zero, so
            # the efference copy contributes exactly nothing and there is
            # nothing to over-subtract; during the jump the eye is not asked.
            # That is why vertebrate gaze is saccadic, and the suppression
            # during the jump is not an engineering convenience -- saccadic
            # suppression is the published behaviour it is named after.
            #
            # UNPUBLISHED FOR THIS SPECIES AS A SEARCH BEHAVIOUR. No saccade
            # amplitude or prey-search fixation dwell has been measured in
            # *Eublepharis macularius*. The two published head-movement results
            # for this species are a stabilising REFLEX (optokinetic head gains,
            # Masseck, Roell & Hoffmann 2008) and ANTIPREDATOR binocular fixation
            # toward a snake -- neither is a saccadic search, and this code does
            # not cite them as if they were. The SHAPE is vertebrate-general; the
            # three numbers are INVENTED.
            if self._fixating:
                if self._t >= self.FIXATE_STEPS:
                    n_steps = int(self.SCAN_HALF_WIDTH_DEG // self.SACCADE_DEG)
                    self._fix_index += int(self._sign)
                    if abs(self._fix_index) > n_steps:
                        self._fix_index -= 2 * int(self._sign)
                        self._sign = -self._sign
                        self._scans += 1
                    self._fix_target = self._fix_index * self.SACCADE_DEG
                    self._fixating = False
                    self._t = 0
            else:
                # The jump. Linear, and short enough that its rate is large --
                # which is the point: it does not look like the world moving.
                frac = min(1.0, self._t / max(self.SACCADE_STEPS, 1))
                self._head += (self._fix_target - self._head) * frac
                if self._t >= self.SACCADE_STEPS:
                    self._head = self._fix_target
                    self._fixating = True
                    self._t = 0
            out = (0.0, self._head, False)
            if self._scans and self._scans != getattr(self, "_last_scan_n", 0):
                self._last_scan_n = self._scans
                self._t = 0
                self._fix_index = 0
                self._fixating = True
                if self._scans % self.SCANS_PER_SITE == 0:
                    self.state = "travel"
                else:
                    self.state = "turn"
                    self._turn_from = trunk

        elif self.state == "turn":
            # Turn until the body HAS turned. Head centred, because a sweeping
            # head during a body turn smears the two rotations together and the
            # eye is worse while moving anyway (#214).
            out = (self.TURN_COMMAND_DEG, 0.0, True)
            turned = abs(_wrap(trunk - (self._turn_from
                                        if self._turn_from is not None else trunk)))
            if turned >= self.TURN_DEG or self._t >= self.TURN_MAX_STEPS:
                self.state = "scan"
                self._t = 0
                self._head = 0.0

        else:  # travel
            out = (0.0, 0.0, True)
            if self._t >= self.TRAVEL_STEPS:
                self.state = "scan"
                self._t = 0
                self._head = 0.0

        # `looking` is False whenever the eye should not be believed: during a
        # saccade, and whenever the body is moving. It is reported rather than
        # enforced, so a caller can ignore it and the cost of ignoring it shows
        # up in the measurement instead of being hidden.
        looking = bool(self.state == "scan" and self._fixating
                       and self._t >= self.SETTLE_STEPS)
        self.last = {"state": self.state, "steps_in_state": self._t,
                     "scans_completed": self._scans,
                     "head_yaw_deg": round(out[1], 2),
                     "looking": looking}
        return out

    def provenance(self):
        return {"module": "SearchPattern",
                "invented": ["SACCADE_DEG", "SACCADE_STEPS", "FIXATE_STEPS",
                             "SETTLE_STEPS", "TURN_DEG",
                             "TURN_COMMAND_DEG", "TURN_MAX_STEPS",
                             "TRAVEL_STEPS", "SCANS_PER_SITE"],
                "from_morphology": ["SCAN_HALF_WIDTH_DEG"],
                "published": [],
                "note": "The SHAPE -- stop to look, then move -- is supported: "
                        "#214 measured this eye firing on 9.3 % of frames "
                        "still against 5.8 % moving, and Minton 1966 describes "
                        "this species slowly stalking. No gecko scan rate, "
                        "amplitude or turn angle is published. The numbers are "
                        "engineering and the coverage claim is checked by "
                        "tools/oracle_free_diagnosis.py, not asserted."}


class PreyEvidence:
    """Several agreeing sightings, or it did not happen.

    ITS FOUNDING ASSUMPTION IS REFUTED AS WRITTEN AND THE REFUTATION IS HERE
    (#245). The first version reasoned: false alarms are scattered, prey is
    not, so a bearing that several frames agree on is a target. Measured with
    the oracle off, over 600 steps:

        false alarms                118, on 19.7 % of frames
        true positives                0
        steps the quorum committed  545, every one of them to noise

    Two things were wrong. The false-alarm rate is twenty times the 1 % this
    was sized for, and at 19.7 % a 25-frame window holds about five reports, so
    three of them landing within a 25 deg tolerance happens by chance nearly
    always. The threshold was not derived from the noise it had to reject.

    IT IS KEPT, WITH ITS THRESHOLD DERIVED INSTEAD OF CHOSEN. `from_noise_rate`
    sizes the quorum from a MEASURED per-frame false-alarm rate so that chance
    agreement stays below a stated probability. That makes the module falsifiable
    -- feed it a measured rate and it either rejects that noise or it does not.

    It is also, honestly, not the binding constraint. With the prey behind the
    camera on 97 % of frames there is no signal for any accumulator to find.
    Search first, re-measure the noise against a target that is actually
    present, then size this from that number.
    """

    #: Reports needed inside the window before a bearing is committed. INVENTED
    #: as a default; prefer `from_noise_rate`, which derives it.
    QUORUM = 3
    #: How many control steps a report stays eligible. INVENTED. At 50 Hz, 25
    #: steps is half a second, the order of the animal's own temporal
    #: integration -- see Retina.motion_window for the flicker-fusion bound.
    WINDOW_STEPS = 25
    #: How far apart two bearings may be and still count as the same target,
    #: degrees. INVENTED.
    AGREE_DEG = 25.0
    #: Steps a commitment survives with no renewal. INVENTED.
    HOLD_STEPS = 40

    def __init__(self, quorum=QUORUM, window_steps=WINDOW_STEPS,
                 agree_deg=AGREE_DEG, hold_steps=HOLD_STEPS):
        self.quorum = int(quorum)
        self.window_steps = int(window_steps)
        self.agree_deg = float(agree_deg)
        self.hold_steps = int(hold_steps)
        self.reset()

    @classmethod
    def from_noise_rate(cls, false_alarms_per_frame, field_deg=70.0,
                        window_steps=WINDOW_STEPS, agree_deg=AGREE_DEG,
                        p_chance=0.01, **kw):
        """Size the quorum from a MEASURED false-alarm rate.

        Treats noise as uniform over the field, which is the conservative
        reading of the measured bearing histogram, and raises the quorum until
        the chance of that many noise reports landing inside one tolerance
        window falls below `p_chance`. Returns an instance carrying the
        derivation in `.derivation`.
        """
        rate = max(float(false_alarms_per_frame), 1e-9)
        # Expected noise reports in the window, and the share of the field one
        # tolerance window covers.
        expected = rate * int(window_steps)
        share = min(1.0, (2.0 * float(agree_deg)) / max(float(field_deg), 1e-9))
        lam = expected * share                      # Poisson mean, in-window
        q, tail = 1, 1.0
        while q < int(window_steps):
            # P(at least q) for a Poisson with mean lam.
            tail = 1.0 - sum(math.exp(-lam) * lam ** k / math.factorial(k)
                             for k in range(q))
            if tail < float(p_chance):
                break
            q += 1
        obj = cls(quorum=q, window_steps=window_steps, agree_deg=agree_deg, **kw)
        obj.derivation = {
            "false_alarms_per_frame": round(rate, 4),
            "window_steps": int(window_steps),
            "agree_deg": float(agree_deg),
            "field_deg": float(field_deg),
            "expected_noise_in_window": round(expected, 3),
            "expected_noise_in_tolerance": round(lam, 3),
            "p_chance_target": float(p_chance),
            "quorum": q,
            "p_chance_achieved": round(tail, 5),
        }
        return obj

    def reset(self):
        self._reports = []          # (step, bearing_deg)
        self._step = 0
        self._committed = None
        self._since_commit = 10 ** 6
        self.last = {}
        return self

    def step(self, bearing_deg):
        """One control step. `bearing_deg` is None when the eye said nothing.

        Returns the committed bearing in degrees, or None. Consumes only the
        tectum's own output.
        """
        self._step += 1
        self._since_commit += 1
        if bearing_deg is not None:
            self._reports.append((self._step, float(bearing_deg)))
        cutoff = self._step - self.window_steps
        self._reports = [r for r in self._reports if r[0] > cutoff]

        agreeing = []
        if len(self._reports) >= self.quorum:
            best = []
            for _, b in self._reports:
                near = [x for _, x in self._reports if abs(x - b) <= self.agree_deg]
                if len(near) > len(best):
                    best = near
            if len(best) >= self.quorum:
                agreeing = best

        if agreeing:
            agreeing = sorted(agreeing)
            mid = len(agreeing) // 2
            median = (agreeing[mid] if len(agreeing) % 2
                      else 0.5 * (agreeing[mid - 1] + agreeing[mid]))
            self._committed = float(median)
            self._since_commit = 0
        elif self._since_commit > self.hold_steps:
            self._committed = None

        self.last = {
            "committed_bearing_deg": self._committed,
            "reports_in_window": len(self._reports),
            "agreeing": len(agreeing),
            "steps_since_commit": (self._since_commit
                                   if self._committed is not None else None),
        }
        return self._committed

    def provenance(self):
        return {"module": "PreyEvidence",
                "invented": ["WINDOW_STEPS", "AGREE_DEG", "HOLD_STEPS"],
                "derived": ["QUORUM, when built via from_noise_rate"],
                "published": [],
                "note": "No gecko evidence-accumulation threshold exists. The "
                        "SHAPE -- several agreeing frames rather than one -- is "
                        "forced by a measured 118/118 false-alarm rate with the "
                        "oracle disconnected. The first threshold was chosen "
                        "and was refuted (#245); the quorum is now sized from "
                        "a measured noise rate."}


class FixationEvidence:
    """A target is something that stays put while the head does.

    WHY THIS BEATS `PreyEvidence`, and it is one idea. That class slid a
    25-frame window across a moving animal and had to allow a 25 deg agreement
    tolerance, because the bearing to a real target genuinely drifts while the
    head turns. A 25 deg window is a third of the visual field, so at the
    measured noise rate chance agreement was near certain (#245).

    Inside a FIXATION the head is not turning. It is commanded to one angle and
    held. A real cricket therefore sits at a bearing that does not move, and the
    tolerance can be cut from a third of the field to a few degrees. That is the
    whole trick, and it is only available because the search saccades instead of
    sweeping (#246) -- the two changes are one change.

    Arithmetic, at the rates this project has actually measured. Eighteen frames
    of fixation, false alarms on 19 % of frames, a +-6 deg tolerance covering
    17 % of a 70 deg field: about 0.58 noise reports expected inside any one
    tolerance band, so four agreeing reports arise by chance on well under 1 %
    of fixations. A real target detected on 21 % of frames yields about 3.8
    reports in the same 18, so roughly half of the fixations that contain prey
    clear the same bar. That ratio -- not the per-frame recall -- is what the
    animal actually hunts on.

    THE DECISION IS PER FIXATION, NOT PER FRAME. The animal looks, decides, and
    either goes or looks somewhere else. Evidence does not carry across a
    saccade, because after a saccade the bearings mean something different.

    Every constant is INVENTED. The published part is the shape: evidence
    accumulating to a threshold within a fixation is what tectal decision
    circuits are generally held to do, and no gecko threshold exists.
    """

    #: Reports needed inside ONE fixation before a bearing is committed.
    #: INVENTED as a default; `from_noise_rate` derives it from a measurement.
    QUORUM = 4
    #: How far apart two bearings may be and still count as the same target,
    #: degrees. INVENTED, and small only because the head is still.
    AGREE_DEG = 6.0
    #: Fixations a commitment survives once the animal starts moving toward it.
    #: INVENTED.
    HOLD_STEPS = 120

    #: Steps of report history the adaptive threshold is estimated over.
    #: INVENTED.
    RATE_WINDOW = 400

    def __init__(self, quorum=QUORUM, agree_deg=AGREE_DEG, hold_steps=HOLD_STEPS,
                 adaptive=False, fixate_steps=18, p_chance=0.01,
                 field_deg=70.0):
        self.quorum = int(quorum)
        self.agree_deg = float(agree_deg)
        self.hold_steps = int(hold_steps)
        # ADAPTIVE THRESHOLD, AND IT USES NOTHING PRIVILEGED. The quorum has to
        # be sized against the noise, and every earlier attempt sized it against
        # a noise rate MEASURED WITH GROUND TRUTH -- which the animal does not
        # have, and which was wrong anyway the moment the scene or the gait
        # changed (#245, #249).
        #
        # An animal can count how often its own tectum fires. It cannot know
        # which of those were crickets, but it does not need to: if almost every
        # report is noise, the report rate IS the noise rate, and if reports are
        # rare the threshold barely matters. So the estimate is the animal's own
        # firing rate over a recent window, which is observable from the inside.
        # This is gain control, which is what sensory systems do with exactly
        # this problem.
        self.adaptive = bool(adaptive)
        self.fixate_steps = int(fixate_steps)
        self.p_chance = float(p_chance)
        self.field_deg = float(field_deg)
        self.derivation = None
        self.reset()

    def _quorum_for(self, rate):
        return self.from_noise_rate(rate, fixate_steps=self.fixate_steps,
                                    field_deg=self.field_deg,
                                    agree_deg=self.agree_deg,
                                    p_chance=self.p_chance).quorum

    @classmethod
    def from_noise_rate(cls, false_alarms_per_frame, fixate_steps,
                        field_deg=70.0, agree_deg=AGREE_DEG, p_chance=0.01,
                        **kw):
        """Size the quorum from a MEASURED per-frame false-alarm rate.

        Noise is treated as uniform over the field, which is the conservative
        reading of the measured bearing histogram. Raises the quorum until the
        chance of that many noise reports landing in one tolerance band within
        one fixation falls below `p_chance`.
        """
        rate = max(float(false_alarms_per_frame), 1e-9)
        share = min(1.0, (2.0 * float(agree_deg)) / max(float(field_deg), 1e-9))
        lam = rate * int(fixate_steps) * share
        # THE NULL HAS TO MATCH WHAT THE TEST ACTUALLY DOES (#249). `step`
        # searches for the DENSEST cluster anywhere in the field; it does not
        # test one band fixed in advance. Scoring it against a single-band
        # Poisson tail is the multiple-comparisons error, and it made the first
        # derivation optimistic by roughly the number of bands -- which is why
        # a quorum "derived" for a 0.3 % chance rate fired on 15 of 24
        # fixations that contained no prey at all.
        bands = max(1.0, float(field_deg) / max(2.0 * float(agree_deg), 1e-9))
        q, tail = 1, 1.0
        while q <= int(fixate_steps):
            band = 1.0 - sum(math.exp(-lam) * lam ** k / math.factorial(k)
                             for k in range(q))
            tail = 1.0 - (1.0 - band) ** bands
            if tail < float(p_chance):
                break
            q += 1
        obj = cls(quorum=q, agree_deg=agree_deg, **kw)
        obj.derivation = {
            "false_alarms_per_frame": round(rate, 4),
            "fixate_steps": int(fixate_steps),
            "agree_deg": float(agree_deg),
            "field_deg": float(field_deg),
            "expected_noise_in_tolerance_band": round(lam, 4),
            "independent_bands_searched": round(bands, 2),
            "p_chance_target": float(p_chance),
            "quorum": q,
            "p_chance_achieved": round(tail, 6),
        }
        return obj

    def reset(self):
        self._bearings = []
        self._committed = None
        self._since_commit = 10 ** 6
        self._was_fixating = False
        self._recent = []           # 1/0 per looking step, for the rate estimate
        self.fixations = 0
        self.commits = 0
        self.last = {}
        return self

    def step(self, bearing_deg, fixating):
        """One control step.

        `bearing_deg` is the tectum's report, or None. `fixating` is the search
        module's own `looking` flag -- True only when the head is held still and
        the body is not walking. Both come from the animal.
        """
        fixating = bool(fixating)
        self._since_commit += 1

        if fixating and not self._was_fixating:
            self._bearings = []          # a new look; the old evidence is stale
            self.fixations += 1
        self._was_fixating = fixating

        if fixating and self.adaptive:
            self._recent.append(1 if bearing_deg is not None else 0)
            if len(self._recent) > self.RATE_WINDOW:
                del self._recent[:-self.RATE_WINDOW]
            if len(self._recent) >= self.fixate_steps:
                self.quorum = self._quorum_for(
                    sum(self._recent) / len(self._recent))

        agreeing = 0
        if fixating:
            if bearing_deg is not None:
                self._bearings.append(float(bearing_deg))
            if len(self._bearings) >= self.quorum:
                best = []
                for b in self._bearings:
                    near = [x for x in self._bearings
                            if abs(x - b) <= self.agree_deg]
                    if len(near) > len(best):
                        best = near
                if len(best) >= self.quorum:
                    agreeing = len(best)
                    best = sorted(best)
                    mid = len(best) // 2
                    self._committed = float(
                        best[mid] if len(best) % 2
                        else 0.5 * (best[mid - 1] + best[mid]))
                    self._since_commit = 0
                    self.commits += 1
        if self._since_commit > self.hold_steps:
            self._committed = None

        self.last = {"committed_bearing_deg": self._committed,
                     "quorum": self.quorum,
                     "report_rate": (round(sum(self._recent) / len(self._recent), 3)
                                     if self._recent else None),
                     "reports_this_fixation": len(self._bearings),
                     "agreeing": agreeing,
                     "fixating": fixating,
                     "steps_since_commit": (self._since_commit
                                            if self._committed is not None
                                            else None)}
        return self._committed

    def provenance(self):
        return {"module": "FixationEvidence",
                "invented": ["AGREE_DEG", "HOLD_STEPS", "RATE_WINDOW"],
                "derived": ["QUORUM, when built via from_noise_rate"],
                "published": [],
                "note": "The tight agreement tolerance is only defensible "
                        "because the head is held still during a fixation, so "
                        "this module and the saccadic search (#246) are one "
                        "change. The quorum is sized from a measured "
                        "false-alarm rate against a stated chance probability, "
                        "so the module is falsifiable rather than tuned."}


class OrientingReflex:
    """See something, point the head at it. The tectum's oldest job.

    WHY THIS EXISTS. Every number the eye produces is a BEARING relative to the
    head, and until now nothing ever used it to move the head. The search swept
    a fixed pattern that ignored the eye entirely, and during a chase the head
    was commanded back to centre -- so the animal walked at a target while
    looking away from it. A detector with no orienting reflex attached is half a
    tectum.

    NOT ONE CONSTANT HERE IS INVENTED, and that is the point of the file. Each
    is fixed by something already measured or by the definition of the act:

      DEADZONE_DEG   one retinal cell. The eye reports bearings quantised to the
                     cell grid -- 70 deg of field over 64 cells -- so a
                     commanded turn smaller than one cell is chasing
                     quantisation noise, not a target. DERIVED FROM THE OPTICS.
      GAIN           1.0. Orienting means putting the target on the optical
                     axis; any other value is a deliberate undershoot, and no
                     undershoot has been measured in any gecko. DEFINITIONAL.
      MAX_DEG        the neck's travel. neck_yaw 40 + head_yaw 30, less margin
                     off the stops. DERIVED FROM THE MORPHOLOGY -- though the
                     morphology's own joint ranges carry no published source
                     (#264), so this is one step removed from measured.
      MOVE_STEPS     18. The neck is a position servo and reaches 90 % of any
                     commanded step in 17-18 control steps whatever its size
                     (#278). DERIVED FROM THE PHYSICS.
      BLIND_STEPS    60. Measured (#279): after a 22 deg head saccade with the
                     body already still, the eye is silent for ~24 steps while
                     the whole-field motion is rejected as "fills the field",
                     then BURSTS to 96 % false alarms as that motion decays
                     through the detector's band, and only returns to its ~20 %
                     floor by about step 60. Believing the eye before then is
                     reading the animal's own head movement.

    WHAT IS STILL UNMEASURED, and this module does not pretend otherwise: no
    voluntary head-saccade amplitude, peak velocity, latency or spontaneous rate
    has ever been published for any gecko. Masseck, Roell & Hoffmann 2008 is the
    only target-species head measurement and it is a STABILISATION reflex to a
    rotating drum -- it explicitly cannot set a parameter for a self-initiated
    gaze shift, and using it here would be the error recorded as #277. The
    SHAPE of this reflex is vertebrate-general neuroscience; its timing comes
    from this animal's own body.
    """

    #: One retinal cell, in degrees. Recomputed from the eye at construction.
    #: The smallest movement worth commanding -- below this the bearing is
    #: quantisation, not a target.
    DEADZONE_DEG = 70.0 / 64.0
    #: How far off-axis a target must be before it is worth orienting to.
    #: DERIVED, and the reason is measured (#281). Aimed recall against
    #: eccentricity, over an oracle-steered approach: 44.5 % at 0-5 deg, 48.2 %
    #: at 5-10, 50.8 % at 10-15, 42.4 % at 15-20, 40.9 % at 20-25 -- FLAT -- then
    #: 30.9 % at 25-30 and 10.3 % at 30-36. The eye is no better at the centre
    #: than it is at 25 degrees out, so centring a target that is already inside
    #: that band buys nothing and costs the measured 78-step blind window.
    #:
    #: The first version used the one-cell deadzone as the trigger and it
    #: STARVED THE ACCUMULATOR: every sighting fired a saccade, every saccade
    #: blinded the animal for 78 steps, and across 3 seeds x 3000 steps it
    #: reached ZERO decisions. Orienting is for bringing in what is falling out
    #: of the field, not for tidying up what is already in it.
    TRIGGER_DEG = 25.0
    GAIN = 1.0
    MAX_DEG = 65.0
    MOVE_STEPS = 18
    BLIND_STEPS = 60

    def __init__(self, field_deg=70.0, cells=64, max_deg=MAX_DEG,
                 move_steps=MOVE_STEPS, blind_steps=BLIND_STEPS,
                 trigger_deg=TRIGGER_DEG):
        self.deadzone_deg = float(field_deg) / max(int(cells), 1)
        self.trigger_deg = float(trigger_deg)
        self.max_deg = float(max_deg)
        self.move_steps = int(move_steps)
        self.blind_steps = int(blind_steps)
        self.reset()

    def reset(self):
        self.head_deg = 0.0
        self._since = 10 ** 6      # steps since the last commanded movement
        self.saccades = 0
        self.last = {}
        return self

    def step(self, bearing_deg=None, allow=True):
        """One control step.

        `bearing_deg` is the eye's report RELATIVE TO THE HEAD, or None.
        `allow` lets the caller veto a movement (e.g. mid-stride).

        Returns (head_yaw_deg, believable), where `believable` is False while
        the head is moving and through the measured blind window after it.
        """
        self._since += 1
        moved = False
        if (allow and bearing_deg is not None
                and abs(float(bearing_deg)) > self.trigger_deg
                and self._since > self.move_steps + self.blind_steps):
            # Orient. The bearing is relative to where the head already points,
            # so the new command is the sum -- an absolute head angle.
            want = self.head_deg + self.GAIN * float(bearing_deg)
            want = max(-self.max_deg, min(self.max_deg, want))
            if abs(want - self.head_deg) > self.deadzone_deg:
                self.head_deg = float(want)
                self._since = 0
                self.saccades += 1
                moved = True

        believable = self._since > self.move_steps + self.blind_steps
        self.last = {"head_yaw_deg": round(self.head_deg, 2),
                     "believable": believable,
                     "steps_since_saccade": self._since,
                     "saccades": self.saccades,
                     "moved_this_step": moved}
        return self.head_deg, believable

    def provenance(self):
        return {"module": "OrientingReflex",
                "invented": [],
                "derived": ["DEADZONE_DEG (one retinal cell, from the optics)",
                            "TRIGGER_DEG (the eccentricity beyond which "
                            "measured recall falls, #281)",
                            "MOVE_STEPS (neck servo step response, #278)",
                            "BLIND_STEPS (post-saccade eye recovery, #279)",
                            "MAX_DEG (joint travel; the ranges themselves are "
                            "untagged, #264)"],
                "definitional": ["GAIN = 1.0, which is what orienting means"],
                "published": [],
                "note": "No gecko voluntary gaze shift has ever been measured. "
                        "Masseck et al. 2008 is a stabilisation reflex and "
                        "cannot calibrate this (#277). The shape is "
                        "vertebrate-general; the timing is this body's own."}
