// Everything between seeing a cricket and eating one, ported from
// brain/search.py (FixationEvidence, OrientingReflex), brain/programs.py
// (MotorPrograms) and envs/prey.py (Prey).
//
// The chain is: the tectum reports a bearing on some frames and nothing on
// others -> the evidence accumulator decides whether those reports are one
// object or noise -> the orienting reflex turns the head toward what has been
// decided real -> the chase program walks the body at it in stalk bouts -> the
// cricket, which is running away.
//
// NOTHING HERE IS TOLD WHERE THE CRICKET IS. The only channel from the world
// into the brain is `site/eye.js`, and the only thing it carries is an angle
// that may or may not be a cricket.
//
// Constants are read off the live Python classes by tools/export_web_brain.py.

// -------------------------------------------------------------- evidence --
//
// A single frame is not evidence. Measured: letting the reflex fire on any
// single-frame report took three seeds of 3000 steps from 144/158/96 decisions
// down to 1/19/0, because the field is +-35 deg, false alarms are spread across
// it, and every noise report bought a saccade plus a 78-step blind window.
export class FixationEvidence {
  constructor(cfg) {
    const h = cfg.hunt.evidence;
    this.quorum = h.quorum;
    this.agreeDeg = h.agree_deg;
    this.holdSteps = h.hold_steps;
    this.rateWindow = h.rate_window;
    this.fixateSteps = h.fixate_steps;
    this.pChance = h.p_chance;
    this.fieldDeg = h.field_deg;
    this.adaptive = h.adaptive;
    this.reset();
  }
  reset() {
    this.bearings = []; this.committed = null; this.sinceCommit = 1e6;
    this.wasFixating = false; this.recent = []; this.fixations = 0; this.commits = 0;
    this.last = {};
    return this;
  }

  // Size the quorum from the animal's OWN firing rate. It cannot know which of
  // its reports were crickets, and it does not need to: if almost every report
  // is noise, the report rate IS the noise rate. The null has to match what
  // `step` actually does -- it searches for the densest cluster ANYWHERE in the
  // field, so the single-band Poisson tail is corrected for the number of bands,
  // which is the multiple-comparisons error that made the first derivation fire
  // on 15 of 24 fixations containing no prey at all.
  _quorumFor(rate) {
    rate = Math.max(rate, 1e-9);
    const share = Math.min(1, (2 * this.agreeDeg) / Math.max(this.fieldDeg, 1e-9));
    const lam = rate * this.fixateSteps * share;
    const bands = Math.max(1, this.fieldDeg / Math.max(2 * this.agreeDeg, 1e-9));
    let q = 1, tail = 1;
    while (q <= this.fixateSteps) {
      let cum = 0, term = Math.exp(-lam);
      for (let k = 0; k < q; k++) { cum += term; term = (term * lam) / (k + 1); }
      const band = 1 - cum;
      tail = 1 - Math.pow(1 - band, bands);
      if (tail < this.pChance) break;
      q += 1;
    }
    return q;
  }

  step(bearingDeg, fixating) {
    fixating = !!fixating;
    this.sinceCommit += 1;
    if (fixating && !this.wasFixating) { this.bearings = []; this.fixations += 1; }
    this.wasFixating = fixating;

    if (fixating && this.adaptive) {
      this.recent.push(bearingDeg !== null && bearingDeg !== undefined ? 1 : 0);
      if (this.recent.length > this.rateWindow) this.recent.splice(0, this.recent.length - this.rateWindow);
      if (this.recent.length >= this.fixateSteps) {
        this.quorum = this._quorumFor(this.recent.reduce((a, b) => a + b, 0) / this.recent.length);
      }
    }

    let agreeing = 0;
    if (fixating) {
      if (bearingDeg !== null && bearingDeg !== undefined) this.bearings.push(bearingDeg);
      if (this.bearings.length >= this.quorum) {
        let best = [];
        for (const b of this.bearings) {
          const near = this.bearings.filter((x) => Math.abs(x - b) <= this.agreeDeg);
          if (near.length > best.length) best = near;
        }
        if (best.length >= this.quorum) {
          agreeing = best.length;
          best = best.slice().sort((a, b) => a - b);
          const mid = Math.floor(best.length / 2);
          this.committed = best.length % 2 ? best[mid] : 0.5 * (best[mid - 1] + best[mid]);
          this.sinceCommit = 0;
          this.commits += 1;
        }
      }
    }
    if (this.sinceCommit > this.holdSteps) this.committed = null;

    this.last = {
      committed_bearing_deg: this.committed, quorum: this.quorum,
      reports_this_fixation: this.bearings.length, agreeing, fixating,
      report_rate: this.recent.length
        ? this.recent.reduce((a, b) => a + b, 0) / this.recent.length : null,
    };
    return this.committed;
  }
}

// ------------------------------------------------------- orienting reflex --
//
// BLIND_STEPS is 60 and it is measured, not chosen: after a 22 deg head saccade
// the eye does not recover for about 78 control steps. Orienting is for bringing
// in what is falling out of the field, not for tidying up what is already in it.
export class OrientingReflex {
  constructor(cfg) {
    const o = cfg.hunt.orienting;
    this.deadzoneDeg = o.field_deg / Math.max(o.cells, 1);
    this.triggerDeg = o.trigger_deg;
    this.maxDeg = o.max_deg;
    this.moveSteps = o.move_steps;
    this.blindSteps = o.blind_steps;
    this.gain = o.gain;
    this.reset();
  }
  reset() { this.headDeg = 0; this.since = 1e6; this.saccades = 0; this.last = {}; return this; }
  step(bearingDeg = null, allow = true) {
    this.since += 1;
    let moved = false;
    if (allow && bearingDeg !== null && bearingDeg !== undefined
        && Math.abs(bearingDeg) > this.triggerDeg
        && this.since > this.moveSteps + this.blindSteps) {
      // The bearing is relative to where the head already points, so the new
      // command is the sum -- an absolute head angle.
      let want = this.headDeg + this.gain * bearingDeg;
      want = Math.max(-this.maxDeg, Math.min(this.maxDeg, want));
      if (Math.abs(want - this.headDeg) > this.deadzoneDeg) {
        this.headDeg = want; this.since = 0; this.saccades += 1; moved = true;
      }
    }
    const believable = this.since > this.moveSteps + this.blindSteps;
    this.last = { head_yaw_deg: this.headDeg, believable,
                  steps_since_saccade: this.since, saccades: this.saccades,
                  moved_this_step: moved };
    return [this.headDeg, believable];
  }
}

// -------------------------------------------------------- motor programs --

export class MotorPrograms {
  constructor(cfg, search, evidence) {
    this.cfg = cfg;
    this.search = search;
    this.evidence = evidence;
    this.orient = new OrientingReflex(cfg);
    const h = cfg.hunt;
    this.PROGRAMS = cfg.world.programs;
    this.LOCOMOTOR = cfg.world.locomotor;
    this.DRIVE_THRESHOLD = cfg.world.drive_threshold;
    this.STALK_MOVE_STEPS = h.stalk_move_steps;
    this.STALK_LOOK_STEPS = h.stalk_look_steps;
    this.ORIENT_HOLD_STEPS = h.orient_hold_steps;
    this.RESUME_SEARCH_ON_LOST = h.resume_search_on_lost;
    this.stalkT = 0;
    this.program = "still";
    this.last = {};
  }

  step(behaviour, { urgency = 1, bearingDeg = null, trunkYawDeg = 0,
                    warmBearingDeg = null, shelterBearingDeg = null,
                    onWarmGround = false } = {}) {
    let program = this.PROGRAMS[behaviour] || "still";
    let drive = (this.LOCOMOTOR[behaviour] || 0) * Math.min(Math.max(urgency, 0), 1);
    if (drive <= this.DRIVE_THRESHOLD) drive = 0;

    let headingDeg = 0;
    let scanHeadDeg = this.orient.headDeg;
    let ranSearch = false;
    if (program !== "chase") this.stalkT = 0;

    // ---- 1. WHERE THE BODY IS GOING ------------------------------------
    if (program === "shelter" || program === "warm") {
      const known = program === "shelter" ? shelterBearingDeg : warmBearingDeg;
      if (known === null || known === undefined) {
        program = "search";              // nothing remembered; brain 7 absent
      } else {
        headingDeg = known;
        // A BASKING ANIMAL LIES DOWN. Walking toward warm ground and walking
        // once you are ON it are not the same act: the second walks you off
        // the far side. Measured -- it closed from 34.0 cm to 3.6 cm, warmed
        // 25.2 -> 28.9 C, then carried on over the edge and cooled again.
        if (program === "warm" && onWarmGround) drive = 0;
      }
    }

    const committedPrior = this.evidence ? this.evidence.last.committed_bearing_deg : null;
    if (program === "chase" && (committedPrior === null || committedPrior === undefined)) {
      program = this.RESUME_SEARCH_ON_LOST ? "search" : "still";
    }

    if (program === "chase") {
      // The committed bearing is HEAD-relative; the walker's heading is
      // TRUNK-relative. The head is not snapped back to centre, so the two
      // differ by wherever the head now points, and the sum is the answer.
      headingDeg = this.orient.headDeg + committedPrior;
      const cycle = this.STALK_MOVE_STEPS + this.STALK_LOOK_STEPS;
      if (this.stalkT % cycle >= this.STALK_MOVE_STEPS) drive = 0;   // the looking half
      this.stalkT += 1;
    } else if (program === "search" && this.search) {
      const sp = this.search.step(trunkYawDeg);
      headingDeg = sp.headingDeg; scanHeadDeg = sp.headYawDeg;
      ranSearch = true;
      if (!sp.moving) drive = 0;
    } else if (program === "still") {
      drive = 0;
    }

    const bodyStill = drive <= 0;

    // ---- 2. WHERE THE HEAD IS LOOKING ----------------------------------
    // Ownership: the reflex owns the head while it is examining a candidate;
    // when idle the scan owns it and the reflex is re-seeded to wherever the
    // scan put it, so its internal angle never drifts from the real one.
    const orientBusy = (this.orient.last.steps_since_saccade ?? 1e6)
      <= this.orient.moveSteps + this.orient.blindSteps + this.ORIENT_HOLD_STEPS;
    if (ranSearch && !orientBusy) this.orient.headDeg = scanHeadDeg;

    const track = (program === "chase" && committedPrior !== null
                   && committedPrior !== undefined && bodyStill)
      ? committedPrior : null;
    const [headYawDeg, orientOk] = this.orient.step(track, bodyStill);

    // ---- 3. IS THE EYE WORTH BELIEVING THIS STEP? ----------------------
    // Three measured conditions, all of which must hold: the body is still
    // (standing more than doubles the signal), the head is past its measured
    // post-saccade blind window, and the scan agrees when the scan is driving.
    const looking = !!(bodyStill && orientOk
                       && (!ranSearch || this.search.looking));

    let committed = null;
    if (this.evidence) {
      committed = this.evidence.step(
        bearingDeg !== null && bearingDeg !== undefined && looking ? bearingDeg : null,
        looking);
      if (program === "chase" && committed !== null) {
        headingDeg = this.orient.headDeg + committed;
      }
    }

    this.program = program;
    this.last = {
      behaviour, program, heading_deg: headingDeg, head_yaw_deg: headYawDeg,
      locomotor_drive: drive, committed_bearing_deg: committed,
      believed_eye: looking, saccades: this.orient.saccades,
      orienting: orientBusy,
      search_state: ranSearch ? this.search.state : program,
    };
    return this.last;
  }
}

// ------------------------------------------------------------------ prey --
//
// A cricket that runs away, which is what makes the camera load-bearing. Every
// number comes from config/proxies.yaml with its own species, source and
// confidence -- nothing is defaulted here, because a default in this file would
// be an invented number wearing a published one's clothes.
//
// What this deliberately does NOT model: prey that turns, evades sideways, uses
// cover, or tires. Real crickets do all of those. The escape is a straight line
// because that is the one thing the literature actually constrains.

// mulberry32: a small deterministic PRNG, so a reload is a different cricket
// but a seeded run is repeatable.
function rng32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MEAN_BOUT_S = 1.6;   // INVENTED; bout structure is unmeasured in any cricket

export class Prey {
  constructor(cfg, seed = (Math.random() * 4294967296) >>> 0) {
    const p = cfg.prey;
    this.escapeSpeed = p.prey_escape_speed_m_s;
    this.fleeRadius = p.prey_flee_radius_m;
    this.escapeLatency = p.prey_escape_latency_s;
    this.captureDistance = p.prey_capture_distance_m;
    this.radius = p.prey_radius_m;
    this.arenaRadius = p.prey_arena_radius_m;
    this.ambientSpeed = p.prey_ambient_speed_m_s;
    this.ambientMoveFraction = p.prey_ambient_move_fraction;
    this.ambientTurnRate = p.prey_ambient_turn_rate_rad_s;
    this.fleeSpeedReference = p.prey_flee_speed_reference_m_s;
    this.fleeSpeedExponent = p.prey_flee_speed_exponent;
    this.fleeRadiusFloor = p.prey_flee_radius_floor_m;
    this.rand = rng32(seed);
    this.captures = 0;
    this.reset([0, 0]);
  }
  _exponential(mean) { return -Math.log(1 - this.rand()) * mean; }

  // Uniform on the arena disc, never inside the flee radius -- Prey._respawn.
  // The sqrt keeps the sample uniform over AREA rather than clustered inward;
  // my first version invented a 0.22-0.40 m annulus instead, which put the
  // cricket a median of 0.65 m away, where a 9 mm insect subtends 0.8 deg and
  // is smaller than one pixel of a 64-pixel eye. The animal was firing on
  // noise 92 % of the time because there was nothing else it could have been
  // firing on.
  _respawn(predatorXY) {
    for (let i = 0; i < 64; i++) {
      const angle = (this.rand() * 2 - 1) * Math.PI;
      const radius = this.arenaRadius * Math.sqrt(this.rand());
      const cx = predatorXY[0] + radius * Math.cos(angle);
      const cy = predatorXY[1] + radius * Math.sin(angle);
      if (Math.hypot(cx - predatorXY[0], cy - predatorXY[1]) > this.fleeRadius) {
        return [cx, cy];
      }
    }
    return [predatorXY[0] + this.arenaRadius, predatorXY[1]];
  }

  reset(predatorXY) {
    const pos = this._respawn(predatorXY);
    this.x = pos[0]; this.y = pos[1];
    this.heading = (this.rand() * 2 - 1) * Math.PI;
    this.walking = false;
    this.boutLeft = 0;
    this.alarm = null;
    this.lastDistance = null;
    this.approachSpeed = 0;
    return this;
  }

  // The flee radius shrinks when the threat is closing slowly. This is the
  // reason a stalk can work at all; all three constants are INVENTED and the
  // registry says so.
  fleeRadiusFor(approachSpeed) {
    const ref = Math.max(this.fleeSpeedReference, 1e-9);
    const scaled = this.fleeRadius * Math.pow(Math.max(approachSpeed, 0) / ref,
                                              this.fleeSpeedExponent);
    return Math.max(this.fleeRadiusFloor, Math.min(this.fleeRadius, scaled));
  }

  step(dt, predatorXY) {
    const ox = this.x - predatorXY[0], oy = this.y - predatorXY[1];
    const distance = Math.hypot(ox, oy);
    // CLOSING SPEED, NOT PREDATOR SPEED. The predator point is the gecko's
    // nose, which sways several centimetres per second with the gait even when
    // the animal is barely advancing -- measured 0.129 m/s of apparent approach
    // during a creep that netted almost nothing. Reading that as a charge made
    // stalking impossible.
    if (this.lastDistance !== null) {
      const closing = Math.max((this.lastDistance - distance) / dt, 0);
      this.approachSpeed = 0.8 * this.approachSpeed + 0.2 * closing;
    }
    this.lastDistance = distance;

    if (distance <= this.captureDistance) {
      this.captures += 1;
      this.reset(predatorXY);
      return { captured: true };
    }

    // GUIDED MODE HOLDS IT STILL. `calm` is set only by the page's declared
    // oracle; with it off the cricket flees exactly as envs/prey.py says.
    if (!this.calm && distance <= this.fleeRadiusFor(this.approachSpeed)) {
      // Latency: a startled animal does not accelerate instantaneously, and a
      // zero-latency prey is uncatchable for reasons that are an artefact of
      // the simulation rather than of the animal.
      if (this.alarm === null) this.alarm = this.escapeLatency;
      this.alarm -= dt;
      if (this.alarm <= 0) {
        const inv = distance > 1e-9 ? 1 / distance : 0;
        const dx = inv ? ox * inv : Math.cos(this.heading);
        const dy = inv ? oy * inv : Math.sin(this.heading);
        this.x += dx * this.escapeSpeed * dt;
        this.y += dy * this.escapeSpeed * dt;
        this.fleeing = true;
      }
    } else {
      this.alarm = null;
      this.fleeing = false;
      this._ambient(dt);
    }

    // KEEPING THE TWO OF THEM IN THE SAME WORLD. Python clamps the prey to
    // twice the arena radius measured from the ORIGIN, because its animal lives
    // in a bounded habitat. This one does not -- the user asked for no walls,
    // and it walks metres. So the bound is measured from the ANIMAL instead,
    // and when a cricket has wandered further than the arena the prey
    // parameters were measured in, a fresh one enters at the edge of it.
    //
    // THIS IS STAGING AND IT SAYS SO. A real desert does not hand a gecko
    // another cricket; it is here because a hunt you can never witness is not
    // worth rendering. Nothing about the DETECTION is staged -- the animal
    // still has to find this one in its own pixels.
    if (Math.hypot(this.x - predatorXY[0], this.y - predatorXY[1]) > this.arenaRadius) {
      this.reset(predatorXY);
      this.replacements = (this.replacements || 0) + 1;
    }
    return { captured: false };
  }

  // Undisturbed: walk in bouts, not a glide. A cricket beyond the flee radius
  // has not noticed the gecko and is going about its business, which is the
  // whole reason the eye has anything to detect at all -- and a bout structure
  // is exactly what a temporal-contrast detector sees.
  _ambient(dt) {
    if (this.ambientSpeed <= 0) return;
    this.boutLeft -= dt;
    if (this.boutLeft <= 0) {
      this.walking = !this.walking;
      const mean = MEAN_BOUT_S * (this.walking ? this.ambientMoveFraction
                                               : 1 - this.ambientMoveFraction);
      this.boutLeft = this._exponential(Math.max(mean, 1e-3));
      if (this.walking) this.heading = (this.rand() * 2 - 1) * Math.PI;
    }
    if (!this.walking) return;
    this.heading += (this.rand() * 2 - 1) * this.ambientTurnRate * dt;
    this.x += Math.cos(this.heading) * this.ambientSpeed * dt;
    this.y += Math.sin(this.heading) * this.ambientSpeed * dt;
  }
}
