// The animal, in JavaScript.
//
// NOTHING HERE IS A NEW MODEL. Every constant comes from media/brain.json,
// which tools/export_web_brain.py reads off the live Python objects rather
// than anyone retyping them. The equations below are transcribed from the
// Python and then CHECKED, not trusted: tools/conformance_web.py runs both
// implementations on the same inputs and compares the results. If this file
// drifts from the Python, that test fails.
//
// The physics is not ported at all. The browser runs the same MuJoCo engine
// (official WebAssembly build) on morphology/gecko_body_web.xml -- the
// certified body with its visual meshes and skin removed, a strip verified as
// physics-neutral at 400 steps under identical controls, maximum joint
// difference exactly 0.

export const FEET = ["HL", "FL", "HR", "FR"];

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

// ---------------------------------------------------------------- the walker

// envs/cpg_residual_controller.py :: _limb_signals
function limbSignals(phi, stance) {
  if (phi < stance) {
    const s = phi / stance;
    return [1 - 2 * s, 0];
  }
  const s = (phi - stance) / (1 - stance);
  return [-1 + 2 * s, Math.sin(Math.PI * s)];
}

// common/hind_stance_geometry.py :: swing_blend
function swingBlend(localPhase, stanceFraction) {
  if (localPhase <= stanceFraction || localPhase === 1) return 1;
  const f = (localPhase - stanceFraction) / (1 - stanceFraction);
  return Math.cos(Math.PI * f) ** 2;
}

// numpy.interp, clamped at both ends -- matching real ctrlrange clipping.
function interp(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] * (1 - t) + ys[hi] * t;
}

export class Walker {
  constructor(cfg) {
    this.w = cfg.walker;
    this.n = this.w.n_act;
  }

  // common/gait_config.py :: LabGaitProfile.phase_fraction, verified against
  // the closed form to 4.4e-16.
  phaseFraction(foot, t) {
    const w = this.w;
    let phase = (t * w.freq_hz + w.phase_fraction_offset[foot]) % 1;
    if (phase < 0) phase += 1;
    if (Math.min(phase, 1 - phase) < 1e-12) return 0;
    const st = w.stance[foot];
    if (Math.abs(phase - st) < 1e-12) return st;
    return phase;
  }

  // envs/cpg_residual_controller.py :: sprawl_signal
  sprawlSignal(limb, t) {
    const p = this.w.lab_parameters;
    const amp = limb.startsWith("H") ? p.hind_sprawl_amplitude : p.fore_sprawl_amplitude;
    if (amp === 0) return 0;
    const phi = this.phaseFraction(limb, t);
    return amp * Math.sin(2 * Math.PI * (phi + p.sprawl_phase));
  }

  // envs/cpg_residual_controller.py :: _lab_limb_signal
  labLimbSignal(limb, role, fa, lift, steer) {
    const w = this.w, p = w.lab_parameters;
    if (role === "fa") {
      let sig = w.lab_fa_amplitude[limb] * fa;
      if (limb.startsWith("H")) sig *= w.tail_coupling_scale;
      if (p.mirror_left_fa && limb.endsWith("L")) sig = -sig;
      if (steer !== 0) sig *= limb.endsWith("L") ? 1 - steer : 1 + steer;
      return sig;
    }
    if (role !== "lift") return p.other_amplitude * fa;
    if (limb.startsWith("H")) {
      const sig = w.amp.lift * lift;
      return p.hind_lift_multiplier === 1 ? sig : sig * p.hind_lift_multiplier;
    }
    const press = limb === "FL" ? w.front_stance_press : w.front_stance_press_fr;
    if (p.continuous_front_lift) {
      return lift > 0 ? press + w.front_swing_lift * lift : press;
    }
    // front_contact is null in this build (open loop), matching base_ctrl's
    // documented "front_contact=None keeps the exact V4.2.7 open-loop behavior".
    if (lift > 0) return w.front_swing_lift * lift;
    return press;
  }

  // envs/cpg_residual_controller.py :: base_ctrl
  baseCtrl(t, headingError = 0) {
    const w = this.w, p = w.lab_parameters;
    const ctrl = Float64Array.from(w.neutral);
    const steer = Math.max(-1, Math.min(1, p.heading_gain * headingError));

    const sig = {};
    for (const f of FEET) sig[f] = limbSignals(this.phaseFraction(f, t), w.stance[f]);

    for (const [aid, limb, role] of w.entries) {
      const [fa, lift] = sig[limb];
      let s = this.labLimbSignal(limb, role, fa, lift, steer);
      if (w.sprawl_ids.includes(aid)) s += this.sprawlSignal(limb, t);
      ctrl[aid] = w.neutral[aid] + w.sign[aid] * s * w.half[aid];
    }

    // The stance compensator, applied after the limb loop exactly as Python
    // does. Its tables are the SAME arrays Python interpolates.
    for (const foot of Object.keys(w.compensator)) {
      const c = w.compensator[foot];
      const phi = this.phaseFraction(foot, t);
      const weight = swingBlend(phi, w.stance[foot]);
      const hip = ctrl[c.drive_aid];
      const cols = c.table[0][0].length;
      for (let k = 0; k < c.free_aids.length && k < cols; k++) {
        const col = c.table.map((row) => row[0][k]);   // sprawl_grid.size === 1
        // Python calls offsets() with local_phase = 0, so ITS internal
        // swing_blend is exactly 1 and the weight is applied once, below.
        // Applying it here as well double-counts it -- which is precisely
        // what the conformance test caught on knee and ankle.
        const value = interp(hip, c.hip_grid, col);
        const aid = c.free_aids[k];
        const target = w.neutral[aid] + value;
        ctrl[aid] = weight * target + (1 - weight) * ctrl[aid];
      }
    }

    if (w.ssl >= 0) ctrl[w.ssl] -= w.shoulder_sprawl_tuck;
    if (w.ssr >= 0) ctrl[w.ssr] += w.shoulder_sprawl_tuck;

    if (w.spine >= 0 && w.spine_amp > 0) {
      const wv = Math.sin(2 * Math.PI * (t * w.freq_hz + w.spine_phase));
      ctrl[w.spine] = w.spine_amp * w.half[w.spine] * wv;
      if (w.tail_amp > 0 && w.tail_l >= 0 && w.tail_r >= 0) {
        const wt = Math.sin(2 * Math.PI * (t * w.freq_hz + w.spine_phase - w.tail_phase_lag));
        ctrl[w.tail_l] = +w.tail_amp * w.half[w.tail_l] * wt;
        ctrl[w.tail_r] = -w.tail_amp * w.half[w.tail_r] * wt;
      }
    }
    return ctrl;
  }
}

// ------------------------------------------------------- the basal ganglia
// brain/prescott_bg.py, extended variant, afferent dopamine.

const rect = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

export class BasalGanglia {
  constructor(cfg) {
    const b = cfg.bg;
    this.p = b.params;
    this.n = b.channels.length;
    this.channels = b.channels;
    this.dopamine = b.dopamine;
    this.gpiTonic = b.gpi_tonic;
    this.FULL = b.FULL; this.PARTIAL = b.PARTIAL;
    this.maxSteps = b.max_steps; this.tol = b.tolerance;
    this.reset();
  }

  reset() {
    const z = () => new Float64Array(this.n);
    this.ctx = z(); this.mot = z(); this.sd1 = z(); this.sd2 = z();
    this.stn = z(); this.gpe = z(); this.gpi = z(); this.thl = z(); this.trn = z();
    this.settled = true;
  }

  step(s) {
    const p = this.p, n = this.n;
    const yCtx = new Float64Array(n), yMot = new Float64Array(n),
          yD1 = new Float64Array(n), yD2 = new Float64Array(n),
          yStn = new Float64Array(n), yGpe = new Float64Array(n),
          yGpi = new Float64Array(n), yThl = new Float64Array(n),
          yTrn = new Float64Array(n);
    // Basic model reads the thalamus against minus its own tonic output;
    // the extended model uses zero. p.extended is true here.
    const thlThr = p.thalamus_threshold_is_tonic ? -this.gpiTonic : 0;
    let stnTotal = 0, trnTotal = 0;
    for (let i = 0; i < n; i++) {
      yCtx[i] = rect(this.ctx[i]);
      yMot[i] = rect(this.mot[i]);
      yD1[i] = rect(this.sd1[i] - p.threshold_striatum);   // afferent mode
      yD2[i] = rect(this.sd2[i] - p.threshold_striatum);
      yStn[i] = rect(this.stn[i] - p.threshold_stn);
      yGpe[i] = rect(this.gpe[i] - p.threshold_gpe);
      yGpi[i] = rect(this.gpi[i] - p.threshold_gpi);
      yThl[i] = rect(this.thl[i] - thlThr);
      yTrn[i] = rect(this.trn[i]);
      stnTotal += yStn[i]; trnTotal += yTrn[i];
    }

    let biggest = 0;
    const relax = p.relax;
    for (let i = 0; i < n; i++) {
      const uCtx = s[i];
      const uMot = yCtx[i] + p.thalamus_to_motor * yThl[i];
      const drive = p.salience_share * yCtx[i] + p.motor_share * yMot[i];
      const uSd1 = (1 + this.dopamine) * drive;
      const uSd2 = (1 - this.dopamine) * drive;
      const uStn = drive - p.gpe_stn * yGpe[i];
      const uGpe = p.stn_out * stnTotal - p.sd2_gpe * yD2[i];
      const uGpi = p.stn_out * stnTotal - p.sd1_gpi * yD1[i] - p.gpe_gpi * yGpe[i];
      let uThl = -yGpi[i] + (p.extended ? yMot[i] : 0);
      if (p.extended) {
        uThl -= p.trn_within * yTrn[i] + p.trn_between * (trnTotal - yTrn[i]);
      }
      const uTrn = p.extended ? (yMot[i] + yThl[i] - p.trn_from_gpi * yGpi[i]) : 0;

      const pairs = [[this.ctx, uCtx], [this.mot, uMot], [this.sd1, uSd1],
                     [this.sd2, uSd2], [this.stn, uStn], [this.gpe, uGpe],
                     [this.gpi, uGpi], [this.thl, uThl], [this.trn, uTrn]];
      for (const [state, target] of pairs) {
        const d = relax * (target - state[i]);
        state[i] += d;
        const ad = Math.abs(d);
        if (ad > biggest) biggest = ad;
      }
    }
    return biggest;
  }

  converge(s) {
    let under = 0;
    for (let i = 0; i < this.maxSteps; i++) {
      if (this.step(s) < this.tol) { if (++under >= 2) { this.settled = true; return i + 1; } }
      else under = 0;
    }
    this.settled = false;
    return this.maxSteps;
  }

  gates() {
    const g = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const y = Math.max(0, this.gpi[i] - this.p.threshold_gpi);
      g[i] = Math.max(0, Math.min(1, 1 - y / this.gpiTonic));
    }
    return g;
  }

  // brain/gecko_selector.py :: selected -- uses the authors' published PARTIAL,
  // not an invented epsilon (#357).
  selected() {
    const g = this.gates();
    let best = 0;
    for (let i = 1; i < this.n; i++) if (g[i] > g[best]) best = i;
    return g[best] < this.PARTIAL ? null : this.channels[best];
  }
}

// --------------------------------------------------------------- the clock
// brain/arousal.py :: arousal_at

export class Clock {
  constructor(cfg) {
    const c = cfg.clock;
    this.DAY = c.DAY_S; this.dark = c.DARK_FRACTION;
    this.ramp = c.DUSK_RAMP_H * 3600;
    this.onset = c.onset_after_dark_s; this.peak = c.peak_window_h * 3600;
    this.t = 0;
  }
  get duskS() { return this.DAY * (1 - this.dark); }
  arousalAt(tod) {
    const DAY = this.DAY;
    let t = ((tod % DAY) + DAY) % DAY;
    let since = t - this.duskS;
    if (since < 0) since += DAY;
    const night = DAY * this.dark;
    if (since > night) {
      const untilDusk = DAY - since;
      if (untilDusk <= this.ramp) return clamp01(1 - untilDusk / this.ramp) * 0.6;
      return 0;
    }
    if (since < this.onset) return 0.6 + 0.4 * (since / Math.max(this.onset, 1e-9));
    if (since <= this.peak) return 1;
    return clamp01(1 - (since - this.peak) / Math.max(night - this.peak, 1e-9));
  }
  step(dt) {
    this.t = (this.t + dt) % this.DAY;
    const a = this.arousalAt(this.t);
    return { timeOfDayH: this.t / 3600, arousal: a, dark: this.t >= this.duskS, asleep: a < 0.1 };
  }
}

// ------------------------------------------------------------- the salience
// brain/gecko_selector.py :: salience_from_drives
export function salienceFromDrives(cfg, { hunger = 0, cold = 0, warm = 0, threat = 0,
                                          preyVisible = 0, arousal = 1 }) {
  return Float64Array.from([
    hunger * preyVisible,                       // hunt
    threat,                                     // flee
    0.35 * hunger * (1 - preyVisible),          // explore
    Math.max(cold, warm),                       // bask
    1 - clamp01(arousal),                       // rest -- the day/night clock
    cfg.bg.groom_tonic,                         // groom -- 0.0, and it was 0.05
  ]);
}

// -------------------------------------------------- the search pattern
// brain/search.py :: SearchPattern, transcribed.
//
// It reads ONLY the animal's own trunk angle -- proprioception -- and nothing
// about prey, which is why it ports honestly with no eye attached. The cycle
// is SCAN -> TURN -> TRAVEL -> SCAN: stand still and sweep the head in 22 deg
// saccades out to +-65 deg, turn the body 120 deg, sweep again, and after
// three sweeps walk to a new place and start over. Three scans separated by
// two turns inspect the whole circle.
//
// Every constant here is INVENTED except SCAN_HALF_WIDTH_DEG, which comes from
// the neck's own travel, and SACCADE_STEPS, which was derived from the servo
// (#250, #282). That is recorded in brain/search.py and repeated here so the
// browser does not look better sourced than the original.

const wrap = (d) => { let x = (d + 180) % 360; if (x < 0) x += 360; return x - 180; };

export class SearchPattern {
  // Not one constant is typed here. They arrive in brain.json, read straight
  // off the Python class by tools/export_web_brain.py, so the browser cannot
  // drift from brain/search.py without the export changing too.
  constructor(cfg) {
    const s = (cfg && cfg.search) || {};
    for (const k of ["SACCADE_DEG", "SACCADE_STEPS", "FIXATE_STEPS",
                     "SETTLE_STEPS", "SCAN_HALF_WIDTH_DEG", "TURN_DEG",
                     "TURN_COMMAND_DEG", "TURN_MAX_STEPS", "TRAVEL_STEPS",
                     "SCANS_PER_SITE"]) {
      if (!(k in s)) throw new Error("brain.json is missing search." + k);
      this[k] = s[k];
    }
    this._lastScanN = 0;   // Python leaves this across reset(); so do we
    this.reset();
  }
  reset() {
    this.state = "scan"; this._t = 0; this._head = 0; this._sign = 1;
    this._scans = 0; this._fixating = true; this._fixTarget = 0; this._fixIndex = 0;
    this._turnFrom = null;
    this.looking = false;
    return this;
  }
  step(trunkYawDeg = 0) {
    this._t += 1;
    const trunk = trunkYawDeg;
    let out;
    if (this.state === "scan") {
      if (this._fixating) {
        if (this._t >= this.FIXATE_STEPS) {
          const n = Math.floor(this.SCAN_HALF_WIDTH_DEG / this.SACCADE_DEG);
          this._fixIndex += this._sign;
          if (Math.abs(this._fixIndex) > n) {
            this._fixIndex -= 2 * this._sign;
            this._sign = -this._sign;
            this._scans += 1;
          }
          this._fixTarget = this._fixIndex * this.SACCADE_DEG;
          this._fixating = false; this._t = 0;
        }
      } else {
        const frac = Math.min(1, this._t / Math.max(this.SACCADE_STEPS, 1));
        this._head += (this._fixTarget - this._head) * frac;
        if (this._t >= this.SACCADE_STEPS) {
          this._head = this._fixTarget; this._fixating = true; this._t = 0;
        }
      }
      out = [0, this._head, false];
      if (this._scans && this._scans !== this._lastScanN) {
        this._lastScanN = this._scans;
        this._t = 0; this._fixIndex = 0; this._fixating = true;
        if (this._scans % this.SCANS_PER_SITE === 0) this.state = "travel";
        else { this.state = "turn"; this._turnFrom = trunk; }
      }
    } else if (this.state === "turn") {
      out = [this.TURN_COMMAND_DEG, 0, true];
      const turned = Math.abs(wrap(trunk - (this._turnFrom == null ? trunk : this._turnFrom)));
      if (turned >= this.TURN_DEG || this._t >= this.TURN_MAX_STEPS) {
        this.state = "scan"; this._t = 0; this._head = 0;
      }
    } else {
      out = [0, 0, true];
      if (this._t >= this.TRAVEL_STEPS) { this.state = "scan"; this._t = 0; this._head = 0; }
    }
    this.looking = this.state === "scan" && this._fixating && this._t >= this.SETTLE_STEPS;
    return { headingDeg: out[0], headYawDeg: out[1], moving: out[2],
             state: this.state, looking: this.looking };
  }
}
