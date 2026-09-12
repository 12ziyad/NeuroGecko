// The eye: retina and tectum, ported from brain/retina.py and brain/tectum.py.
//
// WHY THIS FILE EXISTS. Until now the browser animal had no eye, so `hunt`
// could never release: its salience is `hunger * prey_visible`, and
// `prey_visible` comes from here. The tempting shortcut was to hand the brain
// the cricket's coordinates and let it walk at them. That is an ORACLE, and
// this project spent a long stretch of its ledger taking oracles back out --
// the measured cost of the last one was an eye that fired on 72 % of frames
// with prey in the world and 72 % with no prey in the world at all.
//
// So this does what the animal does. A camera rides on the head. Every control
// step it renders one frame, and this code finds the cricket in the pixels or
// fails to. Nothing downstream is told where the cricket is.
//
// Every constant arrives in brain.json, read off the live Python classes by
// tools/export_web_brain.py. tools/conformance_web.py pushes identical image
// sequences through Python and through this file and compares salience,
// bearing and elevation frame by frame.

const EPS = 1e-9;

// ---------------------------------------------------------------- helpers --

// Separable box mean of width 2w+1 with the edges extended -- retina._blur.
// Python pads by `w`, convolves rows then columns in 'same' mode and crops the
// padding off again, which works out to exactly this.
function blur(plane, n, ratio) {
  const w = Math.max(1, Math.round(ratio));
  if (w <= 1) return plane.slice();
  const at = (i) => (i < 0 ? 0 : i >= n ? n - 1 : i);   // edge extension
  const span = 2 * w + 1;
  const tmp = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let d = -w; d <= w; d++) s += plane[r * n + at(c + d)];
      tmp[r * n + c] = s / span;
    }
  }
  const out = new Float64Array(n * n);
  for (let c = 0; c < n; c++) {
    for (let r = 0; r < n; r++) {
      let s = 0;
      for (let d = -w; d <= w; d++) s += tmp[at(r + d) * n + c];
      out[r * n + c] = s / span;
    }
  }
  return out;
}

// Mean of the neighbourhood EXCLUDING the centre -- tectum._surround. It has to
// be spatially varying: a scalar reference cannot change which cell is largest,
// which is why the version before it was a no-op.
function surround(plane, n, w) {
  const out = new Float64Array(n * n);
  if (w < 1) return out;
  const at = (i) => (i < 0 ? 0 : i >= n ? n - 1 : i);
  const count = (2 * w + 1) * (2 * w + 1) - 1;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let dy = -w; dy <= w; dy++) {
        for (let dx = -w; dx <= w; dx++) {
          if (dx === 0 && dy === 0) continue;
          s += plane[at(r + dy) * n + at(c + dx)];
        }
      }
      out[r * n + c] = s / Math.max(count, 1);
    }
  }
  return out;
}

// Brightness-weighted column of the peak and its immediate neighbours, so the
// reported position moves between cells instead of in whole-cell jumps. Without
// it any speed derived from the position is quantised at 219 deg/s per step and
// the whole 5-200 deg/s band falls between two of its values.
function centroidColumn(plane, n, peakColumn, width = 1) {
  const lo = Math.max(peakColumn - width, 0);
  const hi = Math.min(peakColumn + width + 1, n);
  let total = 0, acc = 0;
  for (let c = lo; c < hi; c++) {
    let wsum = 0;
    for (let r = 0; r < n; r++) wsum += plane[r * n + c];
    total += wsum; acc += wsum * c;
  }
  return total <= 0 ? peakColumn : acc / total;
}

// ----------------------------------------------------------------- retina --

export class Retina {
  constructor(cfg) {
    const e = cfg.eye;
    this.fovyDeg = e.fovy_deg;
    this.renderPx = e.render_pixels;
    this.pixels = e.pixels;
    this.cells = e.cells;
    this.kept = e.kept_channels;           // green and blue; red is dropped
    this.surroundRatio = e.surround_ratio;
    this.motionWindow = Math.max(1, e.motion_window);
    this.opticalLimitPx = e.optical_limit_px;
    const half = (this.fovyDeg * Math.PI) / 180 / 2;
    this.focalPx = this.pixels / 2 / Math.tan(half);
    this.reset();
  }
  reset() {
    this.previous = null;
    this.frames = [];                      // luminance ring, oldest first
    return this;
  }

  // Signed horizontal angle of a (possibly fractional) cell column, degrees.
  // Negative is left of the optical axis. Through the actual projection, not by
  // linear interpolation across the field.
  azimuthOf(column) {
    const centrePx = (column + 0.5) * (this.pixels / this.cells) - this.pixels / 2;
    return (Math.atan(centrePx / this.focalPx) * 180) / Math.PI;
  }
  // Signed vertical angle. Row 0 is the TOP of the frame, so the sign comes out
  // negative for things on the ground, which is where a cricket is.
  elevationOf(row) {
    const centrePx = (row + 0.5) * (this.pixels / this.cells) - this.pixels / 2;
    return -(Math.atan(centrePx / this.focalPx) * 180) / Math.PI;
  }

  // uint8 RGBA or RGB, renderPx square -> luminance at the receptor grid.
  // Red is dropped: this animal's longest-wavelength pigment sits near 521 nm,
  // so a detector that separates prey from floor on red-versus-green is solving
  // a problem the animal cannot solve.
  _receptors(image, stride = 4) {
    const R = this.renderPx, P = this.pixels, K = this.kept;
    const nk = K.length;
    // planes at render resolution, kept channels only, scaled to [0, 1]
    let planes = [];
    for (let ci = 0; ci < nk; ci++) {
      const ch = K[ci], pl = new Float64Array(R * R);
      for (let i = 0; i < R * R; i++) pl[i] = image[i * stride + ch] / 255;
      planes.push(pl);
    }
    if (R !== P) {
      // THE OPTICS, then the sampling, in that order. Sampling first would
      // alias detail the animal cannot resolve into false structure that moves
      // when the animal moves -- indistinguishable from prey to a motion
      // detector.
      const k = R / P;
      planes = planes.map((pl) => {
        const b = blur(pl, R, this.opticalLimitPx);
        const out = new Float64Array(P * P);
        for (let r = 0; r < P; r++) {
          for (let c = 0; c < P; c++) {
            let s = 0;
            for (let dy = 0; dy < k; dy++)
              for (let dx = 0; dx < k; dx++) s += b[(r * k + dy) * R + (c * k + dx)];
            out[r * P + c] = s / (k * k);
          }
        }
        return out;
      });
    }
    const lum = new Float64Array(P * P);
    for (let i = 0; i < P * P; i++) {
      let s = 0;
      for (let ci = 0; ci < nk; ci++) s += planes[ci][i];
      lum[i] = s / nk;
    }
    return lum;
  }

  // Pool to the cell grid. Uniform: every cell gets the same count, because
  // strictly nocturnal geckos completely lack foveae.
  _bin(plane) {
    const P = this.pixels, C = this.cells, k = P / C;
    const out = new Float64Array(C * C);
    for (let r = 0; r < C; r++) {
      for (let c = 0; c < C; c++) {
        let s = 0;
        for (let dy = 0; dy < k; dy++)
          for (let dx = 0; dx < k; dx++) s += plane[(r * k + dy) * P + (c * k + dx)];
        out[r * C + c] = s / (k * k);
      }
    }
    return out;
  }

  step(image, stride = 4) {
    const P = this.pixels, C = this.cells;
    const luminance = this._receptors(image, stride);

    const temporal = new Float64Array(P * P);
    if (this.previous) for (let i = 0; i < P * P; i++) temporal[i] = luminance[i] - this.previous[i];
    this.previous = luminance;

    this.frames.push(luminance);
    if (this.frames.length > this.motionWindow + 1) this.frames.shift();
    let windowed = new Float64Array(P * P), span = 0;
    if (this.frames.length >= 2) {
      const first = this.frames[0];
      for (let i = 0; i < P * P; i++) windowed[i] = luminance[i] - first[i];
      span = this.frames.length - 1;
    }

    const absT = new Float64Array(P * P), absW = new Float64Array(P * P);
    for (let i = 0; i < P * P; i++) { absT[i] = Math.abs(temporal[i]); absW[i] = Math.abs(windowed[i]); }

    const centre = this._bin(luminance);
    const sur = blur(centre, C, this.surroundRatio);
    const contrast = new Float64Array(C * C);
    for (let i = 0; i < C * C; i++) contrast[i] = centre[i] - sur[i];

    return {
      motion: this._bin(absT),
      motion_windowed: this._bin(absW),
      motion_window_span: span,
      luminance: centre,
      contrast,
      pixel_luminance: luminance,
      dt_s: 0,
    };
  }
}

// ----------------------------------------------------------------- tectum --

export class Tectum {
  constructor(cfg, retina) {
    const e = cfg.eye;
    this.retina = retina;
    this.motionWindow = Math.max(1, e.motion_window);
    this.motionFloor = e.motion_floor;
    this.maxTargetFraction = e.max_target_fraction;
    this.surroundCells = e.surround_cells;
    this.salienceScale = e.salience_scale;
    this.flowGainYaw = e.flow_gain_yaw;
    this.flowGainSurge = e.flow_gain_surge;
    this.speedWindow = e.speed_window;
    this.minResolvableDeg = e.min_resolvable_deg;
    this.scaleEfferenceBySpan = e.scale_efference_by_span;
    this.vPeak = e.v_peak; this.vLow = e.v_low; this.vHigh = e.v_high;
    this.subtractField = true;
    this.efferenceCopy = true;
    this.velocityBand = true;
    this.elevationSwitch = true;
    this.history = [];
    this.last = { salience: 0, bearing_deg: 0, elevation_cell: null };
  }
  reset() { this.history = []; this.last = { salience: 0, bearing_deg: 0, elevation_cell: null }; return this; }

  // 1.0 at the published peak, zero at both published cuts. The SHAPE between
  // them is invented -- no tuning curve has been published for any lizard --
  // but 45, 5 and 200 deg/s are not.
  velocityGain(degPerS) {
    const v = Math.abs(degPerS);
    if (!(this.vLow < v && v < this.vHigh)) return 0;
    if (v <= this.vPeak) return (v - this.vLow) / Math.max(this.vPeak - this.vLow, EPS);
    return (this.vHigh - v) / Math.max(this.vHigh - this.vPeak, EPS);
  }

  // Below the horizon is food; above it is not. A hard switch, because that is
  // what was measured. The gaze angle is subtracted first: pitching the head
  // down to keep a cricket in view raises that cricket above the optical axis,
  // and a frame-referenced test then rejects the prey the animal just aimed at.
  elevationGain(row, rows, gazePitchDeg = 0) {
    if (rows < 2) return 1;
    return this.retina.elevationOf(row) - gazePitchDeg <= 0 ? 1 : 0;
  }

  // The retinal motion the animal's own movement should produce: a constant
  // from yaw, plus an expansion from surge that is strongest at the edges.
  // Crude, declared, and the reason the eye can tell a world with prey from a
  // world without.
  _expectedFlow(n, selfMotion) {
    const yaw = Math.abs(selfMotion.yaw_rate_deg_s || 0);
    const surge = Math.abs(selfMotion.forward_m_s || 0);
    const flat = this.flowGainYaw * yaw;
    const centre = (n - 1) / 2;
    const row = new Float64Array(n);
    for (let c = 0; c < n; c++) {
      const radial = Math.abs(c - centre) / Math.max(centre, EPS);
      row[c] = Math.max(0, flat + this.flowGainSurge * surge * radial);
    }
    return row;                       // same for every row
  }

  step(ret, selfMotion = null) {
    const n = this.retina.cells;
    let motion, span = 1;
    if (this.motionWindow > 1 && ret.motion_windowed) {
      motion = ret.motion_windowed.slice();
      span = Math.max(1, ret.motion_window_span || 1);
    } else {
      motion = ret.motion.slice();
    }
    if (selfMotion && this.efferenceCopy) {
      const row = this._expectedFlow(n, selfMotion);
      const k = this.scaleEfferenceBySpan ? span : 1;
      for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++)
          motion[r * n + c] = Math.max(0, motion[r * n + c] - row[c] * k);
    }
    let rawPeak = 0;
    for (let i = 0; i < motion.length; i++) if (motion[i] > rawPeak) rawPeak = motion[i];

    // ORDER MATTERS. "Is this a whole-field event?" is asked of the RAW motion,
    // before anything is subtracted. Asking it afterwards lets self-motion
    // through: 0.066 salience on pure self-motion against 0.000 this way round.
    if (rawPeak > this.motionFloor) {
      let over = 0;
      for (let i = 0; i < motion.length; i++) if (motion[i] > 0.5 * rawPeak) over++;
      if (over / motion.length > this.maxTargetFraction) {
        this.last = { salience: 0, bearing_deg: 0, elevation_cell: null, rejected: "fills the field" };
        return [0, 0];
      }
    }

    let local;
    if (this.subtractField) {
      const sur = surround(motion, n, this.surroundCells);
      local = new Float64Array(n * n);
      for (let i = 0; i < motion.length; i++) local[i] = Math.max(0, motion[i] - sur[i]);
    } else {
      local = motion.slice();
      for (let i = 0; i < local.length; i++) local[i] = Math.max(0, local[i]);
    }

    let peak = 0, best = 0;
    for (let i = 0; i < local.length; i++) if (local[i] > peak) { peak = local[i]; best = i; }
    this.peakLocal = peak;
    this.localMap = local;
    if (peak <= this.motionFloor) {
      this.last = { salience: 0, bearing_deg: 0, elevation_cell: null, peak_local: peak };
      return [0, 0];
    }
    const row = Math.floor(best / n), column = best % n;
    const bearing = this.retina.azimuthOf(column);

    let rejected = null, gain = 1, speed = null;
    const dt = ret.dt_s || 0;
    const fine = this.retina.azimuthOf(centroidColumn(local, n, column));
    this.history.push([fine, dt]);
    if (this.history.length > this.speedWindow) this.history.shift();
    if (this.history.length >= 2) {
      let sp = 0;
      for (let i = 1; i < this.history.length; i++) sp += this.history[i][1];
      const travelled = Math.abs(this.history[this.history.length - 1][0] - this.history[0][0]);
      // REFUSED rather than guessed when the target has not moved a full cell.
      // A refused estimate does not veto the target: treating "not yet
      // resolvable" as "too slow" would blind the animal for the first half
      // second of every encounter.
      if (sp > 0 && travelled >= this.minResolvableDeg) speed = travelled / sp;
    }
    if (this.velocityBand && speed !== null) {
      const g = this.velocityGain(speed);
      gain *= g;
      if (g <= 0) rejected = speed <= this.vLow ? "too slow" : "too fast";
    }
    if (this.elevationSwitch) {
      const gaze = selfMotion ? (selfMotion.gaze_pitch_deg || 0) : 0;
      const g = this.elevationGain(row, n, gaze);
      gain *= g;
      if (g <= 0) rejected = "above the horizon";
    }
    if (gain <= 0) {
      this.last = { salience: 0, bearing_deg: 0, elevation_cell: row,
                    rejected, peak_speed_deg_s: speed };
      return [0, 0];
    }
    // The ABSOLUTE local excess against a fixed scale. A ratio to the frame's
    // own maximum cannot detect anything: when the whole field slides, the
    // quotient parks at a constant, and an empty world scored 0.4034 against a
    // world with prey at 0.4161.
    const salience = Math.min(peak / this.salienceScale, 1) * gain;
    this.last = {
      salience, bearing_deg: bearing, elevation_cell: row,
      // Python stores this one ROUNDED TO 2 DECIMALS in `last`, and the Eye
      // reads `last` for elevation while reading the raw return value for
      // salience and bearing. Copied rather than cleaned up, because a port
      // that is tidier than its original is a port that has drifted. `toFixed`
      // and Python's `round` agree except on an exact decimal tie, which an
      // arctangent does not produce; over the 140 conformance frames they
      // agree at exactly 0.
      elevation_deg: Number(this.retina.elevationOf(row).toFixed(2)),
      velocity_gain: gain, peak_speed_deg_s: speed, rejected: null,
    };
    return [salience, bearing];
  }
}

// -------------------------------------------------------------------- eye --

export class Eye {
  constructor(cfg) {
    this.retina = new Retina(cfg);
    this.tectum = new Tectum(cfg, this.retina);
    this.last = {};
    // The PRETECTUM is not ported. It computes the optokinetic reflex -- a gaze
    // command that stabilises the image -- and the three results it reproduces
    // are published gains for this species. It is absent here rather than
    // approximated, and nothing in the browser reads a gaze command, so the
    // head is steered only by the search pattern and by the tectum's bearing.
    this.gaze_command_deg_s = null;
  }
  reset() { this.retina.reset(); this.tectum.reset(); this.last = {}; return this; }

  step(image, dtS, selfMotion = null, stride = 4) {
    const ret = this.retina.step(image, stride);
    ret.dt_s = dtS;
    const [salience, bearing] = this.tectum.step(ret, selfMotion);
    this.retinaOut = ret;
    this.last = {
      prey_salience: salience,
      // None when nothing was reported, NOT 0.0. A bearing of 0 is a perfectly
      // good bearing -- it means directly ahead -- so returning it for "I saw
      // nothing" made silence indistinguishable from a target dead in front.
      prey_bearing_deg: salience > 0 ? bearing : null,
      prey_elevation_deg: salience > 0 ? this.tectum.last.elevation_deg : null,
      rejected: this.tectum.last.rejected || null,
    };
    return this.last;
  }
}
