// The live gecko: real MuJoCo physics in the browser, driven by the ported brain.
//
// WHAT IS REAL HERE. The physics is the same MuJoCo engine (official
// WebAssembly build) running morphology/gecko_body_web.xml -- the certified
// body with visual meshes stripped, verified physics-neutral at 400 steps,
// maximum joint difference exactly 0. The walker, the basal ganglia and the
// clock are in gecko.js and are checked against Python by
// tools/conformance_web.py: walker 7.3e-15, gates and clock exactly 0.
//
// WHAT IS DEPICTION, AND SAYS SO. In the lower view, colour and brightness are
// mapped from real numbers -- joint speed, the released channel, the gate
// value, arousal -- but the CHOICE of colour, and the pulse that travels down
// the spine when the released behaviour changes, are a drawing. No conduction
// velocity has ever been measured in this animal, and none is claimed here.

import { Walker, BasalGanglia, Clock, salienceFromDrives, SearchPattern, FEET } from "./gecko.js";
import { Eye } from "./eye.js";
import { FixationEvidence, MotorPrograms, Prey } from "./hunt.js";

const THREE = window.THREE;
const CTRL_HZ = 50, PHYS_PER_CTRL = 10;

// mjtGeom
const SPHERE = 2, CAPSULE = 3, ELLIPSOID = 4, BOX = 6;

const CH_COLOUR = {
  hunt: 0xc9705f, flee: 0xd14b45, explore: 0x7fa8b8,
  bask: 0xe3a94a, rest: 0xc79bc2, groom: 0x7a6c5b,
};

// A room the animal can actually live in. Injected into the model XML before
// compile: a floor is already there, so this adds four walls and a ceiling
// light. Dimensions are a stage, not a measurement.
function withRoom(xml) { return xml; }   // the enclosure is gone, see views.js

//: How much of the animal's hunger one cricket removes. INVENTED as a display
//: choice: the project's energetics live in brain/hypothalamus.py and are not
//: ported here, so rather than half-port them this is declared as what it is.
const MEAL_FRACTION = 0.45;
//: Control steps of head-shaking after a swallow. INVENTED. The behaviour is
//: real and named in the published ethogram for this species; its duration is
//: not measured anywhere and this number is a drawing.
const CHEW_STEPS = 90;
//: How much faster the day/night clock runs while the animal is asleep. A
//: VIEWING choice, applied to the clock and to nothing else. INVENTED, and it
//: changes no part of the model -- the arousal curve, the release floor and the
//: selector are untouched, so the animal still sleeps for exactly as many of
//: its own hours as it did before.
const SLEEP_FAST_FORWARD = 14;

// ---------------------------------------------------------------- posture --
// HOW A GECKO TURNS TO LOOK AT SOMETHING, and it is not how a person does it.
// A person keeps the trunk still and rotates the neck. A lizard bends: the
// pelvis stays planted, the trunk curves through `spine_bend` -- which in this
// morphology is a tendon over spine_lat_1, spine_lat_2 and spine_lat_3 at
// coefficients 1.0, 1.0 and 0.8, so the bend is strongest at the front -- and
// the neck and head carry the last of the angle. Then it walks.
//
// The ORDER is the thing: bend, then look, then go. Walking while the trunk is
// still swinging is what made the old render read as a person with a long neck.
//
// EVERY NUMBER IN THIS BLOCK IS INVENTED. No lateral trunk excursion, bend
// rate, or bend-before-step latency has been published for this species or any
// eublepharid. What is published is that the joints exist and how far they go,
// and that is the morphology, not this. The SHAPE is from photographs of the
// animal; the timings are a controller.
const TRUNK_BEND_SHARE = 0.55;    // of a turn carried by the trunk, not the neck
const TRUNK_BEND_MAX = 0.85;      // rad, inside the tendon's own +-1.2
const TRUNK_BEND_RATE = 2.6;      // rad/s toward the commanded bend
const NECK_LAG_STEPS = 14;        // the neck starts after the trunk has begun
const SETTLE_BEND_RAD = 0.12;     // bend is "done" below this much error
const BEND_HOLD_MIN = 18;         // control steps held still before walking on
//: Jaw. Closed is 0 and the morphology's neutral is 0.349 -- half open, which
//: is why it stood there with its mouth ajar forever. A resting gecko's mouth
//: is SHUT. It gapes to swallow and it gapes occasionally at rest, and both the
//: amplitude and the interval here are INVENTED: the published ethogram names
//: the behaviour and measures neither.
const JAW_SHUT = 0.0;
const JAW_GAPE = 0.62;
const JAW_RATE = 5.0;             // rad/s
//: THE CREEP. `flee_radius_for` in envs/prey.py exists for exactly this: the
//: cricket's flight distance scales with how fast the threat is CLOSING, from
//: 7.5 cm at a charge down to a 1.2 cm floor. Measured here, the animal walked
//: at its full stride, the cricket read that as a charge and bolted at 6.8 cm,
//: and the gecko -- which is outrun 2:1 -- could never close the last 3 cm.
//:
//: The behaviour that answers it is PUBLISHED and named in the ethogram for
//: this species: `walk slow motion`, "walking with a strongly reduced speed,
//: mostly in context of prey capture". What is NOT published is how much
//: slower, so this number is INVENTED and says so. It slows the stride clock
//: only -- the gait, the compensator and the accepted walker are untouched.
const CREEP_GAIT_SCALE = 0.34;

export class LiveGecko {
  constructor(cfg, mj, model, data) {
    this.cfg = cfg; this.mj = mj; this.m = model; this.d = data;
    this.walker = new Walker(cfg);
    this.bg = new BasalGanglia(cfg);
    this.clock = new Clock(cfg);
    this.search = new SearchPattern(cfg);
    // THE EYE, AND THE THING IT HAS TO FIND. `setRetinaSource` hands the sim a
    // function that renders one head-camera frame; until then the eye is idle
    // and `hunt` cannot release, which is the honest state rather than a
    // silently faked one.
    this.eye = new Eye(cfg);
    this.evidence = new FixationEvidence(cfg);
    this.programs = new MotorPrograms(cfg, this.search, this.evidence);
    this.prey = new Prey(cfg);
    this.renderRetina = null;
    this.preySalience = 0;
    this.preyBearing = null;
    this.committed = null;
    this.believedEye = false;
    this.captures = 0;
    this.sawCricket = 0;        // frames the tectum reported something
    this.eyeFrames = 0;
    this.nAct = cfg.walker.n_act;

    this.gaitT = 0;            // the walker's own clock, advanced only when walking
    this.compression = 900;    // simulated seconds per real second, for the day
    this.timeOfDayH = 9.5;
    this.clock.t = this.timeOfDayH * 3600;

    this.hunger = 0.9;
    this.bodyC = 30.5;
    this.behaviour = null;
    this.gates = new Float64Array(cfg.bg.channels.length);
    this.salience = new Float64Array(cfg.bg.channels.length);
    this.arousal = this.clock.arousalAt(this.clock.t);
    this.asleep = this.arousal < 0.1;
    this.eyelid = 0;
    this.pulse = 0;            // depiction: decision travelling down the body
    this.signal = -1;          // position of that band, snout 0 -> tail 1
    this.signalStrength = 0;
    this.speed = 0;
    this.heading = 0;          // trunk yaw, so the side/front cameras track it
    this.headYaw = 0;          // where the head is pointed, relative to the trunk
    this.searchState = "scan";
    this.looking = false;
    this.program = "still";
    this.drive = 0;
    this.onWarm = false;
    this.headYawDeg = 0;
    this.lastCapture = -1e9;
    this.chewT = 0;             // the head-shake after a swallow
    this._prevHeading = 0;
    this.trunkBend = 0;         // rad, what the spine tendon is actually at
    this.wantBend = 0;
    this.bendHold = 0;
    this.neckLag = 0;
    this.jaw = JAW_SHUT;
    this.jawWant = JAW_SHUT;
    this.gapeT = 0;
    this.posture = "settled";
    this.assist = true;         // see `assisted()` -- declared on the page
    this.distance = 0;         // ground actually covered, metres
    this._prevXY = [0, 0];

    // NERVE TRAFFIC. Not a timer: how far the brain's command to each joint
    // has actually moved since the renderer last looked. A joint held still
    // produces exactly zero, a joint being driven hard produces a lot, and
    // the view turns that into impulses. Accumulated at control rate so it
    // stays exact however many control steps a rendered frame contains.
    this.simT = 0;
    this.nerveDrive = new Float64Array(this.nAct);
    this._prevCtrl = new Float64Array(this.nAct);
  }

  // Where the animal's nose is, in world coordinates. The cricket is measured
  // against this and not against the trunk: 4 cm of capture distance is most of
  // a head, and using the body centre would let the animal eat through its own
  // neck.
  _snoutXY() {
    if (this._headBody === undefined) {
      this._headBody = Math.max(1, this.cfg.body_names.indexOf("head"));
    }
    // THE SNOUT TIP, not the head body's origin. The origin sits inside the
    // skull, about 3 cm behind the mouth, and `prey_capture_distance_m` is
    // 4.07 cm -- so measuring from the origin made the animal effectively three
    // centimetres shorter than it is and put most of its own head inside the
    // capture radius it could never reach. Same offset the eye uses, out of the
    // morphology's `head_cam`.
    const b = this._headBody * 3, m = this._headBody * 9;
    return [this.d.xpos[b] + this.d.xmat[m] * 0.02958,
            this.d.xpos[b + 1] + this.d.xmat[m + 3] * 0.02958];
  }

  // Hand the sim something that renders one head-camera frame as RGBA bytes.
  setRetinaSource(fn) { this.renderRetina = fn; return this; }

  // The true bearing to the cricket, but ONLY when it is really inside the
  // animal's own field of view and range. Returns null otherwise, which is most
  // of the time -- measured, the cricket is in frame on about 11 % of steps.
  // Used only by guided mode, which the page labels.
  _trueBearingIfVisible() {
    const d = this.d, hb = this.headBodyId ?? 4, p = hb * 3, m = hb * 9;
    const fx = d.xmat[m], fy = d.xmat[m + 3], fz = d.xmat[m + 6];
    const ux = d.xmat[m + 2], uy = d.xmat[m + 5], uz = d.xmat[m + 8];
    const ex = d.xpos[p] + fx * 0.02958 + ux * 0.00682;
    const ey = d.xpos[p + 1] + fy * 0.02958 + uy * 0.00682;
    const ez = d.xpos[p + 2] + fz * 0.02958 + uz * 0.00682;
    const dx = this.prey.x - ex, dy = this.prey.y - ey, dz = 0.007 - ez;
    const L = Math.hypot(dx, dy, dz);
    if (L > this.prey.arenaRadius) return null;                 // out of the arena
    const off = Math.acos(Math.max(-1, Math.min(1, (dx * fx + dy * fy + dz * fz) / L)));
    if (off > (this.cfg.eye.fovy_deg / 2) * Math.PI / 180) return null;   // off screen
    // signed bearing about the head's own up axis, the convention the tectum uses
    const lx = d.xmat[m + 1], ly = d.xmat[m + 4], lz = d.xmat[m + 7];
    const fwd = dx * fx + dy * fy + dz * fz;
    const lat = dx * lx + dy * ly + dz * lz;
    return (Math.atan2(-lat, fwd) * 180) / Math.PI;
  }

  // One 50 Hz control step: eye -> drives -> selector -> motor program -> body.
  controlStep() {
    const dt = 1 / CTRL_HZ;
    // FAST-FORWARDING THE NIGHT, AND ONLY THE CLOCK.
    //
    // This animal is crepuscular and it really does spend most of a day asleep:
    // measured over 6.3 simulated days, 19,771 control steps resting against
    // 5,721 exploring and 4,492 basking. That is the animal, it comes from a
    // published arousal curve, and it is not going to be edited to make a nicer
    // web page -- editing it would be tuning a model until it looked good.
    //
    // What IS a viewing choice is how fast the viewer's clock runs. While the
    // animal is asleep the day advances faster, the way a nature film cuts the
    // night. Nothing else changes: the physics timestep, the walker, the
    // thermostat, the eye and the cricket all run at the same rate they always
    // did. The animal sleeps exactly as long; you just do not sit through it.
    const fast = this.asleep && this.behaviour === "rest" ? SLEEP_FAST_FORWARD : 1;
    this.timeScale = fast;
    const c = this.clock.step(dt * this.compression * fast);
    this.arousal = c.arousal; this.asleep = c.asleep; this.timeOfDayH = c.timeOfDayH;

    // THE THERMOSTAT, and it is the whole reason this animal has more than one
    // thing to do. This species is THIGMOTHERMIC -- it takes heat from the
    // GROUND by lying on it, not from light (Hastings et al. 2023, body temp
    // tracked substrate at r2 = 0.97 against 0.92 for air). So the world has a
    // warm patch of ground, and standing on it is what warms the animal.
    // Every constant here is read from brain.json, straight off
    // GeckoBrainEnv, and the patch is lifted out of the project's own
    // furnished world: 0.14 m across at (0.28, -0.2), surface 30 C.
    const W = this.cfg.world;
    const prefLow = this.cfg.drives.preferred_temperature_C[0];
    this.onWarm = Math.abs(this.d.xpos[3] - W.warm_patch_xy[0]) <= W.warm_patch_half_m
               && Math.abs(this.d.xpos[4] - W.warm_patch_xy[1]) <= W.warm_patch_half_m;
    const substrate = this.onWarm ? W.warm_surface_C : W.ambient_substrate_C;
    const tau = W.thermal_tau_s / W.thermal_time_compression;
    this.bodyC += (substrate - this.bodyC) * (1 - Math.exp(-dt / tau));
    const cold = Math.max(0, Math.min(1, (prefLow - this.bodyC) / 6));

    // GETTING HUNGRY AGAIN. Not an invented decay: hunger in this model is a
    // deficit measured in MEALS, and the published mean inter-meal interval for
    // this species is 2.33 days, over which brain/hypothalamus.py has the animal
    // burn about 71 % of one meal's energy. So the rate is 0.71 meals per 2.33
    // days and it is read off the registry, not chosen. Without it the animal
    // ate once and then had nothing it wanted for the rest of its life --
    // `explore` needs hunger above 0.571 to clear the release floor at all.
    this.hunger = Math.min(1, this.hunger + dt * this.compression * fast
      * (W.meal_burn_fraction / (W.inter_meal_interval_days * 86400)));

    // ---- THE EYE. One rendered frame, and whatever it can find in it.
    // Nothing here is handed the cricket's position: `renderRetina` returns
    // pixels, and `Eye.step` either finds a moving thing below the horizon or
    // does not. The efference copy is the animal's own motor state, which is
    // what lets it tell a cricket from its own walking.
    if (this.renderRetina) {
      const px = this.renderRetina();
      if (px) {
        const out = this.eye.step(px, dt, {
          yaw_rate_deg_s: ((this.heading - this._prevHeading) / dt) * 180 / Math.PI,
          forward_m_s: this.speed,
          gaze_pitch_deg: 0,
        }, 4);
        this.preySalience = out.prey_salience;
        this.preyBearing = out.prey_bearing_deg;
        this.eyeFrames++;
        if (out.prey_salience > 0) this.sawCricket++;
        this.eyeSaidSomething = out.prey_bearing_deg !== null;
      }
    }
    this._prevHeading = this.heading;

    // ---------------------------------------------------------------------
    // GUIDED MODE. This is an ORACLE and it is labelled as one on the page,
    // in the readout, and here.
    //
    // WHAT IT DOES. When the cricket is genuinely inside the animal's own 70
    // degree field of view, the bearing handed to the evidence accumulator is
    // the true one instead of the tectum's guess. Everything else is untouched:
    // the accumulator still has to reach quorum, the orienting reflex still has
    // its blind window, the stalk still stops to look, the cricket still runs
    // away, and the capture still has to happen inside 4.07 cm. It does NOT see
    // through its own body, it does NOT see behind itself, and it is not moved
    // one millimetre closer to anything.
    //
    // WHY IT EXISTS. The real detector fires at roughly its background rate
    // whether or not there is a cricket there (#374) -- that is the model's own
    // documented weakness, measured, and the honest version of this page is
    // unwatchable because of it. So there is a switch. The other position of
    // that switch is the animal as it actually is, and it is one click away.
    //
    // WHAT IT IS NOT. It is not in `brain/`. Not one line of the Python model
    // knows this exists, the conformance suite runs against the unguided path,
    // and nothing measured anywhere in this project was measured with it on.
    // It is a rendering aid for a web page and it says so everywhere it shows.
    this.guided = false;
    if (this.assist && this.renderRetina) {
      const g = this._trueBearingIfVisible();
      if (g !== null) { this.preyBearing = g; this.guided = true; }
    }

    this.salience = salienceFromDrives(this.cfg, {
      hunger: this.hunger, cold, warm: 0, threat: 0,
      preyVisible: this.committed !== null ? 1 : 0, arousal: this.arousal,
    });
    this.bg.converge(this.salience);
    this.gates = this.bg.gates();
    const next = this.bg.selected();
    if (next !== this.behaviour) { this.signal = 0; this.signalStrength = 1; }
    this.behaviour = next;
    // The band runs head to tail once per release, then fades. Real path,
    // drawn timing -- see views.js.
    if (this.signal >= 0) {
      this.signal += dt * 1.9;
      if (this.signal > 1.15) { this.signal = -1; this.signalStrength = 0; }
    }

    // MOTOR PROGRAM -- brain/programs.py, ported whole rather than improvised.
    // A released behaviour is not a gait. `explore` runs SEARCH, which stops
    // and sweeps the head. `bask` runs WARM, which walks to warm ground and
    // then lies down on it. `hunt` runs CHASE, which is a stalk: walk a little,
    // stop and look, walk a little more -- and it only runs at all once the
    // evidence accumulator has decided that what the eye keeps reporting is one
    // object rather than noise.
    const warmBearing = (() => {
      const dx = this.cfg.world.warm_patch_xy[0] - this.d.xpos[3];
      const dy = this.cfg.world.warm_patch_xy[1] - this.d.xpos[4];
      let b = Math.atan2(dy, dx) - this.heading;
      b = ((b + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      return (b * 180) / Math.PI;
    })();

    const cmd = this.programs.step(next, {
      urgency: 1,
      bearingDeg: this.preyBearing,
      trunkYawDeg: (this.heading * 180) / Math.PI,
      warmBearingDeg: warmBearing,
      shelterBearingDeg: null,          // no refuge in this world, and none faked
      onWarmGround: this.onWarm,
    });
    this.program = cmd.program;
    this.searchState = cmd.search_state;
    this.looking = cmd.believed_eye;
    this.committed = cmd.committed_bearing_deg;
    this.believedEye = cmd.believed_eye;
    this.headYaw = cmd.head_yaw_deg;
    let drive = cmd.locomotor_drive;
    const steer = (cmd.heading_deg * Math.PI) / 180;
    this.drive = drive;

    // ---- POSTURE. Bend, then look, then go. ---------------------------
    // The heading the program wants is split: the trunk takes most of it and
    // the neck takes the rest. While the trunk is still swinging into place the
    // animal does not walk -- that gate is the whole difference between a
    // lizard turning and a person pivoting.
    const wantDeg = Math.max(-90, Math.min(90, cmd.heading_deg));
    this.wantBend = Math.max(-TRUNK_BEND_MAX, Math.min(TRUNK_BEND_MAX,
      (wantDeg * Math.PI / 180) * TRUNK_BEND_SHARE));
    const bendErr = this.wantBend - this.trunkBend;
    const bending = Math.abs(bendErr) > SETTLE_BEND_RAD;

    if (drive > 0 && bending && this.bendHold < BEND_HOLD_MIN) {
      drive = 0;                       // stand and turn the body first
      this.bendHold += 1;
      this.posture = "bending";
    } else if (drive > 0) {
      this.posture = "walking";
      // Once it is moving, the walker owns the spine -- it needs it for the
      // gait's own lateral undulation -- so the postural bend unwinds.
      this.wantBend = 0;
    } else {
      this.bendHold = bending ? this.bendHold + 1 : 0;
      this.posture = bending ? "bending" : "settled";
    }
    this.drive = drive;
    const bendStep = TRUNK_BEND_RATE * dt;
    this.trunkBend += Math.max(-bendStep, Math.min(bendStep, this.wantBend - this.trunkBend));
    // The neck follows the trunk rather than leading it.
    this.neckLag = bending ? Math.min(NECK_LAG_STEPS, this.neckLag + 1)
                           : Math.max(0, this.neckLag - 2);
    const neckShare = this.neckLag / NECK_LAG_STEPS;

    // Creep while closing on something it has committed to.
    this.creeping = this.program === "chase" && this.committed !== null;
    let ctrl;
    if (drive > 0) {
      this.gaitT += dt * (this.creeping ? CREEP_GAIT_SCALE : 1);
      ctrl = this.walker.baseCtrl(this.gaitT, steer);
    } else {
      ctrl = Float64Array.from(this.cfg.walker.standing_ctrl);
    }

    // A sleeping gecko shuts its eyes (#356): published for this species --
    // Bergel et al. 2026 recorded sleep with electrodes UNDER EACH EYELID.
    // The angle reuses the declared-invented EYELID_CLOSED_DEG.
    // Gated on the locomotor drive, the same quantity envs/gecko_brain_env.py
    // gates it on: an animal that is walking is not asleep.
    const shut = drive > 0 ? 0 : (1 - Math.min(1, Math.max(0, this.arousal)));
    this.eyelid = shut * this.cfg.eyelid_closed_deg;

    const dd = this.d;
    for (let i = 0; i < this.nAct; i++) dd.ctrl[i] = ctrl[i];
    // eyelid actuators, by name, if present
    if (this._lidIds === undefined) {
      const names = this.cfg.walker.actuator_names;
      this._lidIds = [names.indexOf("eyelid_L"), names.indexOf("eyelid_R")];
    }
    const rad = (this.eyelid * Math.PI) / 180;
    if (this._lidIds[0] >= 0) dd.ctrl[this._lidIds[0]] = -rad;
    if (this._lidIds[1] >= 0) dd.ctrl[this._lidIds[1]] = +rad;

    // THE HEAD ACTUALLY TURNS. envs/gecko_walk_env.py spreads a gaze command
    // across neck_yaw and head_yaw in proportion to their ranges; same here.
    if (this._neckIds === undefined) {
      const nm = this.cfg.walker.actuator_names;
      this._neckIds = [nm.indexOf("neck_yaw"), nm.indexOf("head_yaw")];
      const hi = this.cfg.walker.ctrl_high, lo = this.cfg.walker.ctrl_low;
      this._neckSpan = this._neckIds.map((i) => (i >= 0 ? hi[i] - lo[i] : 0));
    }
    const total = this._neckSpan[0] + this._neckSpan[1];
    if (total > 1e-9) {
      // THE SHAKE AFTER A SWALLOW. Named in the published ethogram for this
      // species; its amplitude and duration are not measured anywhere, and both
      // numbers here are a drawing. It is added to whatever the head was already
      // being asked to do rather than overriding it.
      const shake = this.chewT > 0
        ? 26 * Math.sin(this.simT * 34) * (this.chewT / CHEW_STEPS) : 0;
      // The neck takes whatever the trunk did not, and only once the trunk has
      // started -- `neckShare` ramps in over NECK_LAG_STEPS.
      const carry = (cmd.heading_deg - (this.trunkBend * 180 / Math.PI)) * 0.45 * neckShare;
      const want = ((this.headYaw + shake + carry) * Math.PI) / 180;
      const hi = this.cfg.walker.ctrl_high, lo = this.cfg.walker.ctrl_low;
      for (let k = 0; k < 2; k++) {
        const i = this._neckIds[k];
        if (i < 0) continue;
        dd.ctrl[i] = Math.min(hi[i], Math.max(lo[i], want * (this._neckSpan[k] / total)));
      }
    }

    // THE TRUNK BENDS. Standing still, the postural bend is added straight on
    // to the spine tendon; walking, the walker's own gait undulation is left
    // exactly as it is, because that is the accepted walker and it is not
    // something to improvise over.
    if (this._spineId === undefined) {
      this._spineId = this.cfg.walker.actuator_names.indexOf("spine_bend");
      this._jawId = this.cfg.walker.actuator_names.indexOf("jaw");
      this._tailIds = ["tail_bend_L", "tail_bend_R"].map(
        (n) => this.cfg.walker.actuator_names.indexOf(n));
    }
    if (this._spineId >= 0 && drive <= 0) {
      const lo = this.cfg.walker.ctrl_low[this._spineId];
      const hi = this.cfg.walker.ctrl_high[this._spineId];
      dd.ctrl[this._spineId] = Math.max(lo, Math.min(hi, this.trunkBend));
      // The tail counterbalances the bend, which is what a tail that is a third
      // of the animal's mass is for. Sign and share are INVENTED.
      for (let k = 0; k < 2; k++) {
        const i = this._tailIds[k];
        if (i < 0) continue;
        const tl = this.cfg.walker.ctrl_low[i], th = this.cfg.walker.ctrl_high[i];
        dd.ctrl[i] = Math.max(tl, Math.min(th, -this.trunkBend * (k ? -0.45 : 0.45)));
      }
    }

    // THE MOUTH. Shut, unless it is swallowing or taking a breath-gape.
    if (this._jawId >= 0) {
      if (this.chewT > 0) this.jawWant = JAW_GAPE;
      else if (this.gapeT > 0) { this.jawWant = JAW_GAPE * 0.55; this.gapeT -= 1; }
      else {
        this.jawWant = JAW_SHUT;
        // an occasional gape while awake. Interval INVENTED.
        if (!this.asleep && Math.random() < 0.0009) this.gapeT = 40;
      }
      const js = JAW_RATE * dt;
      this.jaw += Math.max(-js, Math.min(js, this.jawWant - this.jaw));
      const jl = this.cfg.walker.ctrl_low[this._jawId];
      const jh = this.cfg.walker.ctrl_high[this._jawId];
      dd.ctrl[this._jawId] = Math.max(jl, Math.min(jh, this.jaw));
    }

    for (let i = 0; i < this.nAct; i++) {
      this.nerveDrive[i] += Math.abs(dd.ctrl[i] - this._prevCtrl[i]);
      this._prevCtrl[i] = dd.ctrl[i];
    }
    this.simT += dt;

    // THE CRICKET. Advanced against the animal's SNOUT, not its centre, and it
    // reads only the closing speed -- how fast the threat is actually getting
    // nearer -- which is what makes a creep different from a charge and a stalk
    // possible at all.
    const sn = this._snoutXY();
    const pr = this.prey.step(dt, sn);
    if (pr.captured) {
      this.captures += 1;
      this.lastCapture = this.simT;
      this.chewT = CHEW_STEPS;
      // A meal. Hunger falls and the cricket that replaced it is somewhere new.
      this.hunger = Math.max(0, this.hunger - MEAL_FRACTION);
      this.evidence.reset();
      this.eye.reset();
    }
    if (this.chewT > 0) this.chewT -= 1;

    for (let k = 0; k < PHYS_PER_CTRL; k++) this.mj.mj_step(this.m, this.d);

    const x = this.d.xpos[3], y = this.d.xpos[4];
    const step = Math.hypot(x - this._prevXY[0], y - this._prevXY[1]);
    this.speed = step / dt;
    this.distance += step;
    this._prevXY = [x, y];
    // trunk forward axis, from its rotation matrix
    // WHICH WAY THE ANIMAL IS POINTING. `xmat` is ROW-major, so the body's own
    // forward axis expressed in world coordinates is the first COLUMN --
    // (xmat[9], xmat[12], xmat[15]) for the trunk -- not the first row. Reading
    // the row instead returns the world x-axis in body coordinates, which for a
    // yaw is the NEGATIVE of the heading, and a negated heading steers the
    // animal away from whatever it is walking toward. That is exactly what it
    // did: released `bask`, turned its back on the warm ground and walked 7 m
    // in the wrong direction, and the thermostat never closed. Checked against
    // the Python definition, which takes yaw from the quaternion.
    this.heading = Math.atan2(this.d.xmat[12], this.d.xmat[9]);
    this.pulse *= 0.94;                                // depiction decay
  }
}

// --------------------------------------------------------------- rendering

function geomMesh(g, material) {
  const s = g.size;
  let geo;
  if (g.type === SPHERE) geo = new THREE.SphereGeometry(s[0], 14, 10);
  else if (g.type === CAPSULE) geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 12);
  else if (g.type === ELLIPSOID) {
    geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(s[0], s[1], s[2]);
  } else if (g.type === BOX) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  else geo = new THREE.SphereGeometry(0.004, 6, 4);
  return new THREE.Mesh(geo, material);
}

export class View {
  constructor(canvas, cfg, mode) {
    this.mode = mode;                       // "room" | "nerve"
    this.cfg = cfg;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.01, 12);
    this.meshes = [];
    this.gcount = cfg.geoms.length;

    if (mode === "room") {
      this.scene.background = new THREE.Color(0x141210);
      this.scene.add(new THREE.HemisphereLight(0xfff0d8, 0x2a2118, 1.5));
      const key = new THREE.DirectionalLight(0xffe9c4, 2.2);
      key.position.set(0.4, 0.5, 0.9); this.scene.add(key);
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(1.2, 1.2),
        new THREE.MeshStandardMaterial({ color: 0x6b5a44, roughness: 0.95 }));
      this.scene.add(floor);
    } else {
      this.scene.background = new THREE.Color(0x07060a);
      this.scene.add(new THREE.AmbientLight(0xffffff, 0.35));
      const grid = new THREE.GridHelper(1.1, 22, 0x243040, 0x141c26);
      grid.rotation.x = Math.PI / 2; this.scene.add(grid);
    }

    // THE ROOM VIEW DRAWS THE SKIN. The collision solids are the body the
    // physics uses; they are the right thing to show in the nerve view and the
    // wrong thing to call "the animal" (see site/skin.js).
    if (mode !== "room") {
      for (const g of cfg.geoms) {
        const mat = new THREE.MeshBasicMaterial({ color: 0x2a3a4a, transparent: true, opacity: 0.92 });
        const mesh = geomMesh(g, mat);
        this.scene.add(mesh);
        this.meshes.push(mesh);
      }
    }
    this.skin = null;
    this._q = new THREE.Quaternion();
    this._mat = new THREE.Matrix4();
  }

  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // MuJoCo geom_xmat is row-major 3x3; Three wants a Matrix4.
  setPose(mesh, xpos, xmat, i) {
    const p = i * 3, r = i * 9;
    mesh.position.set(xpos[p], xpos[p + 1], xpos[p + 2]);
    this._mat.set(
      xmat[r], xmat[r + 1], xmat[r + 2], 0,
      xmat[r + 3], xmat[r + 4], xmat[r + 5], 0,
      xmat[r + 6], xmat[r + 7], xmat[r + 8], 0,
      0, 0, 0, 1);
    mesh.quaternion.setFromRotationMatrix(this._mat);
  }

  attachSkin(skin) { this.skin = skin; this.scene.add(skin.mesh); }

  update(sim, geomIndex) {
    const d = sim.d, cfg = this.cfg;
    if (this.skin) this.skin.update(d.xpos, d.xquat);
    const trunk = [d.xpos[3], d.xpos[4], d.xpos[5]];
    const ang = this.mode === "room" ? 2.3 : 1.15;
    const dist = this.mode === "room" ? 0.44 : 0.34;
    this.camera.position.set(
      trunk[0] + dist * Math.cos(ang), trunk[1] + dist * Math.sin(ang),
      trunk[2] + (this.mode === "room" ? 0.20 : 0.13));
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(trunk[0], trunk[1], trunk[2]);

    const chCol = CH_COLOUR[sim.behaviour] ?? 0x53627a;
    for (let i = 0; i < this.meshes.length; i++) {
      const gi = geomIndex[i];
      this.setPose(this.meshes[i], d.geom_xpos, d.geom_xmat, gi);
      if (this.mode !== "nerve") continue;

      // NERVE VIEW. Brightness is real per-joint speed; hue is the released
      // channel; the travelling band is depiction and is labelled as such.
      const g = cfg.geoms[i];
      const act = sim.jointActivity[g.body] ?? 0;
      const along = sim.bodyAlong[g.body] ?? 0.5;
      const band = Math.max(0, 1 - Math.abs(along - (1 - sim.pulse)) * 5) * sim.pulse;
      const lit = Math.min(1, act * 2.6) * 0.75 + band * 0.9;
      const base = new THREE.Color(chCol);
      const col = base.clone().multiplyScalar(0.22 + 0.78 * lit);
      if (sim.asleep) col.lerp(new THREE.Color(0x6a4c86), 0.55);
      this.meshes[i].material.color.copy(col);
      this.meshes[i].material.opacity = 0.42 + 0.58 * Math.min(1, lit + 0.25);
    }
    this.renderer.render(this.scene, this.camera);
  }
}

export { withRoom, CH_COLOUR };
