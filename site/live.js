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

export class LiveGecko {
  constructor(cfg, mj, model, data) {
    this.cfg = cfg; this.mj = mj; this.m = model; this.d = data;
    this.walker = new Walker(cfg);
    this.bg = new BasalGanglia(cfg);
    this.clock = new Clock(cfg);
    this.search = new SearchPattern(cfg);
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

  // One 50 Hz control step: clock -> drives -> selector -> motor program -> body.
  controlStep() {
    const dt = 1 / CTRL_HZ;
    const c = this.clock.step(dt * this.compression);
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

    this.salience = salienceFromDrives(this.cfg, {
      hunger: this.hunger, cold, warm: 0, threat: 0,
      preyVisible: 0, arousal: this.arousal,
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

    // MOTOR PROGRAM -- brain/programs.py, ported rather than improvised.
    // A released behaviour is not a gait. The selector says `explore` and the
    // program that runs is SEARCH, which is not walking in a straight line: the
    // animal stops, sweeps its head in saccades, turns its body, walks a
    // little, and scans again. It says `bask` and the program is WARM, which
    // walks toward warm ground and then LIES DOWN on it, because walking once
    // you are on it walks you off the other side. It says `rest` or `groom` and
    // the program is STILL.
    //
    // Two of the six programs cannot run in this browser and are not faked:
    //   chase    needs the eye. The retina and tectum are not ported, so the
    //            animal here has no prey to see and `hunt` never releases.
    //   shelter  needs a threat to flee and a refuge to flee to. This world
    //            has neither, so `flee` never releases.
    // Handing the brain a prey position it did not see would be an oracle, and
    // this project spent #270 through #282 taking oracles back out.
    const P = this.cfg.world.programs, L = this.cfg.world.locomotor;
    let program = P[next] || "still";
    let drive = (L[next] || 0) * 1.0;                       // urgency is 1 here
    if (drive <= this.cfg.world.drive_threshold) drive = 0;

    let steer = 0;
    this.program = program;
    if (program === "warm") {
      // Where the warm ground is. In Python this arrives from the environment
      // as a stand-in for a spatial memory the project has not built -- brain 7
      // is absent -- and it is marked as such there. Same here: this is
      // REMEMBERED, not seen.
      const dx = this.cfg.world.warm_patch_xy[0] - this.d.xpos[3];
      const dy = this.cfg.world.warm_patch_xy[1] - this.d.xpos[4];
      steer = Math.atan2(dy, dx) - this.heading;
      steer = ((steer + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      if (this.onWarm) drive = 0;                           // a basking animal lies down
      this.headYaw += (0 - this.headYaw) * 0.06;
      this.looking = false;
      this.searchState = this.onWarm ? "settled" : "approach";
    } else if (program === "search") {
      const sp = this.search.step((this.heading * 180) / Math.PI);
      this.searchState = sp.state;
      this.looking = sp.looking;
      this.headYaw = sp.headYawDeg;
      if (!sp.moving) drive = 0;
      steer = (sp.headingDeg * Math.PI) / 180;
    } else {
      drive = 0;
      this.headYaw += (0 - this.headYaw) * 0.06;
      this.looking = false;
      this.searchState = "still";
    }
    this.drive = drive;

    let ctrl;
    if (drive > 0) {
      this.gaitT += dt;
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
      const want = (this.headYaw * Math.PI) / 180;
      const hi = this.cfg.walker.ctrl_high, lo = this.cfg.walker.ctrl_low;
      for (let k = 0; k < 2; k++) {
        const i = this._neckIds[k];
        if (i < 0) continue;
        dd.ctrl[i] = Math.min(hi[i], Math.max(lo[i], want * (this._neckSpan[k] / total)));
      }
    }

    for (let i = 0; i < this.nAct; i++) {
      this.nerveDrive[i] += Math.abs(dd.ctrl[i] - this._prevCtrl[i]);
      this._prevCtrl[i] = dd.ctrl[i];
    }
    this.simT += dt;

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
