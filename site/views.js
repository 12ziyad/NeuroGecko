import { poseEyes } from "./eyes.js";
// The two views: the animal in its enclosure, and the same animal as a body
// of joints. Both draw the SAME physics state at the SAME instant.
//
// WHAT IS MEASURED IN THE LOWER VIEW. Joint markers sit at `d.xanchor` and
// point along `d.xaxis` -- the real anchor and axis MuJoCo computes for each
// of the 38 joints. Their brightness and length come from |qvel| at that
// joint's own degree of freedom. Nothing there is animated on a timer: when a
// marker flares, that joint is genuinely moving that fast.
//
// WHAT IS DEPICTION. The colours, and the fact that a travelling wave of
// flares reads as "a signal running down the animal". The wave is real -- it
// is the gait -- but calling it a nerve impulse would be a claim nobody has
// measured in this species, so the page calls it joint speed.

const THREE = window.THREE;
const SPHERE = 2, CAPSULE = 3, ELLIPSOID = 4, BOX = 6;

const CH_COLOUR = {
  hunt: 0xe08268, flee: 0xe05650, explore: 0x7fc4e8,
  bask: 0xf0b552, rest: 0xc79bc2, groom: 0x8a93a2,
};

function geomMesh(g, material) {
  const s = g.size;
  let geo;
  if (g.type === SPHERE) geo = new THREE.SphereGeometry(s[0], 18, 12);
  else if (g.type === CAPSULE) geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16);
  else if (g.type === ELLIPSOID) { geo = new THREE.SphereGeometry(1, 20, 14); geo.scale(s[0], s[1], s[2]); }
  else if (g.type === BOX) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  else geo = new THREE.SphereGeometry(0.003, 6, 4);
  return new THREE.Mesh(geo, material);
}

function poseFromMat(mesh, xpos, xmat, i, m4) {
  const p = i * 3, r = i * 9;
  mesh.position.set(xpos[p], xpos[p + 1], xpos[p + 2]);
  m4.set(xmat[r], xmat[r + 1], xmat[r + 2], 0,
         xmat[r + 3], xmat[r + 4], xmat[r + 5], 0,
         xmat[r + 6], xmat[r + 7], xmat[r + 8], 0, 0, 0, 0, 1);
  mesh.quaternion.setFromRotationMatrix(m4);
}

// ------------------------------------------------------------ the enclosure
// A CRICKET. Its body is `prey_radius_m` -- 9 mm, from the registry -- so the
// thing on screen is the size the detector and the capture distance both use.
function makeCricket(cfg, sc, dark) {
  const r = (cfg.prey && cfg.prey.prey_radius_m) || 0.009;
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(r, 12, 9),
    new THREE.MeshStandardMaterial({ color: dark ? 0x9aa6b4 : 0x3a2d1c,
                                     roughness: 0.6, metalness: 0.05,
                                     emissive: dark ? 0x223344 : 0x000000 }));
  body.scale.set(1.5, 0.75, 0.62);
  body.castShadow = !dark;
  g.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(r * 0.5, 10, 8), body.material);
  head.position.x = r * 1.35; g.add(head);
  const legMat = new THREE.MeshStandardMaterial({ color: dark ? 0x7e8b99 : 0x241b10,
                                                  roughness: 0.8 });
  const legGeo = new THREE.CylinderGeometry(r * 0.08, r * 0.05, r * 1.5, 4);
  for (let i = 0; i < 6; i++) {
    const leg = new THREE.Mesh(legGeo, legMat);
    const side = i % 2 ? 1 : -1, k = Math.floor(i / 2);
    leg.position.set((k - 1) * r * 0.7, side * r * 0.55, -r * 0.25);
    leg.rotation.x = side * 0.9;
    g.add(leg);
  }
  g.position.z = r * 0.75;
  sc.add(g);
  return g;
}


// Put a cricket mesh where the simulated cricket actually is.
function placeCricket(mesh, sim) {
  if (!mesh) return;
  // No cricket in the world for a stretch after a meal, so there is none on
  // screen either -- and the eye therefore cannot find one, which is the point.
  mesh.visible = !!sim.preyPresent;
  if (!sim.prey || !sim.preyPresent) return;
  const r = mesh.userData.r || 0.009;
  mesh.position.x = sim.prey.x;
  mesh.position.y = sim.prey.y;
  mesh.rotation.z = sim.prey.heading || 0;
  // A bob while it is walking. The bout structure is the cricket's, from the
  // registry; the bob is a drawing, and it is what a temporal-contrast detector
  // has to work with.
  mesh.position.z = r * 0.75
    + (sim.prey.walking ? r * 0.22 * Math.abs(Math.sin(sim.simT * 26)) : 0);
}

export class RoomView {
  constructor(canvas, cfg) {
    this.cfg = cfg;
    this.fill = 0.95;
    this.tilt = 0.42;
    this.mode = "follow";
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    const sc = this.scene = new THREE.Scene();
    sc.background = new THREE.Color(0x0e0c0b);
    sc.fog = new THREE.Fog(0x0e0c0b, 0.55, 1.5);
    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.01, 12);

    sc.add(new THREE.HemisphereLight(0xffeede, 0x241d16, 0.85));
    const key = new THREE.DirectionalLight(0xfff0d2, 2.6);
    key.position.set(0.45, 0.35, 0.75);
    key.castShadow = true;
    key.shadow.mapSize.set(512, 512);
    const c = key.shadow.camera;
    c.near = 0.05; c.far = 1.6; c.left = -0.35; c.right = 0.35; c.top = 0.35; c.bottom = -0.35;
    sc.add(key);
    const rim = new THREE.DirectionalLight(0x8fb4d8, 0.75);
    rim.position.set(-0.6, -0.4, 0.3); sc.add(rim);

    // sand floor with a little grain
    const size = 512, cv = document.createElement("canvas");
    cv.width = cv.height = size;
    const g = cv.getContext("2d");
    g.fillStyle = "#8a7150"; g.fillRect(0, 0, size, size);
    // COARSER AND QUIETER THAN IT WAS. The first version speckled 26,000 grains
    // at 1.6 px and half opacity, which at this scale is sand grain detail a
    // gecko's eye cannot resolve -- and unresolvable detail does not vanish, it
    // ALIASES, into false structure that moves when the animal moves. That is
    // indistinguishable from prey to a motion detector, and brain/retina.py
    // names it as the error. Bigger, softer grains carry the same impression of
    // ground at a spatial frequency the eye can actually transmit.
    for (let i = 0; i < 7000; i++) {
      const v = 128 + Math.random() * 60;
      g.fillStyle = `rgba(${v | 0},${(v * 0.86) | 0},${(v * 0.64) | 0},0.28)`;
      g.fillRect(Math.random() * size, Math.random() * size, 3.4, 3.4);
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(120, 120);
    tex.colorSpace = THREE.SRGBColorSpace;
    // 40 m, NOT 6. The eye sees a hard geometric edge where the floor plane
    // stops, and a straight bright-to-dark edge sliding across the frame as the
    // animal turns is the single strongest local motion in the scene. Measured:
    // the tectum's peak landed on cell row 8-9 -- elevation 0 to -2 deg, the
    // horizon -- on 212 of 212 firings, while the cricket was in frame on 0 of
    // 900. The animal was hunting the edge of my floor. A desert does not stop
    // three metres away; the ground now runs past the fog and there is no edge
    // to find.
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 1.0, metalness: 0 }));
    floor.receiveShadow = true;
    sc.add(floor);

    // THE WARM PATCH. Not decoration and not a lamp: this species takes heat
    // from the GROUND by lying on it -- body temperature tracked substrate at
    // r2 = 0.97 against 0.92 for air (Hastings et al. 2023, n = 12), and
    // melanistic pigment made no difference to heating rate, which is evidence
    // against warming by radiation. So the heat is a patch of floor. Its
    // position, its size and its 30 C surface all come out of the project's own
    // furnished world, morphology/gecko_habitat_v1.xml, by way of brain.json.
    // Standing on it is the only thing that warms this animal, and walking to
    // it is what the `bask` channel is for.
    const W = cfg.world;
    if (W && W.warm_patch_xy) {
      const h = W.warm_patch_half_m;
      const warm = new THREE.Mesh(
        new THREE.PlaneGeometry(2 * h, 2 * h),
        new THREE.MeshBasicMaterial({ color: 0xff8c3a, transparent: true,
                                      opacity: 0.16, depthWrite: false }));
      warm.position.set(W.warm_patch_xy[0], W.warm_patch_xy[1], 0.0006);
      sc.add(warm);
      const ring = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-h, -h, 0), new THREE.Vector3(h, -h, 0),
          new THREE.Vector3(h, h, 0), new THREE.Vector3(-h, h, 0)]),
        new THREE.LineBasicMaterial({ color: 0xffa257, transparent: true, opacity: 0.5 }));
      ring.position.set(W.warm_patch_xy[0], W.warm_patch_xy[1], 0.0012);
      sc.add(ring);
      this.warmMat = warm.material;
    }

    // PLANTS. Scenery, and labelled scenery: nothing in this project has ever
    // measured the vegetation of a leopard gecko's range, and these are not
    // collidable -- the physics does not know they exist. They are here so the
    // eye has something to move past and so the ground reads as somewhere
    // rather than as a plane.
    const plantMat = new THREE.MeshStandardMaterial({
      color: 0x5d7040, roughness: 0.85, metalness: 0, side: THREE.DoubleSide });
    const stemMat = new THREE.MeshStandardMaterial({ color: 0x6b5a36, roughness: 1 });
    const bladeGeo = new THREE.PlaneGeometry(0.0055, 0.040);
    bladeGeo.translate(0, 0.020, 0);
    for (let i = 0; i < 46; i++) {
      const a = (i * 2.39996), r = 0.30 + (i % 11) * 0.115;
      const px = Math.cos(a) * r, py = Math.sin(a) * r;
      if (Math.abs(px - W.warm_patch_xy[0]) < 0.12
          && Math.abs(py - W.warm_patch_xy[1]) < 0.12) continue;
      const tuft = new THREE.Group();
      const n = 5 + (i % 4);
      for (let b = 0; b < n; b++) {
        const m = new THREE.Mesh(bladeGeo, plantMat);
        m.rotation.z = (Math.sin(i * 7.1 + b) * 0.5);
        m.rotation.y = (b / n) * Math.PI * 2;
        m.scale.setScalar(0.7 + 0.6 * Math.abs(Math.sin(i * 3.3 + b)));
        m.castShadow = true;
        tuft.add(m);
      }
      const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.0014, 0.0022, 0.012, 6), stemMat);
      stem.position.y = 0.006; tuft.add(stem);
      tuft.position.set(px, py, 0);
      tuft.rotation.x = Math.PI / 2;
      sc.add(tuft);
    }

    // THE CRICKET. Not decoration: this is the object the animal's eye has to
    // find in its own rendered pixels before `hunt` can release at all.
    this.cricket = makeCricket(cfg, sc, false);
    this.cricket.userData.r = (cfg.prey && cfg.prey.prey_radius_m) || 0.009;

    // NO WALLS. The enclosure was mine, not the project's, and all it did was
    // stop the animal walking. Open ground, and the floor follows it.
    // Scattered stones. They are scenery, not biology -- but without a fixed
    // landmark the eye has nothing to measure movement against.
    const stoneMat = new THREE.MeshStandardMaterial({ color: 0x6f6152, roughness: 1 });
    const stones = new THREE.Group();
    let seed = 7;
    const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    for (let i = 0; i < 90; i++) {
      const r = 0.004 + rnd() * 0.012;
      const g2 = new THREE.SphereGeometry(r, 7, 5);
      g2.scale(1, 0.7 + rnd() * 0.5, 0.45 + rnd() * 0.3);
      const st = new THREE.Mesh(g2, stoneMat);
      st.position.set((rnd() - 0.5) * 3.0, (rnd() - 0.5) * 3.0, r * 0.35);
      st.rotation.z = rnd() * 6.28;
      st.castShadow = true; st.receiveShadow = true;
      stones.add(st);
    }
    sc.add(stones);

    this.floor = floor;
    this.skin = null;
    this._m4 = new THREE.Matrix4();
  }
  // The eyes ride the head as their own rigid meshes. See site/eyes.js.
  attachEyes(eyes, headBody) {
    this.eyes = eyes;
    this.eyeHeadBody = headBody;
    this.scene.add(eyes.group);
    return this;
  }

  attachSkin(skin) {
    this.skin = skin;
    skin.mesh.castShadow = true;
    this.scene.add(skin.mesh);
  }
  setMode(mode) { this.mode = mode; }

  cameraFor(t, sim) {
    const R = this.frameRadius, m = this.mode || "follow";
    const heading = sim.heading;
    if (m === "side")  return [t[0] + R * Math.cos(heading + Math.PI / 2),
                               t[1] + R * Math.sin(heading + Math.PI / 2), t[2] + R * 0.12];
    if (m === "front") return [t[0] + R * Math.cos(heading),
                               t[1] + R * Math.sin(heading), t[2] + R * 0.18];
    if (m === "top")   return [t[0] + 0.001, t[1] + 0.001, t[2] + R * 1.25];
    if (m === "low")   return [t[0] + R * Math.cos(sim.camAngle),
                               t[1] + R * Math.sin(sim.camAngle), t[2] + R * 0.06];
    return [t[0] + R * Math.cos(sim.camAngle), t[1] + R * Math.sin(sim.camAngle),
            t[2] + R * this.tilt];
  }


  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // The animal is about 0.16 m nose to tail. Pull back enough that it fits
    // the SHORTER of the two screen axes, so a wide, short pane still frames it.
    const vFov = (this.camera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    this.frameRadius = Math.max(0.16, (0.095 / Math.tan(Math.min(vFov, hFov) / 2)) * this.fill);
  }
  // ------------------------------------------------------------ head camera --
  // The animal's own eye. A camera carried on the head body, pointed where the
  // head points, rendered into an offscreen buffer at the eye's own render
  // resolution. `site/eye.js` reads these pixels and finds the cricket in them
  // or does not -- nothing is told where the cricket is.
  //
  // Field of view is 70 deg, which is the value brain/retina.py was built
  // around. The eye height is the head body's own position, so looking down at
  // the ground is something the animal does with its neck rather than something
  // arranged for it.
  attachHeadCamera(cfg) {
    const px = cfg.eye.render_pixels;
    this.eyeTarget = new THREE.WebGLRenderTarget(px, px, {
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat, type: THREE.UnsignedByteType,
    });
    this.eyeCam = new THREE.PerspectiveCamera(cfg.eye.fovy_deg, 1, 0.004, 4);
    this.eyeCam.up.set(0, 0, 1);
    this.eyeBuf = new Uint8Array(px * px * 4);
    this.eyeFlip = new Uint8Array(px * px * 4);
    this.eyePx = px;
    return this;
  }

  // Render one frame from the head and return it as RGBA bytes, top row first.
  //
  // WHERE THE EYE IS. Not the head body's origin -- that is inside the skull,
  // and a camera there looks at the inside of the animal's own face, which is a
  // large moving textured object filling the frame and is exactly what a motion
  // detector cannot ignore. It is the position the morphology gives for
  // `head_cam`: 29.58 mm forward and 6.82 mm up in the head's own frame, which
  // is the snout.
  //
  // WHERE IT POINTS. Along the head's own x-axis, up along the head's z, read
  // straight out of the head's rotation matrix -- so the neck and head pitch
  // joints aim it, and nothing here invents a downward tilt. The first version
  // did invent one and it was the wrong way to do it.
  renderEye(sim) {
    if (!this.eyeTarget) return null;
    const d = sim.d;
    const hb = (sim.headBodyId ?? 4);
    const p = hb * 3, m = hb * 9;
    // columns of the head's rotation matrix: its own x, y and z in world
    const fx = d.xmat[m],     fy = d.xmat[m + 3], fz = d.xmat[m + 6];   // forward
    const ux = d.xmat[m + 2], uy = d.xmat[m + 5], uz = d.xmat[m + 8];   // up
    const EX = 0.02958, EZ = 0.00682;         // morphology, head_cam pos
    const ex = d.xpos[p] + fx * EX + ux * EZ;
    const ey = d.xpos[p + 1] + fy * EX + uy * EZ;
    const ez = d.xpos[p + 2] + fz * EX + uz * EZ;
    this.eyeCam.position.set(ex, ey, ez);
    this.eyeCam.up.set(ux, uy, uz);
    this.eyeCam.lookAt(ex + fx * 0.4, ey + fy * 0.4, ez + fz * 0.4);
    this.cricket.visible = !!sim.preyPresent;
    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.eyeTarget);
    this.renderer.render(this.scene, this.eyeCam);
    this.renderer.readRenderTargetPixels(this.eyeTarget, 0, 0,
                                         this.eyePx, this.eyePx, this.eyeBuf);
    this.renderer.setRenderTarget(prevTarget);
    // WebGL hands back bottom row first; the retina's row 0 is the TOP of the
    // frame, and the elevation rule depends on that -- row 0 has to be sky or
    // the "below the horizon" switch has its sign inverted and the animal hunts
    // the ceiling.
    const n = this.eyePx, row = n * 4;
    for (let r = 0; r < n; r++) {
      this.eyeFlip.set(this.eyeBuf.subarray((n - 1 - r) * row, (n - r) * row), r * row);
    }
    return this.eyeFlip;
  }

  update(sim) {
    placeCricket(this.cricket, sim);
    if (this.warmMat) {
      this.warmMat.opacity = sim.onWarm
        ? 0.22 + 0.08 * Math.sin(sim.simT * 4.0) : 0.14;
    }
    const d = sim.d;
    if (this.skin) this.skin.update(d.xpos, d.xquat);
    const t = [d.xpos[3], d.xpos[4], d.xpos[5]];
    // THE GROUND DOES NOT FOLLOW. It used to be pinned to the animal, which
    // is exactly why the walk read as a treadmill: nothing ever passed it.
    // It re-centres only in whole texture tiles, so the pattern never slides.
    if (this.floor) {
      const TILE = 40 / 120;   // one texture tile, so the pattern stays in world space
      this.floor.position.set(Math.round(t[0] / TILE) * TILE,
                              Math.round(t[1] / TILE) * TILE, 0);
    }
    if (this.eyes) poseEyes(this.eyes, d, this.eyeHeadBody, this._m4);

    if (this.mode === "eye" && this.eyeCam) {
      // THE SAME CAMERA THE RETINA USES, at screen resolution rather than 64
      // pixels. Not a reconstruction of it: `renderEye` has already placed it
      // this step, so what you are looking at is where the animal is actually
      // pointed, including whatever its neck is doing.
      this.eyeCam.aspect = this.camera.aspect;
      this.eyeCam.updateProjectionMatrix();
      this.renderer.render(this.scene, this.eyeCam);
      return;
    }
    const c = this.cameraFor(t, sim);
    this.camera.position.set(c[0], c[1], c[2]);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(t[0], t[1], t[2] + 0.006);
    this.renderer.render(this.scene, this.camera);
  }
}

// ----------------------------------------------- the body, and its 38 joints
export class NerveView {
  constructor(canvas, cfg) {
    this.cfg = cfg;
    this.fill = 1.02;                 // fit the whole animal, nose to tail tip
    this.tilt = 0.38;
    this.mode = "follow";
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    const sc = this.scene = new THREE.Scene();
    sc.background = new THREE.Color(0x0a0d12);
    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.01, 12);

    sc.add(new THREE.HemisphereLight(0xeaf2fb, 0x16202c, 1.15));
    const k = new THREE.DirectionalLight(0xffffff, 1.9);
    k.position.set(0.5, 0.45, 0.9); sc.add(k);
    const fill = new THREE.DirectionalLight(0x9fc4e8, 0.8);
    fill.position.set(-0.7, -0.3, 0.25); sc.add(fill);

    const grid = new THREE.GridHelper(1.4, 28, 0x1d2836, 0x141b25);
    grid.rotation.x = Math.PI / 2; sc.add(grid);

    // THE CRICKET IS IN THIS PANE TOO, because what the animal is hunting is
    // part of what the animal is doing. Drawn cold rather than warm so it reads
    // as a target in a diagram and not as a second animal.
    this.cricket = makeCricket(cfg, sc, true);
    this.cricket.userData.r = (cfg.prey && cfg.prey.prey_radius_m) || 0.009;

    // WHAT THE EYE IS DOING WITH IT. A line from the head along the bearing the
    // tectum is reporting this frame -- dim while it is one unconfirmed report,
    // solid once the evidence accumulator has committed. This is the only
    // channel from the world into the brain, so it is worth being able to see.
    this.gazeRay = new THREE.Line(
      new THREE.BufferGeometry().setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(6), 3)),
      new THREE.LineBasicMaterial({ color: 0xffcf5c, transparent: true,
                                    opacity: 0.0, depthTest: false }));
    this.gazeRay.frustumCulled = false;
    this.gazeRay.renderOrder = 12;
    sc.add(this.gazeRay);

    // THE 48 SOLIDS ARE THE SKELETON. Not a stand-in for one -- they ARE what
    // this animal is made of, the shapes MuJoCo integrates and the shapes all
    // 14 anatomical checks measure. An earlier build drew an invented skull,
    // invented vertebrae and invented counts over the top of them; it looked
    // better and was a lie, so it was deleted (#360). Bone colour, solid,
    // lit -- and every shape in view is one the physics actually uses.
    this.baseCol = new THREE.Color(0xeae3d4);
    this.meshes = cfg.geoms.map((g) => {
      const mat = new THREE.MeshStandardMaterial({
        color: this.baseCol.clone(), roughness: 0.52, metalness: 0.0,
        emissive: new THREE.Color(0x12100c), emissiveIntensity: 1,
      });
      const mesh = geomMesh(g, mat);
      sc.add(mesh); return mesh;
    });

    // one marker per joint, at its real anchor, along its real axis
    const jm = new THREE.MeshBasicMaterial({ color: 0x39c6ff, transparent: true, opacity: 0.9 });
    const geo = new THREE.CylinderGeometry(0.00055, 0.00055, 1, 6);
    geo.translate(0, 0.5, 0);                       // grow from the anchor
    this.joints = [];
    for (const j of cfg.joints) {
      if (j.type === 0) { this.joints.push(null); continue; }   // skip the free root
      const m = new THREE.Mesh(geo, jm.clone());
      m.visible = false; sc.add(m);
      const hub = new THREE.Mesh(new THREE.SphereGeometry(0.0011, 10, 8), jm.clone());
      hub.visible = false; sc.add(hub);
      this.joints.push({ shaft: m, hub });
    }
    this._buildNerves(cfg);

    this._m4 = new THREE.Matrix4();
    this._up = new THREE.Vector3(0, 1, 0);
    this._ax = new THREE.Vector3();
  }
  setMode(mode) { this.mode = mode; }

  // Hide the bone so only the nervous system is left. The solids are still
  // there and still being posed by the physics; they are simply not drawn.
  setIsolate(on) {
    this.isolate = !!on;
    for (const m of this.meshes) m.visible = !this.isolate;
    if (this.cricket && this.isolate) this.cricket.visible = false;
    return this;
  }

  // ---------------------------------------------------------------- nerve --
  // A motor command does not appear at a joint. It leaves the head, runs down
  // the cord, leaves at the segment that serves that limb, and travels out to
  // the muscle. This draws that route on the animal's OWN tree: `body_parentid`
  // straight out of the MuJoCo model. A fibre to a hind ankle passes through
  // the pelvis because the animal's body does, and the fibre to the jaw is
  // short because the jaw is next to the brain. Nothing here is a straight
  // line through the middle of the gecko.
  //
  // WHAT IS MEASURED AND WHAT IS DRAWN, so nobody has to guess:
  //   route     REAL     the model's own kinematic chain, and each fibre ends
  //                      at the joint's true anchor (`xanchor`).
  //   when      REAL     an impulse launches when the brain's command to that
  //                      joint actually moves. A still joint is silent.
  //   how fast  DRAWN    no conduction velocity has ever been measured in
  //                      *Eublepharis macularius*, and none is claimed. The
  //                      two constants below are INVENTED and say so.
  _buildNerves(cfg) {
    this.NERVE_TRAVEL_S = 0.42;      // INVENTED: brain to toe, seconds
    this.IMPULSES_PER_RAD = 10.0;    // INVENTED: impulses per radian of command
    this.MAX_DOTS = 320;

    const names = cfg.body_names || [];
    const parent = cfg.body_parentid || [];
    this.brainBody = Math.max(0, names.indexOf("head"));
    const up = (b) => {
      const c = []; let g = 0;
      while (b > 0 && g++ < 64) { c.push(b); b = parent[b]; }
      c.push(0); return c;
    };
    const headChain = up(this.brainBody);

    this.nerves = [];
    const fibreMat = new THREE.LineBasicMaterial({
      color: 0x3c7ea3, transparent: true, opacity: 0.38,
      depthWrite: false, depthTest: false,       // the fibre runs INSIDE the body
    });
    for (let i = 0; i < cfg.joints.length; i++) {
      const j = cfg.joints[i];
      if (j.type === 0 || (j.act ?? -1) < 0) continue;   // undriven, or the free root
      const tc = up(j.body);
      let lca = 0;
      for (const b of headChain) if (tc.indexOf(b) >= 0) { lca = b; break; }
      const chain = [];
      for (const b of headChain) { chain.push(b); if (b === lca) break; }
      const down = [];
      for (const b of tc) { if (b === lca) break; down.push(b); }
      down.reverse();
      for (const b of down) chain.push(b);

      const n = chain.length + 1;                        // + the anchor itself
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(n * 3);
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const line = new THREE.Line(geo, fibreMat);
      line.frustumCulled = false;
      this.scene.add(line);
      this.nerves.push({ joint: i, act: j.act, chain, pos, geo,
                         cum: new Float32Array(n), phase: 0, impulses: [] });
    }

    // Every impulse on every fibre in one draw call.
    const dg = new THREE.BufferGeometry();
    this.dotPos = new Float32Array(this.MAX_DOTS * 3);
    this.dotCol = new Float32Array(this.MAX_DOTS * 3);
    dg.setAttribute("position", new THREE.BufferAttribute(this.dotPos, 3));
    dg.setAttribute("color", new THREE.BufferAttribute(this.dotCol, 3));
    dg.setDrawRange(0, 0);
    this.dots = new THREE.Points(dg, new THREE.PointsMaterial({
      size: 4.2, sizeAttenuation: false, vertexColors: true,
      transparent: true, opacity: 1.0, depthWrite: false, depthTest: false,
    }));
    this.dots.renderOrder = 10;
    this.dots.frustumCulled = false;
    this.scene.add(this.dots);

    // THE SELECTOR, IN THE HEAD. Six nodes, one per channel, sitting where the
    // forebrain is -- which is where a lizard's basal ganglia are. Each carries
    // its channel's colour and is lit by that channel's REAL gate value, the
    // same number the bars in the sidebar read, straight off the converged
    // fixed point. The one the selector released sits open; the five it is
    // holding down sit dark, which is what the basal ganglia do: they inhibit
    // everything and release one thing.
    //
    // THE ARRANGEMENT IS A DIAGRAM. Six nodes in a ring is a legible picture,
    // not a claim about where a gecko's striatum sits relative to its pallidum.
    // What is not a diagram is every number driving it.
    this.bgNodes = [];
    const chans = cfg.bg.channels;
    for (let i = 0; i < chans.length; i++) {
      const col = new THREE.Color(CH_COLOUR[chans[i]] ?? 0x8fa6bd);
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.00125, 10, 8),
        new THREE.MeshBasicMaterial({ color: col.clone(), transparent: true,
                                      opacity: 0.5, depthTest: false }));
      mesh.renderOrder = 13;
      this.scene.add(mesh);
      this.bgNodes.push({ mesh, base: col, ch: chans[i] });
    }
    // where the impulses start: a marker sitting in the head
    this.brainDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.0030, 12, 10),
      new THREE.MeshBasicMaterial({ color: 0x8fd4ff, transparent: true,
                                    opacity: 0.35, depthTest: false }));
    this.brainDot.renderOrder = 11;
    this.scene.add(this.brainDot);
    this._dotCol = new THREE.Color();
  }

  _updateNerves(sim, dt) {
    const d = sim.d;
    const chCol = this._dotCol.set(CH_COLOUR[sim.behaviour] ?? 0x6f8296);
    let k = 0;
    const bp = this.brainBody * 3, bm = this.brainBody * 9;
    this.brainDot.position.set(d.xpos[bp], d.xpos[bp + 1], d.xpos[bp + 2]);
    this.brainDot.material.opacity = 0.22 + 0.30 * Math.min(1, sim.speed * 6);

    // the six channels, carried in the head and lit by their own gates
    if (this.bgNodes) {
      const fx = d.xmat[bm], fy = d.xmat[bm + 3], fz = d.xmat[bm + 6];   // head x
      const lx = d.xmat[bm + 1], ly = d.xmat[bm + 4], lz = d.xmat[bm + 7]; // head y
      const ux = d.xmat[bm + 2], uy = d.xmat[bm + 5], uz = d.xmat[bm + 8]; // head z
      const R = 0.0062;
      for (let i = 0; i < this.bgNodes.length; i++) {
        const n = this.bgNodes[i];
        const a = (i / this.bgNodes.length) * Math.PI * 2 - Math.PI / 2;
        const sy = Math.cos(a) * R, sz = Math.sin(a) * R;
        n.mesh.position.set(
          d.xpos[bp] + fx * -0.002 + lx * sy + ux * sz,
          d.xpos[bp + 1] + fy * -0.002 + ly * sy + uy * sz,
          d.xpos[bp + 2] + fz * -0.002 + lz * sy + uz * sz);
        const g = Math.max(0, Math.min(1, (sim.gates[i] || 0)));
        const on = sim.behaviour === n.ch;
        n.mesh.material.opacity = 0.18 + 0.8 * g;
        n.mesh.scale.setScalar(0.75 + 1.05 * g);
        n.mesh.material.color.copy(n.base).multiplyScalar(on ? 1.0 : 0.35 + 0.5 * g);
      }
    }

    for (const nv of this.nerves) {
      // 1. lay the fibre along the body as it is standing RIGHT NOW
      const P = nv.pos, C = nv.cum;
      for (let a = 0; a < nv.chain.length; a++) {
        const b = nv.chain[a] * 3;
        P[a * 3] = d.xpos[b]; P[a * 3 + 1] = d.xpos[b + 1]; P[a * 3 + 2] = d.xpos[b + 2];
      }
      const e = nv.chain.length * 3, ja = nv.joint * 3;
      P[e] = d.xanchor[ja]; P[e + 1] = d.xanchor[ja + 1]; P[e + 2] = d.xanchor[ja + 2];
      C[0] = 0;
      for (let a = 1; a < C.length; a++) {
        C[a] = C[a - 1] + Math.hypot(P[a * 3] - P[a * 3 - 3],
                                     P[a * 3 + 1] - P[a * 3 - 2],
                                     P[a * 3 + 2] - P[a * 3 - 1]);
      }
      nv.geo.attributes.position.needsUpdate = true;
      nv.geo.computeBoundingSphere();

      // 2. launch impulses from the command that actually moved
      const drive = sim.nerveDrive ? sim.nerveDrive[nv.act] : 0;
      nv.phase += drive * this.IMPULSES_PER_RAD;
      while (nv.phase >= 1) {
        nv.phase -= 1;
        if (nv.impulses.length < 5) nv.impulses.push(0);
      }
      if (nv.phase > 2) nv.phase = 2;

      // 3. carry them along it
      const step = dt / this.NERVE_TRAVEL_S;
      const total = C[C.length - 1] || 1;
      for (let q = nv.impulses.length - 1; q >= 0; q--) {
        const u = (nv.impulses[q] += step);
        if (u >= 1) { nv.impulses.splice(q, 1); continue; }
        if (k >= this.MAX_DOTS) continue;
        const want = u * total;
        let a = 1;
        while (a < C.length - 1 && C[a] < want) a++;
        const seg = C[a] - C[a - 1] || 1;
        const f = Math.max(0, Math.min(1, (want - C[a - 1]) / seg));
        const i0 = (a - 1) * 3, i1 = a * 3, o = k * 3;
        this.dotPos[o] = P[i0] + (P[i1] - P[i0]) * f;
        this.dotPos[o + 1] = P[i0 + 1] + (P[i1 + 1] - P[i0 + 1]) * f;
        this.dotPos[o + 2] = P[i0 + 2] + (P[i1 + 2] - P[i0 + 2]) * f;
        const fade = Math.sin(Math.PI * Math.min(1, u * 1.08));
        this.dotCol[o] = chCol.r * fade;
        this.dotCol[o + 1] = chCol.g * fade;
        this.dotCol[o + 2] = chCol.b * fade;
        k++;
      }
    }
    if (sim.nerveDrive) sim.nerveDrive.fill(0);
    this.dots.geometry.setDrawRange(0, k);
    this.dots.geometry.attributes.position.needsUpdate = true;
    this.dots.geometry.attributes.color.needsUpdate = true;
    this.liveDots = k;
  }

  cameraFor(t, sim) {
    const R = this.frameRadius, m = this.mode || "follow";
    const heading = sim.heading;
    if (m === "side")  return [t[0] + R * Math.cos(heading + Math.PI / 2),
                               t[1] + R * Math.sin(heading + Math.PI / 2), t[2] + R * 0.12];
    if (m === "front") return [t[0] + R * Math.cos(heading),
                               t[1] + R * Math.sin(heading), t[2] + R * 0.18];
    if (m === "top")   return [t[0] + 0.001, t[1] + 0.001, t[2] + R * 1.25];
    if (m === "low")   return [t[0] + R * Math.cos(sim.camAngle),
                               t[1] + R * Math.sin(sim.camAngle), t[2] + R * 0.06];
    return [t[0] + R * Math.cos(sim.camAngle), t[1] + R * Math.sin(sim.camAngle),
            t[2] + R * this.tilt];
  }


  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // The animal is about 0.16 m nose to tail. Pull back enough that it fits
    // the SHORTER of the two screen axes, so a wide, short pane still frames it.
    const vFov = (this.camera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    this.frameRadius = Math.max(0.16, (0.095 / Math.tan(Math.min(vFov, hFov) / 2)) * this.fill);
  }

  update(sim) {
    const d = sim.d, cfg = this.cfg;

    // (The snout-to-tail projection that used to live here fed the band, and
    // the band is gone. It also read the trunk's forward axis out of the wrong
    // slice of `xmat`, the same mistake that was steering the animal backwards.
    // Deleted rather than fixed, because nothing reads it now.)

    // THE BONE STAYS BONE. An earlier build tinted every solid as a decision
    // swept past, which repainted the animal and hid the shape it was meant to
    // annotate (#361). The signal now lives on the nerve instead -- a dot, on
    // a fibre, going where the command goes -- so the skeleton can just be the
    // skeleton.
    for (let i = 0; i < this.meshes.length; i++) {
      poseFromMat(this.meshes[i], d.geom_xpos, d.geom_xmat, sim.geomIndex[i], this._m4);
      const m = this.meshes[i].material;
      if (!m.__plain) {
        m.color.copy(this.baseCol); m.emissive.setHex(0x12100c); m.__plain = true;
      }
    }

    placeCricket(this.cricket, sim);

    // WHAT THE EYE IS REPORTING, drawn from the head along the bearing. Dim
    // while it is one unconfirmed report, solid and longer once the evidence
    // accumulator has committed to it. It is drawn from the tectum's own
    // number, so when the animal is wrong the line points at nothing -- which
    // is the useful case to be able to see.
    {
      const b = sim.committed !== null && sim.committed !== undefined
        ? sim.committed : sim.preyBearing;
      const g = this.gazeRay.geometry.attributes.position;
      if (b === null || b === undefined) {
        this.gazeRay.material.opacity = 0;
      } else {
        const committed = sim.committed !== null && sim.committed !== undefined;
        const hb = this.brainBody * 3;
        // head yaw, then the reported bearing, both relative to the trunk
        const ang = sim.heading + ((sim.headYaw + b) * Math.PI) / 180;
        const len = committed ? 0.16 : 0.09;
        g.array[0] = d.xpos[hb]; g.array[1] = d.xpos[hb + 1]; g.array[2] = d.xpos[hb + 2];
        g.array[3] = d.xpos[hb] + Math.cos(ang) * len;
        g.array[4] = d.xpos[hb + 1] + Math.sin(ang) * len;
        g.array[5] = d.xpos[hb + 2] - 0.012;
        g.needsUpdate = true;
        this.gazeRay.geometry.computeBoundingSphere();
        this.gazeRay.material.opacity = committed ? 0.85 : 0.28;
        this.gazeRay.material.color.setHex(committed ? 0xffcf5c : 0x8fa6bd);
      }
    }

    // the nerve, advanced by the animal's own clock so it freezes when paused
    const now = sim.simT || 0;
    const dt = Math.max(0, Math.min(0.25, now - (this._lastT ?? now)));
    this._lastT = now;
    this._updateNerves(sim, dt);

    for (let i = 0; i < cfg.joints.length; i++) {
      const jj = this.joints[i]; if (!jj) continue;
      const j = cfg.joints[i];
      const speed = Math.abs(d.qvel[j.dofadr] || 0);
      const lit = Math.min(1, speed / 5.5);                 // real |qvel|
      const a = i * 3;
      const px = d.xanchor[a], py = d.xanchor[a + 1], pz = d.xanchor[a + 2];
      this._ax.set(d.xaxis[a], d.xaxis[a + 1], d.xaxis[a + 2]).normalize();

      jj.hub.visible = true;
      jj.hub.position.set(px, py, pz);
      jj.hub.material.opacity = 0.22 + 0.60 * lit;
      jj.hub.material.color.setHSL(0.55 - 0.10 * lit, 0.95, 0.45 + 0.35 * lit);
      jj.hub.scale.setScalar(0.9 + 0.8 * lit);

      const len = 0.0035 + 0.0075 * lit;
      jj.shaft.visible = lit > 0.10;
      jj.shaft.position.set(px, py, pz);
      jj.shaft.quaternion.setFromUnitVectors(this._up, this._ax);
      jj.shaft.scale.set(1, len, 1);
      jj.shaft.material.opacity = 0.18 + 0.45 * lit;
      jj.shaft.material.color.setHSL(0.55 - 0.10 * lit, 0.95, 0.45 + 0.35 * lit);
    }

    const t = [d.xpos[3], d.xpos[4], d.xpos[5]];
    const c = this.cameraFor(t, sim);
    this.camera.position.set(c[0], c[1], c[2]);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(t[0], t[1], t[2] + 0.006);
    this.renderer.render(this.scene, this.camera);
  }
}
