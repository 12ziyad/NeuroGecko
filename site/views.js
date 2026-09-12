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
export class RoomView {
  constructor(canvas, cfg) {
    this.cfg = cfg;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    const sc = this.scene = new THREE.Scene();
    sc.background = new THREE.Color(0x0e0c0b);
    sc.fog = new THREE.Fog(0x0e0c0b, 0.9, 2.2);
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
    for (let i = 0; i < 26000; i++) {
      const v = 120 + Math.random() * 90;
      g.fillStyle = `rgba(${v | 0},${(v * 0.86) | 0},${(v * 0.64) | 0},0.5)`;
      g.fillRect(Math.random() * size, Math.random() * size, 1.6, 1.6);
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(5, 5);
    tex.colorSpace = THREE.SRGBColorSpace;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.7),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 1.0, metalness: 0 }));
    floor.receiveShadow = true;
    sc.add(floor);

    // low enclosure walls, drawn (physics has its own in group 4)
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x4a3f34, roughness: 0.9, transparent: true, opacity: 0.55,
      side: THREE.DoubleSide,
    });
    const W = 0.42, H = 0.10, T = 0.012;
    for (const [px, py, sx, sy] of [[0, W, W, T], [0, -W, W, T], [W, 0, T, W], [-W, 0, T, W]]) {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * H), wallMat);
      wall.position.set(px, py, H); sc.add(wall);
    }
    this.skin = null;
    this._m4 = new THREE.Matrix4();
  }
  attachSkin(skin) {
    this.skin = skin;
    skin.mesh.castShadow = true;
    this.scene.add(skin.mesh);
  }
  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // The animal is about 0.16 m nose to tail. Pull back enough that it fits
    // the SHORTER of the two screen axes, so a wide, short pane still frames it.
    const vFov = (this.camera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    this.frameRadius = Math.max(0.20, (0.10 / Math.tan(Math.min(vFov, hFov) / 2)) * 1.05);
  }
  update(sim) {
    const d = sim.d;
    if (this.skin) this.skin.update(d.xpos, d.xquat);
    const t = [d.xpos[3], d.xpos[4], d.xpos[5]];
    const a = sim.camAngle;
    const R = this.frameRadius;
    this.camera.position.set(t[0] + R * Math.cos(a), t[1] + R * Math.sin(a), t[2] + R * 0.42);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(t[0], t[1], t[2] + 0.006);
    this.renderer.render(this.scene, this.camera);
  }
}

// ----------------------------------------------- the body, and its 38 joints
export class NerveView {
  constructor(canvas, cfg) {
    this.cfg = cfg;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    const sc = this.scene = new THREE.Scene();
    sc.background = new THREE.Color(0x0a0d12);
    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.01, 12);

    sc.add(new THREE.HemisphereLight(0xdfe9f5, 0x131922, 1.35));
    const k = new THREE.DirectionalLight(0xffffff, 1.1);
    k.position.set(0.4, 0.4, 0.8); sc.add(k);

    const grid = new THREE.GridHelper(1.4, 28, 0x1d2836, 0x141b25);
    grid.rotation.x = Math.PI / 2; sc.add(grid);

    // the physics body: pale, translucent, like a specimen in fluid
    const shell = new THREE.MeshStandardMaterial({
      color: 0xd6e2ee, roughness: 0.38, metalness: 0.0,
      transparent: true, opacity: 0.42, depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.meshes = cfg.geoms.map((g) => {
      const mesh = geomMesh(g, shell.clone());
      sc.add(mesh); return mesh;
    });

    // one marker per joint, at its real anchor, along its real axis
    const jm = new THREE.MeshBasicMaterial({ color: 0x39c6ff, transparent: true, opacity: 0.9 });
    const geo = new THREE.CylinderGeometry(0.0016, 0.0016, 1, 8);
    geo.translate(0, 0.5, 0);                       // grow from the anchor
    this.joints = [];
    for (const j of cfg.joints) {
      if (j.type === 0) { this.joints.push(null); continue; }   // skip the free root
      const m = new THREE.Mesh(geo, jm.clone());
      m.visible = false; sc.add(m);
      const hub = new THREE.Mesh(new THREE.SphereGeometry(0.0026, 10, 8), jm.clone());
      hub.visible = false; sc.add(hub);
      this.joints.push({ shaft: m, hub });
    }
    this._m4 = new THREE.Matrix4();
    this._up = new THREE.Vector3(0, 1, 0);
    this._ax = new THREE.Vector3();
  }
  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // The animal is about 0.16 m nose to tail. Pull back enough that it fits
    // the SHORTER of the two screen axes, so a wide, short pane still frames it.
    const vFov = (this.camera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    this.frameRadius = Math.max(0.20, (0.10 / Math.tan(Math.min(vFov, hFov) / 2)) * 1.05);
  }

  update(sim) {
    const d = sim.d, cfg = this.cfg;
    for (let i = 0; i < this.meshes.length; i++)
      poseFromMat(this.meshes[i], d.geom_xpos, d.geom_xmat, sim.geomIndex[i], this._m4);

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
      jj.hub.material.opacity = 0.30 + 0.70 * lit;
      jj.hub.material.color.setHSL(0.55 - 0.10 * lit, 0.95, 0.45 + 0.35 * lit);
      jj.hub.scale.setScalar(0.85 + 1.5 * lit);

      const len = 0.006 + 0.020 * lit;
      jj.shaft.visible = lit > 0.02;
      jj.shaft.position.set(px, py, pz);
      jj.shaft.quaternion.setFromUnitVectors(this._up, this._ax);
      jj.shaft.scale.set(1, len, 1);
      jj.shaft.material.opacity = 0.25 + 0.75 * lit;
      jj.shaft.material.color.setHSL(0.55 - 0.10 * lit, 0.95, 0.45 + 0.35 * lit);
    }

    const t = [d.xpos[3], d.xpos[4], d.xpos[5]];
    const a = sim.camAngle;
    const R = this.frameRadius;
    this.camera.position.set(t[0] + R * Math.cos(a), t[1] + R * Math.sin(a), t[2] + R * 0.38);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(t[0], t[1], t[2] + 0.006);
    this.renderer.render(this.scene, this.camera);
  }
}
