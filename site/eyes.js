// The animal's eyes.
//
// These are the project's own meshes -- `eye_L.obj` and `eye_R.obj`, carrying
// `gecko_eye.png` -- exported verbatim by tools/export_web_eyes.py and parented
// to the head body at the local offset MuJoCo itself resolved. Nothing here is
// drawn by hand; the geometry, the placement and the texture all come out of
// the morphology.
//
// WHY THEY WERE MISSING. `export_web_skin.py` exports MuJoCo's *skin*, which is
// the deformable body surface. The eyes are separate rigid meshes, so nothing
// in the browser knew they existed and the animal had blank sockets. An
// eublepharid with no eye is the one thing this render could not afford to get
// wrong: this family is named for having eyelids at all.

export async function loadEyes(base = "media/", v = "") {
  const tag = v ? "?v=" + v : "";

  const head = await fetch(base + "eyes.json" + tag).then((r) => r.json());
  const buf = await fetch(base + "eyes.bin" + tag).then((r) => r.arrayBuffer());

  const tex = await new Promise((resolve) => {
    const l = new THREE.TextureLoader();
    l.load(base + (head.texture || "eye.png") + tag,
      (t) => { t.colorSpace = THREE.SRGBColorSpace; t.flipY = false; resolve(t); },
      undefined, () => resolve(null));
  });

  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    map: tex, color: tex ? 0xffffff : 0x241c12,
    // Wet, unlike the skin. The morphology says so in its own comment: an eye
    // is the one part of this animal that should catch a highlight, and at
    // MuJoCo's default specular a small dark sphere is almost all highlight,
    // which is what used to make them read as cream-coloured blobs.
    roughness: 0.18, metalness: 0.0,
    envMapIntensity: 1.0,
  });

  for (const m of head.meshes) {
    const o = m.offsets;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(
      new Float32Array(buf, o.pos[0], o.pos[1]), 3));
    g.setAttribute("uv", new THREE.BufferAttribute(
      new Float32Array(buf, o.uv[0], o.uv[1]), 2));
    g.setAttribute("normal", new THREE.BufferAttribute(
      new Float32Array(buf, o.nrm[0], o.nrm[1]), 3));
    g.setIndex(new THREE.BufferAttribute(
      new Uint32Array(buf, o.tri[0], o.tri[1]), 1));
    const mesh = new THREE.Mesh(g, material);
    // The geom's own local placement on the head body, quaternion and all.
    mesh.position.set(m.pos[0], m.pos[1], m.pos[2]);
    mesh.quaternion.set(m.quat[1], m.quat[2], m.quat[3], m.quat[0]);  // wxyz -> xyzw
    mesh.castShadow = true;
    mesh.name = m.name;
    group.add(mesh);
  }

  group.matrixAutoUpdate = false;
  return { group, material, meshes: head.meshes };
}

// Ride the head. `xmat` is row-major, so the body's own axes are the COLUMNS.
export function poseEyes(eyes, d, headBody, m4) {
  if (!eyes) return;
  const p = headBody * 3, m = headBody * 9;
  m4.set(d.xmat[m],     d.xmat[m + 1], d.xmat[m + 2], d.xpos[p],
         d.xmat[m + 3], d.xmat[m + 4], d.xmat[m + 5], d.xpos[p + 1],
         d.xmat[m + 6], d.xmat[m + 7], d.xmat[m + 8], d.xpos[p + 2],
         0, 0, 0, 1);
  eyes.group.matrix.copy(m4);
  eyes.group.matrixWorldNeedsUpdate = true;
}
