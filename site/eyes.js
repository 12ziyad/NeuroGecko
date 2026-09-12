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
    roughness: 0.14, metalness: 0.0,
    // A living eye is not a matte ball. This lifts the iris out of the shadow
    // the brow casts over it without washing the texture out.
    emissive: 0x14100a, emissiveIntensity: 1.0,
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

    // THE EYELID RIM. A dark sphere on a dark head reads as a hole, or as one
    // of the stones on the floor -- which is exactly what a viewer said it
    // looked like. This is the pale margin of the lid itself, which is a real
    // feature of this animal and the reason the family has its name: alone
    // among Gekkota, eublepharids have movable eyelids with a scaled margin
    // rather than a fused transparent spectacle.
    //
    // ITS APPEARANCE IS A DRAWING and says so: no lid-margin width, colour or
    // contrast has been published for this species. What it is doing here is
    // separating the eye from the head so the eye can be seen at all.
    const R = m.radius || 0.0036;
    const rim = new THREE.Mesh(
      new THREE.TorusGeometry(R * 0.97, R * 0.14, 8, 32),
      new THREE.MeshStandardMaterial({
        color: 0xdcc9a4, roughness: 0.70, metalness: 0.0,
        emissive: 0x2a2418, emissiveIntensity: 0.42,
      }));
    rim.position.copy(mesh.position);
    rim.quaternion.copy(mesh.quaternion);
    // The torus lies in its own xy plane; the eye looks along the head's +x,
    // so turn the ring to face the same way.
    rim.rotateY(Math.PI / 2);
    // ...and push it very slightly outboard so it rings the eye rather than
    // cutting through the middle of it.
    rim.translateZ(R * 0.30);
    rim.name = m.name + "_rim";
    group.add(rim);
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
