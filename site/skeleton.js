// The skeleton. Appearance only: no mass, no contact, no degree of freedom.
//
// Each bone group is authored in the LOCAL frame of a body the physics already
// has, so posing it is just that body's transform -- the same transform the
// skin rides. Bone LENGTHS are fitted to the certified body (the humerus is
// 11.9 mm because this animal's humerus is). Bone SHAPES, and the vertebral
// counts, are INVENTED and say so in tools/export_web_skeleton.py and on the
// page. No gecko osteology has been read into this project yet.

const THREE = window.THREE;

export class Skeleton {
  constructor(header, buffer) {
    const L = header.layout;
    let off = 0;
    const take = (s) => {
      const C = s.type === "float32" ? Float32Array : Uint32Array;
      const a = new C(buffer, off, s.count); off += s.bytes; return a;
    };
    this.rest = take(L[0]);
    this.face = take(L[1]);
    this.nvert = header.nvert;
    this.bones = header.bones;
    this.counts = header.counts;

    this.posed = new Float32Array(this.nvert * 3);
    this.normal = new Float32Array(this.nvert * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.posed, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(this.normal, 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(this.face), 1));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 2);
    this.geometry = geo;

    this.material = new THREE.MeshStandardMaterial({
      color: 0xf2ece0, roughness: 0.55, metalness: 0.02,
      emissive: new THREE.Color(0x14100a), emissiveIntensity: 1,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;

    this._restN = new Float32Array(this.nvert * 3);
    this._computeRestNormals();
  }

  _computeRestNormals() {
    const v = this.rest, f = this.face, n = this._restN;
    for (let i = 0; i < f.length; i += 3) {
      const a = f[i] * 3, b = f[i + 1] * 3, c = f[i + 2] * 3;
      const ux = v[b] - v[a], uy = v[b + 1] - v[a + 1], uz = v[b + 2] - v[a + 2];
      const wx = v[c] - v[a], wy = v[c + 1] - v[a + 1], wz = v[c + 2] - v[a + 2];
      const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      n[a] += nx; n[a + 1] += ny; n[a + 2] += nz;
      n[b] += nx; n[b + 1] += ny; n[b + 2] += nz;
      n[c] += nx; n[c + 1] += ny; n[c + 2] += nz;
    }
    for (let i = 0; i < n.length; i += 3) {
      const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
      n[i] /= l; n[i + 1] /= l; n[i + 2] /= l;
    }
  }

  // Rigid per-body transform: bone i of body B sits at xpos[B] + R[B] * rest.
  update(xpos, xmat) {
    const out = this.posed, nout = this.normal, rest = this.rest, rn = this._restN;
    for (const b of this.bones) {
      const r = b.body * 9, p = b.body * 3;
      const m0 = xmat[r], m1 = xmat[r + 1], m2 = xmat[r + 2];
      const m3 = xmat[r + 3], m4 = xmat[r + 4], m5 = xmat[r + 5];
      const m6 = xmat[r + 6], m7 = xmat[r + 7], m8 = xmat[r + 8];
      const px = xpos[p], py = xpos[p + 1], pz = xpos[p + 2];
      const end = (b.vertadr + b.vertnum) * 3;
      for (let o = b.vertadr * 3; o < end; o += 3) {
        const x = rest[o], y = rest[o + 1], z = rest[o + 2];
        out[o]     = m0 * x + m1 * y + m2 * z + px;
        out[o + 1] = m3 * x + m4 * y + m5 * z + py;
        out[o + 2] = m6 * x + m7 * y + m8 * z + pz;
        const nx = rn[o], ny = rn[o + 1], nz = rn[o + 2];
        nout[o]     = m0 * nx + m1 * ny + m2 * nz;
        nout[o + 1] = m3 * nx + m4 * ny + m5 * nz;
        nout[o + 2] = m6 * nx + m7 * ny + m8 * nz;
      }
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.normal.needsUpdate = true;
  }
}

export async function loadSkeleton(base = "media/") {
  const header = await (await fetch(base + "skeleton.json")).json();
  const buffer = await (await fetch(base + "skeleton.bin")).arrayBuffer();
  return new Skeleton(header, buffer);
}
