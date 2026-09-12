// The gecko's actual skin, posed by the physics.
//
// This is what the animal LOOKS like: one continuous surface of 21,663
// vertices over 23 bones, carrying the spot texture, built across sessions
// 10-13. The first browser build drew the 48 collision solids instead and the
// result was a chain of capsules -- correct physics, wrong animal.
//
// The skin is pure appearance: ledger #318 measured the trajectory as
// bit-identical with and without it. So the physics still runs the stripped
// 54 KB body, and this poses the surface over it by linear blend skinning,
// using exactly the body transforms MuJoCo produces.
//
// MuJoCo's skin convention, per bone b bound to body B:
//   rest  : bindpos[b], bindquat[b]   -- where that bone sat when the skin was bound
//   live  : xpos[B],   xquat[B]       -- where its body is now
//   vertex: sum over influencing bones of  w * ( q_B * (q_bind^-1 * (v - p_bind)) + p_B )

const THREE = window.THREE;

function quatMulVec(q, v, out) {
  // q = [w,x,y,z] (MuJoCo order), v = [x,y,z]
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  out[0] = v[0] + w * tx + (y * tz - z * ty);
  out[1] = v[1] + w * ty + (z * tx - x * tz);
  out[2] = v[2] + w * tz + (x * ty - y * tx);
  return out;
}

function quatConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }

export class Skin {
  constructor(header, buffer, texture) {
    this.h = header;
    const L = header.layout;
    let off = 0;
    const take = (spec) => {
      const Ctor = spec.type === "float32" ? Float32Array : Uint32Array;
      const a = new Ctor(buffer, off, spec.count);
      off += spec.bytes;
      return a;
    };
    this.vert = take(L[0]);          // rest positions, in the skin's own frame
    this.texcoord = take(L[1]);
    this.face = take(L[2]);
    this.boneVertId = take(L[3]);
    this.boneVertWeight = take(L[4]);

    this.nvert = header.nvert;
    this.posed = new Float32Array(this.nvert * 3);

    // Precompute the inverse bind transform for each bone.
    this.bones = header.bones.map((b) => ({
      body: b.body,
      bindpos: b.bindpos,
      invq: quatConj(b.bindquat),
      vertadr: b.vertadr,
      vertnum: b.vertnum,
    }));

    // Rest normals, computed ONCE. Recomputing them every frame from 39,256
    // faces was the whole cost of the first build: it ran at 6 fps. They are
    // skinned by the same bone rotations as the positions instead.
    this.restNormal = new Float32Array(this.nvert * 3);
    {
      const v = this.vert, f = this.face, n = this.restNormal;
      for (let i = 0; i < f.length; i += 3) {
        const a = f[i]*3, b = f[i+1]*3, c = f[i+2]*3;
        const ux = v[b]-v[a], uy = v[b+1]-v[a+1], uz = v[b+2]-v[a+2];
        const wx = v[c]-v[a], wy = v[c+1]-v[a+1], wz = v[c+2]-v[a+2];
        const nx = uy*wz-uz*wy, ny = uz*wx-ux*wz, nz = ux*wy-uy*wx;
        n[a]+=nx; n[a+1]+=ny; n[a+2]+=nz;
        n[b]+=nx; n[b+1]+=ny; n[b+2]+=nz;
        n[c]+=nx; n[c+1]+=ny; n[c+2]+=nz;
      }
      for (let i = 0; i < n.length; i += 3) {
        const l = Math.hypot(n[i], n[i+1], n[i+2]) || 1;
        n[i]/=l; n[i+1]/=l; n[i+2]/=l;
      }
    }
    this.posedNormal = new Float32Array(this.nvert * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.posed, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(this.posedNormal, 3));
    const uv = new Float32Array(this.nvert * 2);
    uv.set(this.texcoord);
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(this.face), 1));
    this.geometry = geo;

    const mat = new THREE.MeshStandardMaterial({
      map: texture || null,
      color: texture ? 0xffffff : 0xd8a457,
      roughness: 0.72, metalness: 0.03,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 2);

    this._acc = new Float32Array(this.nvert * 3);
    this._nacc = new Float32Array(this.nvert * 3);
    this._wsum = new Float32Array(this.nvert);
    this._tmp = new Float32Array(3);
    this._v = new Float32Array(3);
  }

  // Pose the surface from live body transforms. xpos/xquat are MuJoCo's
  // per-body arrays; nothing here is interpolated or smoothed.
  update(xpos, xquat) {
    const acc = this._acc, nacc = this._nacc, wsum = this._wsum;
    acc.fill(0); nacc.fill(0); wsum.fill(0);
    const v = this._v, t = this._tmp;
    const rn = this.restNormal;

    for (const b of this.bones) {
      const bp = b.bindpos, iq = b.invq, bi = b.body * 3, qi = b.body * 4;
      const px = xpos[bi], py = xpos[bi + 1], pz = xpos[bi + 2];
      const q = [xquat[qi], xquat[qi + 1], xquat[qi + 2], xquat[qi + 3]];
      const end = b.vertadr + b.vertnum;
      for (let k = b.vertadr; k < end; k++) {
        const vid = this.boneVertId[k], w = this.boneVertWeight[k];
        const o = vid * 3;
        v[0] = this.vert[o] - bp[0];
        v[1] = this.vert[o + 1] - bp[1];
        v[2] = this.vert[o + 2] - bp[2];
        quatMulVec(iq, v, t);          // into the bone's rest frame
        quatMulVec(q, t, t);           // out through the body's live rotation
        acc[o] += w * (t[0] + px);
        acc[o + 1] += w * (t[1] + py);
        acc[o + 2] += w * (t[2] + pz);
        // the normal takes the same rotation, no translation
        v[0] = rn[o]; v[1] = rn[o + 1]; v[2] = rn[o + 2];
        quatMulVec(iq, v, t); quatMulVec(q, t, t);
        nacc[o] += w * t[0]; nacc[o + 1] += w * t[1]; nacc[o + 2] += w * t[2];
        wsum[vid] += w;
      }
    }

    const out = this.posed, nout = this.posedNormal;
    for (let i = 0, o = 0; i < this.nvert; i++, o += 3) {
      const w = wsum[i];
      if (w > 1e-9) {
        const iw = 1 / w;
        out[o] = acc[o] * iw; out[o + 1] = acc[o + 1] * iw; out[o + 2] = acc[o + 2] * iw;
        let nx = nacc[o], ny = nacc[o + 1], nz = nacc[o + 2];
        const l = Math.hypot(nx, ny, nz) || 1;
        nout[o] = nx / l; nout[o + 1] = ny / l; nout[o + 2] = nz / l;
      } else {
        out[o] = this.vert[o]; out[o + 1] = this.vert[o + 1]; out[o + 2] = this.vert[o + 2];
        nout[o] = rn[o]; nout[o + 1] = rn[o + 1]; nout[o + 2] = rn[o + 2];
      }
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.normal.needsUpdate = true;
  }
}

export async function loadSkin(base = "media/") {
  const header = await (await fetch(base + "skin.json")).json();
  const buffer = await (await fetch(base + "skin.bin")).arrayBuffer();
  let texture = null;
  if (header.texture) {
    texture = await new Promise((res) => {
      new THREE.TextureLoader().load(base + header.texture.file,
        (t) => { t.colorSpace = THREE.SRGBColorSpace; t.flipY = false; res(t); },
        undefined, () => res(null));
    });
  }
  return new Skin(header, buffer, texture);
}
