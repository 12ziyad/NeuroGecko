import sys, numpy as np, math, mujoco
sys.path.insert(0, ".")
from envs.gecko_walk_env import GeckoWalkEnv
from common.provenance import parameter_value
base = dict(parameter_value("lab_base_parameters"))
def run(overrides, seconds=14.0):
    params = dict(base); params.update(overrides)
    env = GeckoWalkEnv(xml_path="morphology/gecko_world_v1.xml", gait_profile="lab",
                       control_mode="cpg_residual", residual_scale=0.25, privileged_target=True,
                       lab_parameters=params)
    env.reset(seed=0); m, d = env.model, env.data
    sid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("pectoral_center","mid_back","pelvis_center")}
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    yaw, hd, bend, hr_phase, fell, speed = [], [], [], [], False, []
    for i in range(int(seconds / env.dt)):
        fwd = d.xmat[env._trunk].reshape(3,3)[:,0][:2]
        env.target = d.xpos[env._trunk][:2] + fwd * 5.0
        try: _, env._prev_dist, _ = env._target_egocentric()
        except Exception: pass
        _, _, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        if term: fell = True; break
        R = d.xmat[pelvis].reshape(3,3); yaw.append(math.atan2(R[1,0], R[0,0])); hd.append(math.atan2(fwd[1], fwd[0]))
        a, b, c = (d.site_xpos[sid[n]][:2] for n in ("pectoral_center","mid_back","pelvis_center"))
        v1, v2 = b-a, c-b; bend.append(math.degrees(math.atan2(v1[0]*v2[1]-v1[1]*v2[0], v1[0]*v2[0]+v1[1]*v2[1])))
        hr_phase.append(float(env.cpg.foot_phase_fraction("HR", d.time)))
        speed.append(float(info.get("forward_speed", np.nan)))
    env.close()
    if len(yaw) < 300: return None
    rel = np.degrees(np.unwrap(np.array(yaw)) - np.unwrap(np.array(hd))); rel -= rel[200:].mean()
    per = int(round(1.0/1.19/env.dt)); n = len(rel)//per
    exc = [rel[k*per:(k+1)*per].max()-rel[k*per:(k+1)*per].min() for k in range(3, n)]
    bx = [np.array(bend[k*per:(k+1)*per]).max()-np.array(bend[k*per:(k+1)*per]).min() for k in range(3, n)]
    ph = np.array(hr_phase[200:]); r = rel[200:]
    bins = np.linspace(0, 1, 21); centres = 0.5*(bins[1:]+bins[:-1])
    prof = np.array([r[(ph>=lo)&(ph<hi)].mean() if ((ph>=lo)&(ph<hi)).any() else np.nan for lo, hi in zip(bins[:-1], bins[1:])])
    return dict(exc=float(np.mean(exc)), bend=float(np.mean(bx)),
                peak_at=float(centres[np.nanargmax(prof)]), trough_at=float(centres[np.nanargmin(prof)]),
                fell=fell, speed=float(np.nanmean(speed[200:])), prof=prof)
if __name__ == "__main__":
    amps = [float(a) for a in sys.argv[1:]] or [0.30, 0.45, 0.60, 0.75, 0.90]
    print(f"{'spine_amp':>9} {'yaw exc':>8} {'bend pkpk':>10} {'peak@HR':>8} {'trough':>7} {'speed':>7}")
    for amp in amps:
        res = run({"spine_amp": amp})
        if res is None: print(f"{amp:9.2f}   fell early"); continue
        print(f"{amp:9.2f} {res['exc']:8.1f} {res['bend']:10.1f} {res['peak_at']:8.2f} {res['trough_at']:7.2f} {res['speed']*100:6.1f}cm/s {'FELL' if res['fell'] else ''}")
    print("published, same species: pelvic yaw excursion ~50 deg; yaw peaks AWAY from the hindlimb at 0.60-0.65 of its stride, toward it at 0.0")
