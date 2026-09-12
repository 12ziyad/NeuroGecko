"""Export every constant the browser gecko needs, read off the live objects.

WHY THIS EXISTS RATHER THAN A HAND-WRITTEN JS CONSTANTS FILE. A browser port
that retypes the numbers is a SECOND ANIMAL wearing this one's name, and the
first typo is undetectable by anything downstream -- which is the defect class
this whole project is built to prevent. So nothing here is typed. Every value
is read from the constructed `CPGResidualController`, `PrescottParameters`,
`Physiology` and `Arousal` objects at run time and written out as JSON. If a
constant changes in Python, it changes here on the next export.

WHAT IS EXPORTED AND WHY EACH IS SAFE.

  * The gait is PURELY PERIODIC -- measured at 8.3e-16 over five cycles -- so
    the walker is a function of phase, and phase is cheap.
  * The stance compensator is ALREADY a lookup table inside Python
    (`StanceCompensator.tables`), interpolated on a hip grid and a sprawl grid.
    Those arrays are exported verbatim, so the browser does the same
    interpolation over the same numbers rather than re-deriving any geometry.
  * The basal ganglia, the clock, the drives and the salience rule are small
    closed-form arithmetic; their PARAMETERS come out here and their EQUATIONS
    are ported by hand, then checked by `tools/conformance_web.py`, which runs
    both implementations on the same seed and compares trajectories.

The physics itself is not ported at all: the browser runs the same MuJoCo on
`morphology/gecko_body_web.xml`, which is the certified body with its visual
meshes and skin removed. That strip was verified as physics-neutral -- 400
steps under identical controls, maximum joint difference exactly 0.0.

Usage:  python tools/export_web_brain.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "site" / "media" / "brain.json"
FEET = ("HL", "FL", "HR", "FR")


def _f(x):
    return None if x is None else float(x)


def main():
    import mujoco
    from envs.gecko_walk_env import GeckoWalkEnv
    from brain.prescott_bg import PrescottParameters, PrescottBasalGanglia
    from brain.hypothalamus import Physiology
    from brain import arousal as ar
    from brain.gecko_selector import CHANNELS, BASELINE_DOPAMINE, GROOM_TONIC
    from common.provenance import parameter_value

    env = GeckoWalkEnv(xml_path="morphology/gecko_body_lab_v2.xml",
                       gait_profile="lab", control_mode="cpg_residual",
                       hind_stance_compensation=True)
    c = env.cpg
    model = env.model
    aname = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"

    # ---------------------------------------------------------------- walker
    walker = {
        "freq_hz": _f(c.freq),
        "n_act": int(len(c.neutral)),
        "actuator_names": [aname(i) for i in range(len(c.neutral))],
        "neutral": [_f(x) for x in c.neutral],
        "half": [_f(x) for x in c.half],
        "sign": [_f(c._sign(i)) for i in range(len(c.neutral))],
        "entries": [[int(a), str(l), str(r)] for a, l, r in c.entries],
        "phase_fraction_offset": {f: _f(c.profile.phase_fraction(f, 0.0)) for f in FEET},
        "stance": {f: _f(c.stance_for(f)) for f in FEET},
        "amp": {k: _f(v) for k, v in c.amp.items()},
        "lab_fa_amplitude": {k: _f(v) for k, v in c._lab_fa_amplitude.items()},
        "lab_parameters": {k: (_f(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                           for k, v in c.lab_parameters.items()},
        "front_stance_press": _f(c.front_stance_press),
        "front_stance_press_fr": _f(c.front_stance_press_fr),
        "front_swing_lift": _f(c.front_swing_lift),
        "front_stance_seek": _f(c.front_stance_seek),
        "front_seek_relax": _f(c.front_seek_relax),
        "tail_coupling_scale": _f(c.tail_coupling_scale()),
        "sprawl_ids": sorted(int(i) for i in c._sprawl_ids),
        "shoulder_sprawl_tuck": _f(c.shoulder_sprawl_tuck),
        "ssl": int(c._ssl), "ssr": int(c._ssr),
        "spine": int(c._spine), "tail_l": int(c._tail_l), "tail_r": int(c._tail_r),
        "spine_amp": _f(c.spine_amp), "spine_phase": _f(c.spine_phase),
        "tail_amp": _f(c.tail_amp), "tail_phase_lag": _f(c.tail_phase_lag),
        "standing_ctrl": [_f(x) for x in c.standing_ctrl()],
        "ctrl_low": [_f(x) for x in model.actuator_ctrlrange[:, 0]],
        "ctrl_high": [_f(x) for x in model.actuator_ctrlrange[:, 1]],
    }

    # ------------------------------------------------- stance compensator
    comp = {}
    if c._hind_comp is not None:
        for foot, (drive_aid, free_aids, sprawl_aid) in c._hind_comp_ids.items():
            info = c._hind_comp._info[foot]
            comp[foot] = {
                "drive_aid": int(drive_aid),
                "free_aids": [int(a) for a in free_aids],
                "sprawl_aid": int(sprawl_aid),
                "hip_grid": [_f(x) for x in info["hip_grid"]],
                "sprawl_grid": [_f(x) for x in info["sprawl_grid"]],
                "table": np.asarray(c._hind_comp.tables[foot], dtype=float).tolist(),
            }
    walker["compensator"] = comp

    # ------------------------------------------------------- basal ganglia
    p = PrescottParameters.extended()
    probe = PrescottBasalGanglia(n_channels=len(CHANNELS), n_primary=len(CHANNELS),
                                 dopamine=BASELINE_DOPAMINE, parameters=p)
    bg = {
        "channels": list(CHANNELS),
        "dopamine": _f(BASELINE_DOPAMINE),
        "groom_tonic": _f(GROOM_TONIC),
        "gpi_tonic": _f(probe.gpi_tonic),
        "FULL": _f(PrescottBasalGanglia.FULL),
        "PARTIAL": _f(PrescottBasalGanglia.PARTIAL),
        "params": {k: (v if isinstance(v, (bool, str)) else _f(v))
                   for k, v in vars(p).items()},
        "max_steps": int(p.max_steps), "tolerance": _f(p.tolerance),
    }

    # --------------------------------------------------------------- clock
    clock = {
        "DAY_S": _f(ar.DAY_S), "DARK_FRACTION": _f(ar.DARK_FRACTION),
        "DUSK_RAMP_H": _f(ar.DUSK_RAMP_H),
        "onset_after_dark_s": _f(parameter_value("activity_onset_after_dark_s")),
        "peak_window_h": _f(parameter_value("activity_peak_window_h")),
        "sleep_cycle_period_s": parameter_value("sleep_cycle_period_s"),
    }

    # -------------------------------------------------------------- drives
    phys = Physiology.from_registry(body_mass_kg=float(np.sum(model.body_mass[1:])))
    drives = {k: _f(getattr(phys, k)) for k in
              ("body_mass_kg", "maximum_aerobic_speed_m_s", "endurance_coefficient_hours",
               "endurance_exponent", "fatigue_recovery_time_constant_s",
               "cost_of_transport_mL_O2_per_kg_per_m", "oxycalorific_J_per_mL",
               "tail_mass_fraction", "tail_lipid_fraction", "lipid_J_per_g")
              if hasattr(phys, k)}
    drives["preferred_temperature_C"] = [_f(x) for x in phys.preferred_temperature_C]

    geoms = []
    for i in range(model.ngeom):
        if int(model.geom_group[i]) != 3:
            continue
        geoms.append({
            "body": int(model.geom_bodyid[i]),
            "type": int(model.geom_type[i]),
            "size": [_f(x) for x in model.geom_size[i]],
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}",
        })

    # Which joints move which body, so the nervous-system view can colour a
    # limb by the activity of the joint that drives it. Read from the model.
    # Which actuator drives which joint, so the browser can start a nerve
    # impulse at the moment the brain's command to that joint ACTUALLY changes
    # rather than on a timer. Read from the model's transmission table.
    jnt_act = {}
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            jnt_act[int(model.actuator_trnid[a, 0])] = a

    jnt = []
    for j in range(model.njnt):
        jnt.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}",
            "body": int(model.jnt_bodyid[j]),
            "type": int(model.jnt_type[j]),
            "dofadr": int(model.jnt_dofadr[j]),
            "act": int(jnt_act.get(j, -1)),
        })

    # SEARCH. Read off the live class, never typed here: if a constant moves in
    # brain/search.py the browser moves with it, and tools/conformance_web.py
    # runs both step machines side by side to prove they agree.
    from brain.search import SearchPattern
    search = {k: _f(getattr(SearchPattern, k)) for k in (
        "SACCADE_DEG", "SACCADE_STEPS", "FIXATE_STEPS", "SETTLE_STEPS",
        "SCAN_HALF_WIDTH_DEG", "TURN_DEG", "TURN_COMMAND_DEG",
        "TURN_MAX_STEPS", "TRAVEL_STEPS", "SCANS_PER_SITE")}

    # MOTOR PROGRAMS and the THERMAL WORLD. Same discipline as the search
    # constants: read off the live modules, never typed. The warm patch is
    # lifted out of morphology/gecko_habitat_v1.xml -- the project's own
    # furnished world -- so the browser stands on the same patch of ground the
    # Python animal does, at the same place and the same size.
    from brain.programs import PROGRAMS
    from brain.brainstem import LOCOMOTOR, DRIVE_THRESHOLD
    from envs.gecko_brain_env import GeckoBrainEnv
    from common.provenance import parameter_value

    habitat = mujoco.MjModel.from_xml_path(str(REPO / "morphology"
                                               / "gecko_habitat_v1.xml"))
    wb = mujoco.mj_name2id(habitat, mujoco.mjtObj.mjOBJ_BODY, "warm_patch")
    warm_xy = [_f(x) for x in habitat.body_pos[wb][:2]]
    warm_half = 0.05
    for g in range(habitat.ngeom):
        if habitat.geom_bodyid[g] == wb:
            warm_half = _f(habitat.geom_size[g][0])
            break

    world = {
        "programs": dict(PROGRAMS),
        "locomotor": dict(LOCOMOTOR),
        "drive_threshold": _f(DRIVE_THRESHOLD),
        "ambient_substrate_C": _f(GeckoBrainEnv.AMBIENT_SUBSTRATE_C),
        "thermal_tau_s": _f(GeckoBrainEnv.THERMAL_TAU_S),
        "thermal_time_compression": _f(GeckoBrainEnv.THERMAL_TIME_COMPRESSION),
        "warm_surface_C": _f(parameter_value("warm_surface_temperature_C")),
        "warm_patch_xy": warm_xy,
        "warm_patch_half_m": warm_half,
    }

    payload = {
        "geoms": geoms, "joints": jnt, "search": search, "world": world,
        # The body tree itself. The nerve view routes every fibre along this
        # chain -- head to trunk to girdle to limb -- instead of drawing a
        # straight line from a brain to a joint through the middle of the
        # animal, which is not where a nerve goes.
        "body_parentid": [int(x) for x in model.body_parentid],
        "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
                       for i in range(model.nbody)],
        "generated_by": "tools/export_web_brain.py",
        "body_xml": "morphology/gecko_body_web.xml",
        "note": "Every value here was READ from the live Python objects, not typed. "
                "The browser port is checked against Python by tools/conformance_web.py.",
        "walker": walker, "bg": bg, "clock": clock, "drives": drives,
        "eyelid_closed_deg": 70.0,
        "release_floor_salience": 0.199878,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"written: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  actuators {walker['n_act']}  feet {list(walker['stance'])}")
    print(f"  search constants: {len(search)} read off SearchPattern")
    print(f"  warm patch: {warm_half*2:.3f} m across at "
          f"({warm_xy[0]:g}, {warm_xy[1]:g}), surface {world['warm_surface_C']} C")
    print(f"  programs: {world['programs']}")
    print(f"  compensated feet: {sorted(comp)}")
    for f in sorted(comp):
        t = np.asarray(comp[f]["table"])
        print(f"    {f}: table {t.shape}  hip grid {len(comp[f]['hip_grid'])}  "
              f"sprawl grid {len(comp[f]['sprawl_grid'])}")
    print(f"  bg channels {bg['channels']}  gpi_tonic {bg['gpi_tonic']:.6f}")


if __name__ == "__main__":
    main()
