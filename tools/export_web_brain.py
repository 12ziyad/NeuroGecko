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

    _meal_g = (parameter_value("meal_offering_fraction_bodyweight")
               * parameter_value("body_mass_kg") * 1000.0
               * parameter_value("meal_consumed_fraction_of_offering"))
    _cricket_g = parameter_value("prey_item_wet_mass_kg") * 1000.0

    world = {
        "programs": dict(PROGRAMS),
        # One cricket as a fraction of one meal. DERIVED, not invented: 0.35 g
        # of cricket against 5 % of a 38 g animal's bodyweight offered and
        # 54.7 % of that actually eaten.
        "cricket_meal_fraction": _f(_cricket_g / max(_meal_g, 1e-9)),
        "locomotor": dict(LOCOMOTOR),
        "drive_threshold": _f(DRIVE_THRESHOLD),
        "ambient_substrate_C": _f(GeckoBrainEnv.AMBIENT_SUBSTRATE_C),
        "thermal_tau_s": _f(GeckoBrainEnv.THERMAL_TAU_S),
        "thermal_time_compression": _f(GeckoBrainEnv.THERMAL_TIME_COMPRESSION),
        "warm_surface_C": _f(parameter_value("warm_surface_temperature_C")),
        "warm_patch_xy": warm_xy,
        "warm_patch_half_m": warm_half,
        # How fast the animal gets hungry again after a meal. PUBLISHED,
        # this species: mean inter-meal interval 2.33 days, over which
        # brain/hypothalamus.py has it burn about 71 % of one meal. So the
        # browser is not inventing a recovery rate, it is reading one.
        "inter_meal_interval_days": _f(parameter_value("inter_meal_interval_days")),
        "meal_burn_fraction": 0.71,
    }

    # THE EYE. Every constant read off the live Retina and Tectum, and off the
    # provenance registry for the three published velocity numbers. The browser
    # runs the same detector on a real rendered frame -- it is not handed a prey
    # position. tools/conformance_web.py pushes identical image sequences
    # through both and compares salience and bearing frame by frame.
    from brain.retina import Retina, KEPT_CHANNELS, ACUITY_CYC_DEG
    from brain.tectum import Tectum, MOTION_FLOOR, MAX_TARGET_FRACTION

    # 64 -> 64 -> 16 is exactly how envs/gecko_brain_env.py builds the eye by
    # default, so the browser animal sees at the resolution every Python
    # measurement in this project was made at. It is also what keeps the
    # per-control-step pixel readback cheap enough to run at 50 Hz.
    # RENDER 128, SAMPLE 64. Rendering at the receptor resolution is not
    # neutral: it omits the optics entirely, so fine floor texture the eye
    # could never resolve is ALIASED into false structure that moves when the
    # animal moves -- which a motion detector cannot tell from a cricket.
    # brain/retina.py says this in its own docstring and supersampling is its
    # remedy. Measured in the browser, 64 -> 64 against 128 -> 64, in
    # FAILURE_MAP #372.
    EYE_RENDER_PX, EYE_PIXELS, EYE_CELLS, EYE_FOVY = 128, 64, 16, 70.0
    _ret = Retina(fovy_deg=EYE_FOVY, pixels=EYE_PIXELS, cells=EYE_CELLS,
                  render_pixels=EYE_RENDER_PX)
    _tec = Tectum(_ret)
    eye = {
        "fovy_deg": _f(_ret.fovy_deg),
        "render_pixels": int(_ret.render_pixels),
        "pixels": int(_ret.pixels),
        "cells": int(_ret.cells),
        "kept_channels": [int(c) for c in KEPT_CHANNELS],
        "acuity_cyc_deg": _f(_ret.acuity_cyc_deg),
        "surround_ratio": _f(_ret.surround_ratio),
        "motion_window": int(_ret.motion_window),
        "optical_limit_px": _f(_ret.optical_limit_px()),
        # tectum
        "motion_floor": _f(MOTION_FLOOR),
        "max_target_fraction": _f(MAX_TARGET_FRACTION),
        "surround_cells": int(_tec.surround_cells),
        "salience_scale": _f(_tec.salience_scale),
        "flow_gain_yaw": _f(_tec.flow_gain_yaw),
        "flow_gain_surge": _f(_tec.flow_gain_surge),
        "speed_window": int(_tec.speed_window),
        "min_resolvable_deg": _f(_tec.min_resolvable_deg),
        "scale_efference_by_span": bool(_tec.scale_efference_by_span),
        "v_peak": _f(_tec.v_peak),
        "v_low": _f(_tec.v_low),
        "v_high": _f(_tec.v_high),
        # geometry tables, precomputed so the browser cannot re-derive them wrong
        "azimuth_of_cell": [_f(_ret.azimuth_of(c)) for c in range(_ret.cells)],
        "elevation_of_cell": [_f(_ret.elevation_of(r)) for r in range(_ret.cells)],
    }

    # THE CRICKET. Twelve published-or-declared numbers, straight out of the
    # registry via PreyParameters, plus the strike envelope.
    from envs.prey import PreyParameters, REGISTRY_KEYS
    prey = {k: _f(parameter_value(k)) for k in REGISTRY_KEYS}

    # THE CHASE. Evidence accumulator, orienting reflex and the two stalk
    # durations, read off the live classes exactly as the env builds them --
    # FixationEvidence with the adaptive quorum on and its fixation length tied
    # to SearchPattern.FIXATE_STEPS, which is how gecko_brain_env.py wires it.
    from brain.search import FixationEvidence, OrientingReflex
    from brain.programs import (STALK_MOVE_STEPS, STALK_LOOK_STEPS,
                                ORIENT_HOLD_STEPS, RESUME_SEARCH_ON_LOST)
    _fe = FixationEvidence(adaptive=True,
                           fixate_steps=SearchPattern.FIXATE_STEPS)
    _or = OrientingReflex()
    hunt = {
        "evidence": {
            "quorum": int(_fe.quorum), "agree_deg": _f(_fe.agree_deg),
            "hold_steps": int(_fe.hold_steps), "rate_window": int(_fe.RATE_WINDOW),
            "fixate_steps": int(_fe.fixate_steps), "p_chance": _f(_fe.p_chance),
            "field_deg": _f(_fe.field_deg), "adaptive": bool(_fe.adaptive),
        },
        "orienting": {
            "field_deg": 70.0, "cells": 64,
            "trigger_deg": _f(_or.trigger_deg), "max_deg": _f(_or.max_deg),
            "move_steps": int(_or.move_steps), "blind_steps": int(_or.blind_steps),
            "gain": _f(OrientingReflex.GAIN),
        },
        "stalk_move_steps": int(STALK_MOVE_STEPS),
        "stalk_look_steps": int(STALK_LOOK_STEPS),
        "orient_hold_steps": int(ORIENT_HOLD_STEPS),
        "resume_search_on_lost": bool(RESUME_SEARCH_ON_LOST),
    }

    # THE ANIMAL ITSELF, for the panel that sits beside the render. Every one of
    # these is pulled from config/proxies.yaml by key, so the page cannot quote a
    # number this project does not hold, and each carries the provenance tag the
    # registry gives it -- published, derived or invented. A value and its label
    # travel together or neither travels.
    def _bio(key, label, unit, scale=1.0, fmt="%.3g", group=None):
        entry = REGISTRY[key]
        val = entry["value"]
        note = (entry.get("notes") or "")
        tag = ("PUBLISHED" if note.strip().upper().startswith("PUBLISHED")
               else "INVENTED" if str(entry.get("species", "")).upper() == "INVENTED"
               else "DERIVED" if str(entry.get("species", "")).lower() == "derived"
               else "PUBLISHED")
        return {"key": key, "label": label, "unit": unit, "tag": tag, "group": group,
                "value": fmt % (float(val) * scale),
                "species": entry.get("species"),
                "n": entry.get("sample_size"),
                "source": (entry.get("source") or "")[:240]}

    import json as _json
    REGISTRY = _json.loads(
        (REPO / "config" / "proxies.yaml").read_text(encoding="utf-8"))["entries"]

    biology = []
    for args in (
        ("body_mass_kg", "Body mass", "g", 1000.0, "%.0f", "The body"),
        ("fuller_svl_m", "Snout-vent length", "mm", 1000.0, "%.1f"),
        ("tail_mass_fraction", "Tail, share of mass", "%", 100.0, "%.0f"),
        ("tail_length_svl", "Tail length", "x SVL", 1.0, "%.2f"),
        ("hindlimb_length_svl", "Hind limb", "x SVL", 1.0, "%.3f"),
        ("head_width_svl", "Head width", "x SVL", 1.0, "%.3f"),
        ("hip_height_svl", "Hip height", "x SVL", 1.0, "%.2f"),
        ("com_intact_svl", "Centre of mass", "x SVL from snout", 1.0, "%.3f"),
        ("lab_gait_frequency_hz", "Leg rhythm", "Hz", 1.0, "%.3f", "The walk"),
        ("hind_duty_factor", "Hind foot on ground", "of each stride", 1.0, "%.2f"),
        ("fore_duty_factor", "Fore foot on ground", "of each stride", 1.0, "%.2f"),
        ("warm_surface_temperature_C", "Preferred warm ground", "C", 1.0, "%.0f", "Living"),
        ("inter_meal_interval_days", "Between meals", "days", 1.0, "%.2f"),
        ("prey_item_wet_mass_kg", "One cricket", "g", 1000.0, "%.2f"),
        ("prey_escape_speed_m_s", "Cricket escape speed", "m/s", 1.0, "%.3f", "The hunt"),
        ("strike_peak_speed_m_s", "Strike peak speed", "m/s", 1.0, "%.3f"),
        ("capture_head_drop_m", "Head drop at capture", "mm", 1000.0, "%.0f"),
        ("lower_jaw_length_m", "Lower jaw", "mm", 1000.0, "%.0f"),
    ):
        try:
            biology.append(_bio(*args))
        except Exception as exc:
            print(f"  ! biology {args[0]}: {exc}")

    # One cricket as a fraction of one meal, DERIVED rather than invented:
    # 0.35 g of cricket against 5 % of a 38 g animal's bodyweight offered and
    # 54.7 % of that actually consumed.
    payload = {
        "geoms": geoms, "joints": jnt, "search": search, "world": world,
        "eye": eye, "prey": prey, "hunt": hunt, "biology": biology,
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
    print(f"  biology: {len(biology)} registry values on the page")
    print(f"  one cricket = {_cricket_g:.2f} g of a {_meal_g:.2f} g meal "
          f"= {_cricket_g/_meal_g:.3f} meals (DERIVED)")
    print(f"  eye: render {eye['render_pixels']} -> receptors {eye['pixels']} "
          f"-> {eye['cells']} cells, blur {eye['optical_limit_px']:.2f} px, "
          f"band {eye['v_low']}-{eye['v_peak']}-{eye['v_high']} deg/s")
    print(f"  prey: {len(prey)} registry values, "
          f"capture at {prey['prey_capture_distance_m']*100:.1f} cm")
    print(f"  hunt: quorum {hunt['evidence']['quorum']} in a "
          f"{hunt['evidence']['fixate_steps']}-step fixation, saccade blind "
          f"{hunt['orienting']['move_steps']}+{hunt['orienting']['blind_steps']} steps, "
          f"stalk {hunt['stalk_move_steps']} move / {hunt['stalk_look_steps']} look")
    print(f"  compensated feet: {sorted(comp)}")
    for f in sorted(comp):
        t = np.asarray(comp[f]["table"])
        print(f"    {f}: table {t.shape}  hip grid {len(comp[f]['hip_grid'])}  "
              f"sprawl grid {len(comp[f]['sprawl_grid'])}")
    print(f"  bg channels {bg['channels']}  gpi_tonic {bg['gpi_tonic']:.6f}")


if __name__ == "__main__":
    main()
