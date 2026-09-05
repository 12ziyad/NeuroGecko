"""Read-only summary of retained Session2 trials and quaternion steering traces.

No environment, policy, optimizer or training is imported. Outputs are derived
reports; original reports/traces are hashed and never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FEET = ("HL", "FL", "HR", "FR")


def read_hashed(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(data), {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    z = np.mean(np.exp(1j*values))
    return float(np.angle(z)) if abs(z) > 1e-12 else None


def steering_from_trace(trace, *, goal_distance_m=10.):
    """Use actual quaternion orientation, not displacement direction as body yaw.

    goal_distance_m is an explicit historical runner reconstruction (not a
    fitted parameter). The directly logged commanded world angle is reported
    separately, so the inferred finite target point is never hidden.
    """
    s, meta = trace["samples"], trace["metadata"]
    t = np.asarray(s["time_s"], dtype=float)
    xyz = np.asarray(s["trunk_position_m"], dtype=float)
    q = np.asarray(s["trunk_quaternion_wxyz"], dtype=float)
    f = float(meta["commanded_frequency_hz"])
    goal_angle = float(meta["goal_angle_rad"])
    if len(t) < 3 or xyz.shape != (len(t), 3) or q.shape != (len(t), 4):
        raise ValueError("Need time[N], trunk_position[N,3], quaternion_wxyz[N,4].")
    if not np.isfinite(t).all() or not np.all(np.diff(t)>0) or not np.isfinite(xyz).all() or not np.isfinite(q).all() or not np.isfinite([f,goal_angle,goal_distance_m]).all() or f <= 0 or goal_distance_m <= 0:
        raise ValueError("Invalid steering trace values.")
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-norm quaternion cannot define heading.")
    q = q/norms[:,None]
    w,x,y,z = q.T
    forward_xy = np.column_stack([1-2*(y*y+z*z), 2*(x*y+w*z)])
    valid = np.linalg.norm(forward_xy, axis=1)>1e-9
    yaw = np.where(valid, np.arctan2(forward_xy[:,1], forward_xy[:,0]), np.nan)
    target = xyz[0,:2]+goal_distance_m*np.array([np.cos(goal_angle),np.sin(goal_angle)])
    offset = target[None,:]-xyz[:,:2]
    bearing = np.arctan2(offset[:,1],offset[:,0])
    error = wrap(bearing-yaw)
    fixed_error = wrap(goal_angle-yaw)
    first_cycle = int(np.ceil(t[0]*f-1e-9))
    last_cycle = int(np.floor(t[-1]*f+1e-9))-1
    if last_cycle <= first_cycle:
        raise ValueError("Need at least two fully observed commanded cycles.")

    def window(cycle):
        a,b = cycle/f,(cycle+1)/f
        selected = (t >= a-1e-10) & (t < b-1e-10) & valid
        indices = np.flatnonzero(selected)
        if not len(indices):
            return {"cycle_index":cycle, "status":"unavailable: no valid projected heading samples"}
        return {"cycle_index":cycle, "commanded_window_s":[a,b],
                "actual_sample_window_s":[float(t[indices[0]]),float(t[indices[-1]])],
                "sample_count":int(len(indices)), "mean_body_heading_rad":angle_mean(yaw[selected]),
                "mean_finite_goal_bearing_rad":angle_mean(bearing[selected]),
                "mean_signed_goal_bearing_error_rad":angle_mean(error[selected]),
                "mean_absolute_goal_bearing_error_rad":float(np.mean(np.abs(error[selected]))),
                "mean_signed_error_to_recorded_command_angle_rad":angle_mean(fixed_error[selected]),
                "mean_absolute_error_to_recorded_command_angle_rad":float(np.mean(np.abs(fixed_error[selected])))}

    def displacement(start_s):
        indices = np.flatnonzero(t >= start_s-1e-9)
        if not len(indices):
            return {"status":"unavailable after requested cutoff"}
        start = int(indices[0])
        delta = xyz[-1,:2]-xyz[start,:2]
        distance = float(np.linalg.norm(delta))
        heading = float(np.arctan2(delta[1],delta[0])) if distance > 1e-9 else None
        return {"actual_window_s":[float(t[start]),float(t[-1])], "delta_xy_m":delta.tolist(),
                "net_distance_m":distance, "displacement_heading_rad":heading,
                "heading_minus_recorded_goal_angle_rad":float(wrap(heading-goal_angle)) if heading is not None else None}

    windows = {"first_full_commanded_cycle":window(first_cycle), "last_full_commanded_cycle":window(last_cycle)}
    postsettle = int(np.ceil(3.*f-1e-9))
    if postsettle <= last_cycle:
        windows["first_full_commanded_cycle_after_3s"] = window(postsettle)
    first,last = windows["first_full_commanded_cycle"],windows["last_full_commanded_cycle"]
    initial_yaw = float(yaw[np.flatnonzero(valid)[0]]) if np.any(valid) else None
    final_heading = last.get("mean_body_heading_rad")
    desired_turn = float(wrap(goal_angle-initial_yaw)) if initial_yaw is not None else None
    actual_turn = float(wrap(final_heading-initial_yaw)) if initial_yaw is not None and final_heading is not None else None
    direction = (bool(np.sign(actual_turn)==np.sign(desired_turn))
                 if actual_turn is not None and desired_turn is not None and abs(desired_turn)>1e-9 else None)
    improved = (last["mean_absolute_goal_bearing_error_rad"] < first["mean_absolute_goal_bearing_error_rad"]
                if "mean_absolute_goal_bearing_error_rad" in first and "mean_absolute_goal_bearing_error_rad" in last else None)
    return {"goal_angle_rad":goal_angle, "commanded_frequency_hz":f, "commanded_period_s":1/f,
            "quaternion_convention":"wxyz; rotate trunk-local +X into world, project to XY, atan2(y,x). Quaternion sign does not change heading.",
            "maximum_quaternion_norm_deviation":float(np.max(np.abs(norms-1))),
            "undefined_planar_heading_sample_count":int(np.sum(~valid)),
            "goal_reconstruction":{"target_xy_m":target.tolist(), "distance_m":goal_distance_m,
                                   "source":"INFERRED from historical eval/session2_controller.py: initial trunk XY + 10*[cos(goal_angle),sin(goal_angle)]; angle is directly logged, target coordinates/distance are not.",
                                   "assumption":"No goal resampling in these short distant-goal trials; commanded angle comparison is also given independently."},
            "angle_units":"radians; positive error means target lies left/counter-clockwise of body heading",
            "cycle_window_definition":"First and last complete commanded cycles anchored at t=0; final partial cycle excluded. These are clock windows, not inferred contact strides.",
            "windows":windows, "displacement":{"full_run":displacement(float(t[0])), "after_3s":displacement(3.)},
            "directional_check":{"initial_body_heading_rad":initial_yaw, "requested_turn_from_initial_rad":desired_turn,
                                 "last_cycle_body_turn_from_initial_rad":actual_turn, "turn_sign_matches_request":direction,
                                 "mean_absolute_goal_error_reduced_first_to_last_cycle":improved,
                                 "status":"Directional response checks only; no steering tolerance, robustness, biological, or walking pass is inferred."}}


def collect_trials(trials_dir):
    rows=[]
    for path in sorted(Path(trials_dir).glob("*/report.json")):
        r, source = read_hashed(path)
        gait, gate = r["gait"], r["gate2"]
        quality=gait["contact_quality"]
        feet={}
        for foot in FEET:
            limb=gait.get("limbs",{}).get(foot,{})
            diagnostic=limb.get("contact_cycle_diagnostic",{})
            feet[foot]={"observed_contact_cycle_rate_hz":diagnostic.get("observed_contact_cycle_rate_hz"),
                        "commanded_frequency_hz":diagnostic.get("commanded_frequency_hz"),
                        "period_cv":limb.get("stride_period_cv"), "rate_entrainment_pass":diagnostic.get("entrainment_pass"),
                        "complete_contact_cycle_count":diagnostic.get("complete_cycle_count")}
        trace_path=path.parent/"traces/episode_000.json"
        retained=None
        if trace_path.exists():
            trace_bytes=trace_path.read_bytes()
            actual=hashlib.sha256(trace_bytes).hexdigest()
            expected=r.get("trace_sha256")
            if expected is not None and actual != expected:
                raise ValueError("Saved trace hash differs from its original report: "+str(trace_path))
            retained={"path":str(trace_path.resolve()),"sha256":actual,"matches_recorded_report_hash":actual==expected if expected else None}
        timing_fields=[quality.get(k) for k in ("entrainment_pass","single_digit_period_cv_pass","acquisition_at_least_200_hz")]
        rows.append({"case":r["case"], "source_report":source, "retained_trace":retained,
                     "trace_provenance_status":"retained raw trace hash verified" if retained else "raw trace was not retained; saved report hash only",
                     "protocol":r["protocol"], "gate2_pass_as_recorded":gate["pass"], "completed_without_fall":gate["completed_without_fall"],
                     "six_checks_as_recorded":gate["checks"], "requirements_as_recorded":gate["requirements"],
                     "contact_timing":{"per_foot":feet, "rate_entrainment_all_feet":quality.get("entrainment_pass"),
                                       "period_cv_single_digits_all_feet":quality.get("single_digit_period_cv_pass"),
                                       "acquisition_at_least_200_hz":quality.get("acquisition_at_least_200_hz"),
                                       "timing_and_acquisition_valid_recorded":gate.get("timing_and_acquisition_valid"),
                                       "timing_and_acquisition_valid_from_report_components":all(timing_fields) if all(x is not None for x in timing_fields) else None},
                     "sample_rate_hz":gait.get("sample_rate_hz"), "observed_duration_s":gait.get("observed_duration_s")})
    if not rows:
        raise ValueError("No retained trial reports found.")
    return {"schema_version":1,"claim":"Retained CPU-only manual trials; no biological validation. Gate2 remains failed; no further tuning, training, or plant changes are implied.",
            "repeat_unit":"Each case is n=1 deterministic trajectory, not an independent-animal sample. Replays are not pooled and no across-case/replay SD is computed.",
            "selection":"Root selected19_front_height as experimental lab defaults; final_candidate and verified_final are retained confirmations. Later failed alternatives are not hidden.",
            "value_policy":"Copy six gate values/pass flags from each saved report. Early reports may use earlier endpoint masks; source report hashes and missing validity fields remain explicit.",
            "trial_count":len(rows),"gate2_passing_cases":[row["case"] for row in rows if row["gate2_pass_as_recorded"]],"trials":rows}


def fmt(value, factor=1):
    return "missing" if value is None else f"{value*factor:.4f}"


def flag(value):
    return "?" if value is None else "P" if value else "F"


def comparison_markdown(report):
    lines=["# Session2 retained trial comparison", "",report["claim"],"", report["repeat_unit"],"",report["selection"],"",
           "P/F denote saved engineering checks, not biological validation. Paired values are left/right. Loads are raw scalar touch-threshold fractions conditional on the executed held command; they are not world-vertical forces.","",
           "## Six decision metrics", "",
           "Targets: forward ≥0.04m/s; net/path ≥0.5; each hind swing load <0.10; each front stance load ≥0.65; each hind duty0.78±0.05; each ipsilateral phase0.435±0.03. Timing validity and completing without a fall are also required.","",
           "| Case | Forward m/s | Net/path | Hind swing HL/HR | Front stance FL/FR | Hind duty HL/HR | Phase HL→FL/HR→FR | Gate2 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["trials"]:
        c=row["six_checks_as_recorded"]
        def pair(key, feet):
            return "/".join(fmt(c[key]["values"].get(f)) for f in feet)+" "+flag(c[key]["pass"])
        lines.append(f"| {row['case']} | {fmt(c['signed_forward'].get('value_m_s'))} {flag(c['signed_forward']['pass'])} | {fmt(c['net_path'].get('value'))} {flag(c['net_path']['pass'])} | {pair('hind_swing_load',('HL','HR'))} | {pair('front_stance_load',('FL','FR'))} | {pair('hind_duty',('HL','HR'))} | {pair('limb_phase',('HL_to_FL','HR_to_FR'))} | {flag(row['gate2_pass_as_recorded'])} |")
    lines.extend(["","## Contact timing and acquisition", "",
                  "Each foot cell is observed contact cycles/s / periodCV%. These remain contact-cycle diagnostics. Failed entrainment is not repaired by merging events. Missing means unavailable, not zero.","",
                  "| Case | HL rate/CV% | FL rate/CV% | HR rate/CV% | FR rate/CV% | All rates pass | All CV single digits | ≥200Hz | No fall/completed |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["trials"]:
        t=row["contact_timing"]
        cells=[fmt(t["per_foot"][f]["observed_contact_cycle_rate_hz"])+" / "+fmt(t["per_foot"][f]["period_cv"],100) for f in FEET]
        lines.append("| "+" | ".join([row["case"],*cells,flag(t["rate_entrainment_all_feet"]),flag(t["period_cv_single_digits_all_feet"]),flag(t["acquisition_at_least_200_hz"]),flag(row["completed_without_fall"])])+" |")
    lines.extend(["","## Provenance", "",report["value_policy"],"",
                  "The companion JSON contains every original report SHA256, retained-trace SHA256 verification, body/source hashes, exact controller parameters, per-foot rate flags, and recorded requirements. Most exploratory trials retained reports only. No original evidence is changed.",""])
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-dir",type=Path,default=REPO/"artifacts/evidence/session2/trials")
    parser.add_argument("--output-dir",type=Path,default=REPO/"artifacts/evidence/session2")
    args=parser.parse_args(argv)
    report=collect_trials(args.trials_dir)
    by_case={r["case"]:r for r in report["trials"]}
    report["selected_confirmation_equality"]={
        case: {"six_checks_exactly_equal_to_selected19": by_case[case]["six_checks_as_recorded"] == by_case["19_front_height"]["six_checks_as_recorded"],
               "comparison_source_report_sha256":by_case[case]["source_report"]["sha256"]}
        for case in ("final_candidate","verified_final") if case in by_case and "19_front_height" in by_case}
    steering={"schema_version":1,"claim":"Three deterministic steering probes, no pooling/SD and no biological or Gate2 pass claim.","runs":{}}
    matched_conditions={}
    for case in ("final_candidate","steering_left","steering_right"):
        trace,source=read_hashed(args.trials_dir/case/"traces/episode_000.json")
        trial,trial_source=read_hashed(args.trials_dir/case/"report.json")
        if trial.get("trace_sha256") != source["sha256"]:
            raise ValueError("Steering trace/report SHA mismatch: "+case)
        matched_conditions[case]={key:trace["metadata"].get(key) for key in
                                  ("xml_sha256","lab_parameters","commanded_stance_by_foot","phase_offsets_cycle","commanded_frequency_hz","seed","reset_noise","source_sha256")}
        matched_conditions[case]["initial_qpos"]=trace["samples"]["qpos"][0]
        steering["runs"][case]={**steering_from_trace(trace),"original_trace":source,"source_report":trial_source,
                                "xml_sha256":trace["metadata"].get("xml_sha256"),"gate2_pass_as_recorded":trial["gate2"]["pass"],
                                "source_sha256":trace["metadata"].get("source_sha256")}
    center=steering["runs"]["final_candidate"]
    comparisons={}
    for case in ("steering_left","steering_right"):
        item=steering["runs"][case]
        last=item["windows"]["last_full_commanded_cycle"]["mean_body_heading_rad"]
        center_last=center["windows"]["last_full_commanded_cycle"]["mean_body_heading_rad"]
        heading_shift=float(wrap(last-center_last))
        displacement=item["displacement"]["after_3s"]["displacement_heading_rad"]
        center_displacement=center["displacement"]["after_3s"]["displacement_heading_rad"]
        displacement_shift=float(wrap(displacement-center_displacement)) if displacement is not None and center_displacement is not None else None
        goal_shift=float(wrap(item["goal_angle_rad"]-center["goal_angle_rad"]))
        comparisons[case]={"recorded_goal_angle_shift_from_center_rad":goal_shift,
                           "last_cycle_body_heading_shift_from_center_rad":heading_shift,
                           "postsettle_displacement_heading_shift_from_center_rad":displacement_shift,
                           "body_shift_sign_matches_goal_shift":bool(np.sign(heading_shift)==np.sign(goal_shift)),
                           "displacement_shift_sign_matches_goal_shift":bool(np.sign(displacement_shift)==np.sign(goal_shift)) if displacement_shift is not None else None}
    steering["paired_directional_comparisons"]=comparisons
    steering["controlled_comparison"]={
        case:{key: value==matched_conditions["final_candidate"][key] for key,value in matched_conditions[case].items()}
        for case in ("steering_left","steering_right")}
    steering["limitations"]=["One goal per deterministic trial; response direction is not evidence of disturbance rejection, broad steering robustness, or successful locomotion gates.",
                              "First complete cycle includes reset/startup. First complete post3s cycle is additionally reported. Body heading comes from orientation; displacement heading is distinct.",
                              "Reconstructed finite target requires the historical10m target rule; exact command-angle errors do not rely on that distance."]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in (("trial_comparison.json",report),("steering_checks.json",steering)):
        (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (args.output_dir/"trial_comparison.md").write_text(comparison_markdown(report),encoding="utf-8")
    print(json.dumps({"trial_count":report["trial_count"],"passing_cases":report["gate2_passing_cases"],"steering_directional_comparisons":comparisons},indent=2))


if __name__=="__main__":
    main()
