"""Explicit CPU-only, zero-policy controller trial. No optimizer or learning.

Each invocation changes only caller-declared parameters, records all six gates,
and reruns legacy fixtures. Failed trials are retained; this is not validation.
Invoke from repository root: python -m eval.session2_controller --case NAME
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from common.gait_config import get_gait_profile
from common.provenance import parameter_value
from envs.gecko_walk_env import GeckoWalkEnv
from realism_metrics import TraceRecorder, analyze_trace, write_json, sha256, FEET

REPO = Path(__file__).resolve().parents[1]


def gate2(trace, result, diagnosis, completed):
    """All limbs must individually pass, not just the pooled average."""
    spec = parameter_value("session2_gate2")
    limbs = diagnosis["feet"]
    swing = {f: limbs[f]["commanded_swing"]["raw_loaded_fraction"] for f in ("HL", "HR")}
    stance = {f: limbs[f]["commanded_stance"]["raw_loaded_fraction"] for f in ("FL", "FR")}
    duty = {f: result["limbs"][f]["duty_factor"]["mean"] for f in ("HL", "HR")}
    phase = {f: result["limb_phase"][f]["mean_cycle"] for f in ("HL_to_FL", "HR_to_FR")}
    speed = diagnosis["motion"]["signed_body_forward_speed_m_s"].get("mean")
    ratio = diagnosis["motion"]["net_path_ratio"]
    all_test = lambda vals, predicate: all(x is not None and predicate(x) for x in vals)
    checks = {
        "signed_forward": {"value_m_s":speed, "pass": speed is not None and speed >= spec["minimum_forward_m_s"]},
        "net_path": {"value":ratio, "pass":ratio is not None and ratio >= spec["minimum_net_path"]},
        "hind_swing_load": {"values":swing, "pass":all_test(swing.values(), lambda x:x < spec["maximum_hind_swing_load"])},
        "front_stance_load": {"values":stance, "pass":all_test(stance.values(), lambda x:x >= spec["minimum_front_stance_load"])},
        "hind_duty": {"values":duty, "pass":all_test(duty.values(), lambda x:abs(x-spec["hind_duty_target"]) <= spec["hind_duty_tolerance"])},
        "limb_phase": {"values":phase, "pass":all_test(phase.values(), lambda x:abs((x-spec["limb_phase_target"]+.5)%1-.5) <= spec["limb_phase_tolerance"])},
    }
    quality = result["contact_quality"]
    entrained = quality["entrainment_pass"]
    timing_valid = entrained and quality["single_digit_period_cv_pass"] and quality["acquisition_at_least_200_hz"]
    return {"pass":bool(completed and timing_valid and all(c["pass"] for c in checks.values())),
            "completed_without_fall":bool(completed), "observed_contacts_entrained":entrained,
            "timing_and_acquisition_valid":bool(timing_valid),
            "requirements":spec, "checks":checks,
            "timing_caveat":"Duty/phase retain operational contact-cycle values; non-entrained cycles cannot certify walking."}


def run_trial(name, parameters, *, hind_duty=None, limb_phase=None, duration=20, seed=0, goal_angle=0, save_trace=False):
    from eval.base_diagnostics import diagnose_trace
    out = REPO/"artifacts/evidence/session2/trials"/name
    if (out/"report.json").exists():
        raise FileExistsError("Preserve earlier trial evidence: "+str(out))
    regression = subprocess.run([sys.executable, "-m", "unittest", "tests.test_gait_config.LegacyFixtureTests", "tests.test_highrate_trace", "-q"], cwd=REPO, capture_output=True, text=True, timeout=60)
    if regression.returncode:
        raise RuntimeError(regression.stdout+regression.stderr)
    profile = get_gait_profile("lab")
    if hind_duty is not None:
        profile = replace(profile, stance_ratios=(hind_duty,profile.stance_for("FL"),hind_duty,profile.stance_for("FR")))
    if limb_phase is not None:
        profile = replace(profile, touchdown_delays_cycle=(0.,limb_phase,.5,(.5+limb_phase)%1))
    xml = REPO/"morphology/gecko_body_lab_v2.xml"
    env = GeckoWalkEnv(xml_path=xml, gait_profile=profile, control_mode="cpg_residual", reset_noise=0,
                       residual_scale=.25, max_steps=round(duration/.02), lab_parameters=parameters)
    try:
        env.reset(seed=seed)
        env.target = env.data.xpos[env._trunk,:2]+10*np.array([np.cos(goal_angle),np.sin(goal_angle)])
        _, env._prev_dist, _ = env._target_egocentric()
        meta = {"seed":seed,"gait_profile":"lab", "xml_sha256":sha256(xml), "n":1,
                "reset_noise":0,"goal_angle_rad":goal_angle,"lab_parameters":dict(env.cpg.lab_parameters),
                "commanded_stance_by_foot":profile.stance_by_foot,"phase_offsets_cycle":profile.touchdown_delays,
                "source_sha256":{p:sha256(REPO/p) for p in ("envs/cpg_residual_controller.py","envs/gecko_walk_env.py","realism_metrics.py","config/proxies.yaml")}}
        recorder = TraceRecorder(env,meta,physics_substeps=5)
        recorder.record()
        term = trunc = False
        for _ in range(env.max_steps):
            _,_,term,trunc,_ = env.step(np.zeros(env.nu,dtype=np.float32))
            if term or trunc:
                break
        recorder.detach()
        trace = recorder.as_dict()
        if env.data.time > 3:
            result = analyze_trace(trace,settle_s=3)
            diagnosis = diagnose_trace(trace,settle_s=3,entrainment_tolerance_fraction=.10)
        else:
            # An early fall is a result, not an exception that hides evidence.
            result = {"status":"insufficient_post_settling_data", "limbs":{f:{"duty_factor":{"mean":None}} for f in FEET},
                      "limb_phase":{f:{"mean_cycle":None} for f in ("HL_to_FL","HR_to_FR")},
                      "contact_quality":dict(entrainment_pass=False,single_digit_period_cv_pass=False,acquisition_at_least_200_hz=True)}
            diagnosis = {"feet":{f:{k:{"raw_loaded_fraction":None} for k in ("commanded_stance","commanded_swing")} for f in FEET},
                         "motion":{"signed_body_forward_speed_m_s":{"mean":None},"net_path_ratio":None}}
        gate = gate2(trace,result,diagnosis,not term and np.isclose(env.data.time,duration))
        report = {"case":name,"protocol":meta,"gate2":gate,"gait":result,"diagnosis":diagnosis,
                  "legacy_regression":regression.stderr,"claim":"Manual diagnostic trial, no policy training; n=1, no across-repeat SD."}
        if save_trace:
            path = out/"traces/episode_000.json"
            write_json(path,trace)
            report["trace_sha256"] = sha256(path)
        write_json(out/"report.json", report)
        write_json(out/"parameters.json",dict(env.cpg.lab_parameters))
        print(json.dumps({"case":name, "values":{k:v.get("values",v.get("value",v.get("value_m_s"))) for k,v in gate["checks"].items()},
                          "pass":gate["pass"],"entrained":gate["observed_contacts_entrained"]}), flush=True)
        return report
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",required=True)
    p.add_argument("--parameters",type=Path,help="Explicit complete parameter dictionary from a prior trial")
    p.add_argument("--set",action="append",default=[],help="Single declared change KEY=JSON_VALUE")
    p.add_argument("--hind-duty",type=float)
    p.add_argument("--limb-phase",type=float)
    p.add_argument("--goal-angle",type=float,default=0)
    p.add_argument("--save-trace",action="store_true")
    a=p.parse_args()
    if len(a.set)+int(a.hind_duty is not None)+int(a.limb_phase is not None)>1:
        p.error("One change at a time; save each trial before the next.")
    if Path(a.case).name != a.case or a.case in (".",".."):
        p.error("Case must be a simple output name.")
    values=json.loads(a.parameters.read_text()) if a.parameters else dict(parameter_value("lab_base_parameters"))
    for assignment in a.set:
        key,value=assignment.split("=",1)
        values[key]=json.loads(value)
    run_trial(a.case,values,hind_duty=a.hind_duty,limb_phase=a.limb_phase,goal_angle=a.goal_angle,save_trace=a.save_trace)


if __name__=="__main__":
    main()
