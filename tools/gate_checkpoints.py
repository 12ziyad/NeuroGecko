#!/usr/bin/env python3
"""Score every checkpoint of a training run through the official Gate 2.

The reward a policy is trained on and the gates it is judged by are different
quantities, and they disagree. `ep_rew_mean` rising proves the policy is getting
better at what it was *told* to want; only the gates say whether the gait moved
toward the published animal. Session 3g is the standing warning: a harness that
reimplemented limb phase optimised a quantity that diverged from the gate and
reported 0.4497 where the real measure read 0.6534.

So nothing here re-derives a gate value. Each checkpoint is rolled out by
`realism_metrics.py` in a subprocess -- the same entry point the base evidence
used -- and the resulting trace is scored by `eval.session2_controller.gate2`,
the same function `tests/test_session2_gate.py` pins and `docs/BLOCKED.md`
tabulates. This tool only loops, tabulates and ranks.

Ranking is by gates PASSED first. The normalised-distance figure is a tie-break
between equal gate counts, computed from gate2's own reported values against
its own registry spec; it is a sort key, never a claim.

Usage:
    python tools/gate_checkpoints.py --run models/<run> \
        --evidence-dir artifacts/evidence/session4/<run>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.checkpoints import verify_checkpoint_bundle  # noqa: E402
from common.provenance import parameter_value  # noqa: E402
from eval.base_diagnostics import diagnose_trace  # noqa: E402
from eval.session2_controller import gate2  # noqa: E402
from realism_metrics import analyze_trace, sha256, write_json  # noqa: E402
from tools.fit_gate2_official import CV_CEILING  # noqa: E402

CHECK_ORDER = ("signed_forward", "net_path", "hind_swing_load",
               "front_stance_load", "hind_duty", "limb_phase")
# Tie-break scaling only. One unit is roughly "a whole gate's width" for each
# check, so no single check dominates the ordering of equally-passing runs.
TIEBREAK_SCALE = {"signed_forward": 0.02, "net_path": 0.20, "hind_swing_load": 0.05,
                  "front_stance_load": 0.15, "hind_duty": 0.05, "limb_phase": 0.06}


def _bands(spec):
    """The six acceptance bands, read from gate2's own reported requirements."""
    return {
        "signed_forward": (spec["minimum_forward_m_s"], math.inf),
        "net_path": (spec["minimum_net_path"], math.inf),
        "hind_swing_load": (-math.inf, spec["maximum_hind_swing_load"]),
        "front_stance_load": (spec["minimum_front_stance_load"], math.inf),
        "hind_duty": (spec["hind_duty_target"] - spec["hind_duty_tolerance"],
                      spec["hind_duty_target"] + spec["hind_duty_tolerance"]),
        "limb_phase": (spec["limb_phase_target"] - spec["limb_phase_tolerance"],
                       spec["limb_phase_target"] + spec["limb_phase_tolerance"]),
    }


def _outside(name, value, low, high):
    """Distance outside the band, in the check's own units. Zero inside it."""
    if name == "limb_phase":
        # gate2 measures limb phase circularly; a linear gap would misreport a
        # value that has wrapped past 1.0.
        target, tolerance = (low + high) / 2.0, (high - low) / 2.0
        return max(abs((value - target + .5) % 1 - .5) - tolerance, 0.0)
    return max(low - value if low > -math.inf else 0.0,
               value - high if high < math.inf else 0.0, 0.0)


def check_values(gate):
    """One number per check: the WORST limb, in that check's own direction.

    gate2 requires every named limb to pass individually, so the limb that
    decides the check is the one furthest outside its band -- not the minimum,
    which would flatter a 'must stay below' check like hind swing load.
    """
    bands = _bands(gate["requirements"])
    values = {}
    for name in CHECK_ORDER:
        payload = gate["checks"][name]
        if "values" in payload:
            numbers = [v for v in payload["values"].values() if v is not None]
            if not numbers:
                values[name] = None
                continue
            low, high = bands[name]
            values[name] = max(numbers, key=lambda v: _outside(name, v, low, high))
        else:
            values[name] = payload.get("value", payload.get("value_m_s"))
    return values


def tiebreak_distance(gate):
    """Summed normalised worst-limb distance outside each band. Zero when all pass."""
    bands = _bands(gate["requirements"])
    values = check_values(gate)
    total = 0.0
    for name, (low, high) in bands.items():
        value = values[name]
        if value is None:
            return math.inf
        total += _outside(name, value, low, high) / TIEBREAK_SCALE[name]
    return total


def run_metrics(base_flags, model, normalizer, output, trace_dir, video=None):
    """Roll out one checkpoint (or the zero-residual base) and return its report."""
    command = [sys.executable, str(REPO / "realism_metrics.py"), *base_flags,
               "--output", str(output), "--trace-dir", str(trace_dir)]
    if model is None:
        command.append("--zero-residual")
    else:
        command += ["--model", str(model), "--vecnormalize", str(normalizer)]
    if video is not None:
        command += ["--video", str(video)]
    finished = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
    if finished.returncode != 0 or not Path(output).exists():
        return None, (finished.stderr or finished.stdout)[-800:]
    return json.loads(Path(output).read_text(encoding="utf-8")), None


def score_report(report, trace_path, settle):
    """Official Gate 2 on this rollout. No metric is recomputed here."""
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    episode = report["episodes"][0]
    completed = bool(episode.get("completed_requested_duration"))
    if episode["gait"].get("status") != "measured":
        return None, completed
    result = analyze_trace(trace, settle)
    diagnosis = diagnose_trace(trace, settle_s=settle, entrainment_tolerance_fraction=.10)
    gate = gate2(trace, result, diagnosis, completed and not episode.get("terminated"))
    gate["_stride_period_cv"] = result["limbs"]["HL"]["stride_period_cv"]
    return gate, completed


def base_flags_from_config(config, duration, settle):
    """Reconstruct the evaluation protocol from the run's own training config.

    Read from train_config.json rather than retyped, so the body, profile and
    stance compensation a checkpoint is scored under cannot drift from the ones
    it was trained under.
    """
    compensation = config.get("requested_hind_stance_compensation", False)
    flags = ["--xml", config["xml_path_resolved"],
             "--gait-profile", config.get("gait_profile", "legacy"),
             "--episodes", "1", "--seed", "0",
             "--duration", str(duration), "--settle", str(settle)]
    if compensation:
        flags += ["--hind-stance-compensation", "all" if compensation == "all" else "hind"]
    scale = config.get("residual_scale")
    if scale is not None:
        flags += ["--residual-scale", str(scale)]
    return flags


def step_number(bundle):
    name = bundle.name
    digits = "".join(c for c in name.split("step-")[-1] if c.isdigit())
    return int(digits) if digits else -1


def markdown(rows, run_name, protocol):
    lines = [f"# Gate curve - {run_name}", "",
             "Every row is `realism_metrics.py` + `eval.session2_controller.gate2`,",
             "the same code path as the base evidence. Reward is not a column here",
             "on purpose: it is not the quantity being claimed.", "",
             f"A row whose stride-period CV exceeds {CV_CEILING} is REJECTED and cannot be",
             "selected as best, however many gates it passes: an irregular gait can raise",
             "a contact-load check without walking any better. Same rule as",
             "`tools/fit_gate2_official.py`.", "",
             "Protocol: " + json.dumps(protocol, sort_keys=True), "",
             "| step | fwd m/s | net/path | hind swing | front stance | hind duty | limb phase | CV | gates | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.4f}"
    for row in rows:
        label = "base (policy off)" if row["step"] < 0 else f"{row['step']:,}"
        if row.get("error"):
            lines.append(f"| {label} | | | | | | | | - | FAILED: {row['error'][:60]} |")
            continue
        values = row["values"]
        marks = "".join("P" if row["passed"][n] else "." for n in CHECK_ORDER)
        lines.append("| " + " | ".join([
            label, fmt(values["signed_forward"]), fmt(values["net_path"]),
            fmt(values["hind_swing_load"]), fmt(values["front_stance_load"]),
            fmt(values["hind_duty"]), fmt(values["limb_phase"]),
            fmt(row["stride_period_cv"]), f"{row['gates_passed']}/6 {marks}",
            row.get("note", ""),
        ]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="models/<run> directory to score")
    p.add_argument("--evidence-dir", required=True, help="where to write the gate curve")
    p.add_argument("--include-zero-residual", action="store_true", default=True,
                   help="also score the base with the policy off (the comparison row)")
    p.add_argument("--no-zero-residual", dest="include_zero_residual", action="store_false")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--settle", type=float, default=None)
    p.add_argument("--video-best", action="store_true",
                   help="re-run the winning checkpoint to record an mp4")
    args = p.parse_args(argv)

    duration = args.duration if args.duration is not None else float(parameter_value("evaluation_duration_s"))
    settle = args.settle if args.settle is not None else float(parameter_value("evaluation_settle_s"))
    run = Path(args.run).resolve()
    config_path = run / "train_config.json"
    if not config_path.is_file():
        p.error(f"No train_config.json in {run}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    flags = base_flags_from_config(config, duration, settle)
    evidence = Path(args.evidence_dir).resolve()
    evidence.mkdir(parents=True, exist_ok=True)

    bundles = sorted((b for b in (run / "checkpoints").glob("step-*")
                      if (b / "manifest.json").is_file()), key=step_number)
    if not bundles and not args.include_zero_residual:
        p.error(f"No published checkpoint bundles under {run / 'checkpoints'}")

    targets = []
    if args.include_zero_residual:
        targets.append((-1, None, None))
    for bundle in bundles:
        verify_checkpoint_bundle(bundle)
        targets.append((step_number(bundle), bundle / "model.zip", bundle / "vecnormalize.pkl"))

    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for step, model, normalizer in targets:
            tag = "base" if step < 0 else f"step-{step}"
            out = Path(scratch) / f"{tag}.json"
            traces = Path(scratch) / tag
            report, error = run_metrics(flags, model, normalizer, out, traces)
            if report is None:
                rows.append({"step": step, "error": error or "realism_metrics failed"})
                print(f"{tag:>16}  FAILED", flush=True)
                continue
            gate, completed = score_report(report, traces / "episode_000.json", settle)
            if gate is None:
                rows.append({"step": step, "error": "episode not scorable (fell before settling)"})
                print(f"{tag:>16}  UNSCORABLE", flush=True)
                continue
            passed = {n: bool(gate["checks"][n]["pass"]) for n in CHECK_ORDER}
            note = []
            if not completed:
                note.append("did not complete")
            cv = gate["_stride_period_cv"]
            if cv is None or cv > CV_CEILING:
                note.append(f"REJECTED: stride CV {cv} > {CV_CEILING}")
            quality = report["episodes"][0]["gait"]["contact_quality"]
            if not quality["single_digit_period_cv_pass"]:
                note.append("registry CV check failed")
            if not quality["entrainment_pass"]:
                # The 4/6 base itself fails entrainment, so this is reported, not
                # a rejection -- see tools/fit_gate2_official.py on why it cannot
                # be a hard requirement without rejecting the incumbent.
                note.append("not entrained (base is too)")
            rows.append({
                "step": step, "values": check_values(gate), "passed": passed,
                "gates_passed": sum(passed.values()),
                "stride_period_cv": gate["_stride_period_cv"],
                "completed_without_fall": gate["completed_without_fall"],
                "timing_and_acquisition_valid": gate["timing_and_acquisition_valid"],
                "all_six_pass": bool(gate["pass"]),
                "tiebreak_distance": tiebreak_distance(gate),
                "note": "; ".join(note),
                "model_sha256": sha256(model) if model else None,
            })
            print(f"{tag:>16}  {sum(passed.values())}/6  "
                  + "  ".join(f"{n}={check_values(gate)[n]:.4f}" for n in CHECK_ORDER
                              if check_values(gate)[n] is not None), flush=True)

    # Stride-period CV above the ceiling is a HARD rejection, matching
    # tools/fit_gate2_official.py. Session 3g: the previous harness was rewarded
    # for driving CV from 0.0149 to 0.2713 and reported a limb phase the official
    # measure did not agree with. A checkpoint can raise front stance load simply
    # by walking irregularly, so an irregular gait cannot be "best" here either.
    scorable = [r for r in rows
                if not r.get("error") and r["completed_without_fall"]
                and r["stride_period_cv"] is not None
                and r["stride_period_cv"] <= CV_CEILING]
    baseline = next((r for r in rows if r["step"] < 0 and not r.get("error")), None)
    best = None
    # step-0 is published before any learning: it is the setup check that the
    # policy starts at the base, not a trained candidate. Ranking it as "best"
    # would credit training with the base controller's own score.
    trained = [r for r in scorable if r["step"] > 0]
    if trained:
        best = max(trained, key=lambda r: (r["gates_passed"], -r["tiebreak_distance"]))

    protocol = {"duration_s": duration, "settle_s": settle,
                "realism_metrics_flags": flags,
                "scored_by": "eval.session2_controller.gate2 on the rollout trace",
                "train_config_sha256": sha256(config_path)}
    summary = {
        "schema_version": 1,
        "run": str(run),
        "protocol": protocol,
        "stride_period_cv_ceiling": CV_CEILING,
        "rows_rejected_for_irregular_gait": sorted(
            r["step"] for r in rows
            if not r.get("error") and r["completed_without_fall"]
            and (r["stride_period_cv"] is None or r["stride_period_cv"] > CV_CEILING)),
        "rows": rows,
        "baseline_gates_passed": baseline["gates_passed"] if baseline else None,
        "best_step": best["step"] if best else None,
        "best_gates_passed": best["gates_passed"] if best else None,
        "beats_baseline": (bool(best and baseline and best["gates_passed"] > baseline["gates_passed"])
                           if baseline else None),
        "claim": "Instrumented simulation diagnostics; gates measured, reward not used for selection. "
                 "n=1 deterministic rollout per checkpoint; repeats of a frozen policy at zero reset "
                 "noise are the same experiment, not independent samples.",
    }
    write_json(evidence / "gate_curve.json", summary)
    (evidence / "gate_curve.md").write_text(markdown(rows, run.name, protocol), encoding="utf-8")

    print("\n" + markdown(rows, run.name, protocol))
    if baseline:
        print(f"baseline (policy off): {baseline['gates_passed']}/6")
    if best:
        verdict = "BEATS" if summary["beats_baseline"] else "does not beat"
        print(f"best checkpoint: step {best['step']} at {best['gates_passed']}/6 -- {verdict} the base")
    else:
        print("no scorable trained checkpoint")

    if args.video_best and best:
        bundle = run / "checkpoints" / f"step-{best['step']}"
        if not bundle.is_dir():
            bundle = next(b for b in bundles if step_number(b) == best["step"])
        with tempfile.TemporaryDirectory() as scratch:
            run_metrics(flags, bundle / "model.zip", bundle / "vecnormalize.pkl",
                        Path(scratch) / "best.json", Path(scratch) / "best",
                        video=evidence / "best.mp4")
        print(f"video -> {evidence / 'best.mp4'}")
    return summary


if __name__ == "__main__":
    main()
