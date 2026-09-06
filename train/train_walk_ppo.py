"""
train_walk_ppo.py  --  Phase-1 locomotion training (SB3 PPO, CPU, vectorised).

Windows note: SubprocVecEnv uses 'spawn'; this file keeps the entrypoint under
`if __name__ == "__main__":` so child processes import cleanly. For a quick
smoke on Windows use `--vec dummy` (single process, most robust).

Examples (run from the repo root C:\\Users\\ziyad\\GeckoBrain):
    # 10k plumbing smoke (a minute; just proves env->PPO->update->save works)
    python train/train_walk_ppo.py --vec dummy --envs 2 --steps 10000 --n-steps 512 --batch 256 --run smoke

    # V3 CPG sanity run (first run after patch; do not jump to 5M/20M)
    python train/train_walk_ppo.py --vec subproc --envs 8 --steps 200000 --run v3_cpg_sanity_200k

    # continue from an existing V3 checkpoint
    python train/train_walk_ppo.py --vec subproc --envs 16 --steps 5000000 --run v3_cpg_10m_continue --resume-from models/v3_cpg_5m/final.zip --resume-vec models/v3_cpg_5m/vecnormalize.pkl

    # V4.1 clean-gait probe from the V3 10M checkpoint
    python train/train_walk_ppo.py --vec subproc --envs 16 --steps 2000000 --run v4_1_clean_gait_2m --resume-from models/v3_cpg_10m_continue/final.zip --resume-vec models/v3_cpg_10m_continue/vecnormalize.pkl --ent-coef 0.01

    # V4.2 CPG-residual probe from the V3 10M checkpoint
    python train/train_walk_ppo.py --vec subproc --envs 16 --steps 2000000 --run v4_2_cpg_residual_2m --control-mode cpg_residual --residual-scale 0.2 --contact-thresh 0.0564 --resume-from models/v3_cpg_10m_continue/final.zip --resume-vec models/v3_cpg_10m_continue/vecnormalize.pkl --ent-coef 0.01

    # V4.2.1 posture/front-load probe
    python train/train_walk_ppo.py --vec subproc --envs 16 --steps 2000000 --run v4_2_1_posture_2m --control-mode cpg_residual --residual-scale 0.25 --front-stance-press 0.35 --contact-thresh 0.0564 --resume-from models/v4_2_cpg_residual_5m/final.zip --resume-vec models/v4_2_cpg_residual_5m/vecnormalize.pkl --ent-coef 0.004

    # longer runs are only for after the 200k sanity gate passes
"""
from __future__ import annotations
import argparse, sys
from dataclasses import asdict
import inspect
import json
import math
import platform
import subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from envs.gecko_walk_env import GeckoWalkEnv
from envs.cpg_residual_controller import CPGResidualController
from common.gait_config import get_gait_profile
from common.provenance import parameter_value
from common.checkpoints import (
    CheckpointBundleCallback, atomic_json, export_legacy_final,
    sha256_file, verify_checkpoint_bundle,
)


def gait_profile_parameters(profile: str) -> dict:
    """JSON-normalized resolved values, so later registry edits are detectable."""
    return json.loads(json.dumps(asdict(get_gait_profile(profile))))


def resume_gait_contract(model_path: Path | str) -> dict:
    """Old walker checkpoints predate profiles and use the legacy controller.

    New runs always save an explicit profile and resolved parameter snapshot.
    Do not infer a lab profile from a run's name or from unchanged tensor sizes.
    """
    config_path = Path(model_path).parent / "train_config.json"
    source = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if not isinstance(source, dict):
        raise ValueError("Resume train_config.json must be a JSON object")
    profile = source.get("gait_profile", "legacy")
    if profile not in ("legacy", "lab"):
        raise ValueError(f"Unrecognized saved gait profile: {profile!r}")
    return {
        "gait_profile": profile,
        "legacy_inferred_from_missing_metadata": "gait_profile" not in source,
        "parameters": source.get("gait_profile_parameters"),
        "config_path": str(config_path.resolve()) if config_path.is_file() else None,
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
    }


def resolve_training_contact_threshold(gait_profile: str, requested: float | None):
    """Retain the legacy CLI default; let the lab environment derive its own."""
    if requested is not None and (not math.isfinite(requested) or requested < 0):
        raise ValueError("--contact-thresh must be finite and nonnegative")
    if requested is not None:
        return float(requested)
    return .0564 if gait_profile == "legacy" else None


def environment_calibration_snapshot(vector) -> dict:
    """Persist what every created environment actually uses, not just CLI intent."""
    calibrations = vector.get_attr("reward_calibration")
    thresholds = vector.get_attr("contact_threshold")
    rewards = [reward.w for reward in vector.get_attr("reward_fn")]
    # BLOCKED.md: the effective lab controller, not just its timing, must be
    # recorded before lab training or resume. Read from the live workers.
    controllers = vector.get_attr("lab_controller_snapshot")
    if not calibrations:
        raise ValueError("Cannot record calibration for an empty vector environment")
    if any(c != calibrations[0] for c in calibrations) or any(w != rewards[0] for w in rewards):
        raise ValueError("Vector workers resolved different reward calibrations")
    if any(value != thresholds[0] for value in thresholds):
        raise ValueError("Vector workers resolved different contact thresholds")
    if any(c != controllers[0] for c in controllers):
        raise ValueError("Vector workers resolved different lab controllers")
    return json.loads(json.dumps({
        "reward_calibration": calibrations[0],
        "effective_contact_threshold_N": float(thresholds[0]),
        "effective_reward_cfg": rewards[0],
        "effective_lab_controller": controllers[0],
    }, allow_nan=False))


def validate_resume_calibration(args, run_config: dict) -> dict:
    """Never silently reuse a lab policy under different body-derived guards.

    Curriculum weight changes may be intentional and are enumerated in config;
    changes to lab physical inputs, derived guards, or contact are incompatible.
    Legacy runs without calibration metadata retain their recorded inference.
    """
    if not args.resume_from:
        return {}
    source_path = Path(args.resume_from).parent / "train_config.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    old_calibration = source.get("reward_calibration")
    old_contact = source.get("effective_contact_threshold_N", source.get("contact_thresh"))
    if args.gait_profile == "lab":
        if not isinstance(old_calibration, dict):
            raise ValueError("Lab resume requires its saved reward_calibration; start fresh instead of guessing")
        if source.get("xml_sha256") != run_config["xml_sha256"]:
            raise ValueError("Lab resume XML differs from its training body; start a fresh run")
        current = run_config["reward_calibration"]
        for key in ("model_inputs", "reward_overrides"):
            if old_calibration.get(key) != current.get(key):
                raise ValueError(f"Lab resume calibration changed ({key}); restore its configuration or start fresh")
        if old_contact is None or old_contact != run_config["effective_contact_threshold_N"]:
            raise ValueError("Lab resume contact threshold changed; restore its threshold or start fresh")
        old_controller = source.get("effective_lab_controller")
        if old_controller is None:
            raise ValueError("Lab resume requires its saved effective_lab_controller; start fresh "
                             "instead of guessing the base controller it learned against")
        if old_controller != run_config["effective_lab_controller"]:
            raise ValueError("Lab resume effective controller changed (base parameters, stance "
                             "compensation or residual authority); restore it or start fresh")
    changes = {}
    if old_contact is not None and old_contact != run_config["effective_contact_threshold_N"]:
        changes["contact_threshold_N"] = {"saved": old_contact, "requested_effective": run_config["effective_contact_threshold_N"]}
    old_weights = source.get("effective_reward_cfg")
    if isinstance(old_weights, dict):
        for key in set(old_weights) | set(run_config["effective_reward_cfg"]):
            old, new = old_weights.get(key), run_config["effective_reward_cfg"].get(key)
            if old != new:
                changes["reward." + key] = {"saved": old, "requested_effective": new}
    return changes


def validate_training_args(args) -> None:
    """Fail before allocating environments or touching an existing run."""
    if bool(args.resume_from) != bool(args.resume_vec):
        raise ValueError("--resume-from and --resume-vec must be provided together")
    if args.reset_timesteps and not args.resume_from:
        raise ValueError("--reset-timesteps requires a paired resume")
    if not args.run or args.run in (".", "..") or any(c in args.run for c in ("/", "\\", ":")):
        raise ValueError("--run must be a single directory name, not a path")
    for name in ("envs", "steps", "n_steps", "batch", "eval_freq", "checkpoint_freq"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.batch < 2 or args.envs * args.n_steps < 2:
        raise ValueError("PPO requires batch and rollout sizes greater than one")
    requested_gait = gait_profile_parameters(args.gait_profile)
    resolve_training_contact_threshold(args.gait_profile, getattr(args, "contact_thresh", None))
    for name in ("max_wall_seconds", "backup_ack_timeout", "target_kl"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    scale = args.front_lift_residual_scale
    if scale is not None and (not math.isfinite(scale) or scale < 0):
        raise ValueError("--front-lift-residual-scale must be finite and nonnegative")
    if args.gait_profile == "lab" and args.control_mode != "cpg_residual":
        # Raw mode builds no CPG at all, so the lab timing targets would score a
        # gait nothing is generating. Refuse rather than run a different study.
        raise ValueError("--gait-profile lab requires --control-mode cpg_residual")
    if args.gait_profile != "lab":
        # Both are opt-in lab controls; the controller refuses them for legacy.
        if args.hind_stance_compensation != "off":
            raise ValueError("--hind-stance-compensation is a lab-profile control")
        if scale is not None:
            raise ValueError("--front-lift-residual-scale is a lab-profile control")
    if args.resume_from:
        model_path, vec_path = Path(args.resume_from), Path(args.resume_vec)
        if not model_path.is_file() or not vec_path.is_file():
            raise FileNotFoundError("Both resume artifacts must exist")
        # Bundled resumes must use their verified paired normalization state.
        if (model_path.parent / "manifest.json").exists():
            verify_checkpoint_bundle(model_path.parent)
            if model_path.name != "model.zip" or vec_path.resolve() != (model_path.parent / "vecnormalize.pkl").resolve():
                raise ValueError("A checkpoint bundle must resume its own model/normalizer pair")
        source_gait = resume_gait_contract(model_path)
        if source_gait["gait_profile"] != args.gait_profile:
            raise ValueError(
                f"Resume gait profile is {source_gait['gait_profile']!r}, but requested "
                f"{args.gait_profile!r}. Use the matching profile or start a fresh run."
            )
        if source_gait["parameters"] is not None and source_gait["parameters"] != requested_gait:
            raise ValueError("Saved gait parameters differ from the current profile; restore its registry or start fresh")
        if args.gait_profile == "lab" and source_gait["parameters"] is None:
            raise ValueError("Lab resume requires its saved gait_profile_parameters; do not guess its timing")
    if args.xml_path is not None and not Path(args.xml_path).is_file():
        raise FileNotFoundError("--xml-path must be an existing MJCF file")


def reserve_run_directory(models_dir: Path, run: str) -> Path:
    if not run or run in (".", "..") or any(c in run for c in ("/", "\\", ":")):
        raise ValueError("Run must be a single directory name")
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / run
    # mkdir is the atomic reservation; even an empty pre-existing directory is
    # protected. Resume into a NEW named run, never overwrite the source run.
    out.mkdir(exist_ok=False)
    return out


def training_config(args, reward_cfg) -> dict:
    config = dict(vars(args))
    config.update({"reward_cfg": reward_cfg, "python": platform.python_version(),
                   "checkpoint_contract": "atomic paired bundles; off-instance backup requires external verification"})
    config["gait_profile_parameters"] = gait_profile_parameters(args.gait_profile)
    config["contact_thresh_argument_after_legacy_default"] = resolve_training_contact_threshold(
        args.gait_profile, args.contact_thresh
    )
    try:
        config["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        config["git_commit"] = None
    if args.resume_from:
        config["resume_sha256"] = {
            "model": sha256_file(args.resume_from), "vecnormalize": sha256_file(args.resume_vec)
        }
        config["resume_gait_contract"] = resume_gait_contract(args.resume_from)
    xml_path = Path(args.xml_path) if args.xml_path else REPO / "morphology" / "gecko_body_r.xml"
    config["xml_path_resolved"] = str(xml_path.resolve())
    config["xml_sha256"] = sha256_file(xml_path)
    config["requested_hind_stance_compensation"] = COMPENSATION_BY_FLAG[args.hind_stance_compensation]
    return config


def make_env(seed, control_mode="raw", residual_scale=0.25,
             contact_thresh=None, front_stance_press=0.40,
             front_swing_lift=0.40, reward_cfg=None, xml_path=None,
             gait_profile="legacy", hind_stance_compensation=False,
             front_lift_residual_scale=None):
    def _f():
        return GeckoWalkEnv(
            xml_path=xml_path,
            seed=seed,
            control_mode=control_mode,
            residual_scale=residual_scale,
            contact_thresh=resolve_training_contact_threshold(gait_profile, contact_thresh),
            front_stance_press=front_stance_press,
            front_swing_lift=front_swing_lift,
            reward_cfg=reward_cfg,
            gait_profile=gait_profile,
            hind_stance_compensation=hind_stance_compensation,
            front_lift_residual_scale=front_lift_residual_scale,
        )
    return _f


COMPENSATION_BY_FLAG = {"off": False, "hind": True, "all": "all"}

# Lab controller channels added after some evidence reports were written, with
# the value at which each contributes nothing. Listing them explicitly keeps
# "this older report ran the same controller" a proof rather than an assumption.
NO_OP_WHEN = {
    "hind_sprawl_amplitude": 0.0,
    "fore_sprawl_amplitude": 0.0,
    "sprawl_phase": 0.0,
    "tail_hindlimb_coupling": 0.0,
}

# Four of the six Gate 2 checks are recoverable from a realism_metrics report
# alone; front/hind contact LOADS need the trace and are not re-derived here.
EVIDENCE_GATES = (
    ("forward_speed", 0.04, math.inf),
    ("net_over_path", 0.50, math.inf),
    ("hind_duty", 0.73, 0.83),
    ("limb_phase", 0.405, 0.465),
)


def evidence_measurements(report):
    """The report-derivable Gate 2 values, by the gate's own definitions."""
    episode = report["episodes"][0]
    gait = episode["gait"]
    if gait.get("status") != "measured":
        raise ValueError("Lab base evidence episode was not scorable.")
    if episode.get("terminated") or not episode.get("completed_requested_duration"):
        raise ValueError("Lab base evidence episode fell or did not complete its duration.")
    number = lambda v: v["mean"] if isinstance(v, dict) else v
    phases = [gait["limb_phase"][k]["mean_cycle"] for k in ("HL_to_FL", "HR_to_FR")]
    if any(p is None for p in phases):
        raise ValueError("Lab base evidence has no measured limb phase.")
    return {
        "forward_speed": number(gait["forward_speed_m_s"]),
        "net_over_path": gait["net_displacement_m"] / gait["distance_path_m"],
        "hind_duty": min(number(gait["limbs"]["HL"]["duty_factor"]),
                         number(gait["limbs"]["HR"]["duty_factor"])),
        "limb_phase": float(sum(phases) / len(phases)),
    }


def require_lab_training_readiness(args):
    """Session 2 closed lab learning after Gate 2 failed; this is its resolution.

    BLOCKED.md asks for two things before lab training or resume: the failed
    gait gate deliberately resolved, and the exact effective lab controller
    parameters persisted and compared in checkpoint contracts. Both are met here
    by requiring a measured zero-policy base report for this exact body and
    controller, and recording it in train_config.json.

    Gate 2 measures the base controller with the policy switched OFF. It is a
    sanity check that the foundation is not broken, never a finished walker, so
    the requirement is that the base is the known-good one -- not that it
    already passes all six checks. Two checks (front stance load, limb phase)
    are exactly what a learned residual exists to attack, and they are recorded
    as unmet rather than waived silently.

    Legacy training behavior is unchanged: legacy needs no evidence file.
    """
    if args.gait_profile != "lab":
        return None
    if not args.lab_base_evidence:
        raise ValueError("Lab training requires --lab-base-evidence: a realism_metrics report "
                         "measuring this body and base controller with a zero policy. "
                         "See docs/BLOCKED.md.")
    if args.xml_path is None:
        raise ValueError("Lab training requires an explicit --xml-path; the default body is the "
                         "61 g legacy morphology, not the measured lab body.")
    path = Path(args.lab_base_evidence)
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    if protocol.get("gait_profile") != "lab":
        raise ValueError("Lab base evidence must be a lab-profile report.")
    if protocol.get("controller") != "zero residual with contact reflex":
        raise ValueError("Lab base evidence must measure the base with the policy switched off.")
    body_sha = sha256_file(args.xml_path)
    if protocol.get("xml_sha256") != body_sha:
        raise ValueError("Lab base evidence measured a different body than --xml-path.")
    recorded = protocol.get("effective_lab_parameters")
    if not isinstance(recorded, dict):
        raise ValueError("Lab base evidence records no effective lab controller parameters.")
    expected = dict(parameter_value("lab_base_parameters"))
    differing = sorted(k for k in recorded if k not in expected or recorded[k] != expected[k])
    if differing:
        raise ValueError("Lab base evidence used different effective lab controller parameters "
                         "than the current registry " + str(differing) +
                         "; re-measure the base or restore the registry.")
    # A report predating Session 3g records a SUBSET: spine_amp, tail_amp and
    # tail_phase_lag were moved into the registry at the controller's existing
    # constructor defaults, so behaviour was unchanged but the record is older.
    # Accept that only by proving each missing key still equals the default the
    # older run actually executed -- never by assuming it.
    missing = sorted(set(expected) - set(recorded))
    if missing:
        defaults = inspect.signature(CPGResidualController.__init__).parameters
        for key in missing:
            default = defaults[key].default if key in defaults else inspect.Parameter.empty
            if default is not inspect.Parameter.empty and default == expected[key]:
                continue
            # Keys introduced after the evidence was measured are acceptable only
            # at the value that reproduces the behaviour the older run actually
            # had. Session 4 added the sprawl and tail-coupling channels; each is
            # additive and contributes exactly nothing at 0.0.
            if key in NO_OP_WHEN and expected[key] == NO_OP_WHEN[key]:
                continue
            raise ValueError("Lab base evidence omits " + key + " and the registry value is not "
                             "the no-op that the older run actually executed; re-measure the base.")
    duration = float(parameter_value("evaluation_duration_s"))
    if protocol.get("requested_duration_s") != duration:
        raise ValueError("Lab base evidence must be measured at the gate's own %g s duration; "
                         "Session 3h found a 12 s window reads limb phase bimodally." % duration)
    if protocol.get("reset_noise") != 0:
        raise ValueError("Lab base evidence must use the deterministic zero-reset-noise protocol.")
    wanted = COMPENSATION_BY_FLAG[args.hind_stance_compensation]
    if bool(protocol.get("hind_stance_compensation")) != bool(wanted):
        raise ValueError("Lab base evidence stance compensation does not match "
                         "--hind-stance-compensation " + args.hind_stance_compensation + ".")
    measurements = evidence_measurements(report)
    unmet = [name for name, low, high in EVIDENCE_GATES
             if not (low <= measurements[name] <= high)]
    # The SCOPE of compensation (hind-only vs all four limbs) is not recorded by
    # realism_metrics, which stores only a boolean. Hind duty is what separates
    # them on this body, so it is required rather than merely reported.
    if "hind_duty" in unmet:
        raise ValueError("Lab base evidence fails the hind duty-factor check "
                         "(%.4f); this is not the corrected base." % measurements["hind_duty"])
    return {
        "evidence_path": str(path.resolve()),
        "evidence_sha256": sha256_file(path),
        "xml_sha256": body_sha,
        "hind_stance_compensation": args.hind_stance_compensation,
        "parameters_defaulted_in_older_evidence": missing,
        "report_derivable_gate_values": measurements,
        "report_derivable_gates_unmet": unmet,
        "loads_not_rederived": "front/hind contact loads need the trace and are not re-checked here",
        "reason": "Gate 2 measures the base with the policy off; it gates the foundation, "
                  "not the trained walker. See docs/BLOCKED.md and docs/BUILD_LOG.md Session 4.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--run", type=str, default="v3_cpg_sanity_200k")
    p.add_argument("--xml-path", default=None,
                   help="optional candidate MJCF; default keeps the established body path")
    p.add_argument("--gait-profile", choices=["legacy", "lab"], default="legacy",
                   help="lab uses shared measured timing targets; incompatible with legacy-checkpoint resume")
    p.add_argument("--lab-base-evidence", default=None,
                   help="realism_metrics report measuring this body and base controller with a "
                        "zero policy; required for --gait-profile lab (see docs/BLOCKED.md)")
    p.add_argument("--hind-stance-compensation", choices=["off", "hind", "all"], default="off",
                   help="opt-in lab stance compensation; 'all' is the Session 3 corrected base")
    p.add_argument("--front-lift-residual-scale", type=float, default=None,
                   help="residual authority on the FL/FR lift actuators; omitted keeps the "
                        "structural lock at 0.0 (see the controller docstring)")
    p.add_argument("--target-kl", type=float, default=None,
                   help="stop each PPO epoch loop when approximate KL exceeds this; the 10 M-step "
                        "precedent diverged to approx_kl 67 and lost 63%% of eval return")
    p.add_argument("--vec", choices=["subproc", "dummy"], default="subproc")
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.004)
    p.add_argument("--eval-freq", type=int, default=500_000)
    p.add_argument("--checkpoint-freq", type=int, default=100_000,
                   help="aggregate environment steps between atomic paired checkpoints")
    p.add_argument("--checkpoint-mirror-dir", default=None,
                   help="optional filesystem mirror; NOT automatically an off-instance backup")
    p.add_argument("--require-backup-ack", action="store_true",
                   help="fail closed until an external verifier acknowledges each checkpoint, including preflight")
    p.add_argument("--backup-ack-timeout", type=float, default=120.0)
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="stop learning at next callback after this time; does not stop/bound EC2 billing")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--recurrent", action="store_true", help="use RecurrentPPO (needs sb3-contrib)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", type=str, default=None, help="path to PPO model zip to continue")
    p.add_argument("--resume-vec", type=str, default=None, help="path to VecNormalize pkl for resumed run")
    p.add_argument("--reset-timesteps", action="store_true",
                   help="reset SB3 timestep counter when resuming")
    p.add_argument("--control-mode", choices=["raw", "cpg_residual"], default="raw")
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--front-stance-press", type=float, default=0.40)
    p.add_argument("--front-swing-lift", type=float, default=0.40)
    p.add_argument("--contact-thresh", type=float, default=None,
                   help="explicit force threshold in N; omitted means legacy0.0564 or lab body-derived classifier (not skin sensitivity)")
    p.add_argument("--phase", choices=["p1", "p2", "v43", "none"], default="none",
                   help="V4.2.8 curriculum. p1=contact acquisition (low speed "
                        "pressure, early gate), p2=speed restoration. "
                        "none (default) leaves reward weights at code defaults so "
                        "existing commands are unchanged.")
    args = p.parse_args()
    try:
        lab_readiness = require_lab_training_readiness(args)
        validate_training_args(args)
    except (ValueError, FileNotFoundError, KeyError, OSError, json.JSONDecodeError) as exc:
        p.error(str(exc))

    # V4.2.8 curriculum reward configs. --phase none -> reward_cfg=None -> the
    # WalkReward DEFAULTS are used unchanged (backward compatible).
    reward_cfg = None
    if args.phase == "p1":
        reward_cfg = dict(
            progress=7.0, forward=0.6,          # lower speed/progress pressure
            v4_speed_threshold=0.008,           # open speed_gate earlier so gated terms can teach
            front_track=1.30, front_miss=1.90,  # stronger contact acquisition
            front_duty=0.80, front_load=1.20,
        )
    elif args.phase == "p2":
        reward_cfg = dict(
            progress=12.0, forward=1.5,         # restore speed/progress pressure
            v4_speed_threshold=0.025,
            front_track=1.10, front_miss=1.60,  # keep contact tracking active
            front_duty=0.80, front_load=1.20,
        )
    elif args.phase == "v43":
        reward_cfg = dict(
            # V4.5 Claude final: matched front stance + 1.18Hz cadence.
            progress=12.0,
            forward=2.0,
            v4_speed_threshold=0.025,
            front_track=1.20,
            front_miss=1.80,
            front_duty=1.00,
            front_load=1.20,
            target_speed=0.125,
            speed_floor=0.085,
            speed_track_sigma=0.035,
            speed_track=6.0,
            slow_speed=0.112,
            slow_penalty=3.0,
        )

    from stable_baselines3.common.vec_env import (SubprocVecEnv, DummyVecEnv,
                                                  VecNormalize, VecMonitor)
    from stable_baselines3.common.callbacks import EvalCallback
    VecCls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv

    out = reserve_run_directory(REPO / "models", args.run)
    run_config = training_config(args, reward_cfg)
    if lab_readiness is not None:
        run_config["lab_base_evidence"] = lab_readiness
    atomic_json(out / "train_config.json", run_config)
    if run_config.get("resume_gait_contract", {}).get("legacy_inferred_from_missing_metadata"):
        print("[resume] no saved gait-profile metadata; explicitly using historical legacy profile", flush=True)
    tb = REPO / "renders" / "tb"; tb.mkdir(parents=True, exist_ok=True)

    compensation = COMPENSATION_BY_FLAG[args.hind_stance_compensation]
    env_fns = [
        make_env(args.seed + i, args.control_mode, args.residual_scale,
                 args.contact_thresh, args.front_stance_press, args.front_swing_lift,
                 reward_cfg=reward_cfg, xml_path=args.xml_path, gait_profile=args.gait_profile,
                 hind_stance_compensation=compensation,
                 front_lift_residual_scale=args.front_lift_residual_scale)
        for i in range(args.envs)
    ]
    venv = VecCls(env_fns)
    venv = VecMonitor(venv)
    if args.resume_from and args.resume_vec:
        venv = VecNormalize.load(args.resume_vec, venv)
        venv.training = True
        venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    try:
        run_config.update(environment_calibration_snapshot(venv))
        run_config["resume_environment_changes"] = validate_resume_calibration(args, run_config)
        atomic_json(out / "train_config.json", run_config)
        if run_config["resume_environment_changes"]:
            print("[resume] explicitly recorded environment/curriculum changes: "
                  + json.dumps(run_config["resume_environment_changes"], sort_keys=True), flush=True)
    except Exception:
        venv.close()
        raise

    eval_env = DummyVecEnv([
        make_env(10_000, args.control_mode, args.residual_scale,
                 args.contact_thresh, args.front_stance_press, args.front_swing_lift,
                 reward_cfg=reward_cfg, xml_path=args.xml_path, gait_profile=args.gait_profile,
                 hind_stance_compensation=compensation,
                 front_lift_residual_scale=args.front_lift_residual_scale)
    ])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    eval_env.obs_rms = venv.obs_rms

    pol_kwargs = dict(net_arch=[256, 256])
    common = dict(verbose=1, seed=args.seed, device=args.device, n_steps=args.n_steps,
                  batch_size=args.batch, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                  learning_rate=args.lr, clip_range=0.2, ent_coef=args.ent_coef, vf_coef=0.5,
                  max_grad_norm=0.5, target_kl=args.target_kl, tensorboard_log=str(tb))
    if args.resume_from:
        if args.recurrent:
            from sb3_contrib import RecurrentPPO as ResumeAlgorithm
        else:
            from stable_baselines3 import PPO as ResumeAlgorithm
        from stable_baselines3.common.utils import get_schedule_fn
        model = ResumeAlgorithm.load(args.resume_from, env=venv, device=args.device,
                         custom_objects=dict(learning_rate=args.lr, ent_coef=args.ent_coef))
        model.learning_rate = args.lr
        model.lr_schedule = get_schedule_fn(args.lr)
        model.ent_coef = args.ent_coef
        # A resumed model keeps the saved value; the CLI is authoritative here.
        model.target_kl = args.target_kl
    elif args.recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MlpLstmPolicy", venv,
                             policy_kwargs=dict(net_arch=[256], lstm_hidden_size=256), **common)
    else:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", venv, policy_kwargs=pol_kwargs, **common)

    # A resumed SB3 model retains several saved hyperparameters. Record actual
    # values as well as the requested CLI, without silently changing the run.
    run_config["effective_training"] = {
        "algorithm": type(model).__name__, "device": str(model.device),
        "target_kl": model.target_kl,
        "n_steps": model.n_steps, "batch_size": model.batch_size,
        "n_epochs": model.n_epochs, "num_envs": venv.num_envs,
        "observation_shape": list(model.observation_space.shape),
        "action_shape": list(model.action_space.shape),
        "starting_num_timesteps": model.num_timesteps,
    }
    atomic_json(out / "train_config.json", run_config)

    freq = max(args.eval_freq // args.envs, 1)
    checkpoints = CheckpointBundleCallback(
        out, run_config, checkpoint_freq=args.checkpoint_freq,
        max_wall_seconds=args.max_wall_seconds, mirror_dir=args.checkpoint_mirror_dir,
        require_backup_ack=args.require_backup_ack, backup_ack_timeout=args.backup_ack_timeout,
    )
    cbs = [checkpoints,
           EvalCallback(eval_env, best_model_save_path=str(out), eval_freq=freq,
                        n_eval_episodes=5, deterministic=True)]
    try:
        if args.resume_from:
            model.learn(total_timesteps=args.steps, reset_num_timesteps=args.reset_timesteps,
                        callback=cbs, tb_log_name=args.run)
        else:
            model.learn(total_timesteps=args.steps, callback=cbs, tb_log_name=args.run)
        export_legacy_final(checkpoints.last_bundle, out)
        atomic_json(out / "training_result.json", {
            "num_timesteps": model.num_timesteps,
            "stopped_for_wall_limit": checkpoints.stopped_for_wall_limit,
            "checkpoint_bundle": str(checkpoints.last_bundle),
        })
        print("saved ->", out, flush=True)
    finally:
        venv.close()
        eval_env.close()


if __name__ == "__main__":
    main()
