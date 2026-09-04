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
import math
import platform
import subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from envs.gecko_walk_env import GeckoWalkEnv
from common.checkpoints import (
    CheckpointBundleCallback, atomic_json, export_legacy_final,
    sha256_file, verify_checkpoint_bundle,
)


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
    for name in ("max_wall_seconds", "backup_ack_timeout"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.resume_from:
        model_path, vec_path = Path(args.resume_from), Path(args.resume_vec)
        if not model_path.is_file() or not vec_path.is_file():
            raise FileNotFoundError("Both resume artifacts must exist")
        # Bundled resumes must use their verified paired normalization state.
        if (model_path.parent / "manifest.json").exists():
            verify_checkpoint_bundle(model_path.parent)
            if model_path.name != "model.zip" or vec_path.resolve() != (model_path.parent / "vecnormalize.pkl").resolve():
                raise ValueError("A checkpoint bundle must resume its own model/normalizer pair")
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
    xml_path = Path(args.xml_path) if args.xml_path else REPO / "morphology" / "gecko_body_r.xml"
    config["xml_path_resolved"] = str(xml_path.resolve())
    config["xml_sha256"] = sha256_file(xml_path)
    return config


def make_env(seed, control_mode="raw", residual_scale=0.25,
             contact_thresh=1e-6, front_stance_press=0.40,
             front_swing_lift=0.40, reward_cfg=None, xml_path=None):
    def _f():
        return GeckoWalkEnv(
            xml_path=xml_path,
            seed=seed,
            control_mode=control_mode,
            residual_scale=residual_scale,
            contact_thresh=contact_thresh,
            front_stance_press=front_stance_press,
            front_swing_lift=front_swing_lift,
            reward_cfg=reward_cfg,
        )
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--run", type=str, default="v3_cpg_sanity_200k")
    p.add_argument("--xml-path", default=None,
                   help="optional candidate MJCF; default keeps the established body path")
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
    p.add_argument("--contact-thresh", type=float, default=0.0564)
    p.add_argument("--phase", choices=["p1", "p2", "v43", "none"], default="none",
                   help="V4.2.8 curriculum. p1=contact acquisition (low speed "
                        "pressure, early gate), p2=speed restoration. "
                        "none (default) leaves reward weights at code defaults so "
                        "existing commands are unchanged.")
    args = p.parse_args()
    try:
        validate_training_args(args)
    except (ValueError, FileNotFoundError) as exc:
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
    atomic_json(out / "train_config.json", run_config)
    tb = REPO / "renders" / "tb"; tb.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(args.seed + i, args.control_mode, args.residual_scale,
                 args.contact_thresh, args.front_stance_press, args.front_swing_lift,
                 reward_cfg=reward_cfg, xml_path=args.xml_path)
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

    eval_env = DummyVecEnv([
        make_env(10_000, args.control_mode, args.residual_scale,
                 args.contact_thresh, args.front_stance_press, args.front_swing_lift,
                 reward_cfg=reward_cfg, xml_path=args.xml_path)
    ])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    eval_env.obs_rms = venv.obs_rms

    pol_kwargs = dict(net_arch=[256, 256])
    common = dict(verbose=1, seed=args.seed, device=args.device, n_steps=args.n_steps,
                  batch_size=args.batch, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                  learning_rate=args.lr, clip_range=0.2, ent_coef=args.ent_coef, vf_coef=0.5,
                  max_grad_norm=0.5, tensorboard_log=str(tb))
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
