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
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from envs.gecko_walk_env import GeckoWalkEnv


def make_env(seed, control_mode="raw", residual_scale=0.25,
             contact_thresh=1e-6, front_stance_press=0.35):
    def _f():
        return GeckoWalkEnv(
            seed=seed,
            control_mode=control_mode,
            residual_scale=residual_scale,
            contact_thresh=contact_thresh,
            front_stance_press=front_stance_press,
        )
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--run", type=str, default="v3_cpg_sanity_200k")
    p.add_argument("--vec", choices=["subproc", "dummy"], default="subproc")
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.004)
    p.add_argument("--eval-freq", type=int, default=500_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--recurrent", action="store_true", help="use RecurrentPPO (needs sb3-contrib)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", type=str, default=None, help="path to PPO model zip to continue")
    p.add_argument("--resume-vec", type=str, default=None, help="path to VecNormalize pkl for resumed run")
    p.add_argument("--reset-timesteps", action="store_true",
                   help="reset SB3 timestep counter when resuming")
    p.add_argument("--control-mode", choices=["raw", "cpg_residual"], default="raw")
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--front-stance-press", type=float, default=0.35)
    p.add_argument("--contact-thresh", type=float, default=0.0564)
    args = p.parse_args()

    from stable_baselines3.common.vec_env import (SubprocVecEnv, DummyVecEnv,
                                                  VecNormalize, VecMonitor)
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    VecCls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv

    out = REPO / "models" / args.run; out.mkdir(parents=True, exist_ok=True)
    tb = REPO / "renders" / "tb"; tb.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(args.seed + i, args.control_mode, args.residual_scale,
                 args.contact_thresh, args.front_stance_press)
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
                 args.contact_thresh, args.front_stance_press)
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
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import get_schedule_fn
        model = PPO.load(args.resume_from, env=venv, device=args.device,
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

    freq = max(args.eval_freq // args.envs, 1)
    cbs = [CheckpointCallback(save_freq=freq, save_path=str(out),
                              name_prefix="ckpt", save_vecnormalize=True),
           EvalCallback(eval_env, best_model_save_path=str(out), eval_freq=freq,
                        n_eval_episodes=5, deterministic=True)]
    if args.resume_from:
        model.learn(total_timesteps=args.steps, reset_num_timesteps=args.reset_timesteps,
                    callback=cbs, tb_log_name=args.run)
    else:
        model.learn(total_timesteps=args.steps, callback=cbs, tb_log_name=args.run)
    model.save(str(out / "final"))
    venv.save(str(out / "vecnormalize.pkl"))
    print("saved ->", out)


if __name__ == "__main__":
    main()
