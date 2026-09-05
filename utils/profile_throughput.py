"""Component microbenchmarks; do not add unlike-rate measurements as step cost."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if platform.system() == 'Linux' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import torch
from OpenGL import GL
from common.checkpoints import atomic_json
from eval.recovered_brain import load_actor
from envs.gecko_brain_env import GeckoBrainEnv


def benchmark(call, steps, device='cpu'):
    for _ in range(10):
        call()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        call()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    return dict(iterations=steps, total_seconds=elapsed, mean_ms=elapsed*1000/steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--brain-run', type=Path, required=True)
    p.add_argument('--walker-run', default='v4_5b_speed_polish_1m')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.steps <= 100000:
        p.error('steps must be within 1..100000')
    torch.set_num_threads(1)
    actor, _ = load_actor(args.brain_run, args.device)
    env = GeckoBrainEnv(walker_run=args.walker_run, privileged_target=0,
                       policy_camera_mode='sealed')
    try:
        obs, _ = env.reset(seed=100)
        graphics = {}
        for name in ('GL_VENDOR', 'GL_RENDERER', 'GL_VERSION'):
            value = GL.glGetString(getattr(GL, name))
            graphics[name] = value.decode('utf-8', errors='replace') if value else None
        result = dict(platform=platform.platform(), python=platform.python_version(),
                      mujoco=mujoco.__version__, torch=torch.__version__, device=args.device,
                      torch_cuda_gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      graphics=graphics,
                      torch_threads=1, renderer=os.environ.get('MUJOCO_GL', 'default'),
                      physics_dt_s=env.walk_env.model.opt.timestep, control_dt_s=env.walk_env.dt,
                      caveat='Separate component timing, not training throughput: physics evolves at neutral control; rendering uses a reset pose; actor prediction includes data transfer and output synchronization. Torch CUDA availability does not establish GPU-accelerated rendering; inspect GL_RENDERER.')
        result['physics_step'] = benchmark(lambda: mujoco.mj_step(env.walk_env.model, env.walk_env.data), args.steps)
        obs, _ = env.reset(seed=100)
        result['policy_render_64x64'] = benchmark(env._head_cam_image, args.steps)
        result['visual_actor_predict'] = benchmark(lambda: actor.predict(obs, device=args.device), args.steps, args.device)
        result['physics_25_substeps_equivalent_ms'] = result['physics_step']['mean_ms']*env.walk_env.frame_skip
        atomic_json(args.output, result)
        print(result, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
