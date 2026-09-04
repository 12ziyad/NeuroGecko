"""Reproduce the recovered visual student's task scores under named camera modes.

These are legacy marker-proximity eating events (0.10 m radius), not validated
prey-strike biomechanics. Both the source checkpoint and normalizer are trusted
user-owned inputs; the .pt actor is loaded with weights_only=True.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if platform.system() == 'Linux' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from brain.bc_actor import BrainBCActor, build_obs_space
from envs.gecko_brain_env import GeckoBrainEnv
from common.checkpoints import atomic_json, sha256_file


def load_actor(run: Path, device='cpu'):
    config = json.loads((run / 'train_config.json').read_text(encoding='utf-8'))
    if config.get('algo') != 'visual_distillation' or config.get('use_privileged_food') is not False:
        raise ValueError('Expected a visual-distillation student with no privileged input')
    actor = BrainBCActor(build_obs_space(int(config['proprio_dim'])),
                         image_features_dim=config['image_features_dim'],
                         body_features_dim=config['body_features_dim'],
                         fused_features_dim=config['fused_features_dim'],
                         use_privileged=False, action_dim=4).to(device)
    actor.load_state_dict(torch.load(run / 'final.pt', map_location=device, weights_only=True))
    actor.eval()
    return actor, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--brain-run', type=Path, required=True)
    p.add_argument('--walker-run', default='v4_5b_speed_polish_1m')
    p.add_argument('--xml', type=Path)
    p.add_argument('--camera', choices=['legacy', 'sealed'], required=True)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video', type=Path)
    p.add_argument('--max-wall-seconds', type=float, default=600)
    args = p.parse_args()
    if args.episodes <= 0 or args.steps <= 0 or args.max_wall_seconds <= 0:
        p.error('Positive episode, step and wall limits are required')
    torch.set_num_threads(1)
    actor, cfg = load_actor(args.brain_run, args.device)
    env = GeckoBrainEnv(walker_run=args.walker_run, walker_xml_path=args.xml,
                       policy_camera_mode=args.camera, privileged_target=0,
                       max_steps=args.steps, food_spawn_angle_deg=cfg['food_spawn_angle_deg'],
                       food_radius=cfg['food_radius'], eat_radius=cfg['eat_radius'])
    result = dict(kind='legacy_visual_student_task_evaluation', camera=args.camera,
                  actor_sha256=sha256_file(args.brain_run / 'final.pt'),
                  xml_sha256=sha256_file(args.xml or REPO / 'morphology/gecko_body_r.xml'),
                  device=args.device, food_half_angle_deg=cfg['food_spawn_angle_deg'],
                  eat_radius_m=cfg['eat_radius'], privileged_input=False,
                  caveat='Marker-proximity scores, not prey-strike or whole-brain validation',
                  requested_episodes=args.episodes, max_steps=args.steps, episodes=[])
    started = time.monotonic()
    writer = None
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed+episode)
            eats = 0
            visible = []
            if episode == 0 and args.video:
                import imageio.v2 as imageio
                args.video.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(args.video, fps=25, macro_block_size=1)
            for step in range(args.steps):
                if time.monotonic()-started >= args.max_wall_seconds:
                    raise TimeoutError('Evaluation wall-time limit reached')
                if not np.allclose(obs['privileged'], 0):
                    raise RuntimeError('Visual evaluation received privileged input')
                action = actor.predict(obs, device=args.device)
                obs, _, terminated, truncated, info = env.step(action)
                eats += int(info['ate'])
                visible.append(info['food_visible_frac'])
                if writer and step % 2 == 0:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            row = dict(seed=args.seed+episode, steps=step+1, eats=eats,
                       fell=bool(info['fallen']), mean_food_visible_fraction=float(np.mean(visible)))
            result['episodes'].append(row)
            result['elapsed_wall_seconds'] = time.monotonic()-started
            result['total_eats'] = sum(r['eats'] for r in result['episodes'])
            result['episodes_with_eats'] = sum(r['eats'] > 0 for r in result['episodes'])
            result['falls'] = sum(r['fell'] for r in result['episodes'])
            atomic_json(args.output, result)
            print(json.dumps(row), flush=True)
            if writer:
                writer.close()
                writer = None
        result['complete'] = True
    except BaseException as exc:
        result['complete'] = False
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        if writer:
            writer.close()
        env.close()
        result['elapsed_wall_seconds'] = time.monotonic()-started
        atomic_json(args.output, result)


if __name__ == '__main__':
    main()
