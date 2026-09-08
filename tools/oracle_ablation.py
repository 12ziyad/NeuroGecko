"""Is the privileged target channel load-bearing, or merely present?

THE FINDING THIS EXISTS TO MEASURE. `GeckoWalkEnv` puts five numbers into the
92-number proprioception vector -- exact egocentric direction, distance and
bearing to the goal, read straight out of the physics engine, noiseless, at any
range. It defaults to on, `GeckoBrainEnv` never passes the flag, and every
checkpoint in the repository records proprio_dim 92, so every trained policy
carried it. The map records this being "removed everywhere" twice already
(#26, #33); this is the third occurrence.

Knowing it is present is not the same as knowing it matters. A policy can
receive a channel and ignore it. So this zeroes the five slots at inference --
keeping the vector 92 long, so the frozen checkpoint still loads -- and
measures what the walker actually loses:

  * distance travelled toward the goal
  * whether it stays upright
  * whether the gait survives

If the walker is unaffected, the oracle is dead weight and can be retired
without retraining. If it collapses, the accepted walker is a goal-following
policy that was handed the goal, and the locomotion result means less than the
map currently claims. Either answer is worth having; guessing is not.

Usage:  python tools/oracle_ablation.py [--steps 400] [--episodes 5]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Where the five privileged numbers sit inside the 92-vector. Derived from
#: GeckoWalkEnv.observation_layout rather than hard-coded, so a layout change
#: cannot silently move them out from under this test.
def privileged_slice(walk_env):
    offset = 0
    for name, width in walk_env.observation_layout["blocks"]:
        if name == "privileged_target":
            return slice(offset, offset + width)
        offset += width
    return None


def run(zero_oracle: bool, episodes: int, steps: int, seed0: int = 0):
    from envs.gecko_walk_env import GeckoWalkEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run_dir = REPO / "models" / "v4_5b_speed_polish_1m"
    env = GeckoWalkEnv(privileged_target=True)
    norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(str(run_dir / "final.zip"), device="cpu")
    sl = privileged_slice(env)

    rows = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed0 + ep)
            start = env.data.xpos[env._trunk][:2].copy()
            fell = False
            for _ in range(steps):
                vec = np.asarray(obs, dtype=np.float32).copy()
                if zero_oracle:
                    vec[sl] = 0.0
                action, _ = model.predict(norm.normalize_obs(vec[None, :]),
                                          deterministic=True)
                obs, _, term, trunc, info = env.step(action[0])
                if env.data.xmat[env._trunk].reshape(3, 3)[2, 2] < 0.3:
                    fell = True
                if term or trunc:
                    break
            end = env.data.xpos[env._trunk][:2].copy()
            rows.append({
                "episode": ep,
                "distance_travelled_m": round(float(np.linalg.norm(end - start)), 4),
                "toward_goal_m": round(float(np.dot(end - start,
                    (env.target[:2] - start) / (np.linalg.norm(env.target[:2] - start) + 1e-9))), 4),
                "fell": fell,
            })
    finally:
        env.close()
    return rows


def summarise(rows):
    return {
        "episodes": len(rows),
        "mean_distance_m": round(float(np.mean([r["distance_travelled_m"] for r in rows])), 4),
        "mean_toward_goal_m": round(float(np.mean([r["toward_goal_m"] for r in rows])), 4),
        "falls": sum(r["fell"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="artifacts/evidence/session9/oracle_ablation.json")
    args = ap.parse_args()

    print("Does the accepted walker need the goal oracle?\n")
    live = run(False, args.episodes, args.steps)
    zeroed = run(True, args.episodes, args.steps)
    a, b = summarise(live), summarise(zeroed)

    print(f"{'':>10}{'distance':>12}{'toward goal':>14}{'falls':>8}")
    print(f"{'oracle on':>10}{a['mean_distance_m']:>12}{a['mean_toward_goal_m']:>14}{a['falls']:>8}")
    print(f"{'zeroed':>10}{b['mean_distance_m']:>12}{b['mean_toward_goal_m']:>14}{b['falls']:>8}")

    # The question is whether progress TOWARD THE GOAL survives. Walking in a
    # straight line the wrong way is not the walker working.
    retained = (b["mean_toward_goal_m"] / a["mean_toward_goal_m"]) if a["mean_toward_goal_m"] else None
    load_bearing = retained is not None and retained < 0.5

    verdict = (
        "LOAD-BEARING. Zeroing the five privileged numbers costs the walker "
        f"{100 * (1 - retained):.0f}% of its progress toward the goal. The accepted "
        "walker is a goal-following policy that was handed the goal. Retiring the "
        "channel requires retraining, and the locomotion result must be read with "
        "that attached."
        if load_bearing else
        "NOT LOAD-BEARING. The walker keeps "
        f"{100 * retained:.0f}% of its progress with the channel zeroed, so the "
        "policy is not leaning on it. The channel can be retired without "
        "retraining -- and its presence in every checkpoint was carrying a claim "
        "it did not need to make."
        if retained is not None else
        "INCONCLUSIVE -- the baseline made no progress toward the goal, so there "
        "is nothing to ablate against.")

    payload = {
        "schema_version": 1,
        "test": "does zeroing the privileged target channel change what the walker does",
        "generated_by": "tools/oracle_ablation.py",
        "why": "The channel is present in every checkpoint in the repository "
               "(proprio_dim 92 in all 12). Presence is not the same as "
               "dependence, and the difference decides whether it can be "
               "retired without retraining.",
        "checkpoint": "models/v4_5b_speed_polish_1m",
        "oracle_live": {"summary": a, "episodes": live},
        "oracle_zeroed": {"summary": b, "episodes": zeroed},
        "fraction_of_progress_retained": round(retained, 4) if retained is not None else None,
        "load_bearing": load_bearing,
        "verdict": verdict,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n{verdict}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
