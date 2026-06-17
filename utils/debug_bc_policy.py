from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from brain.bc_actor import BrainBCActor, build_obs_space
from envs.gecko_brain_env import GeckoBrainEnv


OBS_KEYS = ("image", "proprio", "drives", "prev_action", "privileged")


def _load_train_config(run_dir: Path) -> dict:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing train_config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_bc_actor(run_dir: Path, train_config: dict) -> BrainBCActor:
    model_path = run_dir / "final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing behavior cloning model: {model_path}")

    algo = str(train_config.get("algo", "")).lower()
    if algo and algo != "behavior_cloning":
        raise ValueError(
            f"Expected behavior_cloning train_config, got algo={train_config.get('algo')!r}"
        )

    proprio_dim = int(train_config.get("proprio_dim", 0))
    if proprio_dim <= 0:
        raise ValueError("train_config missing positive 'proprio_dim'")

    obs_space = build_obs_space(proprio_dim)
    model = BrainBCActor(
        obs_space,
        image_features_dim=int(train_config.get("image_features_dim", 128)),
        body_features_dim=int(train_config.get("body_features_dim", 96)),
        fused_features_dim=int(train_config.get("fused_features_dim", 256)),
        use_privileged=bool(train_config.get("use_privileged_food", True)),
        action_dim=int(train_config.get("action_dim", 4)),
    )
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model


def _as_action_batch(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected action array with shape (N, 4), got {arr.shape}")
    return arr


def _action_metric_arrays(pred: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = _as_action_batch(pred)
    label = _as_action_batch(label)
    if pred.shape != label.shape:
        raise ValueError(f"pred and label shape mismatch: {pred.shape} vs {label.shape}")

    pred_dir = pred[:, :2]
    label_dir = label[:, :2]
    denom = (
        np.linalg.norm(pred_dir, axis=1) * np.linalg.norm(label_dir, axis=1)
    ) + 1e-8
    dir_cos = np.sum(pred_dir * label_dir, axis=1) / denom
    dist_err = np.abs(pred[:, 2] - label[:, 2])
    engage_err = np.abs(pred[:, 3] - label[:, 3])
    return dir_cos.astype(np.float64), dist_err.astype(np.float64), engage_err.astype(np.float64)


def _metric_summary(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    dir_cos, dist_err, engage_err = _action_metric_arrays(pred, label)
    return {
        "dir_cos": float(np.mean(dir_cos)) if len(dir_cos) else float("nan"),
        "dist_err": float(np.mean(dist_err)) if len(dist_err) else float("nan"),
        "engage_err": float(np.mean(engage_err)) if len(engage_err) else float("nan"),
    }


def _print_summary(prefix: str, summary: dict[str, float]) -> None:
    print(
        f"{prefix} "
        f"mean_dir_cos={summary['dir_cos']:.6f} "
        f"mean_dist_err={summary['dist_err']:.6f} "
        f"mean_engage_err={summary['engage_err']:.6f}"
    )


def _fmt_action(action: np.ndarray) -> str:
    return np.array2string(
        np.asarray(action, dtype=np.float32),
        precision=4,
        suppress_small=False,
        floatmode="fixed",
    )


def _print_rows(title: str, pred: np.ndarray, label: np.ndarray, label_name: str) -> None:
    pred = _as_action_batch(pred)
    label = _as_action_batch(label)
    n = min(10, len(pred), len(label))
    print(title)
    for i in range(n):
        print(f"  row={i:02d} pred={_fmt_action(pred[i])} {label_name}={_fmt_action(label[i])}")


def _obs_batch_from_npz(data: np.lib.npyio.NpzFile, indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "image": data["obs_image"][indices],
        "proprio": data["obs_proprio"][indices],
        "drives": data["obs_drives"][indices],
        "prev_action": data["obs_prev_action"][indices],
        "privileged": data["obs_privileged"][indices],
    }


def _predict_batch(model: BrainBCActor, obs_batch: dict[str, np.ndarray]) -> np.ndarray:
    tensors = {
        key: torch.from_numpy(np.asarray(obs_batch[key])).to("cpu")
        for key in OBS_KEYS
    }
    with torch.no_grad():
        pred = model(tensors).cpu().numpy()
    return np.asarray(pred, dtype=np.float32)


def _run_dataset_check(
    model: BrainBCActor,
    dataset_name: str,
    num_samples: int,
    seed: int,
    train_config: dict,
) -> None:
    npz_path = REPO / "data" / "oracle_bc" / f"{dataset_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    print("=" * 72)
    print(f"[dataset] dataset_name={dataset_name}")
    print(f"[dataset] npz_path={npz_path}")

    data = np.load(str(npz_path))
    try:
        missing = [
            key
            for key in (
                "obs_image",
                "obs_proprio",
                "obs_drives",
                "obs_prev_action",
                "obs_privileged",
                "actions",
            )
            if key not in data
        ]
        if missing:
            raise KeyError(f"Dataset missing required keys: {missing}")

        actions = _as_action_batch(data["actions"])
        n_total = len(actions)
        n = min(max(int(num_samples), 1), n_total)
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=n, replace=False)

        expected_proprio = int(train_config["proprio_dim"])
        actual_proprio = int(data["obs_proprio"].shape[1])
        if actual_proprio != expected_proprio:
            raise ValueError(
                "Dataset/model proprio_dim mismatch: "
                f"dataset={actual_proprio} train_config={expected_proprio}"
            )

        obs_batch = _obs_batch_from_npz(data, indices)
        labels = actions[indices]
        pred = _predict_batch(model, obs_batch)

        print(f"[dataset] total_rows={n_total} sampled_rows={n} seed={seed}")
        _print_summary("[dataset]", _metric_summary(pred, labels))
        _print_rows("[dataset] first 10 pred vs label rows:", pred, labels, "label")
    finally:
        data.close()


def _make_env(args: argparse.Namespace, train_config: dict) -> GeckoBrainEnv:
    walker_run = args.walker_run or train_config.get("walker_run", "v4_5b_speed_polish_1m")
    return GeckoBrainEnv(
        walker_run=walker_run,
        max_steps=int(args.steps),
        seed=int(args.seed),
        privileged_target=1.0,
        privileged_food_dropout_prob=0.0,
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        food_radius=float(args.food_radius),
        render_mode=None,
    )


def _append_metric(
    pred: np.ndarray,
    label: np.ndarray,
    dir_cosines: list[float],
    dist_errors: list[float],
    engage_errors: list[float],
) -> tuple[float, float, float]:
    dir_cos, dist_err, engage_err = _action_metric_arrays(pred, label)
    dir_value = float(dir_cos[0])
    dist_value = float(dist_err[0])
    engage_value = float(engage_err[0])
    dir_cosines.append(dir_value)
    dist_errors.append(dist_value)
    engage_errors.append(engage_value)
    return dir_value, dist_value, engage_value


def _run_rollout_check(
    model: BrainBCActor,
    args: argparse.Namespace,
    train_config: dict,
    *,
    teacher_forced: bool,
) -> None:
    mode_name = "oracle_rollout" if teacher_forced else "model_rollout"
    print("=" * 72)
    print(f"[{mode_name}] episodes={args.episodes} steps={args.steps} seed={args.seed}")
    print(
        f"[{mode_name}] food_spawn_angle_deg={args.food_spawn_angle_deg} "
        f"eat_radius={args.eat_radius} food_radius={args.food_radius}"
    )

    env = _make_env(args, train_config)
    all_pred: list[np.ndarray] = []
    all_oracle: list[np.ndarray] = []
    first_pred: np.ndarray | None = None
    first_oracle: np.ndarray | None = None
    first_metrics: tuple[float, float, float] | None = None
    dir_cosines: list[float] = []
    dist_errors: list[float] = []
    engage_errors: list[float] = []
    eat_count = 0
    min_mouth_food_dist = float("inf")
    total_steps = 0

    try:
        for ep in range(int(args.episodes)):
            obs, info = env.reset(seed=int(args.seed) + ep)
            if "mouth_food_dist" in info:
                min_mouth_food_dist = min(min_mouth_food_dist, float(info["mouth_food_dist"]))

            for step in range(int(args.steps)):
                oracle_before = np.asarray(env.oracle_action(), dtype=np.float32)
                pred = np.asarray(model.predict(obs), dtype=np.float32)

                metrics = _append_metric(
                    pred,
                    oracle_before,
                    dir_cosines,
                    dist_errors,
                    engage_errors,
                )
                if first_metrics is None:
                    first_pred = pred.copy()
                    first_oracle = oracle_before.copy()
                    first_metrics = metrics

                if len(all_pred) < 10:
                    all_pred.append(pred.copy())
                    all_oracle.append(oracle_before.copy())

                step_action = oracle_before if teacher_forced else pred
                obs, _, terminated, truncated, info = env.step(step_action)
                total_steps += 1

                eat_count += int(bool(info.get("ate", False)))
                mfd = info.get("mouth_food_dist")
                if mfd is not None:
                    min_mouth_food_dist = min(min_mouth_food_dist, float(mfd))

                if terminated or truncated:
                    print(
                        f"[{mode_name}] episode={ep + 1} ended_after={step + 1} "
                        f"terminated={terminated} truncated={truncated}"
                    )
                    break
    finally:
        env.close()

    if min_mouth_food_dist == float("inf"):
        min_mouth_food_dist = float("nan")

    pred_arr = np.asarray(all_pred, dtype=np.float32)
    oracle_arr = np.asarray(all_oracle, dtype=np.float32)
    summary = {
        "dir_cos": float(np.mean(dir_cosines)) if dir_cosines else float("nan"),
        "dist_err": float(np.mean(dist_errors)) if dist_errors else float("nan"),
        "engage_err": float(np.mean(engage_errors)) if engage_errors else float("nan"),
    }
    print(f"[{mode_name}] compared_steps={total_steps}")
    _print_summary(f"[{mode_name}]", summary)
    if first_metrics is not None and first_pred is not None and first_oracle is not None:
        dcos, derr, eerr = first_metrics
        print(
            f"[{mode_name}] first_step "
            f"dir_cos={dcos:.6f} dist_err={derr:.6f} engage_err={eerr:.6f} "
            f"pred={_fmt_action(first_pred)} oracle={_fmt_action(first_oracle)}"
        )
    _print_rows(f"[{mode_name}] first 10 pred vs oracle rows:", pred_arr, oracle_arr, "oracle")
    print(f"[{mode_name}] eat_count={eat_count} min_mouth_food_dist={min_mouth_food_dist:.6f}")


def _resolve_dataset_name(args: argparse.Namespace, train_config: dict) -> str | None:
    if args.dataset_name:
        return args.dataset_name
    value = train_config.get("dataset_name")
    return str(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug behavior cloning brain policy against dataset labels and live "
            "GeckoBrainEnv oracle actions."
        )
    )
    parser.add_argument("--brain-run", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=["dataset", "oracle_rollout", "model_rollout", "all"],
        default="all",
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--walker-run", type=str, default=None)
    args = parser.parse_args()

    run_dir = REPO / "models" / "brain" / args.brain_run
    train_config = _load_train_config(run_dir)
    model = _load_bc_actor(run_dir, train_config)
    dataset_name = _resolve_dataset_name(args, train_config)
    walker_run = args.walker_run or train_config.get("walker_run", "v4_5b_speed_polish_1m")

    print("=" * 72)
    print(f"[debug] brain_run={args.brain_run}")
    print(f"[debug] run_dir={run_dir}")
    print(f"[debug] algo={train_config.get('algo', 'unknown')}")
    print(f"[debug] walker_run={walker_run}")
    print(f"[debug] proprio_dim={train_config.get('proprio_dim')}")
    print("[debug] privileged_target=1.0 privileged_food_dropout_prob=0.0")
    print(f"[debug] mode={args.mode}")

    if args.mode in ("dataset", "all"):
        if not dataset_name:
            raise ValueError(
                "--dataset-name is required for dataset/all mode when train_config "
                "does not contain dataset_name"
            )
        _run_dataset_check(
            model,
            dataset_name=dataset_name,
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            train_config=train_config,
        )

    if args.mode in ("oracle_rollout", "all"):
        _run_rollout_check(model, args, train_config, teacher_forced=True)

    if args.mode in ("model_rollout", "all"):
        _run_rollout_check(model, args, train_config, teacher_forced=False)


if __name__ == "__main__":
    main()
