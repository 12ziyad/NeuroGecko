from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from brain.bc_actor import BrainBCActor, build_obs_space


DEFAULT_TEACHER = "brain_v1_patch37b_dagger_200k_seed1"


class VisualDistillDataset(Dataset):
    def __init__(self, npz_path: Path, max_rows: int | None = None):
        data = np.load(str(npz_path))
        required = (
            "obs_image",
            "obs_proprio",
            "obs_drives",
            "obs_prev_action",
            "obs_privileged",
            "actions",
        )
        missing = [key for key in required if key not in data]
        if missing:
            data.close()
            raise KeyError(f"Visual distill dataset missing required keys: {missing}")

        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 4:
            data.close()
            raise ValueError(f"Patch38A requires actions shape (N, 4), got {actions.shape}")

        privileged = np.asarray(data["obs_privileged"], dtype=np.float32)
        if privileged.ndim != 2 or privileged.shape[1] != 5:
            data.close()
            raise ValueError(f"Expected obs_privileged shape (N, 5), got {privileged.shape}")
        if not np.allclose(privileged, 0.0):
            max_abs = float(np.max(np.abs(privileged)))
            data.close()
            raise ValueError(
                "Patch38A violation: visual student obs_privileged must be all zeros; "
                f"max_abs={max_abs:.8f}"
            )

        n = int(actions.shape[0])
        if max_rows is not None:
            n = min(n, max(1, int(max_rows)))

        self.image = torch.from_numpy(np.asarray(data["obs_image"][:n], dtype=np.uint8))
        self.proprio = torch.from_numpy(np.asarray(data["obs_proprio"][:n], dtype=np.float32))
        self.drives = torch.from_numpy(np.asarray(data["obs_drives"][:n], dtype=np.float32))
        self.prev_action = torch.from_numpy(
            np.asarray(data["obs_prev_action"][:n], dtype=np.float32)
        )
        self.privileged = torch.from_numpy(np.zeros((n, 5), dtype=np.float32))
        self.actions = torch.from_numpy(actions[:n])
        data.close()

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, idx: int):
        obs = {
            "image": self.image[idx],
            "proprio": self.proprio[idx],
            "drives": self.drives[idx],
            "prev_action": self.prev_action[idx],
            "privileged": self.privileged[idx],
        }
        return obs, self.actions[idx]


def _load_dataset_meta(data_dir: Path, dataset_name: str) -> tuple[Path, dict]:
    npz_path = data_dir / f"{dataset_name}.npz"
    meta_path = data_dir / f"{dataset_name}_meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    return npz_path, json.loads(meta_path.read_text(encoding="utf-8"))


def _parse_dataset_names(value: str | None) -> list[str]:
    names: list[str] = []
    for part in str(value or "").split(","):
        name = part.strip()
        if name:
            names.append(name)
    return names


def _validate_visual_meta(dataset_name: str, meta: dict) -> None:
    if meta.get("observation_mode") != "visual":
        raise ValueError(
            f"Dataset {dataset_name!r} must have observation_mode='visual', "
            f"got {meta.get('observation_mode')!r}"
        )
    if bool(meta.get("use_privileged_food_student", meta.get("use_privileged_food", True))):
        raise ValueError(
            f"Patch38C violation: dataset {dataset_name!r} enables privileged student food"
        )
    if int(meta.get("action_dim", 4)) != 4:
        raise ValueError(
            f"Patch38C violation: dataset {dataset_name!r} action_dim is "
            f"{meta.get('action_dim')!r}, expected 4"
        )


def _validate_compatible_meta(
    main_name: str,
    main_meta: dict,
    extra_name: str,
    extra_meta: dict,
) -> None:
    main_proprio = int(main_meta["proprio_dim"])
    extra_proprio = int(extra_meta["proprio_dim"])
    if extra_proprio != main_proprio:
        raise ValueError(
            "Cannot concatenate visual datasets with different proprio_dim: "
            f"{main_name}={main_proprio}, {extra_name}={extra_proprio}"
        )


def _losses(pred: torch.Tensor, label: torch.Tensor):
    cos_sim = F.cosine_similarity(pred[:, :2], label[:, :2], dim=1, eps=1e-8)
    dir_loss = (1.0 - cos_sim).mean()
    dist_loss = F.smooth_l1_loss(pred[:, 2], label[:, 2])
    engage_loss = F.mse_loss(pred[:, 3], label[:, 3])
    total = 2.0 * dir_loss + dist_loss + 0.5 * engage_loss
    dir_cos = float(cos_sim.mean().detach().cpu())
    dist_mae = float((pred[:, 2] - label[:, 2]).abs().mean().detach().cpu())
    engage_mae = float((pred[:, 3] - label[:, 3]).abs().mean().detach().cpu())
    return total, dir_cos, dist_mae, engage_mae


def _run_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = dir_cos = dist_mae = engage_mae = 0.0
    n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for obs_batch, label_batch in loader:
            obs_batch = {key: value.to(device) for key, value in obs_batch.items()}
            label_batch = label_batch.to(device)
            pred = model(obs_batch)
            if pred.shape[-1] != 4:
                raise RuntimeError(f"Visual student emitted non-4D action shape {pred.shape}")
            loss, cos, dist, engage = _losses(pred, label_batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            dir_cos += cos
            dist_mae += dist
            engage_mae += engage
            n += 1
    denom = max(n, 1)
    return total_loss / denom, dir_cos / denom, dist_mae / denom, engage_mae / denom


def _split_dataset(dataset: Dataset, val_frac: float, seed: int):
    n_total = len(dataset)
    if n_total <= 0:
        raise ValueError("Dataset is empty")
    n_val = int(n_total * max(0.0, min(float(val_frac), 0.9)))
    if n_total > 1 and n_val < 1 and float(val_frac) > 0.0:
        n_val = 1
    n_train = n_total - n_val
    if n_train <= 0:
        return dataset, None
    if n_val <= 0:
        return dataset, None
    return random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(seed)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a visual-only distillation brain actor")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument(
        "--extra-dataset-names",
        type=str,
        default="",
        help="Comma-separated additional visual_distill dataset names to concatenate.",
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--teacher-brain-run", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--image-features-dim", type=int, default=128)
    parser.add_argument("--body-features-dim", type=int, default=96)
    parser.add_argument("--fused-features-dim", type=int, default=256)
    args = parser.parse_args()

    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    data_dir = REPO / "data" / "visual_distill"
    npz_path, meta = _load_dataset_meta(data_dir, args.dataset_name)
    _validate_visual_meta(args.dataset_name, meta)

    extra_dataset_names = _parse_dataset_names(args.extra_dataset_names)
    dataset_names = [args.dataset_name] + extra_dataset_names
    dataset_paths = [npz_path]
    for extra_name in extra_dataset_names:
        extra_npz_path, extra_meta = _load_dataset_meta(data_dir, extra_name)
        _validate_visual_meta(extra_name, extra_meta)
        _validate_compatible_meta(args.dataset_name, meta, extra_name, extra_meta)
        dataset_paths.append(extra_npz_path)

    datasets = [VisualDistillDataset(path) for path in dataset_paths]
    dataset_lengths = [len(ds) for ds in datasets]
    dataset_full = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if args.max_rows is not None:
        max_rows = max(1, int(args.max_rows))
        dataset = Subset(dataset_full, range(min(max_rows, len(dataset_full))))
    else:
        dataset = dataset_full

    proprio_dim = int(meta.get("proprio_dim", datasets[0].proprio.shape[1]))
    if int(datasets[0].proprio.shape[1]) != proprio_dim:
        raise ValueError(
            "Dataset/meta proprio_dim mismatch: "
            f"dataset={datasets[0].proprio.shape[1]} meta={proprio_dim}"
        )

    train_set, val_set = _split_dataset(dataset, float(args.val_frac), int(args.seed))
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True)
    val_loader = (
        DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False)
        if val_set is not None
        else None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BrainBCActor(
        build_obs_space(proprio_dim),
        image_features_dim=int(args.image_features_dim),
        body_features_dim=int(args.body_features_dim),
        fused_features_dim=int(args.fused_features_dim),
        use_privileged=False,
        action_dim=4,
    ).to(device)

    out_dir = REPO / "models" / "brain" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    print("=" * 72)
    print(f"[visual train] run_name={args.run_name}")
    print(f"[visual train] dataset_names={dataset_names}")
    print(
        f"[visual train] rows={len(dataset)} train={len(train_set)} "
        f"by_dataset={dict(zip(dataset_names, dataset_lengths))}"
    )
    if val_set is not None:
        print(f"[visual train] val={len(val_set)}")
    print(f"[visual train] device={device}")
    print("[visual train] FINAL/VISUAL TRAINING: privileged food OFF")
    print("=" * 72)

    best_metric = float("inf")
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        tr_loss, tr_cos, tr_dist, tr_eng = _run_epoch(model, train_loader, optimizer, device)
        if val_loader is not None:
            vl_loss, vl_cos, vl_dist, vl_eng = _run_epoch(model, val_loader, None, device)
            metric = vl_loss
        else:
            vl_loss, vl_cos, vl_dist, vl_eng = tr_loss, tr_cos, tr_dist, tr_eng
            metric = tr_loss
        print(
            f"epoch={epoch:3d} train={tr_loss:.4f} val={vl_loss:.4f} "
            f"dir_cos={vl_cos:.4f} dist_err={vl_dist:.4f} engage_err={vl_eng:.4f}"
        )
        if metric < best_metric:
            best_metric = float(metric)
            best_epoch = int(epoch)
            torch.save(model.state_dict(), str(out_dir / "final.pt"))

    train_config = {
        "run_name": args.run_name,
        "algo": "visual_distillation",
        "observation_mode": "visual",
        "train_obs": "visual",
        "use_privileged_food": False,
        "use_privileged_food_student": False,
        "action_dim": 4,
        "brain_action": ["target_dir_x", "target_dir_y", "target_distance", "engage"],
        "teacher_brain_run": str(meta.get("teacher_brain_run", args.teacher_brain_run)),
        "walker_run": meta.get("walker_run", "v4_5b_speed_polish_1m"),
        "food_spawn_angle_deg": float(meta.get("food_spawn_angle_deg", 60.0)),
        "eat_radius": float(meta.get("eat_radius", 0.10)),
        "food_radius": float(meta.get("food_radius", 0.035)),
        "proprio_dim": int(proprio_dim),
        "image_features_dim": int(args.image_features_dim),
        "body_features_dim": int(args.body_features_dim),
        "fused_features_dim": int(args.fused_features_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_frac": float(args.val_frac),
        "seed": int(args.seed),
        "dataset_name": args.dataset_name,
        "extra_dataset_names": extra_dataset_names,
        "dataset_names": dataset_names,
        "dataset_num_transitions": int(len(dataset)),
        "dataset_total_rows_before_max_rows": int(len(dataset_full)),
        "dataset_row_counts": {
            name: int(length) for name, length in zip(dataset_names, dataset_lengths)
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_metric),
    }
    if train_config["use_privileged_food"] is not False:
        raise RuntimeError("Patch38A violation: visual train_config must disable privileged food")

    model_path = out_dir / "final.pt"
    if not model_path.exists():
        raise RuntimeError(f"Training failed to save model: {model_path}")
    config_path = out_dir / "train_config.json"
    config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    print(f"[visual train] best_epoch={best_epoch} best_val_loss={best_metric:.6f}")
    print(f"[visual train] model  -> {model_path}")
    print(f"[visual train] config -> {config_path}")
    print("[visual train] PASS: use_privileged_food=false action_dim=4")


if __name__ == "__main__":
    main()
