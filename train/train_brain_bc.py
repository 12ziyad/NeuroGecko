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
from torch.utils.data import DataLoader, Dataset, random_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from brain.bc_actor import BrainBCActor, build_obs_space


class OracleBCDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(str(npz_path))
        self.image = torch.from_numpy(data["obs_image"])
        self.proprio = torch.from_numpy(data["obs_proprio"])
        self.drives = torch.from_numpy(data["obs_drives"])
        self.prev_action = torch.from_numpy(data["obs_prev_action"])
        self.privileged = torch.from_numpy(data["obs_privileged"])
        self.actions = torch.from_numpy(data["actions"])

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        obs = {
            "image": self.image[idx],
            "proprio": self.proprio[idx],
            "drives": self.drives[idx],
            "prev_action": self.prev_action[idx],
            "privileged": self.privileged[idx],
        }
        return obs, self.actions[idx]


def _losses(pred: torch.Tensor, label: torch.Tensor):
    """Returns (total_loss, dir_cos, dist_mae, engage_mae)."""
    cos_sim = F.cosine_similarity(pred[:, :2], label[:, :2], dim=1)
    dir_loss = (1.0 - cos_sim).mean()
    dist_loss = F.huber_loss(pred[:, 2], label[:, 2])
    engage_loss = F.mse_loss(pred[:, 3], label[:, 3])
    total = 2.0 * dir_loss + 1.0 * dist_loss + 0.5 * engage_loss
    dir_cos = cos_sim.mean().item()
    dist_mae = (pred[:, 2] - label[:, 2]).abs().mean().item()
    engage_mae = (pred[:, 3] - label[:, 3]).abs().mean().item()
    return total, dir_cos, dist_mae, engage_mae


def _run_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = dir_cos = dist_mae = engage_mae = 0.0
    n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for obs_batch, label_batch in loader:
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            label_batch = label_batch.to(device)
            pred = model(obs_batch)
            loss, cos, dist, eng = _losses(pred, label_batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            dir_cos += cos
            dist_mae += dist
            engage_mae += eng
            n += 1
    m = max(n, 1)
    return total_loss / m, dir_cos / m, dist_mae / m, engage_mae / m


def main() -> None:
    parser = argparse.ArgumentParser(description="Train feedforward BC brain actor")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-features-dim", type=int, default=128)
    parser.add_argument("--body-features-dim", type=int, default=96)
    parser.add_argument("--fused-features-dim", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = REPO / "data" / "oracle_bc"
    npz_path = data_dir / f"{args.dataset_name}.npz"
    meta_path = data_dir / f"{args.dataset_name}_meta.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    proprio_dim = int(meta["proprio_dim"])
    walker_run = meta.get("walker_run", "v4_5b_speed_polish_1m")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = REPO / "models" / "brain" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[bc train] run_name     = {args.run_name}")
    print(f"[bc train] dataset      = {args.dataset_name}")
    print(f"[bc train] proprio_dim  = {proprio_dim}")
    print(f"[bc train] epochs       = {args.epochs}")
    print(f"[bc train] batch_size   = {args.batch_size}")
    print(f"[bc train] lr           = {args.lr}")
    print(f"[bc train] device       = {device}")
    print("=" * 60)

    obs_space = build_obs_space(proprio_dim)
    model = BrainBCActor(
        obs_space,
        image_features_dim=args.image_features_dim,
        body_features_dim=args.body_features_dim,
        fused_features_dim=args.fused_features_dim,
        use_privileged=True,
        action_dim=4,
    ).to(device)

    dataset = OracleBCDataset(npz_path)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[bc train] train={n_train}  val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_cos, tr_dist, tr_eng = _run_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_cos, vl_dist, vl_eng = _run_epoch(model, val_loader, None, device)
        print(
            f"epoch={epoch:3d}  "
            f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
            f"dir_cos={vl_cos:.4f}  dist_err={vl_dist:.4f}  engage_err={vl_eng:.4f}"
        )
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), str(out_dir / "final.pt"))

    print(f"[bc train] best_val_loss = {best_val_loss:.4f}")
    print(f"[bc train] model  -> {out_dir / 'final.pt'}")

    train_config = {
        "run_name": args.run_name,
        "algo": "behavior_cloning",
        "observation_mode": "privileged",
        "use_privileged_food": True,
        "action_dim": 4,
        "brain_action": ["target_dir_x", "target_dir_y", "target_distance", "engage"],
        "walker_run": walker_run,
        "food_spawn_angle_deg": float(meta.get("food_spawn_angle_deg", 180.0)),
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
        "best_val_loss": float(best_val_loss),
    }
    config_path = out_dir / "train_config.json"
    config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    print(f"[bc train] config -> {config_path}")


if __name__ == "__main__":
    main()
