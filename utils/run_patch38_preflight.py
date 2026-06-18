from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _run_step(name: str, cmd: list[str], log_path: Path) -> str:
    header = "\n" + "=" * 88 + f"\n[{name}]\n" + " ".join(cmd) + "\n" + "=" * 88 + "\n"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(header)
    print(header, end="")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout or ""
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(output)
        fh.write(f"\n[{name}] returncode={proc.returncode}\n")
    print(output, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"Patch38A preflight step failed: {name} returncode={proc.returncode}")
    return output


def _check_dataset(dataset_name: str) -> None:
    path = REPO / "data" / "visual_distill" / f"{dataset_name}.npz"
    if not path.exists():
        raise RuntimeError(f"Patch38A preflight dataset missing: {path}")
    data = np.load(str(path))
    try:
        actions = np.asarray(data["actions"])
        privileged = np.asarray(data["obs_privileged"])
        if actions.ndim != 2 or actions.shape[1] != 4:
            raise RuntimeError(f"Patch38A violation: actions shape is {actions.shape}")
        if not np.allclose(privileged, 0.0):
            max_abs = float(np.max(np.abs(privileged)))
            raise RuntimeError(
                "Patch38A violation: student obs_privileged is nonzero; "
                f"max_abs={max_abs:.8f}"
            )
    finally:
        data.close()
    print("[preflight] PASS: dataset actions are 4D and obs_privileged is all zeros")


def _check_model(run_name: str) -> None:
    run_dir = REPO / "models" / "brain" / run_name
    model_path = run_dir / "final.pt"
    config_path = run_dir / "train_config.json"
    if not model_path.exists():
        raise RuntimeError(f"Patch38A preflight model missing: {model_path}")
    if not config_path.exists():
        raise RuntimeError(f"Patch38A preflight train_config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("algo") != "visual_distillation":
        raise RuntimeError(f"Patch38A violation: algo is {config.get('algo')!r}")
    if config.get("observation_mode") != "visual":
        raise RuntimeError(
            f"Patch38A violation: observation_mode is {config.get('observation_mode')!r}"
        )
    if bool(config.get("use_privileged_food", True)):
        raise RuntimeError("Patch38A violation: visual train_config uses privileged food")
    if int(config.get("action_dim", 0)) != 4:
        raise RuntimeError(f"Patch38A violation: action_dim is {config.get('action_dim')!r}")
    print("[preflight] PASS: visual model saved with use_privileged_food=false action_dim=4")


def _check_watcher_output(output: str) -> None:
    if "FINAL/VISUAL MODE: privileged food OFF" not in output:
        raise RuntimeError("Patch38A violation: watcher did not print final visual mode line")
    if "[watch] privileged   = YES" in output:
        raise RuntimeError("Patch38A violation: watcher enabled privileged food")
    print("[preflight] PASS: watcher ran with privileged food OFF")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Patch38A camera/visual-distill preflight")
    parser.add_argument("--teacher-brain-run", type=str, default="brain_v1_patch37b_dagger_200k_seed1")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--audit-samples", type=int, default=24)
    parser.add_argument("--num-transitions", type=int, default=2000)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--train-max-rows", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--watch-episodes", type=int, default=1)
    parser.add_argument("--watch-steps", type=int, default=80)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "logs" / "patch38_preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"patch38_preflight_{stamp}.log"
    dataset_name = f"patch38_preflight_{stamp}"
    run_name = f"brain_v1_patch38a_preflight_{stamp}"

    py = sys.executable
    compile_targets = [
        "utils/audit_camera_visibility.py",
        "utils/collect_visual_distill_dataset.py",
        "train/train_visual_distill.py",
        "utils/run_patch38_preflight.py",
        "utils/watch_trained_brain.py",
    ]
    _run_step("compile", [py, "-m", "py_compile", *compile_targets], log_path)
    _run_step(
        "import",
        [
            py,
            "-c",
            (
                "import importlib; "
                "mods=['utils.audit_camera_visibility','utils.collect_visual_distill_dataset',"
                "'train.train_visual_distill','utils.watch_trained_brain']; "
                "[importlib.import_module(m) for m in mods]; "
                "print('imports_ok')"
            ),
        ],
        log_path,
    )
    _run_step(
        "camera_audit",
        [
            py,
            "utils/audit_camera_visibility.py",
            "--walker-run",
            args.walker_run,
            "--num-samples",
            str(int(args.audit_samples)),
            "--max-steps",
            str(int(args.watch_steps)),
            "--seed",
            str(int(args.seed)),
            "--food-spawn-angle-deg",
            str(float(args.food_spawn_angle_deg)),
            "--eat-radius",
            str(float(args.eat_radius)),
            "--food-radius",
            str(float(args.food_radius)),
            "--save-debug-frames",
            "--debug-frame-count",
            "4",
        ],
        log_path,
    )
    _run_step(
        "collect_visual_distill_dataset",
        [
            py,
            "utils/collect_visual_distill_dataset.py",
            "--teacher-brain-run",
            args.teacher_brain_run,
            "--walker-run",
            args.walker_run,
            "--dataset-name",
            dataset_name,
            "--num-transitions",
            str(int(args.num_transitions)),
            "--seed",
            str(int(args.seed)),
            "--food-spawn-angle-deg",
            str(float(args.food_spawn_angle_deg)),
            "--eat-radius",
            str(float(args.eat_radius)),
            "--food-radius",
            str(float(args.food_radius)),
            "--max-steps",
            "500",
        ],
        log_path,
    )
    _check_dataset(dataset_name)
    _run_step(
        "train_visual_distill",
        [
            py,
            "train/train_visual_distill.py",
            "--dataset-name",
            dataset_name,
            "--run-name",
            run_name,
            "--teacher-brain-run",
            args.teacher_brain_run,
            "--epochs",
            str(int(args.train_epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--max-rows",
            str(int(args.train_max_rows)),
            "--seed",
            str(int(args.seed)),
        ],
        log_path,
    )
    _check_model(run_name)
    watcher_output = _run_step(
        "watch_visual_student",
        [
            py,
            "utils/watch_trained_brain.py",
            "--brain-run",
            run_name,
            "--walker-run",
            args.walker_run,
            "--episodes",
            str(int(args.watch_episodes)),
            "--steps",
            str(int(args.watch_steps)),
            "--seed",
            str(int(args.seed)),
            "--food-spawn-angle-deg",
            str(float(args.food_spawn_angle_deg)),
            "--eat-radius",
            str(float(args.eat_radius)),
            "--food-radius",
            str(float(args.food_radius)),
        ],
        log_path,
    )
    _check_watcher_output(watcher_output)

    print("=" * 88)
    print("[preflight] PASS: Patch38A legal visual pipeline preflight completed")
    print(f"[preflight] log={log_path}")
    print(f"[preflight] dataset_name={dataset_name}")
    print(f"[preflight] brain_run={run_name}")


if __name__ == "__main__":
    main()
