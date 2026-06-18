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
        raise RuntimeError(f"Patch38C smoke step failed: {name} returncode={proc.returncode}")
    return output


def _check_dataset(dataset_name: str) -> None:
    path = REPO / "data" / "visual_distill" / f"{dataset_name}.npz"
    if not path.exists():
        raise RuntimeError(f"Patch38C smoke dataset missing: {path}")
    data = np.load(str(path))
    try:
        actions = np.asarray(data["actions"])
        privileged = np.asarray(data["obs_privileged"])
        if actions.ndim != 2 or actions.shape[1] != 4:
            raise RuntimeError(f"Patch38C violation: actions shape is {actions.shape}")
        if not np.allclose(privileged, 0.0):
            raise RuntimeError(
                "Patch38C violation: obs_privileged is nonzero; "
                f"max_abs={float(np.max(np.abs(privileged))):.8f}"
            )
    finally:
        data.close()
    print("[patch38c smoke] PASS: DAgger dataset is zero-privileged and 4D-labeled")


def _check_model(run_name: str) -> None:
    run_dir = REPO / "models" / "brain" / run_name
    model_path = run_dir / "final.pt"
    config_path = run_dir / "train_config.json"
    if not model_path.exists():
        raise RuntimeError(f"Patch38C smoke model missing: {model_path}")
    if not config_path.exists():
        raise RuntimeError(f"Patch38C smoke config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("algo") != "visual_distillation":
        raise RuntimeError(f"Patch38C violation: algo is {config.get('algo')!r}")
    if bool(config.get("use_privileged_food", True)):
        raise RuntimeError("Patch38C violation: trained visual student uses privileged food")
    if int(config.get("action_dim", 0)) != 4:
        raise RuntimeError(f"Patch38C violation: action_dim is {config.get('action_dim')!r}")
    print("[patch38c smoke] PASS: train_config use_privileged_food=false action_dim=4")


def _check_watcher_output(output: str) -> None:
    if "FINAL/VISUAL MODE: privileged food OFF" not in output:
        raise RuntimeError("Patch38C violation: watcher did not print final visual mode line")
    if "[watch] privileged   = YES" in output:
        raise RuntimeError("Patch38C violation: watcher enabled privileged food")
    if "[watch] oracle_action = YES" in output:
        raise RuntimeError("Patch38C violation: watcher enabled oracle action")
    print("[patch38c smoke] PASS: watcher ran visual eval with privileged OFF and oracle OFF")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch38C visual DAgger smoke runner")
    parser.add_argument("--student-brain-run", type=str, default="brain_v1_patch38b_visual_50k_seed0")
    parser.add_argument("--teacher-brain-run", type=str, default="brain_v1_patch37b_dagger_200k_seed1")
    parser.add_argument("--base-dataset-name", type=str, default="patch38b_visual_50k_seed0")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--num-transitions", type=int, default=3000)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--train-max-rows", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--watch-episodes", type=int, default=1)
    parser.add_argument("--watch-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "logs" / "patch38c_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"patch38c_visual_dagger_smoke_{stamp}.log"
    dagger_dataset_name = f"patch38c_dagger_smoke_{stamp}"
    run_name = f"brain_v1_patch38c_smoke_{stamp}"
    py = sys.executable

    compile_targets = [
        "utils/collect_visual_dagger_dataset.py",
        "train/train_visual_distill.py",
        "utils/watch_trained_brain.py",
        "utils/debug_visual_policy.py",
        "utils/run_patch38c_visual_dagger_smoke.py",
    ]
    _run_step("compile", [py, "-m", "py_compile", *compile_targets], log_path)
    _run_step(
        "import",
        [
            py,
            "-c",
            (
                "import importlib; "
                "mods=['utils.collect_visual_dagger_dataset','train.train_visual_distill',"
                "'utils.watch_trained_brain','utils.debug_visual_policy']; "
                "[importlib.import_module(m) for m in mods]; print('imports_ok')"
            ),
        ],
        log_path,
    )
    _run_step(
        "collect_visual_dagger",
        [
            py,
            "utils/collect_visual_dagger_dataset.py",
            "--student-brain-run",
            args.student_brain_run,
            "--teacher-brain-run",
            args.teacher_brain_run,
            "--dataset-name",
            dagger_dataset_name,
            "--num-transitions",
            str(int(args.num_transitions)),
            "--walker-run",
            args.walker_run,
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
    _check_dataset(dagger_dataset_name)
    _run_step(
        "train_visual_distill_with_dagger",
        [
            py,
            "train/train_visual_distill.py",
            "--dataset-name",
            args.base_dataset_name,
            "--extra-dataset-names",
            dagger_dataset_name,
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
        "watch_visual_eval",
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
    print("[patch38c smoke] PASS: visual DAgger smoke completed")
    print(f"[patch38c smoke] log={log_path}")
    print(f"[patch38c smoke] dagger_dataset_name={dagger_dataset_name}")
    print(f"[patch38c smoke] brain_run={run_name}")


if __name__ == "__main__":
    main()
