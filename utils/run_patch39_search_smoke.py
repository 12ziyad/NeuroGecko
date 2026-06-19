from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run_step(name: str, cmd: list[str]) -> str:
    print("\n" + "=" * 88)
    print(f"[{name}]")
    print(" ".join(cmd))
    print("=" * 88)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout or ""
    print(output, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"Patch39A smoke step failed: {name} returncode={proc.returncode}")
    return output


def _parse_summary(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith("SUMMARY_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("watch_search_brain output did not contain SUMMARY_JSON line")


def _fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch39A scripted visual search smoke runner")
    parser.add_argument("--approach-brain-run", type=str, default="brain_v1_patch38c_visual_dagger_80k_seed0")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--characterize-steps", type=int, default=80)
    args = parser.parse_args()

    py = sys.executable
    compile_targets = [
        "utils/characterize_search_actions.py",
        "utils/watch_search_brain.py",
        "utils/run_patch39_search_smoke.py",
    ]
    _run_step("compile", [py, "-m", "py_compile", *compile_targets])
    _run_step(
        "import",
        [
            py,
            "-c",
            (
                "import importlib; "
                "mods=['utils.characterize_search_actions','utils.watch_search_brain',"
                "'utils.run_patch39_search_smoke']; "
                "[importlib.import_module(m) for m in mods]; print('imports_ok')"
            ),
        ],
    )
    _run_step(
        "characterize_search_actions",
        [
            py,
            "utils/characterize_search_actions.py",
            "--walker-run",
            args.walker_run,
            "--steps",
            str(int(args.characterize_steps)),
            "--seed",
            str(int(args.seed)),
            "--food-radius",
            str(float(args.food_radius)),
            "--eat-radius",
            str(float(args.eat_radius)),
        ],
    )

    summaries: list[dict] = []
    for angle in (60.0, 120.0, 180.0):
        output = _run_step(
            f"search_approach_{int(angle)}deg",
            [
                py,
                "utils/watch_search_brain.py",
                "--approach-brain-run",
                args.approach_brain_run,
                "--walker-run",
                args.walker_run,
                "--episodes",
                str(int(args.episodes)),
                "--steps",
                str(int(args.steps)),
                "--seed",
                str(int(args.seed) + int(angle)),
                "--food-spawn-angle-deg",
                str(float(angle)),
                "--food-radius",
                str(float(args.food_radius)),
                "--eat-radius",
                str(float(args.eat_radius)),
            ],
        )
        summaries.append(_parse_summary(output))

    print("\nPatch39A smoke summary")
    columns = [
        "angle",
        "episodes",
        "episodes_with_eat",
        "total_eats",
        "search_success_rate",
        "mean_time_to_first_visible",
        "handoff_success_rate",
        "falls",
        "mean_search_frac",
        "mean_approach_frac",
        "transition_count",
    ]
    print(" ".join(columns))
    for row in summaries:
        print(" ".join(_fmt(row.get(col)) for col in columns))


if __name__ == "__main__":
    main()
