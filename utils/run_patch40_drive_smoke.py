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
        raise RuntimeError(f"Patch40A smoke step failed: {name} returncode={proc.returncode}")
    return output


def _parse_summary(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith("SUMMARY_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("watch_drive_brain output did not contain SUMMARY_JSON")


def _fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _behavior_fraction(summary: dict, behavior: str) -> float:
    return float(summary.get("behavior_fractions", {}).get(behavior, 0.0))


def _assert_no_falls(name: str, summary: dict) -> None:
    falls = int(summary.get("falls", 0))
    if falls != 0:
        raise RuntimeError(f"{name} failed: falls={falls}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch40A drive-arbiter smoke runner")
    parser.add_argument("--approach-brain-run", type=str, default="brain_v1_patch38c_visual_dagger_80k_seed0")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--rest-steps", type=int, default=120)
    parser.add_argument("--energy-steps", type=int, default=500)
    parser.add_argument("--closed-loop-steps", type=int, default=1200)
    parser.add_argument("--min-low-hunger-ignore-rate", type=float, default=0.70)
    parser.add_argument("--min-high-hunger-eat-rate", type=float, default=0.20)
    parser.add_argument("--min-closed-loop-cycles", type=int, default=0)
    args = parser.parse_args()

    py = sys.executable
    compile_targets = [
        "utils/watch_drive_brain.py",
        "utils/run_patch40_drive_smoke.py",
        "utils/watch_search_brain.py",
    ]
    _run_step("compile", [py, "-m", "py_compile", *compile_targets])
    _run_step(
        "import",
        [
            py,
            "-c",
            (
                "import importlib; "
                "mods=['utils.watch_drive_brain','utils.run_patch40_drive_smoke',"
                "'utils.watch_search_brain']; "
                "[importlib.import_module(m) for m in mods]; print('imports_ok')"
            ),
        ],
    )

    common = [
        "--walker-run",
        args.walker_run,
        "--food-spawn-angle-deg",
        str(float(args.food_spawn_angle_deg)),
        "--eat-radius",
        str(float(args.eat_radius)),
        "--food-radius",
        str(float(args.food_radius)),
    ]

    rest_summary = _parse_summary(
        _run_step(
            "rest_preflight",
            [
                py,
                "utils/watch_drive_brain.py",
                "--force-behavior",
                "REST",
                "--episodes",
                "1",
                "--steps",
                str(int(args.rest_steps)),
                "--seed",
                str(int(args.seed)),
                "--initial-hunger",
                "0.0",
                "--initial-energy",
                "0.20",
                *common,
            ],
        )
    )
    _assert_no_falls("rest_preflight", rest_summary)

    high_summary = _parse_summary(
        _run_step(
            "high_hunger_food_response",
            [
                py,
                "utils/watch_drive_brain.py",
                "--approach-brain-run",
                args.approach_brain_run,
                "--episodes",
                str(int(args.episodes)),
                "--steps",
                str(int(args.steps)),
                "--seed",
                str(int(args.seed) + 100),
                "--initial-hunger",
                "0.90",
                "--initial-energy",
                "1.0",
                *common,
            ],
        )
    )
    _assert_no_falls("high_hunger_food_response", high_summary)

    low_summary = _parse_summary(
        _run_step(
            "low_hunger_ignore_food",
            [
                py,
                "utils/watch_drive_brain.py",
                "--approach-brain-run",
                args.approach_brain_run,
                "--episodes",
                str(int(args.episodes)),
                "--steps",
                str(int(args.steps)),
                "--seed",
                str(int(args.seed) + 100),
                "--initial-hunger",
                "0.02",
                "--initial-energy",
                "1.0",
                *common,
            ],
        )
    )
    _assert_no_falls("low_hunger_ignore_food", low_summary)

    energy_summary = _parse_summary(
        _run_step(
            "energy_override",
            [
                py,
                "utils/watch_drive_brain.py",
                "--approach-brain-run",
                args.approach_brain_run,
                "--episodes",
                str(int(args.episodes)),
                "--steps",
                str(int(args.energy_steps)),
                "--seed",
                str(int(args.seed) + 300),
                "--initial-hunger",
                "0.90",
                "--initial-energy",
                "0.24",
                "--energy-critical-threshold",
                "0.25",
                "--energy-rest-exit-threshold",
                "0.26",
                *common,
            ],
        )
    )
    _assert_no_falls("energy_override", energy_summary)

    closed_summary = _parse_summary(
        _run_step(
            "mini_closed_loop_survival",
            [
                py,
                "utils/watch_drive_brain.py",
                "--approach-brain-run",
                args.approach_brain_run,
                "--episodes",
                "1",
                "--steps",
                str(int(args.closed_loop_steps)),
                "--seed",
                str(int(args.seed) + 500),
                "--initial-hunger",
                "0.42",
                "--initial-energy",
                "1.0",
                "--hungry-threshold",
                "0.30",
                "--sated-threshold",
                "0.15",
                *common,
            ],
        )
    )
    _assert_no_falls("mini_closed_loop_survival", closed_summary)

    high_hunger_eat_success = float(high_summary["episodes_with_eat"]) / max(
        float(high_summary["episodes"]), 1.0
    )
    low_hunger_ignore_rate = _behavior_fraction(low_summary, "REST") + _behavior_fraction(
        low_summary, "EXPLORE"
    )
    energy_rest_compliance = float(energy_summary.get("episodes_started_rest", 0)) / max(
        float(energy_summary["episodes"]), 1.0
    )
    closed_loop_cycles = int(closed_summary.get("closed_loop_cycles", 0))

    if high_hunger_eat_success < float(args.min_high_hunger_eat_rate):
        raise RuntimeError(
            "high_hunger_eat_success below threshold: "
            f"{high_hunger_eat_success:.4f} < {args.min_high_hunger_eat_rate:.4f}"
        )
    if low_hunger_ignore_rate < float(args.min_low_hunger_ignore_rate):
        raise RuntimeError(
            "low_hunger_ignore_rate below threshold: "
            f"{low_hunger_ignore_rate:.4f} < {args.min_low_hunger_ignore_rate:.4f}"
        )
    if energy_rest_compliance < 1.0:
        raise RuntimeError(
            f"energy_rest_compliance failed: {energy_rest_compliance:.4f}; "
            "low-energy episodes must begin in REST"
        )
    if not bool(energy_summary.get("energy_recovered_any", False)):
        raise RuntimeError("energy_override failed: energy did not recover during rest")
    if closed_loop_cycles < int(args.min_closed_loop_cycles):
        raise RuntimeError(
            "closed_loop_cycles below threshold: "
            f"{closed_loop_cycles} < {args.min_closed_loop_cycles}"
        )

    rows = [
        ("rest_preflight", rest_summary),
        ("high_hunger", high_summary),
        ("low_hunger", low_summary),
        ("energy_override", energy_summary),
        ("closed_loop", closed_summary),
    ]
    print("\nPatch40A drive smoke summary")
    print(
        "scenario high_hunger_eat_success low_hunger_ignore_rate energy_rest_compliance "
        "closed_loop_cycles eats falls transition_count mean_hunger final_hunger "
        "mean_energy final_energy REST EXPLORE SEARCH_FOOD APPROACH_FOOD EAT"
    )
    for name, summary in rows:
        print(
            " ".join(
                [
                    name,
                    _fmt(high_hunger_eat_success if name == "high_hunger" else float("nan")),
                    _fmt(low_hunger_ignore_rate if name == "low_hunger" else float("nan")),
                    _fmt(energy_rest_compliance if name == "energy_override" else float("nan")),
                    _fmt(closed_loop_cycles if name == "closed_loop" else float("nan")),
                    _fmt(summary.get("eat_count")),
                    _fmt(summary.get("falls")),
                    _fmt(summary.get("transition_count")),
                    _fmt(summary.get("mean_hunger")),
                    _fmt(summary.get("final_hunger")),
                    _fmt(summary.get("mean_energy")),
                    _fmt(summary.get("final_energy")),
                    _fmt(_behavior_fraction(summary, "REST")),
                    _fmt(_behavior_fraction(summary, "EXPLORE")),
                    _fmt(_behavior_fraction(summary, "SEARCH_FOOD")),
                    _fmt(_behavior_fraction(summary, "APPROACH_FOOD")),
                    _fmt(_behavior_fraction(summary, "EAT")),
                ]
            )
        )

    print("[patch40 smoke] PASS: drive arbiter smoke completed")


if __name__ == "__main__":
    main()
