"""Run reviewed unittest contracts without learning, graphics, or GPU access.

Usage: python -m utils.test_no_training --json artifacts/evidence/session2/tests.json

The exclusions are exact class IDs, not name-substring guesses. BundleTests
uses fake models and remains included. CameraOptionTests only builds a CPU
MjvScene and remains included. TraceReplayTests only tests sample indices.
This is a safety-guarded test runner for trusted repository tests, not a sandbox
for arbitrary Python or a proof that every future optimization is excluded.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXCLUDED_CLASSES = {
    "test_checkpoints.PPOResumeSmokeTests":
        "Actually executes PPO.learn twice, even though on CPU and named smoke.",
    "test_policy_camera.CameraPixelTests":
        "Creates MuJoCo OpenGL renderers and compares rendered pixels.",
}


class CountedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_ids = []

    def addSuccess(self, test):
        self.successful_ids.append(test.id())
        super().addSuccess(test)


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def select_tests(discovered):
    selected, excluded = [], []
    for test in flatten(discovered):
        # unittest discover uses unprefixed modules without tests/__init__.py;
        # explicit loadTestsFromName uses tests.module. Both exact forms agree.
        class_id = test.id().rsplit(".", 1)[0].removeprefix("tests.")
        if class_id in EXCLUDED_CLASSES:
            excluded.append({"id": test.id(), "reason": EXCLUDED_CLASSES[class_id]})
        else:
            selected.append(test)
    return selected, excluded


def install_runtime_guards(stack, blocked_attempts):
    """Fail closed on accidental SB3 learning or graphics in retained tests."""
    # Install before discovery imports test modules or their project imports.
    # Process-local only: no driver, environment, or persistent setting changes.
    stack.enter_context(patch.dict(os.environ, {
        "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "disable",
    }))
    import mujoco
    import stable_baselines3  # Imports supported algorithms, but starts no job.
    from stable_baselines3.common.base_class import BaseAlgorithm
    import torch

    def reject(operation):
        def blocked(*args, **kwargs):
            blocked_attempts.append(operation)
            raise RuntimeError("No-training runner forbids " + operation)
        return blocked

    # Guard all currently imported SB3 algorithm implementations, including
    # overridden learn/train methods. A broad test exception cannot hide an
    # attempted operation: blocked_attempts also makes the whole run fail.
    classes, pending = set(), [BaseAlgorithm]
    while pending:
        cls = pending.pop()
        if cls in classes:
            continue
        classes.add(cls)
        pending.extend(cls.__subclasses__())
    for cls in classes:
        for method in ("learn", "train"):
            if callable(getattr(cls, method, None)):
                stack.enter_context(patch.object(cls, method,
                    reject(cls.__module__ + "." + cls.__name__ + "." + method)))
    for name in ("Renderer", "MjrContext"):
        stack.enter_context(patch.object(mujoco, name, reject("mujoco." + name)))
    stack.enter_context(patch.object(torch.cuda, "_lazy_init", reject("torch.cuda initialization")))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    stack.callback(torch.set_num_threads, old_threads)


def run_tests(*, verbosity=1, list_only=False):
    started = time.monotonic()
    blocked_attempts = []
    with ExitStack() as stack:
        install_runtime_guards(stack, blocked_attempts)
        loader = unittest.TestLoader()
        discovered = loader.discover(str(REPO / "tests"), pattern="test_*.py")
        selected, excluded = select_tests(discovered)
        ids = [test.id() for test in selected]
        if len(set(ids)) != len(ids):
            raise RuntimeError("Duplicate discovered test IDs; refuse misleading totals")
        report = {
            "protocol": "CPU-only unittest contracts; no policy learning or image rendering",
            "discovered": len(selected) + len(excluded),
            "selected": len(selected), "excluded_count": len(excluded),
            "excluded": excluded, "selected_test_ids": ids,
            "discovery_errors": list(loader.errors),
            "list_only": list_only,
            "guard_limit": "Exact reviewed exclusions plus runtime SB3/OpenGL/CUDA guards; trusted code, not a general sandbox.",
        }
        print(f"Discovered {report['discovered']}; selected {len(selected)}; "
              f"excluded {len(excluded)} learning/graphics tests.", flush=True)
        for item in excluded:
            print("EXCLUDED " + item["id"] + ": " + item["reason"], flush=True)
        if list_only:
            for test_id in ids:
                print(test_id)
            report["success"] = not loader.errors and not blocked_attempts
            report["executed"] = 0
        else:
            result = unittest.TextTestRunner(verbosity=verbosity, resultclass=CountedResult).run(unittest.TestSuite(selected))
            report.update({
                "executed": result.testsRun,
                "failures": [{"id": test.id(), "traceback": tb} for test, tb in result.failures],
                "errors": [{"id": test.id(), "traceback": tb} for test, tb in result.errors],
                "skipped": [{"id": test.id(), "reason": reason} for test, reason in result.skipped],
                "expected_failures": [test.id() for test, _ in result.expectedFailures],
                "unexpected_successes": [test.id() for test in result.unexpectedSuccesses],
                "passed": len(result.successful_ids),
                "success": bool(result.wasSuccessful() and not loader.errors
                                and not blocked_attempts and result.testsRun == len(selected)),
            })
        report["blocked_operation_attempts"] = blocked_attempts
        report["wall_seconds"] = time.monotonic() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional machine-readable test evidence")
    parser.add_argument("--list", action="store_true", help="Discover/filter only; execute no tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    report = run_tests(verbosity=2 if args.verbose else 1, list_only=args.list)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
        ("discovered", "selected", "excluded_count", "executed", "success", "wall_seconds")}), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
