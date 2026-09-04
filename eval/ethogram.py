"""Coarse event-count ethogram for traces, never a duration-budget substitute.

Source: Krönke & Xu (2023), Animals 13:3595, doi:10.3390/ani13233595.
The published ethogram uses >=1 snout-to-vent length for walking, <1 SVL with
leg movement for orientation change, and >=3 s inactivity for rest. Thresholds
which operationalize numerical motion/noise and bout boundaries are engineering
proxies, not additional published measurements. This detector cannot recognize
sensory exploration, intention, prey capture, distress, or hide-place context.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def score_ethogram(trace, trunk_activity_m_s=None, joint_activity_rad_s=None,
                   rest_s=None, walk_svl=None, pause_merge_s=None):
    from common.provenance import parameter_value
    from realism_metrics import _runs, validate_times
    defaults = {"trunk_activity_m_s": "ethogram_trunk_activity_m_s",
                "joint_activity_rad_s": "ethogram_joint_activity_rad_s",
                "rest_s": "ethogram_rest_s", "walk_svl": "ethogram_walk_svl",
                "pause_merge_s": "ethogram_pause_merge_s"}
    provided = locals().copy()
    cfg = {k: parameter_value(ref) if provided[k] is None else float(provided[k]) for k, ref in defaults.items()}
    if any(not np.isfinite(v) or v<0 for v in cfg.values()) or cfg["walk_svl"]<=0:
        raise ValueError("Ethogram thresholds must be finite/nonnegative; walking length must be positive.")
    meta, s = trace["metadata"], trace["samples"]
    t, dt = validate_times(s["time_s"])
    xyz = np.asarray(s["trunk_position_m"], dtype=float)
    svl = float(meta["svl_m"])
    if xyz.shape != (len(t), 3) or not np.isfinite(xyz).all() or svl<=0:
        raise ValueError("Need finite trunk positions and positive SVL.")
    # Vertical falling/rising is activity too; walking displacement below remains planar.
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)/dt
    active = speed > cfg["trunk_activity_m_s"]
    leg_active = np.zeros_like(active)
    has_joint_data = "hinge_position_deg" in s and bool(meta.get("hinge_names"))
    if has_joint_data:
        q = np.radians(np.asarray(s["hinge_position_deg"], dtype=float))
        if q.shape != (len(t), len(meta["hinge_names"])) or not np.isfinite(q).all():
            raise ValueError("Hinge trace shape/values are invalid.")
        velocity = np.abs(np.diff(q, axis=0))/dt
        active |= np.any(velocity>cfg["joint_activity_rad_s"], axis=1)
        ids = [i for i, name in enumerate(meta["hinge_names"]) if name.startswith(
            ("hip_", "knee_", "ankle_", "shoulder_", "elbow_", "wrist_"))]
        if ids:
            leg_active = np.any(velocity[:, ids]>cfg["joint_activity_rad_s"], axis=1)
    if "trunk_quaternion_wxyz" in s:
        q = np.asarray(s["trunk_quaternion_wxyz"], dtype=float)
        if q.shape != (len(t), 4) or not np.isfinite(q).all():
            raise ValueError("Quaternion trace shape/values are invalid.")
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("Zero quaternion is invalid.")
        q = q/norms
        angular_speed = 2*np.arccos(np.clip(np.abs(np.sum(q[:-1]*q[1:], axis=1)), 0, 1))/dt
        active |= angular_speed > cfg["joint_activity_rad_s"]
    merged = active.copy()
    for a, b in _runs(active)[1:-1]:
        if not active[a] and (b-a)*dt<cfg["pause_merge_s"]-1e-10:
            merged[a:b] = True
    events = []
    for a, b in _runs(merged):
        duration = float(t[b]-t[a])
        label = None
        max_displacement = float(np.max(np.linalg.norm(xyz[a:b+1, :2]-xyz[a, :2], axis=1)))
        if merged[a]:
            if max_displacement >= cfg["walk_svl"]*svl-1e-10:
                label = "walk_around"
            elif np.any(leg_active[a:b]):
                label = "orientation_change_proxy"
            else:
                label = "unclassified_movement"
        elif duration >= cfg["rest_s"]-1e-10:
            label = "rest_proxy" if has_joint_data else "stationary_unknown_joint_activity"
        if label:
            events.append({"label": label, "start_s": float(t[a]), "end_s": float(t[b]),
                           "duration_s_context_only": duration, "max_displacement_svl": max_displacement/svl,
                           "left_censored": bool(a==0), "right_censored": bool(b==len(active))})
    counts = Counter(e["label"] for e in events)
    elapsed = float(t[-1]-t[0])
    return {"unit": "detected bouts/events, not time fraction", "status": "coarse_automated_proxies_only",
            "counts": dict(counts), "total_detected_events": len(events),
            "event_percentages": {k: 100*v/len(events) for k,v in counts.items()} if events else {},
            "detected_events_per_minute": len(events)/(elapsed/60), "observed_duration_s": elapsed,
            "events": events, "detector_parameters": cfg, "parameter_reference_ids": defaults,
            "unavailable_categories": ["sensory_exploration", "foraging", "interest", "basic_needs", "atypical", "distress", "hide_place_context"],
            "limitations": [
                "One occurrence per detected bout; a continuous long walk is not recounted once per body length.",
                "Movement/bout thresholds are numerical engineering choices; these are not manual animal ethogram annotations.",
                "An orientation proxy needs joint movement, not just a sliding or passively drifting body.",
                "Rest is inactivity, not evidence of sleep; nonzero head/tail/joint motion prevents rest classification.",
                "Endpoint bouts are censored and flagged; their counted occurrence is observed but onset/offset may be outside recording.",
                "No-prey fixed-heading gait tests and short episodes cannot validate published enclosure event proportions or rates.",
                "Unrepresented categories are unavailable, never imputed as zero or used to renormalize published budgets."]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    from realism_metrics import write_json
    result = score_ethogram(json.loads(args.trace.read_text(encoding="utf-8")))
    write_json(args.output, result)
    print(json.dumps({"counts": result["counts"], "status": result["status"]}))


if __name__ == "__main__":
    main()
