"""Offline, force-based base-controller diagnosis. No simulator or learner.

Every value is a property of the supplied simulation trace, not a biological
waveform. Old records require explicit reference provenance; this module never
imports the current controller to guess the controller that made an old trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FEET = ("HL", "FL", "HR", "FR")
REPO = Path(__file__).resolve().parents[1]
TOUCH_SOURCE = "https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-touch"
SESSION1_LEGACY_REFERENCE = {
    "frequency_hz": 1.1888,
    "phase_offsets_cycle": {"HL": 0., "FL": .25, "HR": .5, "FR": .75},
    "phase_convention": "additive",
    "stance_by_foot": {"HL": .62, "FL": .68, "HR": .62, "FR": .70},
    "source": "Explicit historical Session1 legacy controller constants documented in CODEX_PROMPT_2.md and docs/research/beyond_statistics_realism.md section0.3; not inferred from current controller code",
    "allow_control_grid_time_inference": True,
}


def _times(values):
    t = np.asarray(values, dtype=float)
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all():
        raise ValueError("Need at least three finite sample times.")
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-9):
        raise ValueError("Trace timestamps must be strictly increasing and uniform; no upsampling is performed.")
    return t, dt


def _matrix(values, n, width, allow_null_rows=False):
    if allow_null_rows:
        values = [[None]*width if row is None else row for row in values]
    result = np.asarray(values, dtype=float)
    if result.shape != (n, width):
        raise ValueError(f"Expected array shape {(n, width)}, received {result.shape}.")
    return result


def _debounce(signal, dt, minimum_s):
    """Session1 convention: shortest bounded run first, earliest on ties.

    Both contact and flight runs shorter than minimum_s are replaced; exactly
    minimum_s remains. Censored edge runs are unchanged. This removes only
    declared short events, never enough events to fit the commanded frequency.
    """
    x = np.asarray(signal, dtype=bool).copy()
    while len(x):
        starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1])+1]
        ends = np.r_[starts[1:], len(x)]
        candidates = [(int(b-a), int(a), int(b)) for a, b in zip(starts[1:-1], ends[1:-1])
                      if (b-a)*dt < minimum_s-1e-10]
        if not candidates:
            return x
        _, a, b = min(candidates)
        x[a:b] = x[a-1]
    return x


def _circular(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean_cycle": None, "resultant_length": None}
    z = np.mean(np.exp(2j*np.pi*values))
    return {"n": len(values), "mean_cycle": float(np.angle(z)/(2*np.pi) % 1) if abs(z) > 1e-12 else None,
            "resultant_length": float(abs(z))}


def resolve_commands(trace, command_reference=None):
    """Prefer actually recorded held commands; never use current controller data.

    command_reference is an explicit historical dict with frequency_hz,
    phase_offsets_cycle, phase_convention ('additive' or 'touchdown_delay'),
    stance_by_foot and source. Inferring old control timestamps additionally
    requires allow_control_grid_time_inference=True and a matching 50Hz-style
    endpoint grid. It cannot turn low-rate evidence into high-rate evidence.
    """
    meta, s = trace["metadata"], trace["samples"]
    t, dt = _times(s["time_s"])
    n = len(t)
    ref = dict(command_reference or {})
    frequency = meta.get("commanded_frequency_hz", meta.get("frequency_hz", ref.get("frequency_hz")))
    frequency_source = ("metadata.commanded_frequency_hz" if meta.get("commanded_frequency_hz") is not None else
                        "metadata.frequency_hz (historical alias)" if meta.get("frequency_hz") is not None else
                        "explicit historical reference" if frequency is not None else "unavailable")
    if frequency is not None and (not np.isfinite(float(frequency)) or float(frequency) <= 0):
        raise ValueError("Commanded frequency must be finite and positive.")
    if frequency is not None:
        frequency = float(frequency)
    if ref.get("frequency_hz") is not None and frequency is not None and not np.isclose(ref["frequency_hz"], frequency, rtol=0, atol=1e-12):
        raise ValueError("Explicit reference contradicts recorded commanded frequency.")
    command_t = np.full(n, np.nan)
    time_source = "unavailable"
    if "command_time_s" in s:
        command_t = np.asarray(s["command_time_s"], dtype=float)
        time_source = "samples.command_time_s: recorded executed held interval start"
    elif meta.get("gait_profile") == "lab" and "gait_target_time_s" in s:
        command_t = np.asarray(s["gait_target_time_s"], dtype=float)
        time_source = "samples.gait_target_time_s: lab Session1 recorded executed held interval start"
    elif ref.get("allow_control_grid_time_inference"):
        control_dt = float(meta.get("control_dt_s", np.nan))
        if not np.isclose(dt, control_dt, rtol=1e-6, atol=1e-9) or abs(t[0]) > 1e-9:
            raise ValueError("Old clock inference requires one endpoint sample per control interval, starting at reset t=0.")
        # Reproduce the old repeated float64 `_cpg_t += control_dt`, including
        # rounding; do not label an end-of-step timestamp as its start command.
        clock = 0.
        for i in range(1, n):
            command_t[i] = clock
            clock += control_dt
        time_source = "INFERRED from explicit historical reference and repeated control_dt clock; sample i is endpoint of command interval i-1; reset sample excluded"
    if command_t.shape != (n,):
        raise ValueError("Expected command_time_s[N].")

    phase = np.full((n, 4), np.nan)
    commands = np.full((n, 4), np.nan)
    phase_source = "unavailable"
    stance_source = "unavailable"
    stance = meta.get("commanded_stance_by_foot", ref.get("stance_by_foot"))
    if stance is not None and (set(stance) != set(FEET) or any(not np.isfinite(stance[f]) or not 0 < stance[f] < 1 for f in FEET)):
        raise ValueError("Commanded stance fractions require all four feet with finite values in (0,1).")
    offsets = meta.get("phase_offsets_cycle", ref.get("phase_offsets_cycle"))
    convention = ref.get("phase_convention")
    if meta.get("gait_profile") == "lab" and offsets is not None:
        convention = "touchdown_delay"
    elif meta.get("gait_profile") == "legacy" and offsets is not None:
        convention = "additive"
    if "commanded_phase_fraction" in s:
        phase = _matrix(s["commanded_phase_fraction"], n, 4, True)
        phase_source = "samples.commanded_phase_fraction: recorded phase used for held control"
        if np.any((phase[np.isfinite(phase)] < 0) | (phase[np.isfinite(phase)] >= 1)):
            raise ValueError("Recorded local phases must be in [0,1).")
    elif offsets is not None and frequency is not None and convention in ("additive", "touchdown_delay"):
        if set(offsets) != set(FEET) or not np.isfinite(list(offsets.values())).all():
            raise ValueError("Reference phase offsets require all four feet.")
        sign = 1 if convention == "additive" else -1
        phase = (command_t[:, None]*frequency + sign*np.asarray([offsets[f] for f in FEET])) % 1
        phase_source = "reconstructed from held command time and "+convention+" reference; not an observed foot trajectory"
        if convention == "touchdown_delay":
            phase[np.minimum(phase, 1-phase) < 1e-12] = 0.
            if stance:
                for j, foot in enumerate(FEET):
                    phase[np.abs(phase[:, j]-stance[foot]) < 1e-12, j] = stance[foot]
    if "commanded_contacts" in s:
        commands = _matrix(s["commanded_contacts"], n, 4, True)
        stance_source = "samples.commanded_contacts: recorded executed held command"
    elif meta.get("gait_profile") == "lab" and "gait_target_contacts" in s:
        commands = _matrix(s["gait_target_contacts"], n, 4, True)
        stance_source = "samples.gait_target_contacts: lab Session1 executed command; legacy reward targets are deliberately NOT used"
    elif stance is not None:
        commands = np.where(np.isfinite(phase), phase < np.asarray([stance[f] for f in FEET]), np.nan)
        stance_source = "reconstructed from explicit phase and stance references"
    finite_commands = commands[np.isfinite(commands)]
    if not np.isin(finite_commands, [0., 1.]).all():
        raise ValueError("Commanded contacts must be binary or missing.")
    comparable = np.isfinite(phase) & np.isfinite(commands)
    mismatch = (int(np.sum(((phase < np.asarray([stance[f] for f in FEET])) != commands) & comparable))
                if stance is not None else None)
    return {"time_s": command_t, "phase": phase, "contacts": commands, "frequency_hz": frequency,
            "provenance": {"frequency_hz": frequency, "frequency_source": frequency_source,
                           "command_time_source": time_source, "phase_source": phase_source,
                           "stance_source": stance_source, "historical_reference": ref or None,
                           "commanded_stance_by_foot": stance,
                           "recorded_phase_vs_contact_reference_mismatch_count": mismatch,
                           "note": "Desired stance is not observed support. Legacy reward target phase/clock is not treated as controller command."}}


def _cycle_metrics(t, loaded, phase, mask, settle_s, frequency, tolerance, cv_max):
    td = np.flatnonzero(loaded[1:] & ~loaded[:-1])+1
    cycles = []
    for a, b in zip(td[:-1], td[1:]):
        if t[a] < settle_s-1e-9:
            continue
        off = np.flatnonzero(loaded[a:b] & ~loaded[a+1:b+1])+a+1
        if len(off) == 1:
            period = float(t[b]-t[a])
            cycles.append({"start_index": int(a), "end_index": int(b), "liftoff_index": int(off[0]),
                           "period_s": period, "duty_factor": float((t[off[0]]-t[a])/period)})
    periods = [c["period_s"] for c in cycles]
    rate = len(periods)/sum(periods) if periods else None
    cv = float(np.std(periods, ddof=1)/np.mean(periods)) if len(periods) > 1 else None
    error = (rate/frequency-1) if rate is not None and frequency is not None else None
    rate_pass = abs(error) <= tolerance if error is not None else None
    cv_pass = cv < cv_max if cv is not None else None
    passes = bool(rate_pass and cv_pass) if rate_pass is not None and cv_pass is not None else None
    td_keep = td[mask[td]]
    return {"loaded_fraction_including_censored": float(np.mean(loaded[mask])),
            "touchdown_count": int(len(td_keep)), "complete_contact_cycle_count": len(cycles),
            "observed_contact_cycle_rate_hz": rate, "rate_ratio_to_command": rate/frequency if error is not None else None,
            "period_cv": cv, "cycle_duty_mean": float(np.mean([c["duty_factor"] for c in cycles])) if cycles else None,
            "cycle_duty_status": "observed contact-cycle statistic, not validated gait duty",
            "touchdown_command_local_phase": _circular(phase[td_keep]),
            "entrainment": {"rate_within_tolerance": rate_pass, "period_cv_below_limit": cv_pass,
                            "passes_engineering_check": passes,
                            "status": "measured diagnostic" if passes is not None else "unmeasurable: insufficient complete cycles or missing commanded frequency",
                            "rate_tolerance_fraction": tolerance, "period_cv_max_exclusive": cv_max},
            "cycles": cycles}, td


def _condition(forces, raw, debounced, selection):
    n = int(np.sum(selection))
    return {"sample_count": n, "raw_loaded_fraction": float(np.mean(raw[selection])) if n else None,
            "debounced_loaded_fraction": float(np.mean(debounced[selection])) if n else None,
            "mean_scalar_touch_force_N": float(np.mean(forces[selection])) if n else None,
            "mean_scalar_touch_force_when_raw_loaded_N": float(np.mean(forces[selection & raw])) if np.any(selection & raw) else None}


def diagnose_trace(trace, *, settle_s=3., debounce_s=.04, phase_bins=None,
                   command_reference=None, entrainment_tolerance_fraction=None, period_cv_max=None):
    """Pure trace analysis suitable for original50Hz or actually logged high rates.

    Input foot arrays must follow metadata.foot_order == HL,FL,HR,FR. Missing
    command records produce null conditional values, never guesses or zeroes.
    Registry values are only detector configuration, not historical controller
    references. Summary dispersion is temporal/within-trace, never n-animal SEM.
    """
    from common.provenance import parameter_value
    phase_bins = int(parameter_value("diagnostic_phase_bins")) if phase_bins is None else phase_bins
    tolerance = float(parameter_value("entrainment_tolerance_fraction")) if entrainment_tolerance_fraction is None else float(entrainment_tolerance_fraction)
    cv_max = float(parameter_value("contact_period_cv_ceiling")) if period_cv_max is None else float(period_cv_max)
    if not isinstance(phase_bins, int) or phase_bins < 1 or not np.isfinite([settle_s, debounce_s, tolerance, cv_max]).all() or min(debounce_s, tolerance, cv_max) < 0:
        raise ValueError("Invalid detector configuration.")
    meta, s = trace["metadata"], trace["samples"]
    if tuple(meta.get("foot_order", ())) != FEET:
        raise ValueError("Explicit foot_order must be [HL,FL,HR,FR]; do not silently swap limbs.")
    t, dt = _times(s["time_s"])
    n = len(t)
    if settle_s < t[0] or settle_s >= t[-1]:
        raise ValueError("Settling cutoff must leave an observation interval.")
    threshold = float(meta["contact_threshold_N"])
    forces = _matrix(s["foot_force_N"], n, 4)
    xyz = _matrix(s["trunk_position_m"], n, 3)
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(forces).all() or np.any(forces < 0) or not np.isfinite(xyz).all():
        raise ValueError("Need nonnegative finite scalar forces and finite positions/threshold.")
    command = resolve_commands(trace, command_reference)
    raw = forces > threshold
    debounced = np.column_stack([_debounce(raw[:, j], dt, debounce_s) for j in range(4)])
    start = int(np.flatnonzero(t >= settle_s-1e-9)[0])
    mask = t > t[start]+1e-9
    elapsed = float(t[-1]-t[start])
    path = float(np.sum(np.linalg.norm(np.diff(xyz[start:, :2], axis=0), axis=1)))
    displacement = xyz[-1, :2]-xyz[start, :2]
    net = float(np.linalg.norm(displacement))
    result = {"schema_version": 1, "claim": "Simulation diagnosis only; raw contacts are not verified biological strides.",
              "sampling": {"sample_rate_hz": 1/dt, "sample_dt_s": dt, "control_dt_s": meta.get("control_dt_s"),
                           "requested_settle_s": settle_s, "effective_window_start_s": float(t[start]),
                           "end_s": float(t[-1]), "observed_duration_s": elapsed, "endpoint_sample_count": int(np.sum(mask)),
                           "window_convention": "Positions include start; force/velocity endpoints use (start,end], excluding reset and the interval ending exactly at start.",
                           "acquisition_status": "historical low-rate evidence; no interpolation; sub-sample contacts unresolved" if 1/dt < 200-1e-6 else "high-rate trace supplied; acquisition authenticity depends on recorder provenance",
                           "command_phase_bins": phase_bins},
              "identity": {key: meta.get(key) for key in ("xml_sha256", "model_sha256", "normalizer_sha256", "gait_profile", "controller", "seed", "episode", "reset_noise")},
              "repeat_unit": "One deterministic trajectory; repeated identical seeds/runs are not independent samples.",
              "commands": command["provenance"],
              "force_semantics": {"threshold_N": threshold,
                                  "definition": "MuJoCo touch sensor scalar sum of included contact-normal forces; NOT world-vertical force, vector resultant or tissue receptor sensitivity.",
                                  "source": TOUCH_SOURCE, "toe_height_used_for_contact": False},
              "motion": {"path_length_m": path, "net_displacement_m": net, "net_displacement_xy_m": displacement.tolist(),
                         "net_path_ratio": net/path if path > 1e-12 else None, "path_speed_m_s": path/elapsed},
              "debounce": {"minimum_contact_and_flight_s": debounce_s, "method": "shortest bounded run first; ties earliest; censored edges retained; no forcing to one event per commanded cycle"},
              "feet": {}, "observed_limb_phase": {},
              "limitations": ["Endpoint load fraction is a sampled-time estimate, not a continuous contact integral.",
                              "Entrainment is an engineered controller-tracking diagnostic, not biological validation.",
                              "Contact-cycle duty and limb phase remain unvalidated when entrainment fails; all raw events remain counted.",
                              "Phase-binned touch means describe this simulator, not measured muscle or animal force waveforms."]}
    if "forward_speed_m_s" in s:
        velocity = np.asarray(s["forward_speed_m_s"], dtype=float)
        if velocity.shape != (n,) or not np.isfinite(velocity[mask]).all():
            raise ValueError("Expected finite forward_speed_m_s[N] over analysis window.")
        result["motion"]["signed_body_forward_speed_m_s"] = {
            "mean": float(np.mean(velocity[mask])), "temporal_sd": float(np.std(velocity[mask])),
            "min": float(np.min(velocity[mask])), "max": float(np.max(velocity[mask])),
            "source": "recorded trunk-local signed forward velocity, not path speed or fixed-world-axis progress"}
    else:
        result["motion"]["signed_body_forward_speed_m_s"] = {"status": "unavailable; no recorded signed body-forward velocity"}
    touchdown = {kind: {} for kind in ("raw", "debounced")}
    for j, foot in enumerate(FEET):
        phase = command["phase"][:, j]
        cmd = command["contacts"][:, j]
        known = np.isfinite(cmd) & mask
        item = {"command_sample_coverage": float(np.mean(np.isfinite(cmd[mask]))),
                "commanded_stance_sample_fraction": float(np.mean(cmd[known])) if np.any(known) else None,
                "commanded_stance": _condition(forces[:, j], raw[:, j], debounced[:, j], known & (cmd == 1)),
                "commanded_swing": _condition(forces[:, j], raw[:, j], debounced[:, j], known & (cmd == 0)),
                "debounce_changed_sample_count": int(np.sum((raw[:, j] != debounced[:, j]) & mask)), "phase_bins": []}
        for kind, contact in (("raw", raw[:, j]), ("debounced", debounced[:, j])):
            item[kind], touchdown[kind][foot] = _cycle_metrics(t, contact, phase, mask, t[start], command["frequency_hz"], tolerance, cv_max)
        for b in range(phase_bins):
            selection = mask & np.isfinite(phase) & (phase >= b/phase_bins) & (phase < (b+1)/phase_bins)
            item["phase_bins"].append({"local_phase_start": b/phase_bins, "local_phase_end_exclusive": (b+1)/phase_bins,
                                       **_condition(forces[:, j], raw[:, j], debounced[:, j], selection)})
        result["feet"][foot] = item
    for kind in ("raw", "debounced"):
        result["observed_limb_phase"][kind] = {}
        for hind, fore in (("HL", "FL"), ("HR", "FR")):
            values, ambiguous = [], 0
            for cycle in result["feet"][hind][kind]["cycles"]:
                a, b = cycle["start_index"], cycle["end_index"]
                events = touchdown[kind][fore]
                inside = events[(events >= a) & (events < b)]
                if len(inside) == 1:
                    values.append((t[inside[0]]-t[a])/cycle["period_s"])
                else:
                    ambiguous += 1
            valid = all(result["feet"][foot][kind]["entrainment"]["passes_engineering_check"] is True for foot in (hind, fore))
            result["observed_limb_phase"][kind][hind+"_to_"+fore] = {
                **_circular(values), "ambiguous_reference_cycles_excluded": ambiguous,
                "entrainment_supported": valid,
                "status": "observed contact-cycle phase; not biological validation" if valid else "UNVALIDATED: one or both limbs fail/miss entrainment evidence; conditional contact-cycle statistic only",
                "definition": "First fore touchdown delay within same-side hind touchdown cycle; included only if exactly one fore touchdown occurs. Missing/multiple events remain explicitly excluded, not merged."}
    return result


def diagnose_file(path, **kwargs):
    path = Path(path)
    raw = path.read_bytes()
    result = diagnose_trace(json.loads(raw), **kwargs)
    result["original_trace"] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return result


def markdown_report(report):
    lines = ["# Session 2 — base diagnosis before controller edits", "", report["claim"], "",
             "Loads below are scalar MuJoCo touch readings, not world-vertical forces. No toe/site-height contact test is used. "
             f"[MuJoCo sensor definition]({TOUCH_SOURCE}).", "",
             "All trajectories are n = 1 deterministic runs. Dispersion within a trace is not variation between independent runs or animals.", ""]
    for name, r in report["runs"].items():
        m = r["motion"]
        lines.extend(["## "+name, "", f"Original trace SHA256: `{r['original_trace']['sha256']}`.", "",
                      f"XML SHA256: `{r['identity']['xml_sha256']}`; profile: `{r['identity']['gait_profile'] or 'historical legacy reference (trace omitted profile)'}`.", "",
                      f"Window: {r['sampling']['effective_window_start_s']:.3f}–{r['sampling']['end_s']:.3f} s; {r['sampling']['sample_rate_hz']:.1f} Hz; {r['sampling']['endpoint_sample_count']} post-settle endpoints.", "",
                      "Command timing: "+r["commands"]["command_time_source"]+".", "",
                      f"Path {m['path_length_m']:.6f} m; net {m['net_displacement_m']:.6f} m; net/path {m['net_path_ratio']:.4f}; path speed {m['path_speed_m_s']:.6f} m/s; signed body-forward mean {m['signed_body_forward_speed_m_s'].get('mean', float('nan')):.6f} m/s.", "",
                      f"Loaded means touch > {r['force_semantics']['threshold_N']:.9f} N. Raw / debounced fractions are shown separately; debounce removes bounded contact AND flight intervals shorter than {r['debounce']['minimum_contact_and_flight_s']*1000:.0f} ms.", "",
                      "| Foot | Loaded during commanded stance, raw / debounce | Loaded during commanded swing, raw / debounce | Mean force stance / swing (mN) | Debounced overall contact fraction | Contact cycles/s | Period CV | Entrainment check |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"])
        for foot in FEET:
            f = r["feet"][foot]
            stance, swing, cycle = f["commanded_stance"], f["commanded_swing"], f["debounced"]
            def fmt(value, factor=1):
                return "missing" if value is None else f"{value*factor:.3f}"
            lines.append(f"| {foot} | {fmt(stance['raw_loaded_fraction'])} / {fmt(stance['debounced_loaded_fraction'])} | {fmt(swing['raw_loaded_fraction'])} / {fmt(swing['debounced_loaded_fraction'])} | {fmt(stance['mean_scalar_touch_force_N'],1000)} / {fmt(swing['mean_scalar_touch_force_N'],1000)} | {fmt(cycle['loaded_fraction_including_censored'])} | {fmt(cycle['observed_contact_cycle_rate_hz'])} | {fmt(cycle['period_cv'],100)}% | {cycle['entrainment']['passes_engineering_check']} |")
        lines.extend(["", "The JSON includes every local-phase bin, counts, raw/debounced cycles, command provenance, and conditional limb-phase statistics. Contact-cycle values are not promoted to verified biological strides; a failed entrainment check remains a failure.", ""])
    return "\n".join(lines)+"\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", action="store_true", help="Diagnose the two immutable Session1 zero-residual traces with explicit historical legacy reference.")
    p.add_argument("--trace", type=Path, action="append")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path)
    p.add_argument("--settle", type=float, default=3.)
    args = p.parse_args(argv)
    if bool(args.before) == bool(args.trace):
        p.error("Choose --before or one or more --trace, not both.")
    paths = ([REPO/"artifacts/evidence"/name/"traces/episode_000.json" for name in ("legacy_zero_residual", "lab_zero_residual")]
             if args.before else args.trace)
    report = {"schema_version": 1, "claim": "Before-change diagnosis from original saved traces; no new simulation, fitting or training, and no biological validation claimed.", "runs": {}}
    for path in paths:
        name = path.parent.parent.name
        if name in report["runs"]:
            p.error("Trace parent labels must be unique.")
        reference = SESSION1_LEGACY_REFERENCE if args.before and name == "legacy_zero_residual" else None
        report["runs"][name] = diagnose_file(path, settle_s=args.settle, command_reference=reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"Diagnosed {len(paths)} original trace(s); no simulator or training was run.")


if __name__ == "__main__":
    main()
