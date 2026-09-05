"""Build the opt-in lab-cohort body; NEVER overwrite the legacy walker body.

All parameter changes are registry-backed or derived from existing geometry.
Mass fractions within the tail and non-tail groups remain engineering proxies;
this script deliberately does not fit arbitrary densities to make CoM pass.

python -m utils.build_lab_morphology
python -m common.morphology_audit --xml morphology/gecko_body_lab.xml --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.morphology_audit import DEFAULT_XML, REPO_ROOT, _hindlimb_lengths, collect_morphology_metrics, format_table
from common.provenance import DEFAULT_REGISTRY, load_registry

DEFAULT_OUTPUT = REPO_ROOT / "morphology" / "gecko_body_lab.xml"
V2_OUTPUT = REPO_ROOT / "morphology" / "gecko_body_lab_v2.xml"


def vector(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ")


def numbers(values) -> str:
    return " ".join(f"{float(v):.17g}" for v in values)


def canonical_source_text(source: Path) -> str:
    """Canonical UTF-8 text with LF line endings, independent of Git checkout.

    Only newline encoding is normalized. Content and trailing newlines remain
    significant. The raw byte hash is separately retained in build evidence.
    """
    return source.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def canonical_source_sha256(source: Path) -> str:
    return hashlib.sha256(canonical_source_text(source).encode("utf-8")).hexdigest()


def find(root: ET.Element, tag: str, name: str) -> ET.Element:
    result = root.find(f".//{tag}[@name='{name}']")
    if result is None:
        raise ValueError(f"Missing {tag} {name}")
    return result


def shaped_vector(original: np.ndarray, length: float, drop: float) -> np.ndarray:
    """Change inclination while preserving length and original horizontal bearing."""
    if length <= 0 or not 0 <= drop < length:
        raise ValueError("Invalid segment length/drop")
    horizontal = original[:2] / np.linalg.norm(original[:2])
    return np.r_[horizontal * math.sqrt(length * length - drop * drop), -drop]


def reshape_segment(body: ET.Element, child_name: str, length: float, drop: float) -> None:
    child = body.find(f"body[@name='{child_name}']")
    if child is None:
        raise ValueError(f"Missing child {child_name}")
    new_endpoint = shaped_vector(vector(child.get("pos")), length, drop)
    child.set("pos", numbers(new_endpoint))
    # The segment's visual and collision capsules share joint-center endpoints.
    for geom in body.findall("geom"):
        if geom.get("type") == "capsule" and geom.get("fromto"):
            old = vector(geom.get("fromto"))
            if np.linalg.norm(old[:3]) > 1e-12:
                raise ValueError("Expected proximal capsule endpoint at joint origin")
            geom.set("fromto", numbers(np.r_[np.zeros(3), new_endpoint]))


def scale_foot(body: ET.Element, scale: float) -> None:
    if not 0 < scale <= 1:
        raise ValueError("Only conservative foot shortening is supported")
    for element in body:
        if element.tag not in {"geom", "site"}:
            continue
        for attribute in ("pos", "fromto", "size"):
            if element.get(attribute):
                element.set(attribute, numbers(vector(element.get(attribute)) * scale))


def serialize(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def extreme_masses(coordinates, lower, upper, total: float, maximize=False):
    """Exact bounded linear-program solution for a fixed total mass/moment.

    Fill the lowest (or highest) coordinate first. Used to prove whether the
    requested CoM is in the feasible interval before attempting a fit.
    """
    coordinates, lower, upper = map(lambda x: np.asarray(x, dtype=float), (coordinates, lower, upper))
    if np.any(lower < 0) or np.any(upper < lower):
        raise ValueError("Invalid mass bounds")
    if total < lower.sum() - 1e-12 or total > upper.sum() + 1e-12:
        raise ValueError("Total mass is infeasible under selected bounds")
    result = lower.copy()
    remaining = total - result.sum()
    order = np.argsort(coordinates)
    if maximize:
        order = order[::-1]
    for index in order:
        amount = min(remaining, upper[index] - result[index])
        result[index] += amount
        remaining -= amount
    if abs(result.sum() - total) > 1e-10:
        raise RuntimeError("Bounded mass allocator failed to conserve total")
    return result


def bounded_entropy_fit(prior, coordinates, total, target, lower, upper):
    """Minimum relative-entropy mass calibration with box constraints.

    The convex optimum has m_i = clip(prior_i*exp(a+b*x_i), lower_i, upper_i).
    Nested monotone bisections solve the total-mass and first-moment constraints.
    If the requested centroid is infeasible, return the closest feasible edge
    explicitly; never pretend it matched the requested measurement.
    """
    prior, coordinates, lower, upper = map(lambda x: np.asarray(x, dtype=float),
                                           (prior, coordinates, lower, upper))
    if np.any(prior <= 0) or np.any(lower <= 0):
        raise ValueError("Entropy fit requires strictly positive mass priors/bounds")
    smallest = extreme_masses(coordinates, lower, upper, total)
    largest = extreme_masses(coordinates, lower, upper, total, maximize=True)
    minimum, maximum = float(smallest @ coordinates / total), float(largest @ coordinates / total)
    if target <= minimum:
        return smallest, {"target_feasible": bool(target >= minimum - 1e-10), "feasible_centroid_range": [minimum, maximum], "solution": "minimum feasible centroid"}
    if target >= maximum:
        return largest, {"target_feasible": bool(target <= maximum + 1e-10), "feasible_centroid_range": [minimum, maximum], "solution": "maximum feasible centroid"}

    def at_beta(beta):
        log_lo = np.log(lower / prior) - beta * coordinates
        log_hi = np.log(upper / prior) - beta * coordinates
        alpha_lo, alpha_hi = float(log_lo.min() - 1), float(log_hi.max() + 1)
        for _ in range(80):
            alpha = (alpha_lo + alpha_hi) / 2
            masses = np.clip(prior * np.exp(np.clip(alpha + beta * coordinates, -700, 700)), lower, upper)
            if masses.sum() < total:
                alpha_lo = alpha
            else:
                alpha_hi = alpha
        return np.clip(prior * np.exp(np.clip((alpha_lo + alpha_hi) / 2 + beta * coordinates, -700, 700)), lower, upper)

    beta_lo, beta_hi = -1.0, 1.0
    while float(at_beta(beta_lo) @ coordinates / total) > target:
        beta_lo *= 2
    while float(at_beta(beta_hi) @ coordinates / total) < target:
        beta_hi *= 2
    for _ in range(80):
        beta = (beta_lo + beta_hi) / 2
        masses = at_beta(beta)
        if float(masses @ coordinates / total) < target:
            beta_lo = beta
        else:
            beta_hi = beta
    masses = at_beta((beta_lo + beta_hi) / 2)
    if abs(masses.sum() - total) > 1e-10 or abs(float(masses @ coordinates / total) - target) > 1e-9:
        raise RuntimeError("Inverse mass calibration did not satisfy its constraints")
    return masses, {"target_feasible": True, "feasible_centroid_range": [minimum, maximum], "solution": "minimum relative-entropy mass change"}


def calibrate_com(root: ET.Element, entries: dict) -> dict:
    """Constrained mass-only v2 calibration; never move body/geom/site geometry."""
    val = lambda name: entries[name]["value"]
    model = mujoco.MjModel.from_xml_string(serialize(root))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral").id)
    mujoco.mj_forward(model, data)
    nose, vent = data.site("nose_tip").xpos.copy(), data.site("vent").xpos.copy()
    caudal = vent - nose
    svl = float(np.linalg.norm(caudal))
    axial = (data.xipos - nose) @ caudal / (svl * svl)
    tail_ids = [model.body(f"tail{i}").id for i in range(1, 6)]
    total_mass = float(model.body_mass.sum())
    tail_total = float(model.body_mass[tail_ids].sum())
    non_tail_total = total_mass - tail_total
    fraction = tail_total / total_mass
    target_body = float(val("com_tail_excluded_svl"))
    target_whole = float(val("com_intact_svl"))
    target_tail = (target_whole - (1 - fraction) * target_body) / fraction

    groups = {}
    for body_id in range(1, model.nbody):
        if body_id in tail_ids:
            continue
        name = model.body(body_id).name
        key = name[:-2] if name.endswith(("_L", "_R")) else name
        groups.setdefault(key, []).append(body_id)
    group_names = list(groups)
    prior = np.array([model.body_mass[groups[name]].sum() for name in group_names])
    coordinates = np.array([np.average(axial[groups[name]], weights=model.body_mass[groups[name]]) for name in group_names])
    bound_lo, bound_hi = map(float, val("com_fit_non_tail_mass_bounds"))
    body_masses, body_fit = bounded_entropy_fit(prior, coordinates, non_tail_total, target_body, prior * bound_lo, prior * bound_hi)

    volumes = []
    for body_id in tail_ids:
        collision = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id and model.geom_group[g] == 3]
        if len(collision) != 1 or model.geom_type[collision[0]] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError("Tail volume diagnostic currently requires one capsule per segment")
        radius, half_length = model.geom_size[collision[0], :2]
        volumes.append(float(math.pi * radius**2 * 2 * half_length + 4 * math.pi * radius**3 / 3))
    volumes = np.array(volumes)
    source_bulk_density = float(val("com_study_body_mass_kg")) * float(val("tail_mass_fraction")) / float(val("com_study_tail_volume_m3"))
    tail_prior = model.body_mass[tail_ids].copy()
    tail_coordinates = axial[tail_ids]
    tail_lower = tail_prior * float(val("com_fit_tail_min_mass_fraction"))
    guard = source_bulk_density * float(val("com_fit_tail_density_guard_factor"))
    tail_upper = volumes * guard
    tail_masses, tail_fit = bounded_entropy_fit(tail_prior, tail_coordinates, tail_total, target_tail, tail_lower, tail_upper)

    sensitivity = []
    for factor in val("com_fit_density_sensitivity_factors"):
        upper = volumes * source_bulk_density * factor
        case = {"factor": factor, "effective_density_cap_kg_m3": source_bulk_density * factor,
                "total_mass_capacity_kg": float(upper.sum())}
        try:
            chosen, fit = bounded_entropy_fit(tail_prior, tail_coordinates, tail_total, target_tail, tail_lower, upper)
            tail_center = float(chosen @ tail_coordinates / tail_total)
            case.update({"mass_feasible": True, "exact_tail_com_target_feasible": fit["target_feasible"],
                         "minimum_tail_com_svl": fit["feasible_centroid_range"][0],
                         "achieved_tail_com_svl": tail_center,
                         "whole_com_if_body_target_exact_svl": (1 - fraction) * target_body + fraction * tail_center,
                         "segment_masses_kg": chosen.tolist()})
        except ValueError as error:
            case.update({"mass_feasible": False, "reason": str(error)})
        sensitivity.append(case)

    new_body_masses = model.body_mass.copy()
    for name, old_group, new_group in zip(group_names, prior, body_masses):
        new_body_masses[groups[name]] *= new_group / old_group
    new_body_masses[tail_ids] = tail_masses
    for body_id in range(1, model.nbody):
        body = find(root, "body", model.body(body_id).name)
        ratio = new_body_masses[body_id] / model.body_mass[body_id]
        for geom in body.findall("geom"):
            if float(geom.get("mass", "0")) > 0:
                geom.set("mass", numbers([float(geom.get("mass")) * ratio]))

    required_first_fraction = (tail_coordinates[1] - target_tail) / (tail_coordinates[1] - tail_coordinates[0])
    extra_first_mass_from_distal_minima = float(np.sum(tail_lower[2:] * (tail_coordinates[2:] - tail_coordinates[1]))
                                               / (tail_coordinates[1] - tail_coordinates[0]))
    return {"primary_paper": "https://doi.org/10.1242/jeb.110916",
            "source_protocol": "Frozen rigid animals, thread suspension; original tail reattached with glue. Source7females39.4g/SVL121.9mm.",
            "coordinate_assumption": "Normalized longitudinal distance from snout; neutral pose approximates source rigid specimen pose (not provided in paper).",
            "unknown_degrees_of_freedom": {"non_tail_symmetric_mass_groups": len(groups), "non_tail_mass_and_moment_constraints": 2,
                                           "tail_mass_variables": len(tail_ids), "tail_mass_and_moment_constraints_requested": 2},
            "body_fit": {**body_fit, "requested_com_svl": target_body, "achieved_com_svl": float(body_masses @ coordinates / non_tail_total),
                         "bounds_multiple_prior": [bound_lo, bound_hi], "groups": group_names,
                         "prior_group_masses_kg": prior.tolist(), "fitted_group_masses_kg": body_masses.tolist()},
            "tail_fit": {**tail_fit, "requested_com_svl": target_tail, "achieved_com_svl": float(tail_masses @ tail_coordinates / tail_total),
                         "coordinates_svl": tail_coordinates.tolist(), "fitted_masses_kg": tail_masses.tolist(),
                         "capsule_volumes_m3": volumes.tolist(), "effective_density_kg_m3": (tail_masses / volumes).tolist(),
                         "guard_density_kg_m3": guard, "source_approximate_bulk_density_kg_m3": source_bulk_density},
            "geometry_lower_bound": {"minimum_fraction_tail_mass_in_first_segment_for_exact_mean": float(required_first_fraction),
                                     "minimum_first_segment_mass_kg": float(tail_total * required_first_fraction),
                                     "minimum_first_segment_density_kg_m3": float(tail_total * required_first_fraction / volumes[0]),
                                     "minimum_first_density_with_positive_distal_guards_kg_m3": float((tail_total * required_first_fraction + extra_first_mass_from_distal_minima) / volumes[0]),
                                     "assumption": "All remaining tail mass optimistically at segment2 centroid; any mass farther caudal tightens this bound."},
            "density_guard_sensitivity": sensitivity,
            "mass_table": [{"body": model.body(i).name, "prior_mass_kg": float(model.body_mass[i]), "fitted_mass_kg": float(new_body_masses[i]),
                            "neutral_axial_com_svl": float(axial[i])} for i in range(1, model.nbody)],
            "limitations": ["Fit uses calibration targets, not held-out validation.",
                            "Every per-segment mass remains inferred and non-unique; the entropy prior and bounds are invented engineering assumptions.",
                            "Capsules are collision envelopes, not segmented anatomical tissue volumes; overlap makes the summed volume overestimate occupied solid volume.",
                            "Density-cap feasibility is conditional on the engineering guard, not proof of a universal biological impossibility.",
                            "Whole-animal and tail-excluded mean CoMs plus mean tail fraction ignore between-animal covariance; exact pooled means need not obey this identity.",
                            "No geometry, landmarks, segment lengths, total mass, tail fraction, joint topology or actuator settings are moved by the inverse fit."]}


def make_candidate(source: Path = DEFAULT_XML, registry_path: Path = DEFAULT_REGISTRY, fit_com: bool = False) -> tuple[str, dict]:
    entries = load_registry(registry_path)["entries"]
    val = lambda name: entries[name]["value"]
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(canonical_source_text(source), parser=parser)
    original = mujoco.MjModel.from_xml_path(str(source))
    neutral = mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original, neutral, original.key("neutral").id)
    mujoco.mj_forward(original, neutral)
    svl = float(np.linalg.norm(neutral.site("nose_tip").xpos - neutral.site("vent").xpos))
    old_mass = float(original.body_mass.sum())
    new_mass = float(val("body_mass_kg"))
    tail_fraction = float(val("tail_mass_fraction"))
    root.set("model", "gecko_body_lab_candidate_v2" if fit_com else "gecko_body_lab_candidate_v1")
    root.insert(0, ET.Comment(
        f" LAB COHORT CANDIDATE v{2 if fit_com else 1}, opt-in only. Legacy GeckoBody-R physics remains unchanged.\n"
        " Source body: " + source.name + " canonical-LF SHA256 " + canonical_source_sha256(source) + ".\n"
        " 38 g / ~106 mm SVL population choice; segment masses, neutral inclinations and gains are proxies.\n"
        " Femur depression target convention: Jagnandan & Higham 2017 (doi:10.1038/s41598-017-11484-7).\n"
        " Raw hip_sprawl is a local hinge angle, NOT the paper's reconstructed 3-D anatomical angle.\n"
        " No shoulder long-axis joint added; 32 hinges, nq39/nv38/nu25 preserved.\n"
        " This candidate has NOT passed gait validation or fresh residual training. "
    ))
    tail_root = find(root, "body", "tail1")
    tail_geoms = set(tail_root.iter("geom"))
    geoms = list(root.iter("geom"))
    tail_mass = sum(float(g.get("mass", "0")) for g in tail_geoms)
    non_tail_mass = sum(float(g.get("mass", "0")) for g in geoms if g not in tail_geoms)
    for geom in geoms:
        mass = float(geom.get("mass", "0"))
        if mass > 0:
            scale = (new_mass * tail_fraction / tail_mass if geom in tail_geoms
                     else new_mass * (1 - tail_fraction) / non_tail_mass)
            geom.set("mass", numbers([mass * scale]))
    tail_length = sum(float(np.linalg.norm(original.body(f"tail{i}").pos)) for i in range(2, 6))
    tail_length += float(np.linalg.norm(original.site("tail_tip").pos))
    tail_scale = float(val("tail_length_svl")) * svl / tail_length
    # Preserve tail radius profile; shorten only the axial chain dimensions.
    for index in range(1, 6):
        body = find(root, "body", f"tail{index}")
        if index > 1:
            body.set("pos", numbers(vector(body.get("pos")) * [tail_scale, 1, 1]))
        for geom in body.findall("geom"):
            if geom.get("fromto"):
                geom.set("fromto", numbers(vector(geom.get("fromto")) * [tail_scale, 1, 1, tail_scale, 1, 1]))
            if geom.get("pos"):
                geom.set("pos", numbers(vector(geom.get("pos")) * [tail_scale, 1, 1]))
            if geom.get("type") == "ellipsoid":
                geom.set("size", numbers(vector(geom.get("size")) * [tail_scale, 1, 1]))
    tip = find(root, "site", "tail_tip")
    tip.set("pos", numbers(vector(tip.get("pos")) * [tail_scale, 1, 1]))

    length_scale = svl / float(val("fuller_svl_m"))
    femur_length = float(val("fuller_femur_m")) * length_scale
    tibia_length = float(val("fuller_tibia_m")) * length_scale
    hind_total = float(val("hindlimb_length_svl")) * svl
    for side in ("L", "R"):
        foot_parts = _hindlimb_lengths(original, side)
        old_foot_length = foot_parts["ankle_to_fourth_digit_m"] + foot_parts["fourth_digit_m"]
        hind_foot_scale = (hind_total - femur_length - tibia_length) / old_foot_length
        reshape_segment(find(root, "body", f"femur_{side}"), f"tibia_{side}", femur_length, float(val("lab_neutral_femur_drop_m")))
        reshape_segment(find(root, "body", f"tibia_{side}"), f"pes_{side}", tibia_length, float(val("lab_neutral_tibia_drop_m")))
        scale_foot(find(root, "body", f"pes_{side}"), hind_foot_scale)

        # Forelimb proportions are deliberately labelled engineering choices.
        humerus = find(root, "body", f"humerus_{side}")
        forearm = find(root, "body", f"forearm_{side}")
        manus = find(root, "body", f"manus_{side}")
        upper = float(np.linalg.norm(original.body(f"forearm_{side}").pos))
        lower = float(np.linalg.norm(original.body(f"manus_{side}").pos))
        digits = [g for g in manus.findall("geom") if g.get("class") == "visual" and g.get("type") == "capsule"]
        digit = vector(digits[3].get("fromto"))
        fore_foot = float(np.linalg.norm(digit[:3]) + np.linalg.norm(digit[3:] - digit[:3]))
        fore_scale = hind_total * float(val("lab_forelimb_fraction_hind")) / (upper + lower + fore_foot)
        reshape_segment(humerus, f"forearm_{side}", upper * fore_scale, float(val("lab_neutral_humerus_drop_m")))
        reshape_segment(forearm, f"manus_{side}", lower * fore_scale, float(val("lab_neutral_forearm_drop_m")))
        scale_foot(manus, fore_scale)
        pos = vector(humerus.get("pos"))
        pos[2] = float(val("lab_shoulder_attachment_z_m"))
        humerus.set("pos", numbers(pos))

        for name, width in ((f"hip_proret_{side}", float(val("hip_proret_min_range_deg"))),
                            (f"hip_rot_{side}", float(val("lab_hip_rot_range_deg")))):
            # Rotation 30 deg total is the brief's +/-15 deg engineering choice.
            find(root, "joint", name).set("range", numbers([-width / 2, width / 2]))
            find(root, "position", name).set("ctrlrange", numbers(np.deg2rad([-width / 2, width / 2])))
    pectoral = find(root, "site", "pectoral_center")
    position = vector(pectoral.get("pos"))
    position[2] = float(val("lab_shoulder_attachment_z_m"))
    pectoral.set("pos", numbers(position))

    head = find(root, "body", "head")
    half_width = float(val("head_width_svl")) * svl / 2
    half_height = float(val("lab_head_height_svl")) * svl / 2
    main_visual = next(g for g in head.findall("geom") if g.get("class") == "visual" and g.get("material") == "skin")
    main_size = vector(main_visual.get("size"))
    main_size[1:] = [half_width, half_height]
    main_visual.set("size", numbers(main_size))
    collision = next(g for g in head.findall("geom") if g.get("class") == "collision")
    collision.set("type", "ellipsoid")
    collision.attrib.pop("fromto")
    collision.set("pos", main_visual.get("pos"))
    collision.set("size", main_visual.get("size"))
    for geom in head.findall("geom"):
        if geom is not main_visual and geom.get("material") == "skin":
            size = vector(geom.get("size"))
            size[2] *= float(val("lab_snout_height_scale"))
            geom.set("size", numbers(size))
        if geom.get("material") == "eye":
            pos = vector(geom.get("pos"))
            radius = float(geom.get("size"))
            pos[1] = np.sign(pos[1]) * (float(val("interorbital_distance_m")) / 2 + radius)
            geom.set("pos", numbers(pos))

    actuator_scale = new_mass / old_mass
    for position in root.iter("position"):
        for attribute in ("kp", "kv", "forcerange"):
            if position.get(attribute):
                position.set(attribute, numbers(vector(position.get(attribute)) * actuator_scale))
    inverse_fit = calibrate_com(root, entries) if fit_com else None
    if fit_com:
        root.insert(0, ET.Comment(
            " V2 MASS-ONLY INVERSE CALIBRATION. Minimum relative-entropy body fit; density-guarded tail fit.\n"
            " Per-segment masses are inferred engineering proxies, NOT measured animal masses.\n"
            " Exact paired CoM means are geometrically incompatible with the selected tail-density guard.\n"
            " Geometry and landmarks are identical to candidate v1; see generated evidence for feasibility/sensitivity. "
        ))
    # Neutral is an initial release pose; stand is regenerated from physics.
    candidate = mujoco.MjModel.from_xml_string(serialize(root))
    data = mujoco.MjData(candidate)
    mujoco.mj_resetDataKeyframe(candidate, data, candidate.key("neutral").id)
    data.ctrl[:] = 0
    for _ in range(math.ceil(float(val("morphology_settle_s")) / candidate.opt.timestep)):
        mujoco.mj_step(candidate, data)
    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("Candidate became nonfinite; refusing to generate stand keyframe")
    find(root, "key", "stand").set("qpos", " ".join(format(float(x), ".17g") for x in data.qpos))
    result = serialize(root)
    checked = mujoco.MjModel.from_xml_string(result)
    if (checked.nq, checked.nv, checked.nu) != (original.nq, original.nv, original.nu):
        raise RuntimeError("Candidate unexpectedly changed checkpoint dimensions")
    return result, {"source_canonical_lf_sha256": canonical_source_sha256(source),
                    "source_raw_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "registry_raw_file_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                    "tail_length_scale": tail_scale, "hind_foot_scale": hind_foot_scale,
                    "fore_chain_scale": fore_scale, "actuator_scale": actuator_scale,
                    "mass_partition": ("bounded relative-entropy non-tail fit + closest density-guarded tail fit"
                                       if fit_com else val("lab_mass_partition_method")),
                    "inverse_mass_calibration": inverse_fit,
                    "warning": ("CoM acceptance bands can pass although the paired mean targets are infeasible under the selected guard. Calibration is not held-out validation."
                                if fit_com else "Opt-in untrained candidate; CoM gates deliberately not forced by arbitrary mass redistribution.")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--stdout", action="store_true", help="Emit generated XML instead of writing")
    parser.add_argument("--report", type=Path, help="Write candidate measurements and derivations")
    parser.add_argument("--fit-com", action="store_true", help="Build separate v2 with bounded mass-only inverse calibration")
    args = parser.parse_args(argv)
    if args.fit_com and args.output == DEFAULT_OUTPUT:
        args.output = V2_OUTPUT
    if args.output.resolve() == DEFAULT_XML.resolve() or args.output.resolve() == args.source.resolve():
        raise ValueError("Refusing to overwrite source/legacy XML")
    xml, derivations = make_candidate(args.source.resolve(), args.registry.resolve(), args.fit_com)
    if args.stdout:
        print(xml, end="")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml, encoding="utf-8")
    report = collect_morphology_metrics(args.output, registry_path=args.registry)
    report["candidate_derivations"] = derivations
    print(format_table(report))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
