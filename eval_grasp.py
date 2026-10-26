#!/usr/bin/env python3
"""Check GraspNet candidates using point-cloud antipodal friction cones."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from grasp_dataset.pc_geometry import load_registered


def load_cloud(npz_file: Path, scale: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Returns
    -------
    - points: shape [N, 3], in meters
    - normals: shape [N, 3]. Surface normal vectors. normalized to unit length
    - affordance: shape [N], or None. Human grasp affordance score [0, 1]
    """
    # load points, normals, affordance score from npz file
    with np.load(npz_file, allow_pickle=False) as archive:
        if "points" not in archive or "normals" not in archive:
            raise ValueError(f"{npz_file} must contain points and normals")
        source_points = np.asarray(archive["points"], dtype=np.float64)
        source_normals = np.asarray(archive["normals"], dtype=np.float64)
        affordance = np.asarray(archive["affordance_score"], dtype=np.float64).reshape(-1) \
            if "affordance_score" in archive else None
    if source_points.ndim != 2 or source_points.shape[1:] != (3,) \
            or source_normals.shape != source_points.shape:
        raise ValueError("points and normals must have shape [N,3]")
    if not np.isfinite(source_points).all() or not np.isfinite(source_normals).all():
        raise ValueError("points or normals contain non-finite values")
    if affordance is not None and len(affordance) != len(source_points):
        raise ValueError("affordance_score length does not match points")

    # scaling loaded point cloud
    method = scale.get("method")
    if method == "similarity_transform": # no scale in size only rotate and translate
        transform = Path(scale["file"])
        if not transform.is_file():
            raise FileNotFoundError(f"Saved transform is unavailable: {transform}; use --npz only with its original transform")
        points, normals = load_registered(npz_file, transform)
    elif method in ("assumed_longest_extent", "explicit_longest_extent"): # scaling along the longest dimension 
        center = np.asarray(scale["source_center"], dtype=np.float64)
        factor = float(scale["scale_to_m"])
        points = (source_points - center) * factor
        normals = source_normals
    elif method == "source_metric_coordinates": # no scale at all, use source metric
        points, normals = source_points, source_normals
    else:
        raise ValueError(f"Unsupported or missing metric_scale method: {method!r}")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("normals contain zero-length vectors")
    return np.asarray(points), normals / lengths[:, None], affordance


def resolve_npz(payload: dict, override: Path | None) -> Path:
    """Magic resolver helps you find npz file with just object id.
    
    Expect
    ```
    object_id/
        |--affordance.npz
    ```
    """
    if override:
        path = override.resolve()
    else:
        path = Path(payload["source"]["file"])
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud not found at {path}; supply --npz")
    return path


def pair_metrics(p, q, n, m, friction, tolerance_deg, unoriented):
    direction = q - p
    distance = float(np.linalg.norm(direction))
    if distance < 1e-10:
        return None
    direction /= distance
    if unoriented:
        if n @ direction > 0:
            n = -n
        if m @ direction < 0:
            m = -m
    cone_half_angle = min(np.arctan(friction), np.deg2rad(tolerance_deg))
    cone_threshold = float(np.cos(cone_half_angle))
    opposition_threshold = float(np.cos(np.deg2rad(tolerance_deg)))
    left_alignment = float(-n @ direction)
    right_alignment = float(m @ direction)
    opposition = float(-n @ m)
    score = min(left_alignment, right_alignment)
    valid = score >= cone_threshold and opposition >= opposition_threshold
    return {
        "valid": bool(valid), "antipodal_score": score,
        "left_alignment": left_alignment, "right_alignment": right_alignment,
        "normal_opposition": opposition, "contact_distance_m": distance,
        "left_normal": n, "right_normal": m,
        "cone_threshold": cone_threshold, "opposition_threshold": opposition_threshold,
    }


def collision_counts(local: np.ndarray, width: float, height: float,
                     depth: float, args) -> dict[str, int]:
    """Count full-cloud samples inside gripper solids and its approach sweep.
    
    Args
    ----
    `local` is object points in gripper frame.
    - Local X: finger/approach direction
    - Local Y: gripper closing direction
    - Local Z: gripper height direction
    - [0, 0, 0]: GraspNet grasp origin
    """
    x, y, z = local.T
    half_width = width / 2
    half_height = max(0.0, height / 2 - args.collision_clearance)
    inside_height = np.abs(z) < half_height
    finger_back = depth - args.finger_length
    finger_x = (x > finger_back + args.collision_clearance) \
        & (x < depth - args.collision_clearance)
    left = inside_height & finger_x \
        & (y > -half_width - args.finger_thickness + args.collision_clearance) \
        & (y < -half_width - args.collision_clearance)
    right = inside_height & finger_x \
        & (y > half_width + args.collision_clearance) \
        & (y < half_width + args.finger_thickness - args.collision_clearance)
    palm = inside_height \
        & (x > finger_back - args.palm_thickness + args.collision_clearance) \
        & (x < finger_back - args.collision_clearance) \
        & (np.abs(y) < half_width + args.finger_thickness - args.collision_clearance)
    approach = inside_height \
        & (x > finger_back - args.palm_thickness - args.approach_distance) \
        & (x <= finger_back - args.palm_thickness + args.collision_clearance) \
        & (np.abs(y) < half_width + args.finger_thickness - args.collision_clearance)
    return {
        "left_finger": int(left.sum()), "right_finger": int(right.sum()),
        "palm": int(palm.sum()), "approach": int(approach.sum()),
    }


def evaluate_candidate(candidates: np.ndarray, points: np.ndarray, normals: np.ndarray, args) -> dict:
    score, width, height, depth = map(float, candidates[:4])
    rotation = candidates[4:13].reshape(3, 3)
    translation = candidates[13:16]
    if not np.isfinite(candidates).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        return {"accepted": False, "rejection_reasons": ["invalid_grasp_candidates"]}

    local = (points - translation) @ rotation
    collision = collision_counts(local, width, height, depth, args)
    solid_collision = collision["left_finger"] + collision["right_finger"] + collision["palm"]
    reasons = []
    result = {
        "accepted": False, "rejection_reasons": reasons,
        "evaluation_stage": "collision", "antipodal_evaluated": False,
        "graspnet_score": score, "predicted_width_m": width,
        "height_m": height, "depth_m": depth,
        "collision_points": collision,
    }
    if solid_collision > args.collision_point_tolerance:
        reasons.append("full_cloud_gripper_collision")
    if collision["approach"] > args.collision_point_tolerance:
        reasons.append("full_cloud_approach_collision")
    if reasons:
        return result

    x, y, z = local.T
    finger_back = depth - args.finger_length
    region = (
        (x >= finger_back - args.depth_margin) & (x <= depth + args.depth_margin)
        & (np.abs(z) <= height / 2 + args.height_margin)
        & (np.abs(y) <= width / 2 + args.width_tolerance)
    )
    region_indices = np.flatnonzero(region)
    result.update(evaluation_stage="contact_geometry", points_in_closing_volume=int(len(region_indices)))
    if len(region_indices) < 2:
        reasons.append("empty_closing_volume")
        return result

    region_y = y[region_indices]
    left_y, right_y = float(region_y.min()), float(region_y.max())
    contact_width = right_y - left_y
    left_indices = region_indices[np.abs(region_y - left_y) <= args.contact_band]
    right_indices = region_indices[np.abs(region_y - right_y) <= args.contact_band]
    result.update(
        contact_width_m=contact_width,
        left_contact_points=int(len(left_indices)), right_contact_points=int(len(right_indices)))
    if contact_width > width + args.width_tolerance:
        reasons.append("opening_too_narrow")
    if len(left_indices) < args.min_contact_points:
        reasons.append("insufficient_left_contact")
    if len(right_indices) < args.min_contact_points:
        reasons.append("insufficient_right_contact")
    if reasons:
        return result

    # Prefer points closest to each extreme before bounding the cross-product.
    left_indices = left_indices[np.argsort(np.abs(y[left_indices] - left_y))[:args.max_contact_points]]
    right_indices = right_indices[np.argsort(np.abs(y[right_indices] - right_y))[:args.max_contact_points]]
    best = None
    best_rank = -np.inf
    for i in left_indices:
        for j in right_indices:
            metrics = pair_metrics(
                points[i], points[j], normals[i], normals[j], args.friction,
                args.angular_tolerance_deg, not args.oriented_normals)
            if metrics is None:
                continue
            rank = min(metrics["antipodal_score"], metrics["normal_opposition"])
            if metrics["valid"] and (best is None or not best["valid"] or rank > best_rank):
                best, best_rank = metrics, rank
                best["indices"] = (int(i), int(j))
            elif best is None or (not best["valid"] and rank > best_rank):
                best, best_rank = metrics, rank
                best["indices"] = (int(i), int(j))
    if best is None:
        reasons.append("no_distinct_contact_pair")
        return result

    i, j = best.pop("indices")
    left_normal, right_normal = best.pop("left_normal"), best.pop("right_normal")
    result.update(best)
    result.update(
        contact_point_indices=[i, j],
        contact_points_object=points[[i, j]].tolist(),
        contact_normals_object=np.asarray([left_normal, right_normal]).tolist(),
        friction_cone_half_angle_deg=float(np.rad2deg(min(
            np.arctan(args.friction), np.deg2rad(args.angular_tolerance_deg)))),
    )
    if not result.pop("valid"):
        reasons.append("friction_cone_or_antipodal_failure")
    result["evaluation_stage"] = "antipodal"
    result["antipodal_evaluated"] = True
    result["accepted"] = not reasons
    return result


def equal_axes(axis, points):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * 0.7, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def draw(points, affordance, rows, evaluations, label, maximum):
    accepted = [(row, evaluation) for row, evaluation in zip(rows, evaluations)
                if evaluation["accepted"]][:maximum]
    figure = plt.figure(figsize=(11, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(*points.T, c=affordance if affordance is not None else "#7895a8",
                           cmap="turbo" if affordance is not None else None,
                           vmin=0 if affordance is not None else None,
                           vmax=1 if affordance is not None else None,
                           s=2, alpha=0.18, depthshade=False)
    if affordance is not None:
        figure.colorbar(scatter, ax=axis, shrink=0.65, pad=0.08, label="Human grasp affordance")
    colors = plt.colormaps["tab20"](np.linspace(0, 1, max(len(accepted), 1)))
    extent = float(np.ptp(points, axis=0).max())
    for rank, ((row, evaluation), color) in enumerate(zip(accepted, colors)):
        rotation, origin = row[4:13].reshape(3, 3), row[13:16]
        width, depth = float(row[1]), float(row[3])
        center = origin + depth * rotation[:, 0]
        left, right = center - width / 2 * rotation[:, 1], center + width / 2 * rotation[:, 1]
        axis.plot(*np.vstack((left, right)).T, color=color, linewidth=1.3, linestyle=":")
        contacts = np.asarray(evaluation["contact_points_object"])
        axis.plot(*contacts.T, color=color, linewidth=4)
        axis.scatter(*contacts.T, color=[color], edgecolors="black", linewidths=0.8,
                     s=80, depthshade=False)
        axis.quiver(*origin, *rotation[:, 0], length=extent * 0.14,
                    normalize=True, color=color, linewidth=1.5)
        axis.text(*contacts.mean(axis=0), f"{rank}:{evaluation['antipodal_score']:.2f}",
                  color=color, fontsize=8)
    axis.set_title(f"{label} — collision-free, antipodal-verified GraspNet grasps ({len(accepted)} shown)\n"
                   "solid: verified contacts; dotted: predicted jaw opening; arrow: approach")
    axis.set_xlabel("object X [m]")
    axis.set_ylabel("object Y [m]")
    axis.set_zlabel("object Z [m]")
    equal_axes(axis, points)
    return figure


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grasps", type=Path, required=True, help="GraspNet candidate JSON")
    p.add_argument("--npz", type=Path, help="Override source affordance.npz")
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output", type=Path, help="Save accepted-grasp PNG; otherwise open a window")
    p.add_argument("--friction", type=float, default=0.8)
    p.add_argument("--angular-tolerance-deg", type=float, default=30.0)
    p.add_argument("--contact-band", type=float, default=0.003)
    p.add_argument("--depth-margin", type=float, default=0.005)
    p.add_argument("--height-margin", type=float, default=0.003)
    p.add_argument("--width-tolerance", type=float, default=0.003)
    p.add_argument("--finger-length", type=float, default=0.06)
    p.add_argument("--finger-thickness", type=float, default=0.01)
    p.add_argument("--palm-thickness", type=float, default=0.01)
    p.add_argument("--approach-distance", type=float, default=0.05)
    p.add_argument("--collision-clearance", type=float, default=0.0005)
    p.add_argument("--collision-point-tolerance", type=int, default=0,
                   help="Maximum full-cloud points allowed in each gripper solid or approach sweep")
    p.add_argument("--min-contact-points", type=int, default=1)
    p.add_argument("--max-contact-points", type=int, default=32)
    p.add_argument("--oriented-normals", action="store_true",
                   help="Trust stored normal signs; default pair-aligns PCA normal signs")
    p.add_argument("--show-candidates", type=int, default=12)
    p.add_argument("--dpi", type=int, default=180)
    return p


def main(argv=None):
    # Validate parser args
    args = parser().parse_args(argv)
    if args.friction < 0 or not 0 < args.angular_tolerance_deg <= 90:
        raise ValueError("Invalid friction or angular tolerance")
    if any(value < 0 for value in (args.contact_band, args.depth_margin,
                                   args.height_margin, args.width_tolerance,
                                   args.finger_length, args.finger_thickness,
                                   args.palm_thickness, args.approach_distance,
                                   args.collision_clearance, args.collision_point_tolerance)):
        raise ValueError("Contact bands and margins cannot be negative")
    if args.min_contact_points < 1 or args.max_contact_points < 1 or args.show_candidates < 1:
        raise ValueError("Contact and display counts must be positive")

    # read/load file
    grasp_annotations = json.loads(args.grasps.read_text())
    candidates = np.asarray(grasp_annotations.get("graspnet", {}).get("rows", []), dtype=np.float64)
    
    if candidates.ndim != 2 or candidates.shape[1:] != (17,):
        raise ValueError("Input JSON must contain graspnet.rows with shape [N,17]")
    
    npz_file = resolve_npz(grasp_annotations, args.npz)
    points, normals, affordance = load_cloud(npz_file, grasp_annotations.get("metric_scale", {}))
    evaluations = [evaluate_candidate(row, points, normals, args) for row in candidates]
    accepted_indices = [index for index, result in enumerate(evaluations) if result["accepted"]]
    reasons = Counter(reason for result in evaluations for reason in result["rejection_reasons"])

    result = deepcopy(grasp_annotations)
    result["schema"] = "graspnet-antipodal-evaluation/v1"
    result["antipodal"] = {
        "columns": grasp_annotations["graspnet"].get("columns"),
        "rows": candidates[accepted_indices].tolist(),
        "source_indices": accepted_indices,
        "parameters": {
            "friction": args.friction,
            "angular_tolerance_deg": args.angular_tolerance_deg,
            "contact_band_m": args.contact_band,
            "depth_margin_m": args.depth_margin,
            "height_margin_m": args.height_margin,
            "width_tolerance_m": args.width_tolerance,
            "finger_length_m": args.finger_length,
            "finger_thickness_m": args.finger_thickness,
            "palm_thickness_m": args.palm_thickness,
            "approach_distance_m": args.approach_distance,
            "collision_clearance_m": args.collision_clearance,
            "collision_point_tolerance": args.collision_point_tolerance,
            "evaluation_order": ["full_cloud_collision", "contact_geometry", "antipodal_friction_cones"],
            "normal_orientation": "provided" if args.oriented_normals else "pair_aligned",
        },
        "summary": {
            "input": len(candidates), "accepted": len(accepted_indices),
            "rejected": len(candidates) - len(accepted_indices),
            "rejection_reason_counts": dict(sorted(reasons.items())),
        },
        "candidates": evaluations,
    }
    output_json = args.output_json or args.grasps.with_name(args.grasps.stem + "_antipodal.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(accepted_indices)}/{len(candidates)} accepted grasps to {output_json}")
    print(f"rejection_reasons={dict(sorted(reasons.items()))}")

    label = grasp_annotations.get("source", {}).get("object_id", npz_file.parent.name)
    object_type = grasp_annotations.get("source", {}).get("object_type")
    if object_type:
        label = f"{object_type}: {label}"
    figure = draw(points, affordance, candidates, evaluations, label, args.show_candidates)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=args.dpi)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
