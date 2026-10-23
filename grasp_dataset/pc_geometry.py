"""Antipodal grasp generation directly from registered point-cloud normals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Sampling


def load_registered(npz_file: Path, transform_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load points/normals and map them into the metric object frame.

    LASO point clouds are normalized and therefore must never be used directly
    with gripper widths expressed in metres.  The transform must be a uniform
    positive-scale similarity transform (rotation, scale, and translation).
    """
    with np.load(npz_file, allow_pickle=False) as archive:
        if "points" not in archive or "normals" not in archive:
            raise ValueError(f"{npz_file} must contain 'points' and 'normals'")
        points = np.asarray(archive["points"], dtype=np.float64)
        normals = np.asarray(archive["normals"], dtype=np.float64)

    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape [N, 3]")
    if len(points) < 2 or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("point cloud must contain at least two finite points and normals")

    payload = json.loads(transform_file.read_text())
    if isinstance(payload, dict):
        payload = payload.get("annotation_to_object", payload.get("transform"))
    transform = np.asarray(payload, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("point-cloud transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("point-cloud transform must be affine")

    linear = transform[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    scale = float(singular.mean())
    if scale <= 0 or not np.allclose(singular, scale, rtol=1e-5, atol=1e-10):
        raise ValueError("point-cloud transform must use one uniform positive scale")
    rotation = linear / scale
    if np.linalg.det(rotation) < 0 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("point-cloud transform must contain a proper rotation")

    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("point cloud contains zero-length normals")
    points_object = points @ linear.T + transform[:3, 3]
    normals_object = (normals / lengths[:, None]) @ rotation.T
    return points_object, normals_object


def _pair_antipodal(p, q, n, m, mu, tolerance, unoriented):
    direction = q - p
    width = np.linalg.norm(direction)
    if width < 1e-10:
        return False, 0.0, n, m
    direction /= width
    if unoriented:
        # PCA fixes a tangent plane but not its sign.  Choose the signs that
        # oppose the closing forces for this pair.  This also avoids the cloud-
        # centroid orientation failure on inner walls of handles and cavities.
        n = n * (-1.0 if n @ direction > 0 else 1.0)
        m = m * (-1.0 if m @ direction < 0 else 1.0)
    threshold = np.cos(min(np.arctan(mu), np.deg2rad(tolerance)))
    score = min(float(-n @ direction), float(m @ direction))
    valid = score >= threshold and -n @ m >= np.cos(np.deg2rad(tolerance))
    return bool(valid), score, n, m


def generate(points, normals, gripper, settings: Sampling, *, unoriented_normals: bool = True):
    """Generate grasp records without constructing or sampling a surface mesh."""
    s = settings
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape [N, 3]")
    if s.candidates < 1 or s.surface_points < 2 or s.approaches < 1 or s.neighbors < 1 or s.friction < 0 or not 0 < s.angular_tolerance_deg <= 90 or s.min_width < 0:
        raise ValueError("Invalid sampling counts or friction")
    if s.pairs_per_point is not None and s.pairs_per_point < 1:
        raise ValueError("pairs_per_point must be positive")
    if len(points) < 2 or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("point cloud must contain at least two finite points and normals")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("point cloud contains zero-length normals")
    normals = normals / lengths[:, None]

    lo = max(s.min_width, gripper.min_width)
    hi = min(s.max_width if s.max_width is not None else gripper.max_width, gripper.max_width)
    if not 0 <= lo < hi:
        raise ValueError("Requested opening range does not intersect the gripper limits")

    rng = np.random.default_rng(s.seed)
    source_indices = np.arange(len(points))
    if len(points) > s.surface_points:
        source_indices = np.sort(rng.choice(len(points), s.surface_points, replace=False))
        points, normals = points[source_indices], normals[source_indices]

    tree = cKDTree(points)
    records, seen = [], set()
    for i in rng.permutation(len(points)):
        neighbors = tree.query_ball_point(points[i], hi)
        if len(neighbors) > s.neighbors:
            neighbors = rng.choice(neighbors, s.neighbors, replace=False)
        retained_pairs = 0
        for j_value in neighbors:
            j = int(j_value)
            key = tuple(sorted((int(i), j)))
            if key in seen:
                continue
            seen.add(key)
            width = float(np.linalg.norm(points[j] - points[i]))
            if not lo <= width <= hi:
                continue
            valid, score, normal_i, normal_j = _pair_antipodal(
                points[i], points[j], normals[i], normals[j], s.friction,
                s.angular_tolerance_deg, unoriented_normals,
            )
            if not valid:
                continue
            retained_pairs += 1
            y = (points[j] - points[i]) / width
            ref = np.eye(3)[np.argmin(np.abs(y))]
            z0 = np.cross(y, ref)
            z0 /= np.linalg.norm(z0)
            '''
            hoRotate grasp pose around gripper closing axis (y-axis) from a contact pairs.
            '''
            for angle in np.linspace(0, 2 * np.pi, s.approaches, endpoint=False):
                z = z0 * np.cos(angle) + np.cross(y, z0) * np.sin(angle)
                rotation = np.column_stack((np.cross(y, z), y, z))
                pose = np.eye(4)
                pose[:3, :3] = rotation
                pose[:3, 3] = (points[i] + points[j]) / 2 - rotation @ gripper.contact_center_tcp
                records.append(dict(
                    grasp_id=len(records), grasp_pose_object=pose.tolist(), gripper_width=width,
                    contact_points_object=points[[i, j]].tolist(),
                    contact_normals_object=np.asarray([normal_i, normal_j]).tolist(),
                    contact_point_indices=[int(source_indices[i]), int(source_indices[j])],
                    candidate_source="point_cloud_normals",
                    normal_orientation="pair_aligned" if unoriented_normals else "provided",
                    antipodal_score=score, geometry_valid=True, collision_valid=None,
                    rejection_reasons=[], evaluation_status="pending", physical_success=None,
                    physical_success_rate=None, slip_distance=None, affordance_overlap=None,
                    handover_suitable=None, trial_results=[]))
                if len(records) >= s.candidates:
                    return records
            if s.pairs_per_point is not None and retained_pairs >= s.pairs_per_point:
                break
    return records
