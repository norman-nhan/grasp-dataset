"""Bounded, deterministic antipodal surface sampling in the object frame."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import trimesh

@dataclass
class Sampling:
    surface_points: int = 2500
    candidates: int = 100
    approaches: int = 8
    neighbors: int = 96
    min_width: float = 0.005
    max_width: float | None = None
    friction: float = 0.8
    angular_tolerance_deg: float = 30
    seed: int = 42
    pairs_per_point: int | None = None


def antipodal(p, q, n, m, mu, tolerance):
    d = q - p
    length = np.linalg.norm(d)
    if length < 1e-10:
        return False, 0.0
    d /= length
    threshold = np.cos(min(np.arctan(mu), np.deg2rad(tolerance)))
    score = min(float(-n @ d), float(m @ d))
    return bool(score >= threshold and -n @ m >= np.cos(np.deg2rad(tolerance))), score


def generate(mesh, gripper, settings):
    s = settings
    if s.candidates < 1 or s.surface_points < 2 or s.approaches < 1 or s.neighbors < 1 or s.friction < 0 or not 0 < s.angular_tolerance_deg <= 90 or s.min_width < 0:
        raise ValueError('Invalid sampling counts or friction')
    if s.pairs_per_point is not None and s.pairs_per_point < 1:
        raise ValueError('pairs_per_point must be positive')
    lo, hi = max(s.min_width, gripper.min_width), min((s.max_width if s.max_width is not None else gripper.max_width), gripper.max_width)
    if not 0 <= lo < hi:
        raise ValueError('Requested opening range does not intersect the gripper limits')
    rng = np.random.default_rng(s.seed)
    points, faces = trimesh.sample.sample_surface(mesh, s.surface_points, seed=s.seed)
    normals = mesh.face_normals[faces]
    tree = cKDTree(points)
    records, seen = [], set()
    for i in rng.permutation(len(points)):
        neighbors = tree.query_ball_point(points[i], hi)
        if len(neighbors) > s.neighbors:
            neighbors = rng.choice(neighbors, s.neighbors, replace=False)
        retained_pairs = 0
        for j in neighbors:
            key = tuple(sorted((int(i), int(j))))
            if key in seen:
                continue
            seen.add(key)
            width = np.linalg.norm(points[j] - points[i])
            if not lo <= width <= hi:
                continue
            valid, score = antipodal(points[i], points[j], normals[i], normals[j], s.friction, s.angular_tolerance_deg)
            if not valid:
                continue
            retained_pairs += 1
            y = (points[j] - points[i]) / width
            ref = np.eye(3)[np.argmin(np.abs(y))]
            z0 = np.cross(y, ref); z0 /= np.linalg.norm(z0)
            for angle in np.linspace(0, 2*np.pi, s.approaches, endpoint=False):
                z = z0*np.cos(angle) + np.cross(y, z0)*np.sin(angle)
                r = np.column_stack((np.cross(y, z), y, z))
                t = np.eye(4); t[:3, :3] = r
                # Calibrated inner-face center differs from the repository TCP.
                t[:3, 3] = (points[i]+points[j])/2 - r @ gripper.contact_center_tcp
                records.append(dict(grasp_id=len(records), grasp_pose_object=t.tolist(), gripper_width=float(width),
                    contact_points_object=points[[i, j]].tolist(), contact_normals_object=normals[[i, j]].tolist(),
                    antipodal_score=score, geometry_valid=True, collision_valid=None, rejection_reasons=[],
                    evaluation_status='pending', physical_success=None, physical_success_rate=None, slip_distance=None,
                    affordance_overlap=None, handover_suitable=None, trial_results=[]))
                if len(records) >= s.candidates:
                    return records
            if s.pairs_per_point is not None and retained_pairs >= s.pairs_per_point:
                break
    return records
