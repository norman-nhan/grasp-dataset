#!/usr/bin/env python3
"""Generate and visualize LASO point-cloud grasps with GraspNet baseline."""

from __future__ import annotations
from pprint import pprint
import argparse
import json
from pathlib import Path
import sys

import numpy as np

import matplotlib.pyplot as plt

from grasp_dataset.laso_catalog import load as load_catalog, object_type
from grasp_dataset.pc_geometry import load_registered


PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parents[1]
DEFAULT_DATA_ROOT = PROJECT / "data"
DEFAULT_CATALOG = PROJECT / "laso_object_types.json"
DEFAULT_EXTENTS = PROJECT / "laso_physical_dims_assumed.json"
DEFAULT_GRASPNET = WORKSPACE / "not-ros-pkg" / "graspnet-baseline"
DEFAULT_CHECKPOINT = WORKSPACE / "models" / "graspnet" / "checkpoint-rs.tar"
COLUMNS = [
    "score", "width", "height", "depth",
    *[f"rotation_{i}{j}" for i in range(3) for j in range(3)],
    "translation_x", "translation_y", "translation_z", "object_id",
]

def resolve_npz(data_root: Path, object_id: str | None, explicit: Path | None) -> Path:
    """Magic function to find your npz file with just object_id.
    
    Expect:
    ```
    object_id/
        |__affordance.npz
    ```
    """
    if not data_root.exists():
        raise ValueError("Invalid data root path!")
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if not object_id:
        raise ValueError("Supply --object or --npz")

    matches = list(data_root.rglob(f"{object_id}/affordance.npz")) # search directly under data/
    if not len(matches): # between data and object id there is something else
        matches = list(data_root.rglob(f"*/{object_id}/affordance.npz"))
        if not len(matches):
            raise ValueError(f"Unable to resolve {object_id}/affordance.npz. Use --npz")
    
    return matches[0]

def load_npz(npz_file: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return xyz points and affordance score."""
    with np.load(npz_file, allow_pickle=False) as archive:
        if "points" not in archive:
            raise ValueError(f"{npz_file} has no 'points' array")
        points = np.asarray(archive["points"], dtype=np.float32)
        scores = np.asarray(archive["affordance_score"], dtype=np.float32).reshape(-1) \
            if "affordance_score" in archive else None
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
        raise ValueError("points must have shape [N,3] with at least two samples")
    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite values")
    if scores is not None and len(scores) != len(points):
        raise ValueError("affordance_score length does not match points")
    return points, scores


def load_source_metadata(npz_file: Path) -> dict:
    metadata_file = npz_file.with_name("metadata.json")
    if not metadata_file.is_file():
        return {}
    metadata = json.loads(metadata_file.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_file} must contain a JSON object")
    return metadata


def prepare_metric_points(source_points: np.ndarray, npz_file: Path, metadata: dict,
                          category: str | None, args) -> tuple[np.ndarray, dict]:
    """Resolve metric coordinates without rescaling sources already in metres.
    
    Return scaled points and scale_metadata."""
    if args.transform:
        points, _ = load_registered(npz_file, args.transform.resolve())
        return points.astype(np.float32), {
            "method": "similarity_transform", "file": str(args.transform.resolve())}

    if args.longest_extent_m is not None:
        points, center, factor = metricize(source_points, args.longest_extent_m)
        return points, {
            "method": "explicit_longest_extent",
            "longest_extent_m": args.longest_extent_m,
            "source_center": center.astype(float).tolist(), "scale_to_m": factor,
        }

    schema = str(metadata.get("schema", ""))
    units = str(metadata.get("units", "")).strip().lower()
    if schema.startswith("ycb-") and units in ("metres", "meters", "metre", "meter", "m"):
        return source_points.astype(np.float32, copy=True), {
            "method": "source_metric_coordinates", "schema": schema,
            "units": metadata["units"], "scale_to_m": 1.0,
        }

    extents = json.loads(args.class_extents.read_text())
    if category not in extents:
        raise ValueError(
            f"Source {schema or 'without schema'} is not identified as metric YCB "
            "and has no LASO class extent; supply --longest-extent-m or --transform")
    extent = float(extents[category])
    points, center, factor = metricize(source_points, extent)
    return points, {
        "method": "assumed_longest_extent", "longest_extent_m": extent,
        "source_center": center.astype(float).tolist(), "scale_to_m": factor,
        "warning": "LASO has no measured physical scale",
    }


def metricize(points: np.ndarray, longest_extent_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    if not np.isfinite(longest_extent_m) or longest_extent_m <= 0:
        raise ValueError("longest extent must be positive and finite")
    center = points.mean(axis=0)
    source_extent = float(np.ptp(points, axis=0).max())
    if source_extent <= 1e-9:
        raise ValueError("Degenerate point cloud")
    scale = float(longest_extent_m) / source_extent
    return ((points - center) * scale).astype(np.float32), center, scale


def fibonacci_directions(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("--views must be positive")
    indices = np.arange(count, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(golden * indices),
                            radius * np.sin(golden * indices), z))


def camera_basis(view_direction: np.ndarray) -> np.ndarray:
    z = np.asarray(view_direction, dtype=float)
    z /= np.linalg.norm(z)
    up = np.array([0., 0., 1.]) if abs(z[2]) < 0.9 else np.array([0., 1., 0.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def load_model(root: Path, checkpoint: Path):
    import torch

    if not (root / "models" / "graspnet.py").is_file():
        raise FileNotFoundError(f"GraspNet baseline missing at {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GraspNet checkpoint missing at {checkpoint}")
    for child in ("models", "dataset", "utils", "pointnet2", "knn"):
        sys.path.insert(0, str(root / child))
    try:
        from graspnet import GraspNet, pred_decode
    except ImportError as error:
        raise RuntimeError(
            "GraspNet operators are unavailable. Activate "
            "/ros2_ws/.venv/graspnet_env before running this script."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Official GraspNet baseline inference requires CUDA")
    model = GraspNet(
        input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False,
    ).cuda()
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, pred_decode, int(state.get("epoch", -1))


def infer_view(model, decode, points: np.ndarray, basis: np.ndarray, args, rng) -> np.ndarray:
    import torch

    center = points.mean(axis=0)
    offset = np.array([0., 0., args.camera_depth])
    camera = (points - center) @ basis + offset
    chosen = rng.choice(len(camera), args.num_points, replace=len(camera) < args.num_points)
    tensor = torch.from_numpy(camera[chosen][None].astype(np.float32)).cuda()
    with torch.no_grad():
        rows = decode(model({"point_clouds": tensor}))[0].cpu().numpy()
    if args.proposals_per_view and len(rows) > args.proposals_per_view:
        rows = rows[rng.choice(len(rows), args.proposals_per_view, replace=False)]
    rows = rows.copy()
    for row in rows:
        rotation = row[4:13].reshape(3, 3)
        row[4:13] = (basis @ rotation).reshape(-1)
        row[13:16] = center + basis @ (row[13:16] - offset)
    return rows


def filter_predictions(candidates: np.ndarray, points: np.ndarray, args) -> tuple[np.ndarray, dict]:
    finite = np.isfinite(candidates).all(axis=1) # reject infinite/NaN candidates
    score = candidates[:, 0] >= args.min_score # reject candidates that has score lower than min score
    width = (candidates[:, 1] >= args.min_width) & (candidates[:, 1] <= args.max_width) # reject candidates that is bigger than OpenArm gripper max width.
    
    candidates = candidates[finite & score & width] # keep candidates that satisfy all conditions

    log = {
        "raw": int(len(finite)), "nonfinite_rejected": int((~finite).sum()),
        "score_rejected": int((finite & ~score).sum()),
        "width_rejected": int((finite & score & ~width).sum()),
    }

    # all candidates rejected
    if not len(candidates):
        log.update(after_nms=0, collision_rejected=0, retained=0)
        return candidates, log

    # Apply NMS for surviving candidates
    from graspnetAPI import GraspGroup
    group = GraspGroup(candidates.astype(np.float64)).nms(
        translation_thresh=args.nms_translation,
        rotation_thresh=np.deg2rad(args.nms_rotation_deg),
    ).sort_by_score()
    log["after_nms"] = len(group)

    # Check collision after nms
    collision_rejected = 0
    if args.collision_thresh > 0 and len(group):
        root = args.graspnet_root
        if str(root / "utils") not in sys.path:
            sys.path.insert(0, str(root / "utils"))
        from collision_detector import ModelFreeCollisionDetector
        detector = ModelFreeCollisionDetector(points, voxel_size=args.voxel_size)
        collision = detector.detect(
            group, approach_dist=args.approach_distance,
            collision_thresh=args.collision_thresh,
        )
        collision_rejected = int(collision.sum())
        group = group[~collision] # keep candidates that collision = False

    # Keep `max_candidates` from highest score rank
    result = group.grasp_group_array[:args.max_candidates].astype(np.float64)
    log.update(collision_rejected=collision_rejected, retained=len(result))
    return result, log


def equal_axes(axis, points: np.ndarray) -> None:
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * 0.7, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def draw(points, affordance, rows, label, maximum):
    shown = rows[:maximum]
    figure = plt.figure(figsize=(11, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(*points.T, c=affordance if affordance is not None else "#7895a8",
                           cmap="turbo" if affordance is not None else None,
                           vmin=0 if affordance is not None else None,
                           vmax=1 if affordance is not None else None,
                           s=3, alpha=0.45, depthshade=False)
    if affordance is not None:
        figure.colorbar(scatter, ax=axis, shrink=0.65, pad=0.08, label="Human grasp affordance")
    colors = plt.colormaps["tab20"](np.linspace(0, 1, max(len(shown), 1)))
    extent = float(np.ptp(points, axis=0).max())
    for index, (row, color) in enumerate(zip(shown, colors)):
        width, depth = float(row[1]), float(row[3])
        rotation = row[4:13].reshape(3, 3)
        origin = row[13:16]
        center = origin + depth * rotation[:, 0]
        left = center - width / 2 * rotation[:, 1]
        right = center + width / 2 * rotation[:, 1]
        back_left = origin - width / 2 * rotation[:, 1]
        back_right = origin + width / 2 * rotation[:, 1]
        axis.plot(*np.vstack((left, right)).T, color=color, linewidth=2.4)
        axis.plot(*np.vstack((back_left, left)).T, color=color, linewidth=1.4)
        axis.plot(*np.vstack((back_right, right)).T, color=color, linewidth=1.4)
        axis.plot(*np.vstack((back_left, back_right)).T, color=color, linewidth=1.4)
        axis.quiver(*origin, *rotation[:, 0], length=extent * 0.16,
                    normalize=True, color=color, linewidth=1.8)
        axis.text(*center, f"{index}:{row[0]:.2f}", color=color, fontsize=8)
    axis.set_title(f"{label} — GraspNet grasps ({len(shown)}/{len(rows)} shown)\n"
                   "jaw wireframe; arrow: GraspNet +X approach; label: rank:score")
    axis.set_xlabel("object X [m]")
    axis.set_ylabel("object Y [m]")
    axis.set_zlabel("object Z [m]")
    equal_axes(axis, points)
    return figure


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--object", help="LASO object ID below --data-root")
    source.add_argument("--npz", type=Path, help="Explicit affordance.npz")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--object-types", type=Path, default=DEFAULT_CATALOG)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--transform", type=Path, help="Source-to-object metric similarity JSON")
    scale.add_argument("--longest-extent-m", type=float, help="Assumed physical longest extent")
    p.add_argument("--class-extents", type=Path, default=DEFAULT_EXTENTS)
    p.add_argument("--graspnet-root", type=Path, default=DEFAULT_GRASPNET)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--camera-depth", type=float, default=0.5)
    p.add_argument("--num-points", type=int, default=20000)
    p.add_argument("--proposals-per-view", type=int, default=2000)
    p.add_argument("--min-score", type=float, default=0.10)
    p.add_argument("--min-width", type=float, default=0.000)
    p.add_argument("--max-width", type=float, default=0.075)
    p.add_argument("--nms-translation", type=float, default=0.015)
    p.add_argument("--nms-rotation-deg", type=float, default=20.0)
    p.add_argument("--collision-thresh", type=float, default=0.01,
                   help="Set 0 to disable model-free point-cloud collision filtering")
    p.add_argument("--voxel-size", type=float, default=0.005)
    p.add_argument("--approach-distance", type=float, default=0.05)
    p.add_argument("--max-candidates", type=int, default=1000)
    p.add_argument("--show-candidates", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output", type=Path, help="Save PNG instead of opening a window")
    p.add_argument("--dpi", type=int, default=180)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.show_candidates < 1 or args.max_candidates < 1 or args.num_points < 1:
        raise ValueError("point and candidate counts must be positive")
    npz_file = resolve_npz(args.data_root.resolve(), args.object, args.npz)
    source_points, affordance = load_npz(npz_file)
    source_metadata = load_source_metadata(npz_file)
    catalog = load_catalog(args.object_types) if args.object_types.is_file() else {}
    category = object_type(npz_file.parent.name, catalog) if catalog else None
    points, scale_metadata = prepare_metric_points(
        source_points, npz_file, source_metadata, category, args)

    model, decode, epoch = load_model(args.graspnet_root.resolve(), args.checkpoint.resolve())
    rng = np.random.default_rng(args.seed)
    batches = [infer_view(model, decode, points, camera_basis(direction), args, rng)
               for direction in fibonacci_directions(args.views)]
    raw = np.concatenate(batches) if batches else np.empty((0, 17))
    rows, filtering = filter_predictions(raw, points, args)
    if not len(rows):
        raise ValueError(f"GraspNet produced no retained candidates; filtering={filtering}")

    label = f"{category}: {npz_file.parent.name}" if category else npz_file.parent.name
    payload = {
        "schema": "graspnet-candidates/v1",
        "source": {"file": str(npz_file), "object_id": npz_file.parent.name,
                   "object_type": category, "source_schema": source_metadata.get("schema"),
                   "point_count": len(points), "units": "metres"},
        "metric_scale": scale_metadata,
        "model": {"implementation": str(args.graspnet_root.resolve()),
                  "checkpoint": str(args.checkpoint.resolve()), "epoch": epoch},
        "generation": {"views": args.views, "num_points": args.num_points,
                       "seed": args.seed, "filtering": filtering},
        "graspnet": {"format": "GraspNetAPI GraspGroup Nx17", "columns": COLUMNS,
                     "rows": rows.tolist()},
    }
    output_json = args.output_json or npz_file.with_name("graspnet_grasps.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(rows)} GraspNet candidates to {output_json}")
    pprint(f"class={category or 'unknown'} scale={scale_metadata} filtering={filtering}")

    figure = draw(points, affordance, rows, label, min(args.show_candidates, len(rows)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=args.dpi)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
