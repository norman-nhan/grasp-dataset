#!/usr/bin/env python3
"""Generate one grasp annotation by merging GraspNet predictions from occlusion-aware random views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from grasp_dataset.laso_catalog import load as load_catalog, object_type
from generate_graspnet_grasp import (
    COLUMNS,
    DEFAULT_CATALOG,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_EXTENTS,
    DEFAULT_GRASPNET,
    draw,
    filter_predictions,
    load_model,
    load_npz,
    load_source_metadata,
    prepare_metric_points,
    resolve_npz,
)


def random_directions(num_views: int, rng: np.random.Generator,
                      min_elevation_deg: float, max_elevation_deg: float) -> np.ndarray:
    """Sample `num_views` camera positions uniformly in azimuth and spherical elevation band.
    Return [num_views, 3]. 
    """
    if num_views < 1 or not -90 <= min_elevation_deg < max_elevation_deg <= 90:
        raise ValueError("Invalid view count or elevation range")
    azimuth = rng.uniform(0, 2 * np.pi, num_views)
    z = rng.uniform(np.sin(np.deg2rad(min_elevation_deg)),
                    np.sin(np.deg2rad(max_elevation_deg)), num_views)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))


def camera_basis(position_direction: np.ndarray) -> np.ndarray:
    """Return camera-to-object rotation with camera +Z looking at the object."""
    forward = -np.asarray(position_direction, dtype=float)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0., 0., 1.]) if abs(forward[2]) < 0.95 else np.array([0., 1., 0.])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.column_stack((right, up, forward))


def sample_observe_points(points: np.ndarray, camera_position: np.ndarray, basis: np.ndarray,
                   width: int, height: int, fov_y_deg: float, point_radius_px: int,
                   depth_tolerance: float, near: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Sample points from given camera view. Return points in camera and observed points indices.

    Rasterize point splats and return only nearest-depth visible samples.

    This is a perspective z-buffer, not a spatial crop. A back point is removed
    when a nearer point splat covers its projected pixel.
    """
    if width < 2 or height < 2 or not 1 < fov_y_deg < 179 or point_radius_px < 0:
        raise ValueError("Invalid camera raster settings")
    if depth_tolerance < 0 or near <= 0:
        raise ValueError("Depth tolerance must be nonnegative and near plane positive")
    camera = (points - camera_position) @ basis
    z = camera[:, 2]
    focal = height / (2 * np.tan(np.deg2rad(fov_y_deg) / 2))
    safe_z = np.where(z > near, z, 1.0)
    u = np.rint(focal * camera[:, 0] / safe_z + (width - 1) / 2).astype(np.int64)
    v = np.rint(-focal * camera[:, 1] / safe_z + (height - 1) / 2).astype(np.int64)
    projected = (z > near) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_indices = np.flatnonzero(projected)
    if not len(valid_indices):
        return camera, valid_indices

    depth = np.full(width * height, np.inf, dtype=np.float64)
    offsets = [(dx, dy) for dy in range(-point_radius_px, point_radius_px + 1)
               for dx in range(-point_radius_px, point_radius_px + 1)
               if dx * dx + dy * dy <= point_radius_px * point_radius_px]
    for dx, dy in offsets or [(0, 0)]:
        px, py = u[valid_indices] + dx, v[valid_indices] + dy
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        pixels = py[inside] * width + px[inside]
        np.minimum.at(depth, pixels, z[valid_indices[inside]])

    center_pixels = v[valid_indices] * width + u[valid_indices]
    visible = z[valid_indices] <= depth[center_pixels] + depth_tolerance
    return camera, valid_indices[visible]


def infer_partial(model, decode, observed_points: np.ndarray, camera_position: np.ndarray,
                  cam2obj_basis: np.ndarray, args, rng: np.random.Generator) -> np.ndarray:
    """Infer GraspNet output. Return grasp pose in object frame."""
    import torch

    # sample `num_points` from `observed_points` if points are not enough, some points are repeated till satisfy
    chosen = rng.choice(len(observed_points), args.num_points,
                        replace=len(observed_points) < args.num_points)
    point_tensor = torch.from_numpy(observed_points[chosen][None].astype(np.float32)).cuda() # shape: [N, 3]
    with torch.no_grad():
        graspnet_outputs = decode(model({"point_clouds": point_tensor}))[0].cpu().numpy() # shape: [N, 13] read GraspNetAPI for detailed.

    # reduce the amount of predictions to `proposals_per_view`
    if args.proposals_per_view and len(graspnet_outputs) > args.proposals_per_view:
        graspnet_outputs = graspnet_outputs[rng.choice(len(graspnet_outputs), args.proposals_per_view, replace=False)]

    # transform grasp pose from camera frame to object frame
    graspnet_outputs = graspnet_outputs.copy()
    for output in graspnet_outputs:
        cam_rot = output[4:13].reshape(3, 3) # from 4 to 12, rotation matrix with 9 values
        output[4:13] = (cam2obj_basis @ cam_rot).reshape(-1)
        output[13:16] = camera_position + cam2obj_basis @ output[13:16] # from 13 to 15, translation vector with 3 values
    return graspnet_outputs


def find_source_views(final_candidates: np.ndarray, raw_candidates: np.ndarray, raw_view_ids: np.ndarray) -> list[int]:
    """Find/Reconstruct the original view and corresponding grasp candidates from final and raw candidates."""
    result = []
    for candidate in final_candidates:
        exact = np.flatnonzero(np.all(np.isclose(raw_candidates, candidate, rtol=1e-5, atol=1e-6), axis=1))
        if len(exact):
            result.append(int(raw_view_ids[exact[0]]))
            continue
        translation = np.linalg.norm(raw_candidates[:, 13:16] - candidate[13:16], axis=1)
        score = np.abs(raw_candidates[:, 0] - candidate[0])
        result.append(int(raw_view_ids[np.argmin(translation + score)]))
    return result # shape: len(final_candidates). result[i] return source view of candidate i-th in final_candidates


def draw_view_preview(points: np.ndarray, view_records: list[dict], maximum: int):
    shown = view_records[:maximum]
    columns = min(4, len(shown))
    rows = int(np.ceil(len(shown) / columns))
    figure = plt.figure(figsize=(4.4 * columns, 4.0 * rows), constrained_layout=True)
    extent = float(np.ptp(points, axis=0).max())
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(extent * 0.62, 0.01)
    for slot, record in enumerate(shown, 1):
        axis = figure.add_subplot(rows, columns, slot, projection="3d")
        visible = points[np.asarray(record["visible_indices"], dtype=int)]
        axis.scatter(*visible.T, c=visible[:, 2], cmap="viridis", s=4, depthshade=False)
        camera = np.asarray(record["camera_position_m"])
        axis.scatter(*camera, marker="^", color="red", s=45)
        axis.plot(*np.vstack((camera, center)).T, color="red", linewidth=0.8)
        axis.set_title(f"view {record['view_id']}: {len(visible)}/{len(points)} visible")
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
    return figure


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--object")
    source.add_argument("--npz", type=Path)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--object-types", type=Path, default=DEFAULT_CATALOG)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--transform", type=Path)
    scale.add_argument("--longest-extent-m", type=float)
    p.add_argument("--class-extents", type=Path, default=DEFAULT_EXTENTS)
    p.add_argument("--graspnet-root", type=Path, default=DEFAULT_GRASPNET)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--min-elevation-deg", type=float, default=-35)
    p.add_argument("--max-elevation-deg", type=float, default=65)
    p.add_argument("--camera-distance", type=float,
                   help="Object-center distance in metres; default max(0.5, 2.5*object extent)")
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--fov-y-deg", type=float, default=60)
    p.add_argument("--point-radius-px", type=int, default=3,
                   help="Depth splat radius closing gaps between sparse point samples")
    p.add_argument("--depth-tolerance", type=float, default=0.002)
    p.add_argument("--min-visible-points", type=int, default=128)
    p.add_argument("--num-points", type=int, default=20000)
    p.add_argument("--proposals-per-view", type=int, default=2000)
    p.add_argument("--min-score", type=float, default=0.10)
    p.add_argument("--min-width", type=float, default=0.008)
    p.add_argument("--max-width", type=float, default=0.075)
    p.add_argument("--nms-translation", type=float, default=0.015)
    p.add_argument("--nms-rotation-deg", type=float, default=20)
    p.add_argument("--collision-thresh", type=float, default=0.01)
    p.add_argument("--voxel-size", type=float, default=0.005)
    p.add_argument("--approach-distance", type=float, default=0.05)
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--show-candidates", type=int, default=12)
    p.add_argument("--preview-views", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output", type=Path, help="Combined grasp PNG")
    p.add_argument("--views-output", type=Path, help="PNG of depth-visible partial clouds")
    p.add_argument("--dpi", type=int, default=180)
    return p


def main(argv=None):
    args = parser().parse_args(argv)

    if args.num_points < 1 or args.max_candidates < 1 or args.min_visible_points < 1:
        raise ValueError("Point and candidate counts must be positive")
    
    npz_file = resolve_npz(args.data_root.resolve(), args.object, args.npz)
    source_points, affordance = load_npz(npz_file)
    source_metadata = load_source_metadata(npz_file)
    catalog = load_catalog(args.object_types) if args.object_types.is_file() else {}
    category = object_type(npz_file.parent.name, catalog) if catalog else None
    full_cloud, scale_metadata = prepare_metric_points(
        source_points, npz_file, source_metadata, category, args)

    extent = float(np.ptp(full_cloud, axis=0).max()) # extent = object longest dimension 

    # Generate random camera views in a sphere around object
    distance = args.camera_distance or max(0.5, 2.5 * extent)
    if distance <= extent / 2:
        raise ValueError("Camera distance places the camera inside the object bounds")
    center = full_cloud.mean(axis=0) # object point cloud center
    rng = np.random.default_rng(args.seed)
    directions = random_directions(
        args.views, rng, args.min_elevation_deg, args.max_elevation_deg)

    # Load Graspnet model and its prediction decoder
    model, decode, epoch = load_model(args.graspnet_root.resolve(), args.checkpoint.resolve())

    batches, batch_view_ids, view_records = [], [], []
    for view_id, direction in enumerate(directions):
        camera_position = center + direction * distance
        basis = camera_basis(direction)

        camera_points, observed_indices = sample_observe_points(
            full_cloud, camera_position, basis, args.image_width, args.image_height,
            args.fov_y_deg, args.point_radius_px, args.depth_tolerance)
        
        record = {
            "view_id": view_id, "direction": direction.tolist(),
            "camera_position_m": camera_position.tolist(),
            "camera_to_object_rotation": basis.tolist(),
            "visible_count": int(len(observed_indices)),
            "visible_indices": observed_indices.astype(int).tolist(),
        }

        # skip if there not enough points
        if len(observed_indices) < args.min_visible_points:
            record.update(status="skipped_too_few_visible_points", raw_predictions=0)
            view_records.append(record)
            continue

        # predict grasp candidates
        predictions = infer_partial(
            model, decode, camera_points[observed_indices], camera_position, basis, args, rng)
        
        record.update(status="inferred", raw_predictions=int(len(predictions)))
        view_records.append(record)
        batches.append(predictions)
        batch_view_ids.append(np.full(len(predictions), view_id, dtype=np.int32))
        print(f"view={view_id} observed points num={len(observed_indices)}/{len(full_cloud)} raw grasp candidates={len(predictions)}")
    
    if not batches:
        raise ValueError("No view had enough points for GraspNet prediction")

    raw_grasp_candidates = np.concatenate(batches)
    raw_view_ids = np.concatenate(batch_view_ids)

    '''Final candidates are filtred with these rules:
    1. Reject infinite or NaN candidates
    2. Reject candidates that has score lower than arg min score
    3. Reject candidates that has width bigger than OpenArm gripper width (arg max width)
    4. Apply NMS on remains
    5. Check collision after NMS, reject candidates that collision = True
    6. Keep top arg max candidates based on score
    '''
    final_candidates, log = filter_predictions(raw_grasp_candidates, full_cloud, args)
    
    if not len(final_candidates):
        raise ValueError(f"No merged candidates survived filtering: {log}")
    
    candidate_views = find_source_views(final_candidates, raw_grasp_candidates, raw_view_ids)

    label = f"{category}: {npz_file.parent.name}" if category else npz_file.parent.name
    payload = {
        "schema": "multiview-graspnet-candidates/v1",
        "source": {"file": str(npz_file), "object_id": npz_file.parent.name,
                   "object_type": category, "source_schema": source_metadata.get("schema"),
                   "point_count": len(full_cloud), "units": "metres"},
        "metric_scale": scale_metadata,
        "model": {"implementation": str(args.graspnet_root.resolve()),
                  "checkpoint": str(args.checkpoint.resolve()), "epoch": epoch},
        "generation": {
            "method": "perspective_point_splat_z_buffer",
            "warning": "Visibility is approximated from discrete point splats, not a measured depth image or watertight mesh",
            "seed": args.seed, "requested_views": args.views,
            "camera": {"distance_m": distance, "width": args.image_width,
                       "height": args.image_height, "fov_y_deg": args.fov_y_deg,
                       "point_radius_px": args.point_radius_px,
                       "depth_tolerance_m": args.depth_tolerance},
            "views": view_records, "filtering": log,
            "collision_filter_cloud": "complete registered object point cloud",
        },
        "graspnet": {"format": "GraspNetAPI GraspGroup Nx17", "columns": COLUMNS,
                     "rows": final_candidates.tolist(), "source_view_ids": candidate_views},
    }
    output_json = args.output_json or npz_file.with_name("multiview_graspnet_grasps.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(final_candidates)} merged candidates from {len(batches)} views to {output_json}")
    print(f"filtering={log}")

    interactive = not args.output and not args.views_output
    if args.output or interactive:
        figure = draw(full_cloud, affordance, final_candidates, label, min(args.show_candidates, len(final_candidates)))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.output, dpi=args.dpi)
            print(f"Saved {args.output}")
    if args.views_output:
        figure = draw_view_preview(full_cloud, view_records, min(args.preview_views, len(view_records)))
        args.views_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.views_output, dpi=args.dpi)
        print(f"Saved {args.views_output}")
    if interactive:
        plt.show()


if __name__ == "__main__":
    main()
