"""Reconstruct LASO point clouds using multi-radius Ball Pivoting."""

from pathlib import Path
import argparse

import numpy as np
import pymeshlab
from scipy.spatial import cKDTree

from generate_graspnet_grasp import resolve_npz

DEFAULT_DATA_ROOT = Path("data/laso_grasp_only_affordance").resolve()

def auto_ball_pivoting_radii(points: np.ndarray, alpha: float = 2.0,) -> list[float]:
    """Return Ball Pivoting radii as bounding-box percentages."""
    if len(points) < 3:
        raise ValueError("At least three points are required")
    if alpha <= 0:
        raise ValueError("--alpha must be positive")

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    # Column zero is the distance from each point to itself.
    nearest_distances = distances[:, 1]
    median_spacing = float(np.median(nearest_distances))
    bbox_diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if bbox_diagonal <= 0:
        raise ValueError("Point cloud has zero bounding-box diagonal")

    # For example:
    # [base, base + 0.5, ..., base + 2.0]
    base_radius_percent = 100.0 * alpha * median_spacing / bbox_diagonal
    offsets = np.arange(0.0, 2.01, 0.5)
    return [float(base_radius_percent + offset) for offset in offsets]


def reconstruct(npz_file: Path, output_file: Path, args: argparse.Namespace,) -> None:
    """Reconstruct one point cloud and save its mesh."""

    with np.load(npz_file, allow_pickle=False) as data:
        if "points" not in data:
            raise ValueError(f"{npz_file} does not contain points")
        if "normals" not in data:
            raise ValueError(f"{npz_file} does not contain normals")
        points = np.asarray(data["points"], dtype=np.float64)
        normals = np.asarray(data["normals"], dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{npz_file}: points must have shape [N, 3]")
    if normals.shape != points.shape:
        raise ValueError(f"{npz_file}: normals must match points")
    if not np.isfinite(points).all():
        raise ValueError(f"{npz_file}: points contain invalid values")
    if not np.isfinite(normals).all():
        raise ValueError(f"{npz_file}: normals contain invalid values")

    cloud = pymeshlab.Mesh(
        vertex_matrix=points,
        v_normals_matrix=normals,
    )

    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(cloud, "point_cloud")

    radius_percentages = auto_ball_pivoting_radii(points, args.alpha)
    print(f"INFO: {npz_file}: Ball Pivoting radii={radius_percentages}")

    # Progressive, multi-radius Ball Pivoting.
    for index, radius in enumerate(radius_percentages):
        meshes.generate_surface_reconstruction_ball_pivoting(
            ballradius=pymeshlab.PercentageValue(radius),
            deletefaces=index == 0,
        )

    if args.max_hole_size > 0:
        meshes.meshing_close_holes(
            maxholesize=args.max_hole_size,
            selected=False,
            newfaceselected=True,
            selfintersection=True,
            refinehole=True,
        )

    if args.step_smooth > 0:
        meshes.apply_coord_taubin_smoothing(
            stepsmoothnum=args.step_smooth,
            lambda_=0.5,
            mu=-0.53,
        )

    meshes.compute_normal_per_vertex()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    meshes.save_current_mesh(str(output_file))

    mesh = meshes.current_mesh()
    print(f"Saved {output_file}: vertices={mesh.vertex_number()} faces={mesh.face_number()}")

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--object", help="LASO object ID below --data-root")
    source.add_argument("--npz", type=Path, help="Explicit affordance.npz")
    source.add_argument("--all", action="store_true", help="Reconstruct every affordance.npz under --data-root")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--outpath", type=Path, help=(
        "Output directory for a single object. "
        "By default, save beside affordance.npz."))
    p.add_argument("--alpha", type=float, default=2.0, help=(
        "Multiplier applied to median nearest-neighbor "
        "spacing when calculating the initial radius"))
    p.add_argument("--max-hole-size", type=int, default=20, help=(
        "Close boundary loops containing at most this "
        "number of edges; use 0 to disable"))
    p.add_argument("--step-smooth", type=int, default=0, help=("Taubin smoothing iterations; use 0 to disable"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.data_root = args.data_root.resolve()

    if args.max_hole_size < 0:
        raise ValueError("--max-hole-size cannot be negative")
    if args.step_smooth < 0:
        raise ValueError("--step-smooth cannot be negative")
    
    if args.all:
        if args.outpath is not None:
            print("WARNING: --outpath is ignored with --all")
        
        npz_files = sorted(args.data_root.rglob("affordance.npz"))
        if not npz_files:
            raise ValueError(f"No affordance.npz found under {args.data_root}")
        print(f"INFO: Reconstructing {len(npz_files)} point clouds")
        for npz_file in npz_files:
            output_file = npz_file.with_name("reconstructed_mesh.ply")
            try:
                reconstruct(npz_file, output_file, args)
            except Exception as error:
                print(f"FAILED: {npz_file}: {error}")
        return
    else:
        npz_file = resolve_npz(args.data_root, args.object, args.npz)
        if args.outpath is None:
            output_file = npz_file.with_name("reconstructed_mesh.ply")
        else:
            output_file = (args.outpath.resolve() / "reconstructed_mesh.ply")

        reconstruct(npz_file, output_file, args)

if __name__ == "__main__":
    main()