"""
Since LASO only contains normalized point cloud.
This program can reconstruct mesh from point cloud using multi-radius Ball Pivoting for LASO dataset.
"""

from pathlib import Path
import argparse

import numpy as np
import pymeshlab
from scipy.spatial import cKDTree

DEFAULT_LASO_ROOT = Path("data/laso_grasp_only_affordance").resolve()

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


def ball_pivot_reconstruct(ply_file: Path, outpath: Path, args: argparse.Namespace,) -> None:
    """Reconstruct mesh from point cloud using bal pivoting method."""
    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(str(ply_file))

    cloud = meshes.current_mesh()
    points = cloud.vertex_matrix()

    # safe guards
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{ply_file}: points must have shape [N, 3]")
    if not np.isfinite(points).all():
        raise ValueError(f"{ply_file}: points contain invalid values")
    # Check that the PLY contains vertex normals.
    if not cloud.has_vertex_normal():
        raise ValueError(f"{ply_file}: point cloud does not contain vertex normals")
    # ====

    radius_percentages = auto_ball_pivoting_radii(points, args.alpha)
    print(f"INFO: {ply_file}: Ball Pivoting radii={radius_percentages}")

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

    outpath.parent.mkdir(parents=True, exist_ok=True)
    meshes.save_current_mesh(str(outpath))

    mesh = meshes.current_mesh()
    print(f"Saved {outpath}: vertices={mesh.vertex_number()} faces={mesh.face_number()}")

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--ply", type=Path, help="Explicit affordance.ply file")
    source.add_argument("--all", action="store_true", help="Reconstruct every object under --data-root")
    p.add_argument("--data-root", type=Path, default=DEFAULT_LASO_ROOT)
    p.add_argument("--outpath", type=Path, help=("Explicit output visual.obj path. By default, save beside ply file."))
    p.add_argument("--alpha", type=float, default=2.0, help=("Multiplier applied to median nearest-neighbor spacing when calculating the initial radius"))
    p.add_argument("--max-hole-size", type=int, default=20, help=("Close boundary loops containing at most this number of edges; use 0 to disable"))
    p.add_argument("--step-smooth", type=int, default=0, help=("Taubin smoothing iterations; use 0 to disable"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    # safe guard
    if args.max_hole_size < 0:
        raise ValueError("--max-hole-size cannot be negative")
    if args.step_smooth < 0:
        raise ValueError("--step-smooth cannot be negative")
    # =====
    if args.all:
        if args.outpath is not None:
            print("WARNING: --outpath is ignored with --all")
        all_files = sorted(data_root.rglob("affordance.ply"))
        if not all_files:
            raise ValueError(f"No affordance.ply found under {args.data_root}")
        # continue from unfinished objects
        for file in all_files:
            if file.parent.rglob("visual.obj"):
                print(f"INFO: Skipping {file} because mesh already reconstructed.")
                continue
            else:
                print(f"INFO: Reconstructing mesh from {file}.")
                try:
                    outpath = file.parent / 'visual.obj'
                    ball_pivot_reconstruct(ply_file=file, outpath=outpath, args=args)
                except Exception as error:
                    print(f"ERROR: {file}: {error}")
        return
    else:
        try:
            file = args.ply
            outpath = file.parent / 'visual.obj'
            ball_pivot_reconstruct(ply_file=file, outpath=outpath, args=args)
        except Exception as error:
            print(f"ERROR: {file}: {error}")

if __name__ == "__main__":
    main()