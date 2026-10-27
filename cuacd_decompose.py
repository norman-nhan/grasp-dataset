"""Decompose non-convex meshes into convex OBJ parts using GPU cuACD."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import cuacd
import numpy as np
import trimesh

from reconstruct_mesh import DEFAULT_DATA_ROOT


def save_obj(parts: list[tuple[np.ndarray, np.ndarray]], outdir: Path) -> None:
    """Save each convex hull as a separate OBJ for MuJoCo collision geoms."""
    outdir.mkdir(parents=True, exist_ok=True)
    for index, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            process=False,
        )
        part_mesh.export(outdir / f"part_{index:02d}.obj")
    print(f"INFO: Saved {len(parts)} convex OBJ files to {outdir.resolve()}")


def normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize a mesh to largest extent 2, as required by cuACD."""
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) / 2.0
    scale = float((upper - lower).max()) / 2.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Mesh has zero or invalid spatial extent")
    normalized = np.ascontiguousarray(
        (vertices - center) / scale,
        dtype=np.float32,
    )
    return normalized, center, scale


def cuacd_decompose_parts(
    file: Path,
    args: argparse.Namespace,
    context: cuacd.Context,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run cuACD and return convex hulls transformed to input coordinates."""
    mesh = trimesh.load(file, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {file}")

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    normalized, center, scale = normalize_vertices(vertices)
    audit = context.check_mesh(normalized, faces)
    print(
        f"INFO: Decomposing {file} "
        f"vertices={len(vertices)}, faces={len(faces)}, "
        f"watertight={audit['watertight']}, "
        f"manifold={audit['manifold']}, oriented={audit['oriented']}"
    )

    should_preprocess = args.preprocess == "on" or (
        args.preprocess == "auto" and audit["needs_remesh"]
    )
    if should_preprocess:
        print(
            "INFO: Remeshing invalid input before decomposition "
            f"at resolution {args.preprocess_resolution}."
        )
        normalized, faces = context.preprocess(
            normalized,
            faces,
            resolution=args.preprocess_resolution,
        )

    parts = context.lookahead_decompose(
        normalized,
        faces,
        threshold=args.threshold,
        max_iters=args.iters,
        width=args.width,
        width2=args.width2,
        depth=args.depth,
        quick_depth=args.quick_depth,
        n_concave_edges=args.n_concave_edges,
        decompose_components=not args.no_decompose_components,
        merge_hulls=not args.no_merge_hulls,
    )
    if not parts:
        raise RuntimeError(f"cuACD produced no parts for {file}")

    # Each result is (cut_vertices, cut_faces, hull_vertices, hull_faces).
    # Export the convex hull, transformed back to the input coordinates.
    hulls = []
    for _, _, hull_vertices, hull_faces in parts:
        world_vertices = (
            np.asarray(hull_vertices, dtype=np.float64) * scale + center
        )
        hulls.append((world_vertices, np.asarray(hull_faces, dtype=np.int32)))
    return hulls


def process_file(
    file: Path,
    outdir: Path,
    args: argparse.Namespace,
    context: cuacd.Context,
) -> None:
    file = file.expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(file)
    parts = cuacd_decompose_parts(file, args, context)
    save_obj(parts, outdir)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--ply", type=Path, help="Explicit reconstructed_mesh.ply")
    source.add_argument(
        "--all",
        action="store_true",
        help="Decompose every object under --data-root",
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument(
        "--outdir",
        type=Path,
        help=(
            "Explicit output directory. By default, save in collision_meshes/ "
            "beside the input mesh. Ignored with --all."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Normalized concavity tolerance; lower values produce more parts",
    )
    p.add_argument("--iters", type=int, default=100, help="Maximum outer iterations")
    p.add_argument("--width", type=int, default=30, help="Full-level candidate cuts")
    p.add_argument("--width2", type=int, default=5, help="Deeper-level candidate cuts")
    p.add_argument("--depth", type=int, default=2, help="Full expansion levels")
    p.add_argument("--quick-depth", type=int, default=0)
    p.add_argument("--n-concave-edges", type=int, default=16)
    p.add_argument(
        "--preprocess",
        choices=("auto", "on", "off"),
        default="auto",
        help="GPU-remesh invalid input automatically, always, or never",
    )
    p.add_argument(
        "--preprocess-resolution",
        type=int,
        default=64,
        help="Power-of-two remeshing resolution",
    )
    p.add_argument("--no-decompose-components", action="store_true")
    p.add_argument("--no-merge-hulls", action="store_true")
    p.add_argument(
        "--device",
        type=int,
        default=-1,
        help="CUDA device ordinal; -1 reuses an existing CUDA context",
    )
    p.add_argument(
        "--pool",
        type=int,
        default=0,
        help="GPU heap pool size in bytes; 0 uses 70%% of free VRAM",
    )
    p.add_argument(
        "--failed-log",
        type=Path,
        default=Path("cuacd_failed_objects.txt"),
        help="Failure log written by --all",
    )
    p.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Do not retry cuACD failures with conservative CPU CoACD settings",
    )
    p.add_argument("--fallback-threshold", type=float, default=0.1)
    p.add_argument("--fallback-iters", type=int, default=5)
    p.add_argument("--fallback-preprocess-resolution", type=int, default=20)
    return p


def single_file_command(file: Path, args: argparse.Namespace) -> list[str]:
    """Build an isolated single-mesh invocation for robust batch processing."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--ply",
        str(file),
        "--threshold",
        str(args.threshold),
        "--iters",
        str(args.iters),
        "--width",
        str(args.width),
        "--width2",
        str(args.width2),
        "--depth",
        str(args.depth),
        "--quick-depth",
        str(args.quick_depth),
        "--n-concave-edges",
        str(args.n_concave_edges),
        "--preprocess",
        args.preprocess,
        "--preprocess-resolution",
        str(args.preprocess_resolution),
        "--device",
        str(args.device),
        "--pool",
        str(args.pool),
    ]
    if args.no_decompose_components:
        command.append("--no-decompose-components")
    if args.no_merge_hulls:
        command.append("--no-merge-hulls")
    return command


def cpu_fallback_command(file: Path, args: argparse.Namespace) -> list[str]:
    """Build the conservative CPU CoACD fallback invocation."""
    return [
        sys.executable,
        str(Path(__file__).with_name("coacd_decompose.py").resolve()),
        "--ply",
        str(file),
        "--threshold",
        str(args.fallback_threshold),
        "--iters",
        str(args.fallback_iters),
        "--preprocess-resolution",
        str(args.fallback_preprocess_resolution),
        "--max-convex-hull",
        "64",
        "--no-merge-hulls",
    ]


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.threshold <= 0.0:
        raise ValueError("--threshold must be positive")

    if args.all:
        if args.outdir is not None:
            print("WARNING: --outdir is ignored with --all")

        data_root = args.data_root.expanduser().resolve()
        ply_files = sorted(data_root.rglob("reconstructed_mesh.ply"))
        if not ply_files:
            raise ValueError(f"No reconstructed_mesh.ply found under {data_root}!")

        done_files = []
        not_done_files = []
        for file in ply_files:
            collision_dir = file.parent / "collision_meshes"
            is_done = collision_dir.is_dir() and any(
                collision_dir.glob("part_*.obj")
            )
            if is_done:
                done_files.append(file)
            else:
                not_done_files.append(file)

        print(f"INFO: Found {len(ply_files)} reconstructed meshes.")
        print(f"INFO: Skipping {len(done_files)} already decomposed meshes.")
        if not not_done_files:
            print(f"INFO: All meshes in {data_root} are already decomposed.")
            return
        print(f"INFO: Decomposing {len(not_done_files)} remaining meshes.")

        # A CUDA illegal-memory-access poisons the current GPU context. Run each
        # object in its own process so one native cuACD failure cannot terminate
        # the complete batch.
        failures = []
        for index, file in enumerate(not_done_files, start=1):
            print(f"[{index}/{len(not_done_files)}] {file}", flush=True)
            result = subprocess.run(single_file_command(file, args), check=False)
            if result.returncode != 0:
                print(
                    f"WARNING: cuACD failed for {file} "
                    f"with exit code {result.returncode}.",
                    flush=True,
                )
                if args.no_cpu_fallback:
                    failures.append((file, result.returncode))
                    continue

                print("INFO: Retrying this mesh with CPU CoACD fallback.", flush=True)
                fallback = subprocess.run(
                    cpu_fallback_command(file, args),
                    check=False,
                )
                if fallback.returncode != 0:
                    failures.append((file, fallback.returncode))
                    print(
                        f"ERROR: CPU fallback also failed for {file}; continuing.",
                        flush=True,
                    )

        failed_log = args.failed_log.expanduser().resolve()
        if failures:
            failed_log.parent.mkdir(parents=True, exist_ok=True)
            failed_log.write_text(
                "".join(f"{file}\texit_code={code}\n" for file, code in failures),
                encoding="utf-8",
            )
            print(
                f"WARNING: {len(failures)} of {len(not_done_files)} meshes "
                f"failed. See {failed_log}"
            )
        else:
            failed_log.parent.mkdir(parents=True, exist_ok=True)
            failed_log.write_text("# No cuACD failures.\n", encoding="utf-8")
            print(f"INFO: Successfully decomposed all {len(not_done_files)} meshes.")
    else:
        file = args.ply.expanduser().resolve()
        if args.outdir is not None:
            outdir = args.outdir.expanduser().resolve()
        else:
            outdir = file.parent / "collision_meshes"

        with cuacd.Context(device=args.device, pool_bytes=args.pool) as context:
            process_file(file, outdir, args, context)


if __name__ == "__main__":
    main()
