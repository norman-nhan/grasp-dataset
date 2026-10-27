'''Decompose non-convex mesh using CoACD.'''
import re

from networkx import is_empty
import numpy as np
import coacd
from reconstruct_mesh import DEFAULT_DATA_ROOT
import trimesh
from pathlib import Path
import argparse


def save_obj(parts, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    for index, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            process=False,
        )
        outpath = outdir / f'part_{index:02d}.obj'
        part_mesh.export(outpath)
    print(f'Saved {len(parts)} convex OBJ files to {outdir.resolve()}')


def coacd_decompose_parts(file, args):
    mesh = trimesh.load(file, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {file}")
    print(f"INFO: Decomposing {file} vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")
    
    coacd_mesh = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        preprocess_resolution=args.preprocess_resolution,
        resolution=1500,
        mcts_nodes=10,
        mcts_iterations=args.iters,
        mcts_max_depth=3,
        merge=not args.no_merge_hulls,
        seed=args.seed,
    )

    if not parts:
        raise RuntimeError(f"CoACD produced no parts for {file}")

    # Or use real metric mode (threshold in meters)
    # parts = coacd.run_coacd(mesh, threshold=0.01, real_metric=True)
    return parts

def process_file(file: Path, outdir: Path, args) -> None:
    file = file.expanduser().resolve()

    if not file.is_file():
        raise FileNotFoundError(file)

    parts = coacd_decompose_parts(file=file, args=args)
    save_obj(parts=parts, outdir=outdir)

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--ply", type=Path, help="Explicit reconstructed_mesh.ply")
    source.add_argument("--all", action="store_true", help="Decompose every object under --data-root")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--outdir", type=Path, help="Explicit output dir path. By default, save in `collision_meshes/` beside mesh file.")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--max-convex-hull", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.03, help="CoACD decomposition threshold")
    p.add_argument("--preprocess-resolution", type=int, default=40)
    p.add_argument("--no-merge-hulls", action="store_true")
    p.add_argument("--iters", type=int, default=30, help="Number of MCTS iterations")
    return p

def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)

    if args.all:
        if args.outdir is not None:
            print("WARNING: --outdir is ignored with --all")

        data_root = args.data_root.expanduser().resolve()
        ply_files = sorted(data_root.rglob("reconstructed_mesh.ply"))
        if not ply_files:
            raise ValueError(f"No reconstructed_mesh.ply found under {data_root}!")
        # Checking if object mesh is already decomposed
        done_files = []
        not_done_files = []
        for file in ply_files:
            collision_dir = file.parent / "collision_meshes"
            is_done = collision_dir.is_dir() and any(collision_dir.glob("part_*.obj"))
            if is_done:
                done_files.append(file)
            else:
                not_done_files.append(file)

        print(f"INFO: Found {len(ply_files)} reconstructed meshes.")
        print(f"INFO: Skipping {len(done_files)} already decomposed meshes.")
        if not not_done_files: # end program if there is no file to decompose
            print(f'INFO: All meshes in {data_root} already decomposed.')
            return
        
        print(f"INFO: Decomposing {len(not_done_files)} remaining meshes.")
        
        for index, file in enumerate(not_done_files, start=1):
            print(f"[{index}/{len(not_done_files)}] {file}")

            outdir = file.parent / "collision_meshes"
            process_file(file, outdir, args)

    else:
        file = args.ply.expanduser().resolve()

        if args.outdir is not None:
            outdir = args.outdir.expanduser().resolve()
        else:
            outdir = file.parent / "collision_meshes"

        process_file(file, outdir, args)
        

if __name__ == "__main__":
    main()
