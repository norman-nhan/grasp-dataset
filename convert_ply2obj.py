"""Convert reconstructed PLY meshes into OBJ visual meshes for MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pymeshlab

from reconstruct_mesh import DEFAULT_LASO_ROOT


def convert_mesh(source: Path, destination: Path) -> None:
    """Load one PLY with PyMeshLab and save it as an OBJ without rescaling."""
    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(str(source))

    mesh = meshes.current_mesh()
    if mesh.vertex_number() == 0 or mesh.face_number() == 0:
        raise ValueError(
            f"Mesh is empty: vertices={mesh.vertex_number()}, "
            f"faces={mesh.face_number()}"
        )

    meshes.save_current_mesh(str(destination))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_LASO_ROOT)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace visual.obj files that already exist",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)

    ply_files = sorted(data_root.rglob("reconstructed_mesh.ply"))
    if not ply_files:
        raise ValueError(f"No reconstructed_mesh.ply found under {data_root}")

    pending = [
        source
        for source in ply_files
        if args.overwrite or not (source.parent / "visual.obj").is_file()
    ]
    skipped = len(ply_files) - len(pending)

    print(f"INFO: Found {len(ply_files)} reconstructed meshes.")
    print(f"INFO: Skipping {skipped} existing visual OBJ files.")
    print(f"INFO: Converting {len(pending)} remaining meshes.")

    failures: list[tuple[Path, str]] = []
    for index, source in enumerate(pending, start=1):
        destination = source.parent / "visual.obj"
        print(f"[{index}/{len(pending)}] {source} -> {destination}")
        try:
            convert_mesh(source, destination)
        except Exception as error:
            failures.append((source, str(error)))
            print(f"ERROR: {source}: {error}")

    converted = len(pending) - len(failures)
    print(
        f"INFO: Converted {converted}, skipped {skipped}, "
        f"failed {len(failures)}."
    )
    if failures:
        details = "\n".join(f"  {path}: {reason}" for path, reason in failures)
        raise RuntimeError(f"Failed to convert {len(failures)} meshes:\n{details}")


if __name__ == "__main__":
    main()
