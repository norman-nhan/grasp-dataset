#!/usr/bin/env python3
"""Create metric MuJoCo model.xml assets from prepared LASO meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from reconstruct_mesh import DEFAULT_LASO_ROOT


PROJECT = Path(__file__).resolve().parent
DEFAULT_EXTENTS = PROJECT / "laso_physical_dims_assumed.json"


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in value)


def source_identity(object_dir: Path) -> tuple[str, str]:
    metadata_file = object_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    category = str(metadata.get("class", object_dir.parent.name))
    object_id = str(metadata.get("object_id", object_dir.name))
    return category, object_id


def reference_geometry(object_dir: Path, visual: trimesh.Trimesh) -> tuple[np.ndarray, str]:
    """Use the original LASO points so scaling matches grasp generation."""
    npz_file = object_dir / "affordance.npz"
    if npz_file.is_file():
        with np.load(npz_file, allow_pickle=False) as data:
            points = np.asarray(data["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid points in {npz_file}")
        return points, str(npz_file.resolve())
    return np.asarray(visual.vertices, dtype=np.float64), str((object_dir / "visual.obj").resolve())


def collision_files(object_dir: Path) -> list[Path]:
    candidates = [
        sorted((object_dir / "collision_meshes").glob("part_*.obj")),
        sorted((object_dir / "collision_meshes").glob("collision_*.obj")),
        sorted(object_dir.glob("collision_*.obj")),
    ]
    for files in candidates:
        if files:
            return files
    raise ValueError(
        f"No collision OBJ files found for {object_dir}; expected "
        "collision_meshes/part_*.obj or collision_*.obj"
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=True)
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid mesh: {path}")
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def transform_mesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> None:
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * scale


def write_model(
    output_dir: Path,
    model_name: str,
    visual: trimesh.Trimesh,
    hulls: list[trimesh.Trimesh],
    mass_kg: float,
    friction: list[float],
) -> Path:
    visual.export(output_dir / "visual.obj")
    volumes = np.asarray([hull.volume for hull in hulls], dtype=float)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Collision components must have positive finite volume")
    mass_fractions = volumes / volumes.sum()

    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(
        root, "compiler", angle="radian", meshdir=".", autolimits="true"
    )
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        gravity="0 0 -9.81",
    )
    visual_settings = ET.SubElement(root, "visual")
    ET.SubElement(
        visual_settings,
        "headlight",
        ambient="0.35 0.35 0.35",
        diffuse="0.75 0.75 0.75",
        specular="0.2 0.2 0.2",
    )
    defaults = ET.SubElement(root, "default")
    ET.SubElement(
        defaults,
        "geom",
        condim="4",
        friction=" ".join(str(value) for value in friction),
        solref="0.01 1",
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="object_visual", file="visual.obj")
    for index, hull in enumerate(hulls):
        filename = f"collision_{index:03d}.obj"
        hull.export(output_dir / filename)
        ET.SubElement(
            asset, "mesh", name=f"collision_{index:03d}", file=filename
        )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="key",
        pos="-0.5 -0.5 1.5",
        dir="0.3 0.3 -1",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0.6 0.6 0.02",
        rgba="0.3 0.3 0.32 1",
    )
    ET.SubElement(
        world,
        "camera",
        name="overview",
        pos="0.45 -0.45 0.35",
        xyaxes="0.707 0.707 0 -0.35 0.35 0.866",
    )
    body_height = -float(visual.bounds[0, 2]) + 0.005
    body = ET.SubElement(
        world, "body", name=model_name, pos=f"0 0 {body_height:.12g}"
    )
    ET.SubElement(body, "freejoint", name=f"{model_name}_freejoint")
    ET.SubElement(
        body,
        "geom",
        name=f"{model_name}_visual",
        type="mesh",
        mesh="object_visual",
        contype="0",
        conaffinity="0",
        mass="0",
        rgba="0.65 0.7 0.75 1",
        group="2",
    )
    for index, fraction in enumerate(mass_fractions):
        ET.SubElement(
            body,
            "geom",
            name=f"{model_name}_collision_{index:03d}",
            type="mesh",
            mesh=f"collision_{index:03d}",
            mass=f"{mass_kg * fraction:.12g}",
            rgba="0.7 0.7 0.7 0",
            group="3",
        )

    ET.indent(root, space="  ")
    model_file = output_dir / "model.xml"
    ET.ElementTree(root).write(model_file, encoding="unicode")
    return model_file


def convert_object(
    visual_file: Path,
    output_dir: Path,
    extent_m: float,
    mass_kg: float,
    friction: list[float],
) -> dict:
    object_dir = visual_file.parent
    category, object_id = source_identity(object_dir)
    visual = load_mesh(visual_file)
    points, reference_file = reference_geometry(object_dir, visual)
    center = points.mean(axis=0)
    source_extent = float(np.ptp(points, axis=0).max())
    if source_extent <= 1e-12:
        raise ValueError(f"Degenerate source extent for {object_dir}")
    scale = extent_m / source_extent

    files = collision_files(object_dir)
    hulls = []
    for file in files:
        hull = load_mesh(file)
        transform_mesh(hull, center, scale)
        if not hull.is_watertight or not hull.is_convex or hull.volume <= 0:
            raise ValueError(
                f"Invalid collision component {file}: watertight={hull.is_watertight}, "
                f"convex={hull.is_convex}, volume={hull.volume}"
            )
        hulls.append(hull)
    transform_mesh(visual, center, scale)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(f"laso_{category}_{object_id}")
    model_file = write_model(
        output_dir, model_name, visual, hulls, mass_kg, friction
    )
    compiled = mujoco.MjModel.from_xml_path(str(model_file))

    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    metadata = {
        "schema": "openarm-laso-mujoco-object/v1",
        "object_id": object_id,
        "class": category,
        "model": "model.xml",
        "units": "metres",
        "source_visual_mesh": str(visual_file.resolve()),
        "source_collision_meshes": [str(file.resolve()) for file in files],
        "scale_reference": reference_file,
        "scale_method": "assumed longest extent",
        "assumed_longest_extent_m": extent_m,
        "source_center": center.tolist(),
        "source_extent": source_extent,
        "scale_to_m": scale,
        "source_to_object": transform.tolist(),
        "mesh_extents_m": visual.extents.tolist(),
        "mass_kg": mass_kg,
        "mass_source": "configured simulation assumption; not measured LASO mass",
        "friction": friction,
        "friction_source": "configured simulation assumption; not measured LASO material",
        "collision_method": "precomputed convex decomposition",
        "collision_components": len(hulls),
        "compiled_nbody": int(compiled.nbody),
        "compiled_ngeom": int(compiled.ngeom),
        "limitations": [
            "Physical scale is class-level assumed, not measured per object.",
            "Mass, friction and inertia are simulation assumptions.",
            "Collision accuracy depends on reconstructed topology and decomposition.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_LASO_ROOT)
    p.add_argument("--class-extents", type=Path, default=DEFAULT_EXTENTS)
    p.add_argument(
        "--longest-extent-m",
        type=float,
        help="Override the assumed longest extent for every selected object",
    )
    p.add_argument("--mass-kg", type=float, default=0.1)
    p.add_argument(
        "--friction",
        type=float,
        nargs=3,
        default=[1.2, 0.01, 0.001],
        metavar=("SLIDE", "TORSION", "ROLL"),
    )
    p.add_argument(
        "--output-dir-name",
        default="mujoco",
        help="Output subdirectory created beside each visual.obj",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    if args.mass_kg <= 0 or not np.isfinite(args.mass_kg):
        raise ValueError("--mass-kg must be positive and finite")
    if any(value < 0 or not np.isfinite(value) for value in args.friction):
        raise ValueError("--friction values must be nonnegative and finite")
    if args.longest_extent_m is not None and (
        args.longest_extent_m <= 0 or not np.isfinite(args.longest_extent_m)
    ):
        raise ValueError("--longest-extent-m must be positive and finite")

    class_extents = json.loads(args.class_extents.expanduser().read_text())
    visual_files = sorted(data_root.rglob("visual.obj"))
    visual_files = [
        file for file in visual_files if args.output_dir_name not in file.parts
    ]
    if not visual_files:
        raise ValueError(f"No visual.obj found under {data_root}")

    pending = []
    skipped = 0
    for visual_file in visual_files:
        output_dir = visual_file.parent / args.output_dir_name
        if (output_dir / "model.xml").is_file() and not args.overwrite:
            skipped += 1
        else:
            pending.append((visual_file, output_dir))

    print(f"INFO: Found {len(visual_files)} LASO visual meshes.")
    print(f"INFO: Skipping {skipped} existing MuJoCo models.")
    print(f"INFO: Converting {len(pending)} remaining objects.")

    failures = []
    converted = 0
    for index, (visual_file, output_dir) in enumerate(pending, start=1):
        category, object_id = source_identity(visual_file.parent)
        extent = args.longest_extent_m
        if extent is None:
            if category not in class_extents:
                failures.append((visual_file, f"No assumed extent for class {category!r}"))
                continue
            extent = float(class_extents[category])
        print(
            f"[{index}/{len(pending)}] class={category} object={object_id} "
            f"longest_extent_m={extent}"
        )
        try:
            metadata = convert_object(
                visual_file,
                output_dir,
                extent,
                args.mass_kg,
                list(args.friction),
            )
            converted += 1
            print(
                f"INFO: Saved {metadata['collision_components']} collision parts "
                f"to {output_dir}"
            )
        except Exception as error:
            failures.append((visual_file, str(error)))
            print(f"ERROR: {visual_file}: {error}")

    print(
        f"INFO: Converted {converted}, skipped {skipped}, failed {len(failures)}."
    )
    if failures:
        failure_file = data_root / "laso_mujoco_failed_objects.txt"
        failure_file.write_text(
            "".join(f"{file}\t{reason}\n" for file, reason in failures),
            encoding="utf-8",
        )
        print(f"WARNING: See {failure_file}")


if __name__ == "__main__":
    main()
