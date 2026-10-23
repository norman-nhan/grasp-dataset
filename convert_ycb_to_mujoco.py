#!/usr/bin/env python3
"""Convert downloaded YCB Google-16k meshes into directly loadable MuJoCo models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import coacd
import mujoco
import numpy as np
import trimesh


PROJECT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT / "assets" / "ycb-affordance"
DEFAULT_OUTPUT = PROJECT / "assets" / "ycb-affordance-mujoco"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indent(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False) + "\n"


def load_mesh(source: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(source, force="scene").to_mesh()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid or empty mesh: {source}")
    return mesh


def downsample_collision_mesh(mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
    """Simplify only the mesh passed to CoACD; preserve the visual mesh."""
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("--downsampling-ratio must be in the interval (0, 1]")
    if ratio == 1:
        return mesh.copy()
    target_faces = max(4, int(round(len(mesh.faces) * ratio)))
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    simplified.update_faces(simplified.nondegenerate_faces())
    simplified.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(simplified, multibody=True)
    if len(simplified.faces) < 4 or not np.isfinite(simplified.vertices).all():
        raise ValueError("Mesh downsampling produced invalid collision geometry")
    return simplified


def convex_decompose(mesh: trimesh.Trimesh, output: Path, seed: int) -> list[dict]:
    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=0.05,
        preprocess_resolution=40,
        resolution=2000,
        mcts_nodes=10,
        mcts_iterations=30,
        mcts_max_depth=3,
        max_convex_hull=32,
        seed=seed,
    )
    results = []
    volumes = []
    meshes = []
    for vertices, faces in parts:
        part = trimesh.Trimesh(vertices, faces, process=False)
        volume = abs(float(part.volume))
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("CoACD produced a collision component with invalid volume")
        meshes.append(part)
        volumes.append(volume)
    if not meshes:
        raise ValueError("CoACD produced no collision components")
    fractions = np.asarray(volumes) / np.sum(volumes)
    for index, (part, fraction) in enumerate(zip(meshes, fractions)):
        filename = f"collision_{index:03d}.stl"
        part.export(output / filename)
        results.append({"file": filename, "mass_fraction": float(fraction)})
    return results


def build_xml(object_id: str, mesh: trimesh.Trimesh, parts: list[dict], mass: float) -> str:
    root = ET.Element("mujoco", model=object_id)
    ET.SubElement(root, "compiler", angle="radian", meshdir=".", texturedir=".", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.75 0.75 0.75", specular="0.2 0.2 0.2")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")

    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", condim="4", friction="1.0 0.01 0.001", solref="0.01 1")

    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", name="object_texture", type="2d", file="texture_map.png")
    ET.SubElement(assets, "material", name="object_material", texture="object_texture", specular="0.15", shininess="0.2")
    ET.SubElement(assets, "mesh", name="object_visual", file="textured.obj")
    for index, part in enumerate(parts):
        ET.SubElement(assets, "mesh", name=f"collision_{index:03d}", file=part["file"])

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="-0.5 -0.5 1.5", dir="0.3 0.3 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0.6 0.6 0.02", rgba="0.3 0.3 0.32 1")
    ET.SubElement(world, "camera", name="overview", pos="0.45 -0.45 0.35", xyaxes="0.707 0.707 0 -0.35 0.35 0.866")

    z_offset = -float(mesh.bounds[0, 2]) + 0.002
    body = ET.SubElement(world, "body", name=object_id, pos=f"0 0 {z_offset:.9g}")
    ET.SubElement(body, "freejoint", name=f"{object_id}_freejoint")
    ET.SubElement(
        body,
        "geom",
        name=f"{object_id}_visual",
        type="mesh",
        mesh="object_visual",
        material="object_material",
        contype="0",
        conaffinity="0",
        mass="0",
        group="2",
    )
    for index, part in enumerate(parts):
        ET.SubElement(
            body,
            "geom",
            name=f"{object_id}_collision_{index:03d}",
            type="mesh",
            mesh=f"collision_{index:03d}",
            mass=f"{mass * part['mass_fraction']:.12g}",
            rgba="0.7 0.7 0.7 0.0",
            group="3",
        )
    return indent(root)


def convert_object(
    source_dir: Path,
    output_root: Path,
    mass: float,
    seed: int,
    downsampling_ratio: float,
    overwrite: bool,
) -> dict:
    object_id = source_dir.name
    google = source_dir / "google_16k"
    required = [google / "textured.obj", google / "textured.mtl", google / "texture_map.png"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Google-16k assets: " + ", ".join(missing))

    output = output_root / object_id
    model_file = output / "model.xml"
    if output.exists() and not overwrite:
        if model_file.is_file():
            metadata_file = output / "metadata.json"
            metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
            expected = (mass, seed, downsampling_ratio)
            actual = (
                metadata.get("mass_kg"),
                metadata.get("coacd_seed"),
                metadata.get("downsampling_ratio", 1.0),
            )
            if actual != expected:
                raise ValueError(
                    f"Existing conversion settings {actual} differ from requested {expected}; "
                    "use --overwrite"
                )
            mujoco.MjModel.from_xml_path(str(model_file))
            return {"object_id": object_id, "status": "existing", "model": str(model_file.resolve())}
        raise FileExistsError(f"Incomplete output exists: {output}; use --overwrite")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for source in required:
        shutil.copy2(source, output / source.name)
    mesh = load_mesh(output / "textured.obj")
    collision_input = downsample_collision_mesh(mesh, downsampling_ratio)
    parts = convex_decompose(collision_input, output, seed)
    model_file.write_text(build_xml(object_id, mesh, parts, mass))

    # Loading is the authoritative structural validation for generated MJCF.
    model = mujoco.MjModel.from_xml_path(str(model_file))
    metadata = {
        "schema": "openarm-ycb-mujoco-object/v1",
        "object_id": object_id,
        "source_directory": str(source_dir.resolve()),
        "source_mesh_sha256": sha256(required[0]),
        "model": "model.xml",
        "units": "metres",
        "mass_kg": mass,
        "mass_source": "configured simulation assumption; not measured YCB mass",
        "visual_mesh": "textured.obj",
        "texture": "texture_map.png",
        "collision_method": "CoACD convex decomposition",
        "collision_components": len(parts),
        "coacd_version": importlib.metadata.version("coacd"),
        "coacd_seed": seed,
        "downsampling_ratio": downsampling_ratio,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "collision_input_vertices": int(len(collision_input.vertices)),
        "collision_input_faces": int(len(collision_input.faces)),
        "mesh_extents_m": mesh.extents.tolist(),
        "mujoco_version": mujoco.__version__,
        "compiled_nbody": int(model.nbody),
        "compiled_ngeom": int(model.ngeom),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "object_id": object_id,
        "status": "converted",
        "model": str(model_file.resolve()),
        "collision_components": len(parts),
    }


def discover(input_root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(input_root.iterdir())
        if path.is_dir() and (path / "google_16k").is_dir()
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--objects", nargs="+", help="Object IDs; default converts every discovered object")
    p.add_argument("--mass", type=float, default=0.1, help="Assumed mass in kg for every object")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--downsampling-ratio",
        type=float,
        default=1.0,
        help="Target fraction of visual-mesh faces used as CoACD input, in (0, 1]; visual mesh stays unchanged",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not np.isfinite(args.mass) or args.mass <= 0:
        raise ValueError("--mass must be positive and finite")
    if not np.isfinite(args.downsampling_ratio) or not 0 < args.downsampling_ratio <= 1:
        raise ValueError("--downsampling-ratio must be in the interval (0, 1]")
    available = discover(input_root)
    requested = args.objects or sorted(available)
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError("Requested objects are not downloaded: " + ", ".join(missing))
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for object_id in requested:
        print(f"Converting {object_id}...", flush=True)
        try:
            result = convert_object(
                available[object_id],
                output_root,
                args.mass,
                args.seed,
                args.downsampling_ratio,
                args.overwrite,
            )
        except Exception as error:
            result = {"object_id": object_id, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    manifest = {
        "schema": "openarm-ycb-mujoco-manifest/v1",
        "input": str(input_root),
        "output": str(output_root),
        "mass_kg": args.mass,
        "seed": args.seed,
        "downsampling_ratio": args.downsampling_ratio,
        "requested": requested,
        "results": results,
        "counts": {
            "requested": len(results),
            "converted": sum(item["status"] == "converted" for item in results),
            "existing": sum(item["status"] == "existing" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
    if manifest["counts"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
