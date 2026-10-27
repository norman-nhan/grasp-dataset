#!/usr/bin/env python3
"""Check GraspNet candidates with the actual OpenArm gripper in MuJoCo."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import trimesh
import yaml

from grasp_dataset.scene import Scene
from grasp_dataset.simulation import Environment, Physics


GRASPNET_TO_TCP = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])


def metric_mesh(mesh_file: Path, scale: dict) -> trimesh.Trimesh:
    """Scale mesh to real metric [meters]"""
    mesh = trimesh.load(mesh_file, force="mesh", process=True)
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid mesh: {mesh_file}")
    method = scale.get("method")
    if method in ("assumed_longest_extent", "explicit_longest_extent"):
        mesh.vertices = (
            mesh.vertices - np.asarray(scale["source_center"], dtype=float)
        ) * float(scale["scale_to_m"])
    elif method == "source_metric_coordinates":
        pass
    elif method == "similarity_transform":
        payload = json.loads(Path(scale["file"]).read_text())
        if isinstance(payload, dict):
            payload = payload.get("annotation_to_object", payload.get("transform"))
        transform = np.asarray(payload, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("Similarity transform must be 4x4")
        mesh.apply_transform(transform)
    else:
        raise ValueError(f"Unsupported metric_scale method: {method!r}")
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def prepare_scene(mesh_file: Path, collision_dir: Path, scale: dict,
                  asset_dir: Path, object_id: str, overwrite: bool) -> Path:
    config = asset_dir / "object.yaml"
    collision_files = sorted(collision_dir.glob("part_*.obj"))
    if not collision_files:
        raise ValueError(
            f"No part_*.obj collision meshes found in {collision_dir}. "
            "Run coacd_decompose.py first."
        )

    if config.is_file() and not overwrite:
        existing = yaml.safe_load(config.read_text()) or {}
        provenance = existing.get("provenance", {})
        if (
            provenance.get("collision_method") == "precomputed convex OBJ parts"
            and provenance.get("collision_source_dir") == str(collision_dir.resolve())
        ):
            return config

    asset_dir.mkdir(parents=True, exist_ok=True)
    mesh = metric_mesh(mesh_file, scale)
    mesh.export(asset_dir / "surface.obj")

    hulls = []
    for collision_file in collision_files:
        hull = metric_mesh(collision_file, scale)
        if not hull.is_watertight or not hull.is_convex or hull.volume <= 0:
            raise ValueError(
                f"Invalid convex collision component {collision_file}: "
                f"watertight={hull.is_watertight}, convex={hull.is_convex}, "
                f"volume={hull.volume}"
            )
        hulls.append(hull)

    total_volume = sum(h.volume for h in hulls)
    collision = []
    for index, hull in enumerate(hulls):
        filename = f"collision_{index:03d}.obj"
        hull.export(asset_dir / filename)
        collision.append({"file": filename, "mass_fraction": float(hull.volume / total_volume)})

    safe_object_id = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in object_id
    )
    target = {
        "name": f"object_{safe_object_id[-16:]}", "target": True, "movable": True,
        "type": "mesh", "mass": 0.1, "friction": [1.2, 0.01, 0.001],
        "visual_meshes": [{"file": "surface.obj"}],
        "mujoco_collision_meshes": collision,
    }
    provenance = {
        "source_mesh": str(mesh_file.resolve()),
        "source_mesh_watertight": bool(mesh.is_watertight),
        "metric_scale": scale,
        "collision_method": "precomputed convex OBJ parts",
        "collision_source_dir": str(collision_dir.resolve()),
        "collision_source_files": [str(file.resolve()) for file in collision_files],
        "collision_components": len(hulls),
        "warning": "Collision validity depends on reconstructed LASO topology and assumed scale",
    }
    config.write_text(yaml.safe_dump(
        {"standalone": True, "objects": [target], "provenance": provenance},
        sort_keys=False))
    return config


def graspnet_record(row: np.ndarray, env: Environment, index: int) -> dict:
    width, depth = float(row[1]), float(row[3])
    rotation = row[4:13].reshape(3, 3)
    origin = row[13:16]
    contact_center = origin + depth * rotation[:, 0]
    tcp_rotation = rotation @ GRASPNET_TO_TCP
    tcp = np.eye(4)
    tcp[:3, :3] = tcp_rotation
    tcp[:3, 3] = contact_center - tcp_rotation @ env.contact_center_tcp
    contacts = np.vstack((
        contact_center - width / 2 * rotation[:, 1],
        contact_center + width / 2 * rotation[:, 1],
    ))
    return {
        "grasp_id": index,
        "grasp_pose_object": tcp.tolist(),
        "gripper_width": width,
        "contact_points_object": contacts.tolist(),
    }


def view_grasp(env: Environment, record: dict) -> None:
    """Open the native MuJoCo viewer at one candidate's final grasp pose."""
    env.reset()
    target = env.object_pose() @ np.asarray(record["grasp_pose_object"])
    width = min(env.max_width, record["gripper_width"] + 0.012)
    env.place(target, width)

    center = env.data.xpos[env.object].copy()
    extent = np.asarray(env.scene.mesh.extents, dtype=float)
    print(
        f"Opening MuJoCo viewer at grasp {record['grasp_id']}. "
        "Space pauses/runs; Esc closes the window."
    )
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(0.35, float(np.linalg.norm(extent)) * 3.5)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        # Group 0 contains visuals; group 3 contains collision geometry.
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[3] = 0
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.03)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grasps", required=True, type=Path, help="GraspNet annotation JSON")
    p.add_argument("--mesh", type=Path, help="Reconstructed mesh; default beside source NPZ")
    p.add_argument(
        "--collision-dir",
        type=Path,
        help="Directory containing part_*.obj; default collision_meshes/ beside mesh",
    )
    p.add_argument("--asset-dir", type=Path, help="Generated MuJoCo collision asset directory")
    p.add_argument("--output-json", type=Path)
    p.add_argument("--overwrite-assets", action="store_true")
    p.add_argument("--max-candidates", type=int)
    p.add_argument(
        "--view-grasp",
        type=int,
        metavar="INDEX",
        help="Open the native MuJoCo viewer at this source candidate index",
    )
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    source = json.loads(args.grasps.read_text())
    rows = np.asarray(source.get("graspnet", {}).get("rows", []), dtype=float)
    if rows.ndim != 2 or rows.shape[1:] != (17,):
        raise ValueError("Input JSON must contain graspnet.rows with shape [N,17]")
    if args.max_candidates is not None:
        rows = rows[:args.max_candidates]
    npz_file = Path(source["source"]["file"])
    mesh_file = (args.mesh or npz_file.with_name("reconstructed_mesh.ply")).resolve()
    if not mesh_file.is_file():
        raise FileNotFoundError(mesh_file)
    collision_dir = (
        args.collision_dir or mesh_file.parent / "collision_meshes"
    ).expanduser().resolve()
    if not collision_dir.is_dir():
        raise FileNotFoundError(collision_dir)
    object_id = str(source["source"]["object_id"])
    asset_dir = (args.asset_dir or
                 args.grasps.parent / f"{args.grasps.stem}_mujoco_assets").resolve()
    scene_file = prepare_scene(
        mesh_file,
        collision_dir,
        source["metric_scale"],
        asset_dir,
        object_id,
        args.overwrite_assets,
    )
    scene = Scene(scene_file, environment="minimal", support="none")
    env = Environment(scene, Physics())

    evaluations = []
    retained = []
    for index, row in enumerate(rows):
        record = graspnet_record(row, env, index)
        report = env.screen_record(record)
        evaluations.append({
            "source_index": index,
            "accepted": bool(report["valid"]),
            "rejection_reasons": report["rejection_reasons"],
            "gripper_width_m": record["gripper_width"],
            "grasp_pose_object": record["grasp_pose_object"],
            "approach_screen": report,
        })
        if report["valid"]:
            retained.append(index)
        print(f"grasp={index} collision_free={report['valid']} "
              f"reasons={report['rejection_reasons']}", flush=True)

    reasons = Counter(reason for item in evaluations for reason in item["rejection_reasons"])
    result = deepcopy(source)
    result["schema"] = "graspnet-mujoco-collision-evaluation/v1"
    result["mujoco_collision"] = {
        "source_mesh": str(mesh_file),
        "collision_mesh_dir": str(collision_dir),
        "scene": str(scene_file),
        "gripper": "actual OpenArm right-gripper collision geometry",
        "pose_conversion": "GraspNet +X approach/+Y closing to TCP +Z approach/+Y closing",
        "rows": rows[retained].tolist(),
        "source_indices": retained,
        "summary": {
            "input": len(rows), "accepted": len(retained),
            "rejected": len(rows) - len(retained),
            "rejection_reason_counts": dict(sorted(reasons.items())),
        },
        "candidates": evaluations,
    }
    output = args.output_json or args.grasps.with_name(args.grasps.stem + "_mujoco_collision.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(retained)}/{len(rows)} collision-free grasps to {output}")

    if args.view_grasp is not None:
        if not 0 <= args.view_grasp < len(rows):
            raise ValueError(
                f"--view-grasp must be between 0 and {len(rows) - 1}"
            )
        view_grasp(
            env,
            graspnet_record(rows[args.view_grasp], env, args.view_grasp),
        )


if __name__ == "__main__":
    main()
