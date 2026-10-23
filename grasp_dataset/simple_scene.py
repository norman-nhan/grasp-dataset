"""Open a converted YCB model.xml together with the OpenArm gripper in MuJoCo."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import trimesh

from .scene import Scene, SHARE, vec
from .viewer import write_png


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = PROJECT / "assets" / "ycb-affordance-mujoco"


def catalog(models_root: Path) -> list[Path]:
    return sorted(path for path in models_root.glob("*/model.xml") if path.is_file())


def resolve_model(models_root: Path, query: str | None, model: Path | None = None) -> Path:
    if model is not None:
        result = model.resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    if not query:
        raise ValueError("--object or --model is required unless --list is used")
    direct = (models_root / query).resolve()
    if direct.is_dir():
        direct = direct / "model.xml"
    if direct.is_file():
        return direct
    matches = [path for path in catalog(models_root) if path.parent.name == query]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No converted model for {query!r} under {models_root}; use --list")
    raise ValueError(f"Converted model name {query!r} is ambiguous")


def load_geometry(model_file: Path) -> tuple[trimesh.Trimesh, dict]:
    metadata_file = model_file.parent / "metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    source = ET.parse(model_file).getroot()
    visual = source.find("asset/mesh[@name='object_visual']")
    if visual is None or not visual.get("file"):
        raise ValueError(f"{model_file} has no object_visual mesh asset")
    visual_file = (model_file.parent / visual.get("file")).resolve()
    mesh = trimesh.load(visual_file, force="scene").to_mesh()
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid visual mesh referenced by {model_file}")
    return mesh, metadata


def namespace_object_model(source: ET.Element, model_file: Path) -> tuple[list[ET.Element], ET.Element]:
    """Copy model assets/body with absolute files and collision defaults applied."""
    source_assets = source.find("asset")
    source_body = source.find("worldbody/body")
    if source_assets is None or source_body is None:
        raise ValueError(f"{model_file} must contain asset and worldbody/body sections")
    prefix = source_body.get("name", model_file.parent.name) + "__"
    name_maps: dict[str, dict[str, str]] = {"mesh": {}, "texture": {}, "material": {}}
    for asset in source_assets:
        if asset.tag in name_maps and asset.get("name"):
            name_maps[asset.tag][asset.get("name")] = prefix + asset.get("name")

    assets = []
    for source_asset in source_assets:
        asset = deepcopy(source_asset)
        if asset.tag in name_maps and asset.get("name"):
            asset.set("name", name_maps[asset.tag][asset.get("name")])
        if asset.get("file"):
            asset.set("file", str((model_file.parent / asset.get("file")).resolve()))
        for attribute, kind in (("texture", "texture"), ("material", "material"), ("mesh", "mesh")):
            if asset.get(attribute) in name_maps[kind]:
                asset.set(attribute, name_maps[kind][asset.get(attribute)])
        assets.append(asset)

    body = deepcopy(source_body)
    collision_defaults = source.find("default/geom")
    for element in body.iter():
        if element is not body and element.get("name"):
            element.set("name", prefix + element.get("name"))
        for attribute, kind in (("material", "material"), ("mesh", "mesh"), ("texture", "texture")):
            if element.get(attribute) in name_maps[kind]:
                element.set(attribute, name_maps[kind][element.get(attribute)])
        if element.tag == "geom" and collision_defaults is not None:
            for attribute, value in collision_defaults.attrib.items():
                if element.get(attribute) is None:
                    element.set(attribute, value)
    return assets, body


def position_gripper(root: ET.Element, object_center: np.ndarray, extent: np.ndarray) -> None:
    """Put the open gripper beside the object with its approach axis facing it."""
    distance = max(0.13, float(extent[1]) / 2 + 0.09)
    position = object_center + np.array([0.0, -distance, max(0.04, extent[2] * 0.35)])
    quaternion = np.array([2**-0.5, -(2**-0.5), 0.0, 0.0])
    for name in ("openarm_right_hand", "hand_target"):
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"OpenArm base model is missing {name}")
        body.set("pos", vec(position))
        body.set("quat", vec(quaternion))


def combined_xml(model_file: Path) -> tuple[str, np.ndarray, np.ndarray, dict]:
    """Merge the converted object MJCF into the corrected gripper-only model."""
    mesh, metadata = load_geometry(model_file)
    object_source = ET.parse(model_file).getroot()
    imported_assets, imported_body = namespace_object_model(object_source, model_file)

    base_scene = Scene(SHARE / "config" / "scene_cube.yaml", support="plane")
    root = ET.fromstring(base_scene.minimal_xml())
    world = root.find("worldbody")
    target = world.find(f"body[@name='{base_scene.name}']")
    if target is None:
        raise ValueError("Could not remove the base scene's placeholder object")
    world.remove(target)
    root.find("asset").extend(imported_assets)
    world.append(imported_body)

    body_position = np.fromstring(imported_body.get("pos", "0 0 0"), sep=" ")
    object_center = mesh.bounds.mean(axis=0) + body_position
    position_gripper(root, object_center, mesh.extents)
    return ET.tostring(root, encoding="unicode"), object_center, mesh.extents, metadata


def build_model(model_file: Path) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, np.ndarray, dict]:
    xml, center, extent, metadata = combined_xml(model_file)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nu:
        data.ctrl[:] = model.actuator_ctrlrange[:, 1]
    mujoco.mj_forward(model, data)
    return model, data, center, extent, metadata


def camera_for(center: np.ndarray, extent: np.ndarray) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = center + np.array([0.0, -0.03, extent[2] * 0.15])
    camera.distance = max(0.42, float(np.linalg.norm(extent)) * 3.5)
    camera.azimuth = 135
    camera.elevation = -25
    return camera


def render_png(model: mujoco.MjModel, data: mujoco.MjData, center: np.ndarray, extent: np.ndarray, path: Path) -> None:
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera_for(center, extent))
        write_png(path, renderer.render())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", help="Converted object ID, for example 025_mug")
    p.add_argument("--model", type=Path, help="Explicit converted model.xml path")
    p.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    p.add_argument("--list", action="store_true", help="List available converted object IDs and exit")
    p.add_argument("--render", type=Path, help="Render one PNG instead of opening the GUI")
    p.add_argument("--save-xml", type=Path, help="Save the combined OpenArm + object MJCF")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    models_root = args.models_root.resolve()
    if args.list:
        try:
            for path in catalog(models_root):
                print(path.parent.name)
        except BrokenPipeError:
            sys.stdout = open("/dev/null", "w")
        return

    model_file = resolve_model(models_root, args.object, args.model)
    xml, center, extent, metadata = combined_xml(model_file)
    if args.save_xml:
        args.save_xml.parent.mkdir(parents=True, exist_ok=True)
        args.save_xml.write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nu:
        data.ctrl[:] = model.actuator_ctrlrange[:, 1]
    mujoco.mj_forward(model, data)

    print(f"object={metadata.get('object_id', model_file.parent.name)} model={model_file}")
    print(
        "geometry=converted_ycb_model "
        f"collision_components={metadata.get('collision_components')} "
        f"downsampling_ratio={metadata.get('downsampling_ratio', 1.0)}"
    )
    if args.render:
        render_png(model, data, center, extent, args.render)
        print(f"Saved {args.render}")
        return

    print("Opening MuJoCo viewer. Space pauses/runs; Esc closes the window.")
    camera = camera_for(center, extent)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
