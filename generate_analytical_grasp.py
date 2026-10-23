#!/usr/bin/env python3
"""Generate and visualize analytical grasps from an affordance point cloud."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import numpy as np

# Keep Matplotlib's cache out of a potentially read-only container home.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"openarm-matplotlib-{os.getuid()}")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

from grasp_dataset.geometry import Sampling
from grasp_dataset.laso_catalog import load as load_catalog, object_type
from grasp_dataset.pc_geometry import generate, load_registered


PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT / "data"
DEFAULT_OBJECT_TYPES = PROJECT / "laso_object_types.json"


def resolve_npz(data_root: Path, object_id: str | None, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if not object_id:
        raise ValueError("Supply --object or --npz")
    direct = data_root / object_id
    if direct.is_dir():
        direct = direct / "affordance.npz"
    if direct.is_file():
        return direct.resolve()
    matches = list(data_root.glob(f"**/{object_id}/affordance.npz"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one affordance.npz for {object_id!r} under {data_root}, "
            f"found {len(matches)}; use --npz"
        )
    return matches[0].resolve()


def load_source(npz_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_file, allow_pickle=False) as archive:
        if "points" not in archive or "normals" not in archive:
            raise ValueError(f"{npz_file} must contain 'points' and 'normals'")
        points = np.asarray(archive["points"], dtype=np.float64)
        normals = np.asarray(archive["normals"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape [N, 3]")
    if len(points) < 2 or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("point cloud must contain at least two finite points and normals")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("point cloud contains zero-length normals")
    return points, normals / lengths[:, None]


class AnalyticalGripper:
    """Only the geometric quantities needed to construct candidate poses."""

    def __init__(self, min_width: float, max_width: float, contact_center_tcp: np.ndarray):
        self.min_width = min_width
        self.max_width = max_width
        self.contact_center_tcp = contact_center_tcp


def affordance_scores(npz_file: Path, count: int) -> np.ndarray | None:
    with np.load(npz_file, allow_pickle=False) as archive:
        if "affordance_score" not in archive:
            return None
        scores = np.asarray(archive["affordance_score"], dtype=float).reshape(-1)
    return np.clip(scores, 0, 1) if len(scores) == count else None


def equal_axes(axis, points: np.ndarray) -> None:
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * 0.65, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def selected_records(records: list[dict], grasp_ids: list[int] | None, maximum: int) -> list[dict]:
    if grasp_ids:
        by_id = {record["grasp_id"]: record for record in records}
        missing = [value for value in grasp_ids if value not in by_id]
        if missing:
            raise ValueError(f"Unknown --grasp-id values: {missing}")
        return [by_id[value] for value in grasp_ids]
    if maximum < 1:
        raise ValueError("--show-candidates must be positive")
    return records[:maximum]


def draw(points: np.ndarray, scores: np.ndarray | None, records: list[dict], axis_length: float, units: str, label: str):
    figure = plt.figure(figsize=(11, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=scores if scores is not None else "#8ca6b8",
        cmap="turbo" if scores is not None else None,
        vmin=0 if scores is not None else None,
        vmax=1 if scores is not None else None,
        s=3, alpha=0.55, depthshade=False,
    )
    if scores is not None:
        figure.colorbar(scatter, ax=axis, shrink=0.65, pad=0.08, label="Human grasp affordance")

    colors = plt.colormaps["tab20"](np.linspace(0, 1, max(len(records), 1)))
    for color, record in zip(colors, records):
        contacts = np.asarray(record["contact_points_object"], dtype=float)
        normals = np.asarray(record["contact_normals_object"], dtype=float)
        pose = np.asarray(record["grasp_pose_object"], dtype=float)
        midpoint = contacts.mean(axis=0)
        tcp = pose[:3, 3]
        approach = pose[:3, 2]
        opening_axis = pose[:3, 1]

        # Closing line and both selected surface samples.
        axis.plot(*contacts.T, color=color, linewidth=2.4)
        axis.scatter(*contacts.T, color=[color], s=30, depthshade=False)
        # Contact normals point into the free space expected by the fingers.
        axis.quiver(*contacts[0], *normals[0], length=axis_length * 0.55,
                    normalize=True, color=color, linewidth=1.0)
        axis.quiver(*contacts[1], *normals[1], length=axis_length * 0.55,
                    normalize=True, color=color, linewidth=1.0)
        # TCP-to-contact-center offset, approach (+Z), and closing (+Y) axes.
        axis.plot(*np.vstack((tcp, midpoint)).T, color=color, linestyle=":", linewidth=1.0)
        axis.quiver(*tcp, *approach, length=axis_length, normalize=True,
                    color=color, linewidth=2.0)
        axis.quiver(*tcp, *opening_axis, length=axis_length * 0.7, normalize=True,
                    color=color, linewidth=1.2, linestyle="--")
        axis.text(*tcp, str(record["grasp_id"]), color=color, fontsize=8)

    axis.set_title(
        f"{label} — analytical point-cloud grasps ({len(records)} displayed)\n"
        "solid: contacts/closing line; long arrow: +Z approach; dashed arrow: +Y closing axis"
    )
    axis.set_xlabel(f"object X [{units}]")
    axis.set_ylabel(f"object Y [{units}]")
    axis.set_zlabel(f"object Z [{units}]")
    equal_axes(axis, points)
    return figure


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", help="Point-cloud object ID below --data-root")
    p.add_argument("--npz", type=Path, help="Explicit affordance.npz")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--object-types", type=Path, default=DEFAULT_OBJECT_TYPES)
    p.add_argument("--transform", type=Path, help="Optional source-to-object 4x4 JSON; when omitted, use normalized source coordinates")
    p.add_argument("--num-candidates", type=int, default=100)
    p.add_argument("--surface-points", type=int, default=2500)
    p.add_argument("--approaches", type=int, default=8)
    p.add_argument("--neighbors", type=int, default=96)
    p.add_argument("--pairs-per-point", type=int, default=1)
    p.add_argument("--min-width", type=float, default=0.0, help="Minimum contact separation; units follow the point cloud")
    p.add_argument("--max-width", type=float, help="Maximum contact separation; default is the point-cloud extent")
    p.add_argument("--contact-center-tcp", type=float, nargs=3, default=(0, 0, 0), metavar=("X", "Y", "Z"), help="Optional TCP-to-contact-center offset; default places TCP at contact midpoint")
    p.add_argument("--friction", type=float, default=0.8)
    p.add_argument("--angular-tolerance-deg", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oriented-normals", action="store_true", help="Trust stored normal signs instead of pair-aligning PCA normals")
    p.add_argument("--grasp-id", type=int, action="append", help="Display this grasp ID; repeat to select several")
    p.add_argument("--show-candidates", type=int, default=12, help="Number displayed when --grasp-id is omitted")
    p.add_argument("--axis-length", type=float, help="Pose-axis length in point units; default 18%% of object extent")
    p.add_argument("--output-json", type=Path, help="Save all analytical grasp records")
    p.add_argument("--output", type=Path, help="Save visualization PNG instead of opening a window")
    p.add_argument("--dpi", type=int, default=180)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    npz_file = resolve_npz(args.data_root.resolve(), args.object, args.npz)
    category = None
    if args.object_types.is_file():
        category = object_type(npz_file.parent.name, load_catalog(args.object_types))
    if args.transform:
        transform_file = args.transform.resolve()
        if not transform_file.is_file():
            raise FileNotFoundError(transform_file)
        points, normals = load_registered(npz_file, transform_file)
        units = "m"
    else:
        transform_file = None
        points, normals = load_source(npz_file)
        units = "source units"
    extent = float(np.ptp(points, axis=0).max())
    maximum_width = args.max_width if args.max_width is not None else extent
    gripper = AnalyticalGripper(
        min_width=args.min_width, max_width=maximum_width,
        contact_center_tcp=np.asarray(args.contact_center_tcp, dtype=float),
    )
    settings = Sampling(
        surface_points=args.surface_points, candidates=args.num_candidates,
        approaches=args.approaches, neighbors=args.neighbors,
        min_width=args.min_width, max_width=args.max_width,
        friction=args.friction, angular_tolerance_deg=args.angular_tolerance_deg,
        seed=args.seed, pairs_per_point=args.pairs_per_point,
    )
    records = generate(
        points, normals, gripper, settings,
        unoriented_normals=not args.oriented_normals,
    )
    if not records:
        raise ValueError("No antipodal pairs found; inspect normals, friction cone, and width limits")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "source": str(npz_file),
                "object_id": npz_file.parent.name, "object_type": category,
                "transform": str(transform_file) if transform_file else None,
                "units": units, "analytical_only": True, "sampling": asdict(settings),
                "point_count": len(points), "width_limits": [gripper.min_width, gripper.max_width],
                "contact_center_tcp": gripper.contact_center_tcp.tolist(),
                "normal_orientation": "provided" if args.oriented_normals else "pair_aligned",
            },
            "grasps": records,
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved {len(records)} candidates to {args.output_json}")

    shown = selected_records(records, args.grasp_id, args.show_candidates)
    axis_length = args.axis_length if args.axis_length is not None else extent * 0.18
    if not np.isfinite(axis_length) or axis_length <= 0:
        raise ValueError("--axis-length must be positive and finite")
    label = f"{category}: {npz_file.parent.name}" if category else npz_file.parent.name
    figure = draw(points, affordance_scores(npz_file, len(points)), shown, axis_length, units, label)
    print(
        f"source={npz_file} class={category or 'unknown'} points={len(points)} candidates={len(records)} "
        f"displayed={len(shown)} widths=[{min(r['gripper_width'] for r in records):.4f}, "
        f"{max(r['gripper_width'] for r in records):.4f}] {units}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=args.dpi)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
