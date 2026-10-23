#!/usr/bin/env python3
"""Visualize object points and surface normals stored in affordance.npz."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from grasp_dataset.laso_catalog import load as load_catalog, object_type


PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT / "data"
DEFAULT_OBJECT_TYPES = PROJECT / "laso_object_types.json"


def catalog(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("*/**/affordance.npz"))


def resolve_npz(data_root: Path, query: str | None, npz: Path | None) -> Path:
    if npz is not None:
        result = npz.resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    if not query:
        raise ValueError("--object or --npz is required unless --list is used")
    direct = (data_root / query).resolve()
    if direct.is_dir():
        direct = direct / "affordance.npz"
    if direct.is_file():
        return direct
    matches = [path for path in catalog(data_root) if path.parent.name == query]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No affordance.npz for {query!r} under {data_root}; use --list")
    choices = ", ".join(str(path.relative_to(data_root)) for path in matches)
    raise ValueError(f"Object ID {query!r} is ambiguous: {choices}")


def load_arrays(npz_file: Path) -> dict[str, np.ndarray]:
    with np.load(npz_file, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "points" not in arrays or "normals" not in arrays:
        raise ValueError(f"{npz_file} must contain 'points' and 'normals'")
    points = np.asarray(arrays["points"], dtype=np.float64)
    normals = np.asarray(arrays["normals"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape [N, 3]")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("points or normals contain non-finite values")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("normals contain zero-length vectors")
    arrays["points"] = points
    arrays["normals"] = normals / lengths[:, None]
    return arrays


def normal_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("--max-normals must be positive")
    if count <= maximum:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def colors(arrays: dict[str, np.ndarray], mode: str) -> tuple[np.ndarray | str, str | None]:
    count = len(arrays["points"])
    if mode == "score":
        if "affordance_score" not in arrays:
            raise ValueError("--color score requires affordance_score in the NPZ")
        values = np.asarray(arrays["affordance_score"], dtype=float).reshape(-1)
        if len(values) != count:
            raise ValueError("affordance_score length does not match points")
        return values, "Human grasp affordance score"
    if mode == "mask":
        if "affordance_mask" not in arrays:
            raise ValueError("--color mask requires affordance_mask in the NPZ")
        mask = np.asarray(arrays["affordance_mask"], dtype=bool).reshape(-1)
        if len(mask) != count:
            raise ValueError("affordance_mask length does not match points")
        return np.where(mask[:, None], [1.0, 0.2, 0.05, 1.0], [0.2, 0.65, 0.9, 0.45]), None
    return "#4ba3d3", None


def set_equal_axes(axis, points: np.ndarray) -> None:
    center = points.min(axis=0) / 2 + points.max(axis=0) / 2
    radius = max(float(np.ptp(points, axis=0).max()) / 2, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def draw(
    arrays: dict[str, np.ndarray],
    title: str,
    color_mode: str,
    max_normals: int,
    normal_length: float,
    point_size: float,
    seed: int,
    invert_normals: bool,
):
    if normal_length is None:
        normal_length = float(np.ptp(arrays["points"], axis=0).max()) * 0.06
    if not np.isfinite(normal_length) or normal_length <= 0:
        raise ValueError("--normal-length must be positive and finite")
    points = arrays["points"]
    normals = arrays["normals"] * (-1 if invert_normals else 1)
    selected = normal_indices(len(points), max_normals, seed)
    point_colors, colorbar_label = colors(arrays, color_mode)

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=point_colors, cmap="turbo" if color_mode == "score" else None,
        vmin=0 if color_mode == "score" else None,
        vmax=1 if color_mode == "score" else None,
        s=point_size, depthshade=False,
    )
    axis.quiver(
        points[selected, 0], points[selected, 1], points[selected, 2],
        normals[selected, 0], normals[selected, 1], normals[selected, 2],
        length=normal_length, normalize=True, color="black", linewidth=0.55,
        arrow_length_ratio=0.25,
    )
    if colorbar_label:
        figure.colorbar(scatter, ax=axis, shrink=0.72, pad=0.08, label=colorbar_label)
    axis.set_title(f"{title}\n{len(points):,} points; {len(selected):,} displayed normals")
    axis.set_xlabel("X [source units]")
    axis.set_ylabel("Y [source units]")
    axis.set_zlabel("Z [source units]")
    set_equal_axes(axis, points)
    return figure, normal_length


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", help="Object ID, for example 025_mug")
    p.add_argument("--npz", type=Path, help="Explicit affordance.npz path")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--object-types", type=Path, default=DEFAULT_OBJECT_TYPES)
    p.add_argument("--list", action="store_true", help="List available dataset/object paths and exit")
    p.add_argument("--color", choices=["score", "mask", "uniform"], default="score")
    p.add_argument("--max-normals", type=int, default=500, help="Maximum normal arrows to draw")
    p.add_argument(
        "--normal-length", type=float,
        help="Arrow length in data units; default is 6%% of the object's largest extent",
    )
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42, help="Deterministic arrow subsampling seed")
    p.add_argument("--invert-normals", action="store_true", help="Display the opposite normal direction")
    p.add_argument("--output", type=Path, help="Save PNG instead of opening an interactive window")
    p.add_argument("--dpi", type=int, default=180)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    data_root = args.data_root.resolve()
    catalog_data = load_catalog(args.object_types) if args.object_types.is_file() else {}
    if args.list:
        try:
            for path in catalog(data_root):
                category = object_type(path.parent.name, catalog_data)
                print(f"{category or 'unknown'}: {path.parent.relative_to(data_root)}")
        except BrokenPipeError:
            sys.stdout = open("/dev/null", "w")
        return
    npz_file = resolve_npz(data_root, args.object, args.npz)
    arrays = load_arrays(npz_file)
    category = None
    if catalog_data:
        category = object_type(npz_file.parent.name, catalog_data)
    label = f"{category}: {npz_file.parent.name}" if category else npz_file.parent.name
    figure, normal_length = draw(
        arrays, label, args.color, args.max_normals,
        args.normal_length, args.point_size, args.seed, args.invert_normals,
    )
    print(
        f"source={npz_file} class={category or 'unknown'} points={len(arrays['points'])} "
        f"max_normals={args.max_normals} normal_length={normal_length:.6g}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=args.dpi)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
