#!/usr/bin/env python3
"""Build an explicitly approximate, metric mug from trusted local LASO points.

LASO provides neither triangle meshes nor measured dimensions. This example uses
a point-distance isosurface, not the global convex hull (which fills the cup).
"""
import argparse
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import coacd
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.measure import marching_cubes
import trimesh
import yaml

DEFAULT_SHAPE = '34ae0b61b0d8aaf2d7b20fded0142d7a'


def reconstruct(points, radius=.0025, pitch=.00075):
    origin = points.min(0) - radius - 4*pitch
    count = np.ceil((points.max(0) - origin + radius + 4*pitch)/pitch).astype(int) + 1
    if np.prod(count) > 20_000_000:
        raise ValueError('Reconstruction grid exceeds 20 million cells; increase voxel pitch')
    grid = np.indices(count).reshape(3, -1).T*pitch + origin
    distance = cKDTree(points).query(grid)[0].reshape(count)
    field = gaussian_filter(distance, .75)
    # Fill only enclosed air pockets; the open mouth and handle hole stay open.
    solid = gaussian_filter(binary_fill_holes(field <= radius).astype(float), .75)
    vertices, faces, _, _ = marching_cubes(solid, level=.5, spacing=(pitch,)*3)
    mesh = trimesh.Trimesh(vertices + origin, faces, process=True)
    trimesh.repair.fix_normals(mesh, multibody=True)
    components = mesh.split()
    if not mesh.is_watertight or len(components) != 1 or mesh.volume <= 0:
        volumes = sorted((float(p.volume) for p in components),reverse=True)
        raise ValueError(f'Reconstruction must be one watertight positive-volume solid; '
            f'watertight={mesh.is_watertight}, components={len(components)}, '
            f'largest_component_volumes={volumes[:5]}; inspect shell radius')
    return mesh


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=root.parent/'laso-reviewer/data/LASO_dataset')
    p.add_argument('--shape-id', default=DEFAULT_SHAPE)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--longest-extent', type=float, default=.085, help='Assumed metres, not measured LASO scale')
    p.add_argument('--shell-radius', type=float, default=.0025, help='Assumed material radius around source points, metres')
    p.add_argument('--voxel-pitch', type=float, default=.00075)
    p.add_argument('--mass', type=float, default=.1, help='Assumed kg')
    p.add_argument('--directory', type=Path)
    args = p.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (args.longest_extent, args.shell_radius, args.voxel_pitch, args.mass)):
        p.error('Dimensions, shell radius, voxel pitch and mass must be finite and positive')
    directory = (args.directory or root/'assets/laso'/(args.shape_id+'-shell')).resolve()
    if directory.exists():
        raise FileExistsError(f'Use a fresh directory: {directory}')
    source = args.dataset/f'objects_{args.split}.pkl'
    with source.open('rb') as handle:
        objects = pickle.load(handle)
    with (args.dataset/f'anno_{args.split}.pkl').open('rb') as handle:
        annotations = pickle.load(handle)
    if not any(str(a['shape_id']) == args.shape_id and a['class'] == 'mug' for a in annotations):
        raise ValueError('Requested shape is not a LASO mug')
    xyz = np.asarray(objects[args.shape_id], dtype=float)[:, :3]
    # This example's LASO mug is Y-up. Rotate Y to Z, preserving handedness.
    rotation = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    transform = np.eye(4)
    transform[:3, :3] = rotation * (args.longest_extent / np.ptp(xyz, axis=0).max())
    points = xyz @ transform[:3, :3].T
    transform[:3, 3] = -(points.min(0) + points.max(0)) / 2
    points += transform[:3, 3]
    mesh = reconstruct(points, args.shell_radius, args.voxel_pitch)
    cloud_distance = trimesh.proximity.closest_point(mesh, points)[1]
    if cloud_distance.max() > 2*args.shell_radius:
        raise ValueError('Reconstruction is too far from some source points')
    # Preserve mug's open central cavity through both surface reconstruction and
    # convex decomposition: probe the body center, well above its bottom.
    cavity = np.array([[0., .015, 0.]])
    if mesh.contains(cavity).any():
        raise ValueError('Example cavity probe is filled; inspect reconstruction')
    coacd.set_log_level('error')
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=.02,
        preprocess_resolution=60, resolution=3000, mcts_nodes=15,
        mcts_iterations=50, mcts_max_depth=4, max_convex_hull=64, seed=42)
    hulls = [trimesh.Trimesh(v, f, process=True) for v, f in parts]
    if not hulls or any(h.volume <= 0 for h in hulls):
        raise ValueError('Invalid collision decomposition')
    if any(h.contains(cavity).any() for h in hulls):
        raise ValueError('Collision decomposition fills the example cavity probe')
    # For the supplied example, test the actual entry rays as well as a point.
    cavity_validation = {}
    if args.shape_id == DEFAULT_SHAPE and args.longest_extent == .085:
        for label, models in [('surface', [mesh]), ('collision', hulls)]:
            hits = [np.asarray(h.ray.intersects_location([[0, .015, .1]],
                [[0, 0, -1]])[0]).reshape(-1, 3) for h in models]
            hits = np.concatenate(hits)
            if not len(hits) or hits[:, 2].max() >= -.02:
                raise ValueError(f'{label} blocks entry through the cup mouth')
            handle_hits = sum(len(h.ray.intersects_location([[-.1, -.03, 0]],
                [[1, 0, 0]])[0]) for h in models)
            if handle_hits:
                raise ValueError(f'{label} fills the handle hole')
            cavity_validation[label] = dict(mouth_ray_highest_hit_z_m=float(hits[:, 2].max()),
                handle_hole_ray_intersections=handle_hits)
    directory.mkdir(parents=True)
    mesh.export(directory/'surface.stl')
    trimesh.points.PointCloud(points).export(directory/'source_points.ply')
    (directory/'annotation_to_object.json').write_text(json.dumps(transform.tolist(), indent=2)+'\n')
    total_volume = sum(h.volume for h in hulls)
    specs = []
    for i, hull in enumerate(hulls):
        filename = f'collision_{i:03d}.stl'
        hull.export(directory/filename)
        specs.append(dict(file=filename, mass_fraction=float(hull.volume/total_volume)))
    provenance = dict(dataset='LASO', split=args.split, shape_id=args.shape_id,
        source_points_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
        geometry='approximate smoothed point-distance shell of 2048 source points; not original mesh',
        shell_radius_m=args.shell_radius, voxel_pitch_m=args.voxel_pitch,
        smoothing_sigma_voxels=.75, assumed_longest_extent_m=args.longest_extent,
        enclosed_voxel_air_pockets_filled=True,
        physical_scale='assumed; LASO normalized points have no measured metric scale',
        mass_kg=args.mass, mass_source='simulation assumption, not measured',
        source_point_count=len(points), surface_vertices=len(mesh.vertices),
        source_to_surface_max_m=float(cloud_distance.max()), extents_m=mesh.extents.tolist(),
        collision_method='CoACD', collision_components=len(hulls), seed=42,
        coacd_version=importlib.metadata.version('coacd'),
        coacd_settings=dict(threshold=.02, preprocess_resolution=60, resolution=3000,
            mcts_nodes=15, mcts_iterations=50, mcts_max_depth=4, max_convex_hull=64),
        cavity_probe_object_m=cavity[0].tolist(), cavity_probe_clear=True,
        entry_ray_validation=cavity_validation,
        annotation_to_object=transform.tolist())
    target = dict(name='laso_mug_'+args.shape_id[:8], target=True, movable=True,
        type='mesh', mass=args.mass, friction=[1.2, .01, .001],
        visual_meshes=[dict(file='surface.stl')], mujoco_collision_meshes=specs)
    (directory/'object.yaml').write_text(yaml.safe_dump(dict(standalone=True,
        objects=[target], provenance=provenance), sort_keys=False))
    (directory/'SOURCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))
    print(f'Prepared {directory / "object.yaml"}')


if __name__ == '__main__':
    main()
