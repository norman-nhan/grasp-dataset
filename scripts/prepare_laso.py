#!/usr/bin/env python3
"""Prepare one approximate LASO bottle, knife or mug, with explicit assumptions."""
import argparse
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import coacd
import numpy as np
import trimesh
import yaml
from prepare_laso_mug import reconstruct

EXTENTS = {'bottle': .18, 'knife': .22, 'mug': .085}


def validate_source_coverage(mesh, points, radius):
    distance = trimesh.proximity.closest_point(mesh, points)[1]
    far = distance > 2*radius
    if far.any() and not mesh.contains(points[far]).all():
        raise ValueError('Some source points lie outside the solid beyond twice shell radius')
    return distance, far


def prepare(dataset, split, shape_id, category, directory):
    directory = Path(directory)
    config = directory/'object.yaml'
    if config.exists():
        document = yaml.safe_load(config.read_text())
        existing = document['provenance']
        if (existing['shape_id'], existing['split'], existing['category']) != (shape_id, split, category):
            raise ValueError('Existing asset identity does not match requested object')
        # MuJoCo's STL decoder rejects >200k faces. OBJ preserves the same
        # triangles without that STL-specific limit; no decimation is needed.
        visual = document['objects'][0]['visual_meshes'][0]
        if visual['file']=='surface.stl':
            import struct
            with (directory/'surface.stl').open('rb') as f:
                f.seek(80)
                faces = struct.unpack('<I',f.read(4))[0]
            if faces > 200000:
                trimesh.load(directory/'surface.stl').export(directory/'surface.obj')
                visual['file']='surface.obj'
                temporary=config.with_suffix('.yaml.tmp')
                temporary.write_text(yaml.safe_dump(document,sort_keys=False))
                temporary.replace(config)
        return config
    with (dataset/f'objects_{split}.pkl').open('rb') as f:
        objects = pickle.load(f)
    with (dataset/f'anno_{split}.pkl').open('rb') as f:
        annotations = pickle.load(f)
    matches = [a for a in annotations if str(a['shape_id']) == shape_id]
    if not matches or any(a['class'] != category for a in matches):
        raise ValueError('LASO category/shape mismatch')
    xyz = np.asarray(objects[shape_id], dtype=float)[:, :3]
    if not np.isfinite(xyz).all() or np.ptp(xyz, axis=0).max() <= 0:
        raise ValueError('Invalid source coordinates')
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1,0,0],[0,0,-1],[0,1,0]])*EXTENTS[category]/np.ptp(xyz,axis=0).max()
    points = xyz @ transform[:3, :3].T
    transform[:3, 3] = -(points.min(0)+points.max(0))/2
    points += transform[:3, 3]
    radius = .0015 if category == 'knife' else .0025
    pitch = .00075
    attempts = []
    # Only geometry connectivity/coverage selects a reconstruction radius. No
    # physical outcome or affordance mask participates in this choice.
    for factor in [1., 1.25, 1.5]:
        try:
            actual_radius = radius*factor
            mesh = reconstruct(points, actual_radius, pitch)
            # Filling enclosed voxel air can bury inner source surfaces in the
            # assumed material. A buried point is represented, not omitted.
            distance, far = validate_source_coverage(mesh, points, actual_radius)
            break
        except ValueError as error:
            attempts.append(dict(radius_m=radius*factor, error=str(error)))
    else:
        raise ValueError('Reconstruction failed: '+json.dumps(attempts))
    coacd.set_log_level('error')
    settings = dict(threshold=.02, preprocess_resolution=40, resolution=2000,
        mcts_nodes=10, mcts_iterations=30, mcts_max_depth=3,
        max_convex_hull=64, seed=42)
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **settings)
    hulls = [trimesh.Trimesh(v, f, process=True) for v, f in parts]
    if not hulls or any(not h.is_watertight or h.volume <= 0 for h in hulls):
        raise ValueError('Invalid collision decomposition')
    directory.mkdir(parents=True, exist_ok=True)
    mesh.export(directory/'surface.obj')
    trimesh.points.PointCloud(points).export(directory/'source_points.ply')
    (directory/'annotation_to_object.json').write_text(json.dumps(transform.tolist(),indent=2)+'\n')
    total = sum(h.volume for h in hulls)
    collision = []
    for i, hull in enumerate(hulls):
        name = f'collision_{i:03d}.stl'
        hull.export(directory/name)
        collision.append(dict(file=name,mass_fraction=float(hull.volume/total)))
    provenance = dict(dataset='LASO', split=split, shape_id=shape_id, category=category,
        available_affordances=sorted({a['affordance'] for a in matches}),
        representation='approximate smoothed point-distance shell; not original source mesh',
        source_points_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
        assumed_longest_extent_m=EXTENTS[category], assumed_mass_kg=.1,
        scale_and_mass='configured simulation assumptions; not measured LASO properties',
        orientation='source Y mapped to world Z; assumed canonical frame',
        shell_radius_m=actual_radius, voxel_pitch_m=pitch, smoothing_sigma_voxels=.75,
        enclosed_voxel_air_pockets_filled=True, reconstruction_attempts=attempts,
        source_to_surface_max_m=float(distance.max()), extents_m=mesh.extents.tolist(),
        source_points_deeper_than_two_radii_inside_solid=int(far.sum()),
        watertight=bool(mesh.is_watertight), collision_components=len(hulls),
        coacd_settings=settings, coacd_version=importlib.metadata.version('coacd'),
        annotation_to_object=transform.tolist(),
        limits='Source holes and thin regions may be altered; no universal mug cavity guarantee')
    target = dict(name=f'laso_{category}_{shape_id}',target=True,movable=True,type='mesh',
        mass=.1,friction=[1.2,.01,.001],visual_meshes=[dict(file='surface.obj')],
        mujoco_collision_meshes=collision)
    (directory/'SOURCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    temporary = config.with_suffix('.yaml.tmp')
    temporary.write_text(yaml.safe_dump(dict(standalone=True,objects=[target],provenance=provenance),sort_keys=False))
    temporary.replace(config)
    return config


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--split',choices=['train','val','test'],required=True)
    p.add_argument('--shape-id',required=True)
    p.add_argument('--category',choices=list(EXTENTS),required=True)
    p.add_argument('--directory',type=Path,required=True)
    a = p.parse_args()
    print(prepare(a.dataset,a.split,a.shape_id,a.category,a.directory),flush=True)
