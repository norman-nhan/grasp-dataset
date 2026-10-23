"""LASO labels remain independent of physical outcomes and sampling."""
import json
import pickle
from pathlib import Path
import numpy as np
import trimesh
from .scene import points_transform

def load_points(annotation, transform, affordance='grasp', split='val', shape_id=None):
    """The required 4x4 similarity maps source annotation coordinates into object metres."""
    source = Path(annotation)
    t = np.asarray(transform, dtype=float)
    if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1]) or np.linalg.det(t[:3,:3]) <= 0:
        raise ValueError('annotation_to_object must be a finite, invertible 4x4 transform')
    gram = t[:3,:3].T @ t[:3,:3]
    if not np.allclose(gram, np.eye(3)*np.trace(gram)/3, atol=1e-8):
        raise ValueError('Annotation transform must be a rigid transform with uniform positive scale')
    if source.is_dir():
        # Only explicitly supplied trusted local LASO pickle files are accepted.
        with (source/f'objects_{split}.pkl').open('rb') as f:
            objects = pickle.load(f)
        with (source/f'anno_{split}.pkl').open('rb') as f:
            annotations = pickle.load(f)
        if shape_id is None:
            raise ValueError('LASO pickle input requires --shape-id')
        points = np.asarray({str(k):v for k,v in objects.items()}[shape_id])[:,:3]
        masks = [np.asarray(a['mask']).reshape(-1)>0.5 for a in annotations if str(a['shape_id']) == shape_id and a['affordance'] == affordance]
        if not masks or any(len(mask) != len(points) for mask in masks):
            raise ValueError('Missing affordance or mismatched LASO masks')
        points = points[np.logical_or.reduce(masks)]
    else:
        meta = json.loads(source.read_text())
        if meta.get('schema') != 'laso-point-cloud-export/v1':
            raise ValueError('Expected repository LASO export schema')
        cloud = trimesh.load(source.parent/meta['geometry']['file'], process=False)
        masks = [a['point_indices'] for a in meta['affordances'] if a['affordance'] == affordance]
        if not masks:
            raise ValueError(f'No {affordance!r} region in annotation')
        points = np.asarray(cloud.vertices)[np.unique(np.concatenate(masks)).astype(int)]
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('Empty or invalid LASO region')
    return points_transform(points, t)

def overlap(points_object, record, environment, clearance=0.01):
    if clearance < 0:
        raise ValueError('Clearance must be nonnegative')
    # Place object and gripper in a common world frame, independent of initial YAML pose.
    environment.reset(np.eye(4))
    environment.place(np.asarray(record['grasp_pose_object']), record['gripper_width'])
    blocked = np.zeros(len(points_object), dtype=bool)
    for g in environment.gripper_geoms:
        mesh = environment.geom_mesh(g)
        distance = trimesh.proximity.closest_point(mesh, points_object)[1]
        blocked |= (distance <= clearance) | mesh.contains(points_object)
    return float(blocked.mean())
