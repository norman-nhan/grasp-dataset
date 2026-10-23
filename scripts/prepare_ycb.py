#!/usr/bin/env python3
"""Download official YCB Google 16k scans and build standalone object configurations."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tarfile
import urllib.request
import coacd
import numpy as np
import trimesh
import yaml

OBJECTS = ['006_mustard_bottle','025_mug','032_knife','037_scissors']
AVAILABLE_OBJECTS = OBJECTS + ['021_bleach_cleanser', '022_windex_bottle']
BASE = 'https://ycb-benchmarks.s3.amazonaws.com/data/google/'

def prepare(root, name, mass, seed):
    directory = root/name
    config = directory/'object.yaml'
    if config.exists():
        print(f'Already prepared: {config}',flush=True)
        return config
    directory.mkdir(parents=True,exist_ok=True)
    archive = root/(name+'_google_16k.tgz')
    url = BASE+archive.name
    if not archive.exists():
        print(f'Downloading {url}',flush=True)
        urllib.request.urlretrieve(url,archive)
    # Read only expected model files; never extract arbitrary archive paths.
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and '/google_16k/' in member.name and Path(member.name).name in ('textured.obj','textured.mtl','texture_map.png'):
                contents=tar.extractfile(member).read()
                (directory/Path(member.name).name).write_bytes(contents)
    source=directory/'textured.obj'
    if not source.exists():
        raise ValueError(f'{archive} has no google_16k/textured.obj')
    mesh=trimesh.load(source,force='scene').to_mesh()
    mesh.update_faces(mesh.nondegenerate_faces()); mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh,multibody=True)
    coacd.set_log_level('error')
    print(f'Decomposing {name}: {len(mesh.faces)} faces',flush=True)
    parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),threshold=0.05,
        preprocess_resolution=40,resolution=2000,mcts_nodes=10,mcts_iterations=30,mcts_max_depth=3,
        max_convex_hull=32,seed=seed)
    volumes=np.array([trimesh.Trimesh(v,f,process=False).volume for v,f in parts])
    if len(volumes)==0 or not np.isfinite(volumes).all() or np.any(volumes<=0):
        raise ValueError('Invalid convex decomposition')
    specs=[]
    for i,((v,f),fraction) in enumerate(zip(parts,volumes/volumes.sum())):
        file=directory/f'collision_{i:03d}.stl'
        trimesh.Trimesh(v,f,process=False).export(file)
        specs.append(dict(file=file.name,mass_fraction=float(fraction)))
    target=dict(name=name,target=True,movable=True,type='mesh',mass=mass,
        mass_source='configured simulation assumption; not a measured YCB mass',friction=[1.2,.01,.001],
        visual_meshes=[dict(file='textured.obj',texture='texture_map.png')],mujoco_collision_meshes=specs)
    metadata=dict(source_url=url,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        license='CC BY 4.0',citation='Calli et al., The YCB Object and Model Set, ICAR 2015',
        collision_method='CoACD',coacd_version=importlib.metadata.version('coacd'),seed=seed,
        threshold=.05,preprocess_resolution=40,resolution=2000,mcts_nodes=10,mcts_iterations=30,mcts_max_depth=3,max_convex_hull=32,
        collision_components=len(parts),inertia='MuJoCo inferred from convex components with volume-proportional mass fractions')
    config.write_text(yaml.safe_dump(dict(standalone=True,objects=[target],provenance=metadata),sort_keys=False))
    (directory/'SOURCE.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Prepared {config}: {len(parts)} convex collision components',flush=True)
    return config

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects',nargs='+',choices=AVAILABLE_OBJECTS,default=OBJECTS)
    p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'assets/ycb')
    p.add_argument('--mass',type=float,default=.1,help='Explicit simulation mass assumption, kg (default 0.1)')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if not np.isfinite(a.mass) or a.mass<=0:
        p.error('--mass must be finite and positive')
    for name in a.objects:
        prepare(a.directory.resolve(),name,a.mass,a.seed)
