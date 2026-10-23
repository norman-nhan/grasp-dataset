"""Load repository scenes without ROS; reuse the generated, URDF-corrected MJCF."""
from copy import deepcopy
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

REPO = Path(__file__).resolve().parents[3]
SHARE = REPO / 'src/openarm_mujoco_bringup'

def vec(x):
    return ' '.join(str(float(v)) for v in x)

def pose(position=(0, 0, 0), quaternion_wxyz=(1, 0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    t[:3, 3] = position
    return t

def quat(t):
    return np.roll(Rotation.from_matrix(t[:3, :3]).as_quat(), 1)

def points_transform(points, t):
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]

def resolve_asset(file, scene):
    p = Path(file)
    for candidate in (p, scene.parent / p, SHARE / p):
        if candidate.is_file():
            return candidate.resolve()
    # Generated XML may contain Docker install paths. Only map known package assets.
    if '/assets/' in str(file):
        candidate = SHARE / 'assets' / str(file).split('/assets/', 1)[1]
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot resolve asset {file!r} from {scene}')

class Scene:
    def __init__(self, file, object_name=None, model_file=None, environment="minimal", support="plane"):
        self.environment, self.support = environment, support
        self.file = Path(file).resolve()
        self.config = yaml.safe_load(self.file.read_text())
        objects = self.config['objects']
        matches = [o for o in objects if o.get('name') == object_name] if object_name else [o for o in objects if o.get('target')]
        if len(matches) != 1:
            raise ValueError('Select exactly one target with --object')
        self.object = matches[0]
        self.name = self.object['name']
        self.standalone = bool(self.config.get('standalone'))
        self.model_file = Path(model_file).resolve() if model_file else (SHARE/'config/scene_cube/scene_cube.xml' if self.standalone else self.file.with_suffix('') / (self.file.stem + '.xml'))
        if not self.model_file.is_file():
            raise FileNotFoundError(f'{self.model_file}: build this repository scene first or supply --model with its generated XML')
        self.source = ET.parse(self.model_file).getroot()
        if not self.standalone:
            self.validate_generated_target()
        self.mesh = self.load_mesh()
        self.initial_pose = pose(self.object.get('position', [0, 0, 0]), self.object.get('quaternion_wxyz', [1, 0, 0, 0]))
        if self.environment == 'minimal':
            world_vertices = points_transform(self.mesh.vertices, self.initial_pose)
            self.initial_pose[:2,3] = -world_vertices[:,:2].mean(axis=0) + self.initial_pose[:2,3]
            self.initial_pose[2,3] += -world_vertices[:,2].min() + (0.15 if self.support == 'none' else 0.003)

    def validate_generated_target(self):
        """Fail clearly when YAML and its generated simulation model have drifted."""
        body = self.source.find(f".//body[@name='{self.name}']")
        if body is None:
            raise ValueError('Target is absent from generated scene XML')
        actual = pose(np.fromstring(body.get('pos','0 0 0'),sep=' '), np.fromstring(body.get('quat','1 0 0 0'),sep=' '))
        expected = pose(self.object.get('position',[0,0,0]), self.object.get('quaternion_wxyz',[1,0,0,0]))
        if not np.allclose(actual, expected, atol=1e-6):
            raise ValueError('Scene YAML object pose differs from generated XML; rebuild the scene')
        if self.object['type'] == 'box':
            geom = body.find("geom[@type='box']")
            if geom is None or not np.allclose(np.fromstring(geom.get('size'),sep=' ')*2,self.object['size']):
                raise ValueError('Scene YAML box dimensions differ from generated XML; rebuild the scene')
        elif self.object['type'] == 'mesh':
            geoms = [g for g in body.findall('geom') if g.get('contype') == '0' and g.get('type') == 'mesh']
            specs = self.visual_parts()
            if len(geoms) != len(specs):
                raise ValueError('Scene YAML mesh components differ from generated XML; rebuild the scene')
            for spec, geom in zip(specs, geoms):
                asset = self.source.find(f"asset/mesh[@name='{geom.get('mesh')}']")
                if not np.allclose(np.fromstring(asset.get('scale','1 1 1'),sep=' '),spec.get('scale',[1,1,1])):
                    raise ValueError('Scene YAML mesh scale differs from generated XML; rebuild the scene')
                if Path(asset.get('file')).name != Path(spec['file']).name:
                    raise ValueError('Scene YAML mesh file differs from generated XML; rebuild the scene')
                actual = pose(np.fromstring(geom.get('pos','0 0 0'),sep=' '), np.fromstring(geom.get('quat','1 0 0 0'),sep=' '))
                expected = pose(spec.get('position',[0,0,0]),spec.get('quaternion_wxyz',[1,0,0,0]))
                if not np.allclose(actual, expected, atol=1e-6):
                    raise ValueError('Scene YAML mesh transform differs from generated XML; rebuild the scene')

    def visual_parts(self):
        o = self.object
        return o.get('visual_meshes') or [dict(file=o['visual_mesh'], scale=o.get('scale', [1, 1, 1]), quaternion_wxyz=o.get('mesh_quaternion_wxyz', [1, 0, 0, 0]))]

    def load_mesh(self):
        o = self.object
        if o['type'] == 'box':
            mesh = trimesh.creation.box(extents=o['size'])
        elif o['type'] == 'mesh':
            parts = []
            for spec in self.visual_parts():
                m = trimesh.load(resolve_asset(spec['file'], self.file), force='scene').to_mesh()
                m.apply_scale(spec.get('scale', [1, 1, 1]))
                m.apply_transform(pose(spec.get('position', [0, 0, 0]), spec.get('quaternion_wxyz', [1, 0, 0, 0])))
                parts.append(m)
            mesh = trimesh.util.concatenate(parts)
        else:
            raise ValueError('Sampling currently supports box and mesh targets')
        if not np.isfinite(mesh.vertices).all() or len(mesh.faces) == 0:
            raise ValueError('Nonfinite or empty triangle geometry')
        self.degenerate_faces_removed = int(np.count_nonzero(mesh.area_faces < 1e-16))
        if self.degenerate_faces_removed:
            import warnings
            warnings.warn(f'Removing {self.degenerate_faces_removed} degenerate triangles from {self.name}', stacklevel=2)
            mesh.update_faces(mesh.area_faces >= 1e-16)
            mesh.remove_unreferenced_vertices()
        if len(mesh.faces) == 0:
            raise ValueError('Mesh has no nondegenerate faces')
        trimesh.repair.fix_normals(mesh, multibody=True)
        if not mesh.is_winding_consistent:
            raise ValueError('Inconsistent mesh normals')
        self.mesh_watertight = bool(mesh.is_watertight)
        return mesh

    def minimal_xml(self, timestep=0.002):
        root = ET.Element('mujoco', model='openarm_gripper_dataset')
        ET.SubElement(root, 'compiler', angle='radian')
        ET.SubElement(root, 'option', timestep=str(timestep), integrator='implicitfast', gravity='0 0 -9.81')
        root.append(deepcopy(self.source.find('default')))
        asset = ET.SubElement(root, 'asset')
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', pos='0 0 3', dir='0 0 -1', diffuse='0.8 0.8 0.8')
        hand = deepcopy(self.source.find(".//body[@name='openarm_right_hand']"))
        if hand is None:
            raise ValueError('Generated model lacks openarm_right_hand')
        # URDF-corrected scenes place fingers as siblings of the hand.
        parents = {c: p for p in self.source.iter() for c in p}
        def source_pose(body):
            t = np.eye(4)
            while body.tag == 'body':
                t = pose(np.fromstring(body.get('pos', '0 0 0'), sep=' '), np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')) @ t
                body = parents[body]
            return t
        hand_source = self.source.find(".//body[@name='openarm_right_hand']")
        for name in ('openarm_right_right_finger', 'openarm_right_left_finger'):
            if hand.find(f".//body[@name='{name}']") is None:
                original = self.source.find(f".//body[@name='{name}']")
                finger = deepcopy(original)
                t = np.linalg.inv(source_pose(hand_source)) @ source_pose(original)
                finger.set('pos', vec(t[:3, 3])); finger.set('quat', vec(quat(t)))
                hand.append(finger)
        hand.set('pos', '0 0 1')
        hand.set('quat', '1 0 0 0')
        hand.insert(0, ET.Element('freejoint', name='hand_free'))
        world.append(hand)
        ET.SubElement(world, 'body', name='hand_target', mocap='true', pos='0 0 1')
        if self.standalone:
            target = ET.Element('body', name=self.name)
            ET.SubElement(target, 'freejoint', name=self.name+'_freejoint')
            specs = self.visual_parts()
            for i, spec in enumerate(specs):
                name = f'dataset_visual_{i}'
                mesh_path = resolve_asset(spec['file'], self.file)
                ET.SubElement(asset, 'mesh', name=name, file=str(mesh_path), scale=vec(spec.get('scale',[1,1,1])))
                material = None
                if spec.get('texture'):
                    texture = f'dataset_texture_{i}'; material = f'dataset_material_{i}'
                    ET.SubElement(asset, 'texture', name=texture, type='2d', file=str(resolve_asset(spec['texture'],self.file)))
                    ET.SubElement(asset, 'material', name=material, texture=texture)
                geom = ET.SubElement(target,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',
                    pos=vec(spec.get('position',[0,0,0])),quat=vec(spec.get('quaternion_wxyz',[1,0,0,0])))
                if material:
                    geom.set('material',material)
            for i,spec in enumerate(self.object['mujoco_collision_meshes']):
                name=f'dataset_collision_{i}'
                ET.SubElement(asset,'mesh',name=name,file=str(resolve_asset(spec['file'],self.file)),scale=vec(spec.get('scale',[1,1,1])))
                ET.SubElement(target,'geom',name=name,type='mesh',mesh=name,mass=str(self.object['mass']*spec['mass_fraction']),
                    pos=vec(spec.get('position',[0,0,0])),quat=vec(spec.get('quaternion_wxyz',[1,0,0,0])),
                    friction=vec(self.object.get('friction',[1.2,.01,.001])),rgba='0.7 0.7 0.7 1',group='3')
        else:
            target = deepcopy(self.source.find(f".//body[@name='{self.name}']"))
            if target is None or target.find('freejoint') is None:
                raise ValueError('Target must have a generated freejoint body')
        if self.environment == 'minimal':
            target.set('pos',vec(self.initial_pose[:3,3])); target.set('quat',vec(quat(self.initial_pose)))
            if self.support == 'plane':
                ET.SubElement(world,'geom',name='dataset_support',type='plane',size='0.5 0.5 0.01',friction='1.2 .01 .001',rgba='.35 .35 .35 1')
        world.append(target)
        if self.environment == 'source':
            for other in self.config['objects']:
                if other['name'] != self.name:
                    body = self.source.find(f".//body[@name='{other['name']}']")
                    if body is None:
                        raise ValueError('Nearby scene object absent from generated XML')
                    world.append(deepcopy(body))
            for spec in self.config.get('scene_objects', []):
                body = self.source.find(f".//body[@name='{spec['name']}']")
                if body is None:
                    body = self.source.find(f"worldbody/geom[@name='{spec['name']}']")
                if body is None:
                    raise ValueError(f"Generated model missing environment body {spec['name']}")
                world.append(deepcopy(body))
        # Optional scene-specific fingertip pads already belong to the copied hand.
        eq = ET.SubElement(root, 'equality')
        ET.SubElement(eq, 'weld', body1='hand_target', body2='openarm_right_hand', solref='0.01 1')
        for e in self.source.findall('equality/joint'):
            if 'openarm_right_finger' in e.get('joint1', ''):
                eq.append(deepcopy(e))
        acts = ET.SubElement(root, 'actuator')
        for a in self.source.findall('actuator/*'):
            if a.get('joint', '').startswith('openarm_right_finger') or a.get('tendon') == 'split_right':
                acts.append(deepcopy(a))
        if not len(acts) or any(a.tag != 'position' for a in acts):
            raise ValueError('Expected existing right-finger position actuators')
        tendon = ET.SubElement(root, 'tendon')
        for e in self.source.findall('tendon/fixed'):
            if e.get('name') == 'split_right':
                tendon.append(deepcopy(e))
        contact = ET.SubElement(root, 'contact')
        for e in self.source.findall('contact/exclude'):
            if e.get('body1', '').startswith('openarm_right_hand'):
                contact.append(deepcopy(e))
        # Retain only assets referenced by this environment, following material dependencies.
        available = {a.get('name'): a for a in self.source.find('asset')}
        needed = {e.get(k) for e in root.iter() for k in ('mesh', 'material', 'texture') if e.get(k)}
        while needed:
            name = min(needed)
            needed.remove(name)
            if any(a.get('name') == name for a in asset):
                continue
            if name not in available:
                # Unused inherited visual class in upstream default has a missing material.
                if name == 'visualgeom':
                    ET.SubElement(asset, 'material', name=name, rgba='0.6 0.6 0.6 1')
                    continue
                raise ValueError(f'Missing model asset {name}')
            a = deepcopy(available[name])
            if a.get('file'):
                file = a.get('file')
                if a.tag == 'mesh' and not Path(file).is_absolute() and not file.startswith('assets/'):
                    file = str(SHARE / 'assets/openarm_mujoco_v1/meshes' / file)
                a.set('file', str(resolve_asset(file, self.file)))
            if a.get('texture'):
                needed.add(a.get('texture'))
            asset.append(a)
        return ET.tostring(root, encoding='unicode')

    def fingerprint(self):
        h = hashlib.sha256(self.file.read_bytes() + self.minimal_xml().encode())
        for asset in ET.fromstring(self.minimal_xml()).find('asset'):
            if asset.get('file'):
                h.update(Path(asset.get('file')).read_bytes())
        h.update(np.asarray(self.mesh.vertices).tobytes())
        h.update(np.asarray(self.mesh.faces).tobytes())
        return h.hexdigest()
