"""Browse actual saved grasps with native MuJoCo; overlay all object-relative poses."""
from collections import deque
import struct
import time
import zlib
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from .scene import points_transform
from .accessibility import EVALUATOR_VERSION


def color(record, threshold=1):
    if record.get('approach_valid') is False:
        return [.55,.55,.55,1]
    if record['affordance_overlap'] is not None and record['affordance_overlap']>threshold:
        return [0.9,0.2,0.9,1]
    if record['physical_success'] is True:
        return [0.2,0.9,0.2,1]
    if record['physical_success'] is False:
        return [0.95,0.15,0.1,1]
    if record['evaluation_status']=='rejected_geometry':
        return [.55,.55,.55,1]
    return [.15,.55,1,1]


def select(dataset, result_filter='all'):
    def matches(g):
        return (result_filter=='all' or result_filter=='accessible' and g.get('approach_valid') is True or
            result_filter=='inaccessible' and g.get('approach_valid') is False or result_filter=='success' and g['physical_success'] is True or
            result_filter=='failure' and g['physical_success'] is False or
            result_filter=='rejected' and g['evaluation_status']=='rejected_geometry' or
            result_filter=='pending' and g['evaluation_status'] in ('pending','partial'))
    records=[g for g in dataset['grasps'] if matches(g)]
    if not records:
        raise ValueError(f'No candidates match filter {result_filter!r}')
    return records


def decorate(scene, dataset, records, obj, selected, all_candidates):
    """Overlays are contact segments and approach stems, not collision geometry."""
    annotation=dataset['metadata'].get('annotation',{})
    threshold=annotation.get('overlap_threshold',1)
    def sphere(point,radius,rgba):
        if scene.ngeom>=scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius]*3),np.asarray(point),np.eye(3).ravel(),np.array(rgba,dtype=np.float32))
        scene.ngeom+=1
    def segment(a,b,rgba,arrow=False):
        if scene.ngeom>=scene.maxgeom or np.linalg.norm(a-b)<1e-10:
            return
        geom=scene.geoms[scene.ngeom]
        kind=mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
        mujoco.mjv_initGeom(geom,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(rgba,dtype=np.float32))
        mujoco.mjv_connector(geom,kind,0.0006,np.asarray(a),np.asarray(b))
        scene.ngeom+=1
    if annotation.get('points_object'):
        for point in points_transform(annotation['points_object'],obj):
            sphere(point,.0015,[1,.8,0,.8])
    for g in (records if all_candidates else [selected]):
        t=obj@np.asarray(g['grasp_pose_object'])
        contacts=points_transform(g['contact_points_object'],obj)
        midpoint=contacts.mean(axis=0)
        rgba=color(g,threshold)
        segment(contacts[0],contacts[1],rgba)
        segment(midpoint-t[:3,2]*.035,midpoint,rgba,arrow=True)
    report=selected.get('approach_screen',{})
    if report.get('pregrasp_pose_object'):
        pre=obj@np.asarray(report['pregrasp_pose_object'])
        target=obj@np.asarray(report['target_pose_object'])
        segment(pre[:3,3],target[:3,3],[0,.8,1,1] if report['valid'] else [1,.45,0,1],arrow=True)
        if report.get('blocked_distance_from_target') is not None:
            point=target[:3,3]-target[:3,2]*report['blocked_distance_from_target']
            sphere(point,.004,[1,0,0,1])
        sphere(pre[:3,3],.003,[.1,.3,1,1])
    for point in points_transform(selected['contact_points_object'],obj):
        sphere(point,.003,[0,.8,1,1])


def place(env,dataset,g):
    env.reset()
    obj=env.object_pose()
    env.place(obj@np.asarray(g['grasp_pose_object']),g['gripper_width'])
    rgba=color(g,dataset['metadata'].get('annotation',{}).get('overlap_threshold',1))
    for geom in env.gripper_geoms:
        env.model.geom_rgba[geom]=rgba
    return obj


def show(dataset,env,grasp_id,all_candidates=False,result_filter='all',render=None):
    if dataset['metadata'].get('evaluator_version')!=EVALUATOR_VERSION and any(g.get('trial_results') for g in dataset['grasps']):
        print('Historical physical labels: these trials predate the swept-approach evaluator. Nominal accessibility labels are separate.',flush=True)
    records=select(dataset,result_filter)
    matches=[i for i,g in enumerate(records) if g['grasp_id']==grasp_id]
    if not matches and result_filter=='all':
        raise ValueError(f'Unknown grasp ID {grasp_id}')
    index=matches[0] if matches else 0
    obj=place(env,dataset,records[index])
    camera=mujoco.MjvCamera();camera.lookat[:]=obj[:3,3]
    camera.distance=max(.3,np.linalg.norm(env.scene.mesh.extents)*3)
    camera.azimuth=135;camera.elevation=-25
    options=mujoco.MjvOption();options.geomgroup[2]=0;options.geomgroup[3]=1
    def announce():
        g=records[index]
        print(f"[{index+1}/{len(records)}] grasp={g['grasp_id']} status={g['evaluation_status']} accessible={g.get('approach_valid')} success={g['physical_success']} overlap={g['affordance_overlap']} approach_reasons={g.get('approach_screen',{}).get('rejection_reasons',[])}",flush=True)
    announce()
    if render:
        with mujoco.Renderer(env.model,480,640) as renderer:
            renderer.update_scene(env.data,camera=camera,scene_option=options)
            decorate(renderer.scene,dataset,records,obj,records[index],all_candidates)
            pixels=renderer.render()
            write_png(render,pixels)
        print(f'Saved {render}')
        return
    print('N / Right: next; P / Left: previous; A: toggle all pose overlays. Green=success; red=failure; gray=inaccessible/rejected; blue=pending; magenta=overlap. Cyan=selected contacts/clear path; blue sphere=exterior start; orange path/red marker=blocked approach. Paused inspection.')
    events=deque()
    with mujoco.viewer.launch_passive(env.model,env.data,key_callback=events.append) as viewer:
        viewer.opt.geomgroup[:]=options.geomgroup
        viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance
        viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
        while viewer.is_running():
            with viewer.lock():
                changed=False
                while events:
                    key=events.popleft()
                    if key in (78,262):
                        index=(index+1)%len(records);changed=True
                    elif key in (80,263):
                        index=(index-1)%len(records);changed=True
                    elif key==65:
                        all_candidates=not all_candidates
                if changed:
                    obj=place(env,dataset,records[index]);announce()
                viewer.user_scn.ngeom=0
                decorate(viewer.user_scn,dataset,records,obj,records[index],all_candidates)
            viewer.sync();time.sleep(.03)


def write_png(path,pixels):
    """Write renderer RGB pixels using the standard library."""
    def chunk(tag,data):
        return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data))
    height,width,_=pixels.shape
    raw=b''.join(b'\x00'+row.tobytes() for row in pixels)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(png)
