"""Physical trials using extracted OpenArm geometry and actuators, without an arm."""
from dataclasses import dataclass
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
from .scene import pose, quat, points_transform
from .accessibility import ApproachChecker

@dataclass
class Physics:
    timestep: float = 0.002
    settle: float = 0.8
    approach: float = 0.6
    close: float = 0.8
    lift_time: float = 1.0
    hold: float = 0.8
    pregrasp_distance: float = 0.06
    max_pregrasp_distance: float = 0.5
    exterior_clearance: float = 0.01
    approach_clearance: float = 0.0002
    approach_max_step: float = 0.005
    approach_max_checks: int = 4096
    terminal_contact_distance: float = 0.001
    terminal_contact_step: float = 0.00005
    contact_patch_radius: float = 0.025
    contact_penetration_tolerance: float = 0.0001
    max_approach_speed: float = 0.1
    lift_distance: float = 0.08
    min_lift: float = 0.04
    max_slip: float = 0.015
    max_rotation_deg: float = 25
    penetration_tolerance: float = 0.001
    trials: int = 1
    orientation_deg: float = 0
    loading_tilt_deg: float = 0
    approach_noise_m: float = 0
    approach_noise_deg: float = 0
    friction_low: float = 1
    friction_high: float = 1
    mass_low: float = 1
    mass_high: float = 1
    seed: int = 42

class Environment:
    def __init__(self, scene, settings=None):
        self.scene, self.settings = scene, settings or Physics()
        self.xml = scene.minimal_xml(self.settings.timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hand = m.body('openarm_right_hand').id
        self.tcp = m.body('openarm_right_hand_tcp').id
        self.object = m.body(scene.name).id
        if not np.isclose(m.body_mass[self.object], scene.object['mass']):
            raise ValueError('Scene YAML mass differs from generated XML; rebuild the scene')
        self.mocap = m.body('hand_target').mocapid[0]
        self.hand_q = m.joint('hand_free').qposadr[0]
        self.object_q = m.jnt_qposadr[m.body_jntadr[self.object]]
        self.finger_joints = [m.joint(f'openarm_right_finger_joint{i}').id for i in (1, 2)]
        self.finger_q = m.jnt_qposadr[self.finger_joints]
        self.finger_bodies = {m.body(f'openarm_right_{side}_finger').id for side in ('left', 'right')}
        self.finger_contact_index = {m.body('openarm_right_right_finger').id:0, m.body('openarm_right_left_finger').id:1}
        self.gripper_bodies = self.finger_bodies | {self.hand, self.tcp}
        self.gripper_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.gripper_bodies and m.geom_contype[g]]
        self.object_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.object and m.geom_contype[g]]
        self.base_gravity = m.opt.gravity.copy()
        self.base_friction = m.geom_friction.copy()
        self.base_mass, self.base_inertia = m.body_mass.copy(), m.body_inertia.copy()
        self.tcp_hand = pose(m.body_pos[self.tcp], m.body_quat[self.tcp])
        self.reset()
        self.data.qpos[self.finger_q] = self.model.jnt_range[self.finger_joints, 0]
        mujoco.mj_forward(self.model, self.data)
        # Calibrate opposing inner faces from actual finger collision vertices.
        bounds = []
        for side in ('right', 'left'):
            geoms = [g for g in self.gripper_geoms if m.geom_bodyid[g] == m.body(f'openarm_right_{side}_finger').id]
            vertices = np.concatenate([self.geom_mesh(g).vertices for g in geoms])
            vertices = points_transform(vertices, np.linalg.inv(self.hand_pose()))
            bounds.append(np.array([vertices.min(axis=0), vertices.max(axis=0)]))
        self.closed_gap = float(bounds[1][0, 1] - bounds[0][1, 1])
        # Distal finger region, derived from mesh bounds; avoids centering on finger root.
        center = np.array([(bounds[0][:, 0].mean()+bounds[1][:, 0].mean())/2,
                           (bounds[0][1, 1]+bounds[1][0, 1])/2,
                           min(bounds[0][1, 2], bounds[1][1, 2]) - min(np.ptp(bounds[0], axis=0)[2], np.ptp(bounds[1], axis=0)[2])*0.2])
        self.contact_center_tcp = points_transform(center[None], np.linalg.inv(self.tcp_hand))[0]
        limits = m.jnt_range[self.finger_joints]
        self.min_width = max(0., self.closed_gap + limits[:, 0].sum())
        self.max_width = self.closed_gap + limits[:, 1].sum()
        self.approach_checker = ApproachChecker(self)

    def geom_mesh(self, g):
        m, d = self.model, self.data
        kind = m.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fs, fc = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            mesh = trimesh.Trimesh(m.mesh_vert[start:start+count].copy(), m.mesh_face[fs:fs+fc].copy(), process=False)
            # MuJoCo collisions use the convex hull of each mesh component.
            mesh = mesh.convex_hull
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2*m.geom_size[g])
        else:
            raise ValueError(f'Unsupported gripper collision geom type {kind}')
        t = np.eye(4); t[:3,:3] = d.geom_xmat[g].reshape(3,3); t[:3,3] = d.geom_xpos[g]
        mesh.apply_transform(t)
        return mesh

    def body_pose(self, body):
        d = self.data
        return pose(d.xpos[body], d.xquat[body])

    def hand_pose(self):
        return self.body_pose(self.hand)

    def object_pose(self):
        return self.body_pose(self.object)

    def reset(self, object_pose=None, friction=1., mass=1.):
        m, d = self.model, self.data
        m.body_gravcomp[self.object] = 1.0 if self.scene.environment == 'minimal' and self.scene.support == 'none' else 0.0
        m.opt.gravity[:] = self.base_gravity
        m.geom_friction[:] = self.base_friction
        m.geom_friction[self.object_geoms + self.gripper_geoms] *= friction
        # Preserve the original COM and scale the original inertia with mass.
        m.body_mass[:] = self.base_mass; m.body_inertia[:] = self.base_inertia
        m.body_mass[self.object] *= mass; m.body_inertia[self.object] *= mass
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        t = self.scene.initial_pose if object_pose is None else object_pose
        d.qpos[self.object_q:self.object_q+7] = np.r_[t[:3,3], quat(t)]
        d.qpos[self.finger_q] = m.jnt_range[self.finger_joints, 1]
        d.ctrl[:] = m.actuator_ctrlrange[:, 1]
        mujoco.mj_forward(m, d)
        self.place(pose([0, 0, 2]))

    def command(self, tcp_pose):
        hand = tcp_pose @ np.linalg.inv(self.tcp_hand)
        self.data.mocap_pos[self.mocap] = hand[:3, 3]
        self.data.mocap_quat[self.mocap] = quat(hand)

    def place(self, tcp_pose, width=None):
        """Initialization only. Active trials exclusively move the mocap target."""
        self.command(tcp_pose)
        hand = tcp_pose @ np.linalg.inv(self.tcp_hand)
        self.data.qpos[self.hand_q:self.hand_q+7] = np.r_[hand[:3,3], quat(hand)]
        if width is not None:
            self.data.qpos[self.finger_q] = np.clip((width-self.closed_gap)/2, self.model.jnt_range[self.finger_joints,0], self.model.jnt_range[self.finger_joints,1])
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def contacts(self, record=None, phase=None, remaining=None):
        out = []
        for contact_id, c in enumerate(self.data.contact):
            b1, b2 = (int(self.model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            if not ({b1,b2} & self.gripper_bodies):
                continue
            force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, contact_id, force)
            # Contact frame rows are axes; force is on geom2, normal is axis 0.
            world_force = c.frame.reshape(3,3).T @ force[:3]
            finger_target = self.object in (b1,b2) and bool({b1,b2} & self.finger_bodies)
            intended = finger_target
            patch_distance = None
            if record is not None and finger_target:
                finger = next(iter({b1,b2} & self.finger_bodies))
                patch = points_transform(np.asarray(record['contact_points_object']),self.object_pose())[self.finger_contact_index[finger]]
                patch_distance = float(np.linalg.norm(c.pos-patch))
                permitted_phase = phase in ('close','lift','hold') or (phase=='approach' and remaining is not None and remaining<=self.settings.terminal_contact_distance)
                intended = bool(permitted_phase and patch_distance<=self.settings.contact_patch_radius)
                if phase=='approach' and c.dist < -self.settings.contact_penetration_tolerance:
                    intended=False
            self_contact = b1 in self.gripper_bodies and b2 in self.gripper_bodies
            out.append(dict(geom1=int(c.geom1), geom2=int(c.geom2), geom1_name=self.model.geom(c.geom1).name,
                geom2_name=self.model.geom(c.geom2).name, position=c.pos.tolist(), distance=float(c.dist),
                effective_friction=c.friction.tolist(), patch_distance=patch_distance, finger_target=finger_target, force_contact=force.tolist(), force_on_geom2_world=world_force.tolist(), intended=intended,
                finger_body=next(iter({b1,b2} & self.finger_bodies), None), unwanted=not intended and not self_contact))
        return out

    def screen(self, tcp_pose, width):
        self.place(tcp_pose, width)
        reasons = []
        for c in self.contacts():
            if c['distance'] < -self.settings.penetration_tolerance:
                if c['unwanted']:
                    reasons.append('palm_or_environment_penetration')
                elif c['intended']:
                    reasons.append('finger_penetration')
        return sorted(set(reasons))

    def screen_record(self, record):
        """Nominal classification, separate from perturbed physical trial labels."""
        self.reset()
        if self.scene.environment != 'minimal' or self.scene.support != 'none':
            for _ in range(round(self.settings.settle/self.settings.timestep)):
                mujoco.mj_step(self.model,self.data)
        target=self.object_pose() @ np.asarray(record['grasp_pose_object'])
        width=min(self.max_width,record['gripper_width']+.012)
        return self.approach_checker.prepare(target,width,record)

    def run(self, record, trial_id):
        s, m, d = self.settings, self.model, self.data
        rng = np.random.default_rng(np.random.SeedSequence([s.seed, record['grasp_id'], trial_id]))
        initial = self.scene.initial_pose.copy()
        perturb = Rotation.from_rotvec(rng.normal(size=3)*np.deg2rad(s.orientation_deg)).as_matrix()
        initial[:3,:3] = perturb @ initial[:3,:3]
        friction = float(rng.uniform(s.friction_low, s.friction_high))
        mass = float(rng.uniform(s.mass_low, s.mass_high))
        self.reset(initial, friction, mass)
        for _ in range(round(s.settle/s.timestep)):
            mujoco.mj_step(m,d)
        # Always get current object pose after settling, never substitute the YAML pose.
        object_start = self.object_pose()
        target = object_start @ np.asarray(record['grasp_pose_object'])
        target[:3,3] += rng.normal(size=3)*s.approach_noise_m
        target[:3,:3] = target[:3,:3] @ Rotation.from_rotvec(rng.normal(size=3)*np.deg2rad(s.approach_noise_deg)).as_matrix()
        open_width = min(self.max_width, record['gripper_width']+0.012)
        approach_check=self.approach_checker.prepare(target,open_width,record)
        reasons=approach_check['rejection_reasons']
        pre=np.asarray(approach_check['pregrasp_pose_world'])
        result = dict(trial_id=trial_id, seed=[s.seed, record['grasp_id'], trial_id],
            evaluation_kind='loading_stability' if s.loading_tilt_deg else ('free_space_stability' if self.scene.support == 'none' and self.scene.environment == 'minimal' else 'gripper_accessibility_and_stability'), orientation_perturbation=perturb.tolist(),
            friction_scale=friction, mass_scale=mass, mass_kg=float(m.body_mass[self.object]),
            object_friction=m.geom_friction[self.object_geoms].tolist(), object_pose_start=object_start.tolist(),
            status='rejected_geometry' if reasons else 'completed', success=None, rejection_reasons=reasons,
            approach_screen=approach_check,approach_valid=approach_check['valid'],failure_reasons=[])
        if reasons:
            return result
        self.place(pre, open_width)
        d.ctrl[:] = np.clip((open_width-self.closed_gap)/2, m.actuator_ctrlrange[:,0], m.actuator_ctrlrange[:,1])
        trace, unwanted = [], False
        runtime_violation = None
        hold_checks = []
        def motion(a, b, duration, phase):
            nonlocal unwanted, runtime_violation
            steps = max(1, round(duration/s.timestep))
            for k in range(steps):
                t = a.copy(); t[:3,3] = a[:3,3]+(b[:3,3]-a[:3,3])*((k+1)/steps)
                self.command(t); mujoco.mj_step(m,d)
                cs = self.contacts(record,phase,float(np.linalg.norm(b[:3,3]-t[:3,3])) if phase=='approach' else None)
                forbidden = [c for c in cs if c['unwanted'] and c['distance']<0]
                unwanted |= any(c['unwanted'] and c['distance'] < 0 for c in cs)
                if phase == 'hold':
                    rel = np.linalg.inv(self.body_pose(self.tcp)) @ self.object_pose()
                    fingers = {c['finger_body'] for c in cs if c['intended'] and np.linalg.norm(c['force_contact'][:3]) > 0.01}
                    support = any(self.object in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]) and any(m.geom_bodyid[gg] not in self.gripper_bodies|{self.object} for gg in (c.geom1,c.geom2)) for c in d.contact)
                    hold_checks.append((float(np.linalg.norm(rel[:3,3]-reference[:3,3])),
                        float(np.rad2deg(Rotation.from_matrix(rel[:3,:3] @ reference[:3,:3].T).magnitude())),
                        float(d.xpos[self.object,2]-object_start[2,3]), len(fingers)==2, support))
                if forbidden or k % max(1, round(0.02/s.timestep)) == 0 or k == steps-1:
                    trace.append(dict(support_contact=any(self.object in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]) and any(m.geom_bodyid[gg] not in self.gripper_bodies|{self.object} for gg in (c.geom1,c.geom2)) for c in d.contact), time=float(d.time), phase=phase, object_pose=self.object_pose().tolist(),
                        tcp_pose=self.body_pose(self.tcp).tolist(), finger_qpos=d.qpos[self.finger_q].tolist(),
                        actuator_force=d.actuator_force.tolist(), contacts=cs))
                if forbidden:
                    runtime_violation=dict(phase=phase,time=float(d.time),contact=forbidden[0])
                    return False
            return True
        approach_duration=max(s.approach,approach_check['pregrasp_distance']/s.max_approach_speed)
        result['approach_duration']=approach_duration
        active=motion(pre,target,approach_duration,'approach')
        if active:
            d.ctrl[:] = m.actuator_ctrlrange[:,0]
            active=motion(target,target,s.close,'close')
        m.body_gravcomp[self.object] = 0.0
        result['temporary_fixture'] = self.scene.environment == 'minimal' and self.scene.support == 'none'
        reference = np.linalg.inv(self.body_pose(self.tcp)) @ self.object_pose()
        # Isolate loading robustness: rotate gravity only after closure, with the same grasp.
        loading = Rotation.from_rotvec(rng.normal(size=3)*np.deg2rad(s.loading_tilt_deg)).as_matrix()
        m.opt.gravity[:] = loading @ self.base_gravity
        result['loading_gravity_world'] = m.opt.gravity.tolist()
        lift = target.copy(); lift[2,3] += s.lift_distance
        if active:
            active=motion(target,lift,s.lift_time,'lift')
        if active:
            motion(lift,lift,s.hold,'hold')
        if runtime_violation:
            result['runtime_violation']=runtime_violation
            result['failure_reasons']=['unexpected_'+runtime_violation['phase']+'_contact']
        slips = [v[0] for v in hold_checks]
        angles = [v[1] for v in hold_checks]
        heights = [v[2] for v in hold_checks]
        success = bool(hold_checks and min(heights) >= s.min_lift and max(slips) <= s.max_slip and
            max(angles) <= s.max_rotation_deg and all(v[3] and not v[4] for v in hold_checks) and not unwanted)
        result.update(success=success, slip_distance=max(slips, default=None), rotation_slip_deg=max(angles, default=None),
            min_hold_lift=min(heights, default=None), unwanted_contacts=unwanted, trace=trace,
            joint_positions_final=d.qpos[self.finger_q].tolist())
        return result
