"""Straight-line accessibility using exterior backoff and signed-distance advancement.

The path has fixed orientation and opening. State queries use scratch MjData;
validation never teleports or resets the active simulation. Convex component
signed distances bound safe translation steps. Terminal localized contacts use
small steps and an explicit penetration tolerance; this is not a general SE(3)
planner or exact continuous collision detection for arbitrary triangle meshes.
"""
import hashlib
import itertools
import json
import numpy as np
import mujoco
from .scene import points_transform, quat

APPROACH_VERSION = 'exterior-sweep-v1'
EVALUATOR_VERSION = 'approach-contact-v2'
FIELDS = ('settle','timestep','pregrasp_distance','max_pregrasp_distance','exterior_clearance',
          'approach_clearance','approach_max_step','approach_max_checks',
          'terminal_contact_distance','terminal_contact_step',
          'contact_patch_radius','contact_penetration_tolerance')


class ApproachChecker:
    def __init__(self, env):
        self.env, self.model = env, env.model
        self.data = mujoco.MjData(self.model)
        self.obstacles = [g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] not in env.gripper_bodies and self.model.geom_contype[g]]
        self.pairs = [(g,h) for g in env.gripper_geoms for h in self.obstacles
            if (self.model.geom_contype[g] & self.model.geom_conaffinity[h]) or
               (self.model.geom_contype[h] & self.model.geom_conaffinity[g])]
        self.signs = np.asarray(list(itertools.product((-1,1), repeat=3)))

    @property
    def signature(self):
        settings = {k:getattr(self.env.settings,k) for k in FIELDS}
        return hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()

    def place(self, tcp, width):
        env, m, d = self.env, self.model, self.data
        hand = tcp @ np.linalg.inv(env.tcp_hand)
        d.qpos[env.hand_q:env.hand_q+7] = np.r_[hand[:3,3],quat(hand)]
        d.qpos[env.finger_q] = np.clip((width-env.closed_gap)/2,
            m.jnt_range[env.finger_joints,0],m.jnt_range[env.finger_joints,1])
        mujoco.mj_kinematics(m,d)

    def corners(self, geoms):
        result=[]
        for g in geoms:
            center, half = self.model.geom_aabb[g,:3], self.model.geom_aabb[g,3:]
            local = center + self.signs*half
            result.append(local @ self.data.geom_xmat[g].reshape(3,3).T + self.data.geom_xpos[g])
        return np.concatenate(result)

    def prepare(self, target, width, record):
        env, m, s = self.env, self.model, self.env.settings
        mujoco.mj_copyData(self.data,m,env.data)
        self.place(target,width)
        direction = np.asarray(target[:3,2])
        if not np.isclose(np.linalg.norm(direction),1) or not np.isfinite(target).all():
            raise ValueError('Approach requires a finite rigid TCP pose')
        obj_pose = env.object_pose()
        patches = points_transform(record['contact_points_object'],obj_pose)
        gripper = self.corners(env.gripper_geoms)
        obj = self.corners(env.object_geoms)
        distance = max(s.pregrasp_distance,
            float((gripper@direction).max()-(obj@direction).min()+s.exterior_clearance))
        pre = target.copy();pre[:3,3] -= direction*distance
        report=dict(version=APPROACH_VERSION,settings_fingerprint=self.signature,valid=False,
            exterior_pregrasp=False,pregrasp_distance=distance,checks=0,min_clearance=None,
            pregrasp_pose_world=pre.tolist(),target_pose_world=target.tolist(),
            pregrasp_pose_object=(np.linalg.inv(obj_pose)@pre).tolist(),
            target_pose_object=(np.linalg.inv(obj_pose)@target).tolist(),rejection_reasons=[])
        if distance>s.max_pregrasp_distance:
            report['rejection_reasons']=['exterior_pregrasp_exceeds_limit']
            return report
        self.place(pre,width)
        report['exterior_pregrasp'] = bool((self.corners(env.gripper_geoms)@direction).max()
            <= (obj@direction).min()-s.exterior_clearance+1e-9)
        if not report['exterior_pregrasp']:
            report['rejection_reasons']=['exterior_pregrasp_unavailable']
            return report
        traveled=0.
        for _ in range(s.approach_max_checks):
            tcp=pre.copy();tcp[:3,3] += direction*traveled
            self.place(tcp,width)
            remaining=distance-traveled
            terminal=remaining<=s.terminal_contact_distance+1e-9
            budget=np.inf
            nearest=None
            for g,h in self.pairs:
                segment=np.zeros(6)
                signed=float(mujoco.mj_geomDistance(m,self.data,g,h,2.0,segment))
                if report['min_clearance'] is None or signed<report['min_clearance']:
                    report['min_clearance']=signed
                finger=int(m.geom_bodyid[g]);is_target=h in env.object_geoms
                allowed = (terminal and is_target and finger in env.finger_contact_index and
                    np.linalg.norm(segment[3:]-patches[env.finger_contact_index[finger]])<=s.contact_patch_radius)
                threshold=-s.contact_penetration_tolerance if allowed else s.approach_clearance
                gap=signed-threshold
                if gap < -1e-8:
                    report['checks']+=1
                    kind=('finger' if finger in env.finger_bodies else 'palm') if is_target else 'environment'
                    report.update(rejection_reasons=[f'approach_{kind}_collision'],
                        blocking_geoms=[m.geom(g).name,m.geom(h).name],
                        blocked_distance_from_target=remaining)
                    return report
                if gap<budget:
                    budget=gap;nearest=[m.geom(g).name,m.geom(h).name]
            report['checks']+=1
            if remaining<=1e-9:
                report['valid']=True
                return report
            # For fixed-orientation translation, separation changes no faster
            # than path length. Never impose a minimum step across small gaps.
            step=min(s.approach_max_step,remaining,max(0.,budget)*.8)
            if terminal:
                step=min(step,s.terminal_contact_step)
            elif not terminal:
                step=min(step,remaining-s.terminal_contact_distance)
            if step<1e-8:
                report.update(rejection_reasons=['approach_clearance_blocked'],blocking_geoms=nearest,
                    blocked_distance_from_target=remaining)
                return report
            traveled+=step
        report['rejection_reasons']=['approach_check_limit']
        return report
