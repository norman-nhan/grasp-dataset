"""Terminal entry point for each independently runnable dataset stage."""
#!/usr/bin/env python3

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from . import __version__
from .accessibility import APPROACH_VERSION, EVALUATOR_VERSION, FIELDS
from .scene import Scene, SHARE
from .geometry import Sampling, generate
from .simulation import Environment, Physics
from .storage import SCHEMA, save, load, aggregate, externalize_trace
from .affordance import load_points, overlap


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['candidates', 'evaluate', 'label', 'full', 'visualize', 'inspect', 'screen'])
    p.add_argument('--scene', type=Path, default=SHARE/'config/scene_cube.yaml')
    p.add_argument('--model', type=Path, help='Existing URDF-corrected generated scene XML')
    p.add_argument('--object')
    p.add_argument('--environment', choices=['minimal','source'], help='Default: minimal; source is only for reproducing old scene experiments')
    p.add_argument('--support', choices=['plane','none'], help='Minimal environment support; none uses a temporary gravity-compensated fixture until closure')
    p.add_argument('--all', action='store_true', help='Show all candidates with keyboard browsing')
    p.add_argument('--output', type=Path, required=True, help='Dataset JSON (also the input for later stages)')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--trace-storage', choices=['inline','external'], help='Compressed external traces avoid large checkpoint rewrites; default inline')
    p.add_argument('--num-candidates', type=int, default=100)
    p.add_argument('--num-trials', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--surface-points', type=int, default=2500)
    p.add_argument('--approaches', type=int, default=8)
    p.add_argument('--neighbors', type=int, default=96)
    p.add_argument('--pairs-per-point', type=int, default=1, help='Limit retained pairs per surface point for spatial coverage')
    p.add_argument('--min-width', type=float, default=0.005)
    p.add_argument('--max-width', type=float)
    p.add_argument('--friction', type=float, default=0.8, help='Sampling Coulomb coefficient')
    p.add_argument('--angular-tolerance-deg', type=float, default=30)
    p.add_argument('--physics', type=Path, help='JSON overrides for Physics settings; num-trials and seed come from CLI')
    p.add_argument('--annotation', type=Path, help='Repository .laso.json export or trusted LASO pickle directory')
    p.add_argument('--annotation-transform', type=Path, help='JSON 4x4 annotation-to-object similarity, REQUIRED for labeling')
    p.add_argument('--affordance', default='grasp')
    p.add_argument('--shape-id')
    p.add_argument('--split', default='val', choices=['train','val','test'])
    p.add_argument('--clearance', type=float, default=0.01)
    p.add_argument('--overlap-threshold', type=float, default=0.2)
    p.add_argument('--grasp-id', type=int, default=0)
    p.add_argument('--filter', choices=['all','success','failure','rejected','pending','accessible','inaccessible'], default='all')
    p.add_argument('--render', type=Path, help='Render visualization to PNG without opening a window (use MUJOCO_GL=egl headlessly)')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == 'inspect':
        data = load(args.output)
        print(json.dumps(dict(metadata=data['metadata'], candidates=len(data['grasps']),
            successful=sum(g['physical_success'] is True for g in data['grasps']),
            measured_failures=sum(g['physical_success'] is False for g in data['grasps']),
            unevaluated=sum(g['physical_success'] is None for g in data['grasps'])), indent=2))
        return
    previous = None
    if args.output.exists():
        if args.mode in ('candidates', 'full') and not args.resume:
            raise ValueError('Output exists; use --resume or a new output path')
        previous = load(args.output)
        # Saved scene/model identities take precedence for later stage commands.
        args.scene = Path(previous['metadata']['scene'])
        args.model = Path(previous['metadata']['model'])
        args.object = previous['metadata']['object_id']
    elif args.mode not in ('candidates', 'full'):
        raise FileNotFoundError(args.output)
    environment = args.environment or (previous['metadata'].get('environment','source') if previous else 'minimal')
    trace_storage = args.trace_storage or (previous['metadata'].get('trace_storage','inline') if previous else 'inline')
    if previous and trace_storage != previous['metadata'].get('trace_storage','inline'):
        raise ValueError('Trace storage changed; use a new output dataset')
    support = args.support or (previous['metadata'].get('support','plane') if previous else 'plane')
    scene = Scene(args.scene, args.object, args.model, environment, support)
    overrides = json.loads(args.physics.read_text()) if args.physics else {}
    physics = Physics(**overrides)
    physics.trials, physics.seed = args.num_trials, args.seed
    if previous and args.mode in ('evaluate', 'full'):
        saved = previous['metadata'].get('physics')
        if saved and not args.physics:
            physics = Physics(**saved)
            physics.trials = max(physics.trials, args.num_trials)
        if saved and any(saved[k] != asdict(physics)[k] for k in saved if k != 'trials'):
            raise ValueError('Physics or seed changed; use a new output dataset')
        if saved and physics.trials < saved['trials']:
            raise ValueError('Cannot reduce trial count on resume')
    if previous and args.mode=='screen':
        physics=Physics(**((previous['metadata'].get('physics') or {})|overrides))
    if not all(np.isfinite(v) for v in asdict(physics).values()):
        raise ValueError('All physics settings must be finite')
    if any(getattr(physics,k) < 0 for k in ('settle','pregrasp_distance','max_slip','max_rotation_deg','penetration_tolerance','orientation_deg','loading_tilt_deg','approach_noise_m','approach_noise_deg')) or physics.min_lift <= 0:
        raise ValueError('Invalid negative physics tolerance, duration, or perturbation')
    if any(getattr(physics,k)<=0 for k in ('max_pregrasp_distance','exterior_clearance','approach_max_step','terminal_contact_distance','terminal_contact_step','contact_patch_radius','max_approach_speed')) or physics.approach_clearance<0 or physics.contact_penetration_tolerance<0:
        raise ValueError('Invalid approach checking distance, clearance or speed')
    if not isinstance(physics.approach_max_checks,int) or physics.approach_max_checks<1:
        raise ValueError('approach_max_checks must be a positive integer')
    if physics.max_pregrasp_distance<physics.pregrasp_distance:
        raise ValueError('Maximum pregrasp distance is smaller than minimum backoff')
    if physics.loading_tilt_deg and (physics.orientation_deg or physics.approach_noise_m or physics.approach_noise_deg):
        raise ValueError('Run loading stability separately from orientation/approach accessibility perturbations')
    if physics.trials < 1 or any(getattr(physics, k) <= 0 for k in ('timestep','approach','close','lift_time','hold','lift_distance')):
        raise ValueError('Trial count, integration timestep, and phase durations must be positive')
    if not 0 < physics.friction_low <= physics.friction_high or not 0 < physics.mass_low <= physics.mass_high:
        raise ValueError('Invalid friction/mass ranges')
    env = Environment(scene, physics)
    if previous:
        data = previous
        if args.mode in ('evaluate','full') and any(g['trial_results'] for g in data['grasps']) and data['metadata'].get('evaluator_version')!=EVALUATOR_VERSION:
            raise ValueError('Approach evaluator changed; keep historical trials and generate a new output dataset. Use screen to classify existing poses without changing old physical labels.')
        if data['metadata']['mujoco_version'] != mujoco.__version__ or data['metadata']['version'] != __version__:
            raise ValueError('Simulator or generator version changed; start a new dataset')
        if data['metadata']['scene_fingerprint'] != scene.fingerprint():
            raise ValueError('Scene/mesh/model changed since generation; start a new dataset')
    else:
        sampling = Sampling(args.surface_points,args.num_candidates,args.approaches,args.neighbors,args.min_width,args.max_width,
            args.friction,args.angular_tolerance_deg,args.seed,args.pairs_per_point)
        records = generate(scene.mesh, env, sampling)
        if not records:
            raise ValueError('No antipodal pairs found; inspect mesh scale and opening limits')
        data = dict(schema=SCHEMA, metadata=dict(version=__version__, scene=str(scene.file), model=str(scene.model_file),
            evaluator_version=EVALUATOR_VERSION, environment=environment, support=support, trace_storage=trace_storage, scene_fingerprint=scene.fingerprint(), object_id=scene.name, object_config=scene.object, object_provenance=scene.config.get("provenance"),
            mesh_watertight=scene.mesh_watertight, degenerate_faces_removed=scene.degenerate_faces_removed, gripper_configuration='extracted-openarm-right/'+scene.fingerprint(),
            gripper_limits_m=[env.min_width,env.max_width], contact_center_tcp=env.contact_center_tcp.tolist(),
            hand_T_tcp=env.tcp_hand.tolist(), sampling=asdict(sampling), physics=None, mujoco_version=mujoco.__version__,
            frames='metres; right handed; object_T_tcp; TCP axes match hand (+Y closing, +Z approach); quaternion wxyz',
            labels='physical_success=any completed trial; physical_success_rate=success/completed; execution_success_rate=success/all; handover_suitable=overlap<=threshold'), grasps=records)
        for record in records:
            record['object_id'] = scene.name
        save(args.output, data)
    if args.mode in ('screen','evaluate','full'):
        for index,g in enumerate(data['grasps']):
            existing=g.get('approach_screen',{})
            if existing.get('version')!=APPROACH_VERSION or existing.get('settings_fingerprint')!=env.approach_checker.signature:
                g['approach_screen']=env.screen_record(g)
                g['approach_valid']=g['approach_screen']['valid']
                if index%16==15:
                    save(args.output,data)
                print(f"grasp={g['grasp_id']} accessible={g['approach_valid']} reasons={g['approach_screen']['rejection_reasons']}",flush=True)
        data['metadata']['approach_screen_version']=APPROACH_VERSION
        data['metadata']['nominal_approach_settings']={k:getattr(physics,k) for k in FIELDS}
        save(args.output,data)
    if args.mode in ('evaluate', 'full'):
        data['metadata']['evaluator_version']=EVALUATOR_VERSION
        for record in data['grasps']:
            for trial in record['trial_results']:
                if trial.get('trace_file') and not (args.output.parent/trial['trace_file']).is_file():
                    raise FileNotFoundError(f"Missing saved trial trace: {trial['trace_file']}")
        data['metadata']['physics'] = asdict(physics)
        for record in data['grasps']:
            aggregate(record, physics.trials)
        save(args.output, data)
        for g in data['grasps']:
            completed = {t['trial_id'] for t in g['trial_results']}
            for trial in range(physics.trials):
                if trial in completed:
                    continue
                result = env.run(g, trial)
                if trace_storage == 'external':
                    externalize_trace(args.output, g['grasp_id'], result)
                g['trial_results'].append(result)
                aggregate(g, physics.trials)
                save(args.output, data)
                print(f"grasp={g['grasp_id']} trial={trial} status={g['trial_results'][-1]['status']} success={g['trial_results'][-1]['success']}", flush=True)
    if args.mode == 'label' or (args.mode == 'full' and args.annotation):
        if not args.annotation or not args.annotation_transform:
            raise ValueError('Labeling requires --annotation and --annotation-transform; no registration is assumed')
        if not 0 <= args.overlap_threshold <= 1:
            raise ValueError('Overlap threshold must lie in [0,1]')
        transform = json.loads(args.annotation_transform.read_text())
        points = load_points(args.annotation, transform, args.affordance, args.split, args.shape_id)
        data['metadata']['annotation'] = dict(file=str(args.annotation.resolve()), transform=transform,
            affordance=args.affordance, split=args.split, shape_id=args.shape_id, clearance=args.clearance,
            overlap_threshold=args.overlap_threshold, points_object=points.tolist(),
            points_sha256=hashlib.sha256(points.tobytes()).hexdigest(), metric='fraction of LASO region points inside or within clearance of gripper convex collision components at nominal width')
        for g in data['grasps']:
            g['affordance_overlap'] = overlap(points, g, env, args.clearance)
            g['handover_suitable'] = g['affordance_overlap'] <= args.overlap_threshold
        save(args.output, data)
    if args.mode == 'visualize':
        from .viewer import show
        show(data, env, args.grasp_id, all_candidates=args.all, result_filter=args.filter, render=args.render)
    else:
        print(f"Saved {len(data['grasps'])} candidates to {args.output}")
        if args.mode == 'full' and not args.annotation:
            print('LASO labels are null: no annotation registration supplied.')

if __name__ == '__main__':
    main()
