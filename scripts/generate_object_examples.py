#!/usr/bin/env python3
"""Resumable bottle/knife/mug generation for YCB and local LASO splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time

from grasp_dataset.storage import load, save

ROOT = Path(__file__).resolve().parents[1]
YCB = {'006_mustard_bottle':'bottle', '021_bleach_cleanser':'bottle',
       '022_windex_bottle':'bottle', '025_mug':'mug', '032_knife':'knife'}


def runner_active(output):
    with (output/'runner.lock').open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle,fcntl.LOCK_UN)
    return False


def catalog(dataset, splits):
    jobs = [dict(key='ycb/'+name,source='ycb',object=name,category=category)
        for name,category in YCB.items()]
    for split in splits:
        with (dataset/f'anno_{split}.pkl').open('rb') as f:
            annotations = pickle.load(f)
        shapes = {(str(a['shape_id']),a['class']) for a in annotations
            if a['class'] in ('bottle','knife','mug')}
        for shape,category in sorted(shapes,key=lambda x:(x[1],x[0])):
            affordances=sorted({a['affordance'] for a in annotations if str(a['shape_id'])==shape})
            jobs.append(dict(key=f'laso/{split}/{category}/{shape}', source='laso',
                object=shape,category=category,split=split,affordances=affordances))
    return jobs


def summary(path):
    records = load(path)['grasps']
    return dict(candidates=len(records),accessible=sum(g.get('approach_valid') is True for g in records),
        successes=sum(g['physical_success'] is True for g in records),
        measured_failures=sum(g['physical_success'] is False for g in records),
        unmeasured=sum(g['physical_success'] is None for g in records),
        laso_labeled=sum(g.get('affordance_overlap') is not None for g in records),
        successful_keepfree=sum(g['physical_success'] is True and g.get('handover_suitable') is True for g in records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT.parent/'laso-reviewer/data/LASO_dataset')
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['val','test','train'])
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/bottle-knife-mug')
    p.add_argument('--num-candidates',type=int,default=384)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,help='Restrict object count for smoke tests; recorded as a different batch scope')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--status',action='store_true')
    a = p.parse_args()
    if a.workers < 1 or a.num_candidates < 1 or (a.limit is not None and a.limit < 1):
        p.error('Counts and workers must be positive')
    if len(set(a.splits)) != len(a.splits):
        p.error('Do not repeat split names')
    output = a.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest_file = output/'manifest.json'
    if a.status:
        manifest = json.loads(manifest_file.read_text())
        from collections import Counter
        print(json.dumps(dict(runner_active=runner_active(output),
            statuses=dict(Counter(j['status'] for j in manifest['jobs'])),
            completed_candidates=sum(j.get('summary',{}).get('candidates',0) for j in manifest['jobs']),
            successes=sum(j.get('summary',{}).get('successes',0) for j in manifest['jobs'])),indent=2))
        return
    # An OS lock survives no crashed process and prevents two resumptions writing
    # the same datasets concurrently.
    with (output/'runner.lock').open('w') as lock_file:
        fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        settings = dict(dataset=str(a.dataset.resolve()),splits=a.splits,
            num_candidates=a.num_candidates,seed=a.seed,limit=a.limit,
            approaches=12,support='none',trials=1,physics='unchanged defaults',
            laso_extent_assumptions_m=dict(bottle=.18,knife=.22,mug=.085))
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            if manifest['settings'] != settings:
                raise ValueError('Batch settings changed; use a new output directory')
        else:
            jobs = catalog(a.dataset,a.splits)
            if a.limit:
                jobs = jobs[:a.limit]
            for job in jobs:
                job.update(status='queued',dataset=str(output/(job['key']+'.json')),
                    log=str(output/'logs'/(job['key']+'.log')))
            manifest = dict(schema='openarm-object-examples/v1',settings=settings,jobs=jobs,
                notes=['YCB annotations remain null: no registered LASO mask for those objects',
                    '022_windex_bottle is an incomplete mesh in the official YCB catalogue',
                    'LASO geometries, sizes, thicknesses and masses are assumed reconstructions'])
        manifest['runner_pid'] = os.getpid()
        manifest['workers'] = a.workers
        manifest['finished'] = False
        # Enrich resumable plans created before affordance availability was
        # recorded. Missing human-grasp masks must not invalidate physics.
        available={j['key']:j.get('affordances',[]) for j in catalog(a.dataset,a.splits)}
        for job in manifest['jobs']:
            if job['status'] in ('preparing','evaluating','rendering'):
                job['status']='queued'
            job['affordances']=available[job['key']]
            job['annotation_status']=('registered_grasp' if 'grasp' in job['affordances']
                else 'missing_grasp_affordance') if job['source']=='laso' else 'unregistered'
        save(manifest_file,manifest)
        print(f"Objects: {len(manifest['jobs'])}; candidates/object: {a.num_candidates}",flush=True)
        if a.plan_only:
            return
        mutex = threading.Lock()
        def update(job,**values):
            with mutex:
                job.update(values)
                save(manifest_file,manifest)
        def execute(command,stream,timeout):
            result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                timeout=timeout,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl'))
            if result.returncode:
                raise RuntimeError(f'Command exited {result.returncode}; see log')
        def run(job):
            started = time.monotonic()
            log = Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
            stage = 'preparing'
            update(job,status=stage,error=None)
            try:
                with log.open('a',buffering=1) as stream:
                    if job['source']=='ycb':
                        scene=ROOT/'assets/ycb'/job['object']/'object.yaml'
                        execute([sys.executable,str(ROOT/'scripts/prepare_ycb.py'),
                            '--objects',job['object']],stream,1200)
                        label_args=[]
                    else:
                        asset=ROOT/'assets/laso-batch-v1'/job['split']/job['category']/job['object']
                        scene=asset/'object.yaml'
                        execute([sys.executable,str(ROOT/'scripts/prepare_laso.py'),
                            '--dataset',str(a.dataset.resolve()),'--split',job['split'],
                            '--shape-id',job['object'],'--category',job['category'],
                            '--directory',str(asset)],stream,1200)
                        label_args=(['--annotation',str(a.dataset.resolve()),'--split',job['split'],
                            '--shape-id',job['object'],'--annotation-transform',str(asset/'annotation_to_object.json')]
                            if 'grasp' in job['affordances'] else [])
                    stage='evaluating';update(job,status=stage,scene=str(scene))
                    execute([sys.executable,str(ROOT/'generate_grasp_dataset.py'),'full',
                        '--scene',str(scene),'--support','none','--num-candidates',str(a.num_candidates),
                        '--approaches','12','--min-width','0.001','--seed',str(a.seed),
                        '--trace-storage','external','--output',job['dataset'],'--resume']+label_args,stream,1800)
                    result_summary=summary(job['dataset'])
                    saved=load(Path(job['dataset']))
                    saved['metadata']['batch_annotation_status']=job['annotation_status']
                    save(Path(job['dataset']),saved)
                    stage='rendering';update(job,status=stage,summary=result_summary)
                    records=load(Path(job['dataset']))['grasps']
                    selected=next((g for g in records if g['physical_success'] is True),
                        next((g for g in records if g.get('approach_valid') is True),records[0]))
                    preview=str(Path(job['dataset']).with_suffix('.png'))
                    execute([sys.executable,str(ROOT/'generate_grasp_dataset.py'),'visualize',
                        '--output',job['dataset'],'--grasp-id',str(selected['grasp_id']),
                        '--all','--render',preview],stream,120)
                update(job,status='completed',summary=result_summary,preview=preview,
                    elapsed_seconds=round(time.monotonic()-started,2))
            except Exception as error:
                update(job,status='failed',failed_stage=stage,error=str(error),
                    elapsed_seconds=round(time.monotonic()-started,2))
            print(job['key'],job['status'],job.get('summary',job.get('error')),flush=True)
        pending=[j for j in manifest['jobs'] if j['status']!='completed']
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run,job) for job in pending]
            for future in as_completed(futures):
                future.result()
        manifest['finished']=True
        manifest['all_completed']=all(j['status']=='completed' for j in manifest['jobs'])
        save(manifest_file,manifest)
        if not manifest['all_completed']:
            raise SystemExit('Batch finished with recorded failures; inspect manifest and logs')


if __name__=='__main__':
    main()
