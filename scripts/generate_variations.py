#!/usr/bin/env python3
"""Generate resumable, separate approach- and loading-variation datasets for YCB."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from grasp_dataset.storage import load, save
from prepare_ycb import OBJECTS


def summarize(path):
    data=load(path)
    records=data['grasps']; trials=[t for g in records for t in g['trial_results']]
    return dict(candidates=len(records),trials=len(trials),
        measured_trials=sum(t['status']=='completed' for t in trials),
        successful_trials=sum(t.get('success') is True for t in trials),
        measured_failed_trials=sum(t.get('success') is False for t in trials),
        rejected_trials=sum(t['status']=='rejected_geometry' for t in trials),
        candidates_with_success=sum(g['physical_success'] is True for g in records),
        trace_files=sum('trace_file' in t for t in trials))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/ycb-varied'))
    parser.add_argument('--objects',nargs='+',choices=OBJECTS,default=OBJECTS)
    parser.add_argument('--num-candidates',type=int,default=384)
    parser.add_argument('--num-trials',type=int,default=5)
    parser.add_argument('--loading-candidates',type=int,default=96)
    parser.add_argument('--loading-trials',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261016)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    root=Path(__file__).resolve().parents[1]
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for kind,count,trials in [('approach',args.num_candidates,args.num_trials),('loading',args.loading_candidates,args.loading_trials)]:
        for obj in args.objects:
            dataset=output/kind/(obj+'.json')
            log=output/'logs'/(kind+'_'+obj+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            config=root/'examples'/('ycb_approach_variation.json' if kind=='approach' else 'ycb_loading_variation.json')
            command=[sys.executable,str(root/'generate_grasp_dataset.py'),'full','--scene',str(root/'assets/ycb'/obj/'object.yaml'),
                '--environment','minimal','--support','none','--num-candidates',str(count),'--num-trials',str(trials),
                '--surface-points','12000','--neighbors','192','--approaches','12','--pairs-per-point','1',
                '--min-width','0.001','--seed',str(args.seed),'--physics',str(config),'--trace-storage','external',
                '--output',str(dataset),'--resume']
            tasks.append(dict(kind=kind,object=obj,dataset=str(dataset),log=str(log),command=command,status='queued'))
    manifest=dict(schema='openarm-grasp-batch/v1',settings=vars(args)|{'output_dir':str(output)},jobs=tasks)
    save(output/'manifest.json',manifest)
    def run(task):
        start=time.monotonic()
        with Path(task['log']).open('a') as stream:
            result=subprocess.run(task['command'],cwd=root,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        return task,result.returncode,time.monotonic()-start
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,task):task for task in tasks}
        for future in as_completed(futures):
            task,code,elapsed=future.result()
            task['status']='completed' if code==0 else 'failed'
            task['elapsed_seconds']=round(elapsed,2)
            if code==0:
                task['summary']=summarize(task['dataset'])
            save(output/'manifest.json',manifest)
            print(task['kind'],task['object'],task['status'],task.get('summary',task['log']),flush=True)
    if any(t['status']=='failed' for t in tasks):
        raise SystemExit('Some jobs failed; see logs and rerun the same command to resume.')
    print(f'Completed {output}/manifest.json',flush=True)

if __name__=='__main__':
    main()
