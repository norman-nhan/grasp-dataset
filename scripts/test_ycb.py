#!/usr/bin/env python3
"""Run the four prepared YCB objects in the minimal evaluation environment."""
import argparse
from pathlib import Path
from grasp_dataset.cli import main
from grasp_dataset.storage import load
from prepare_ycb import OBJECTS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-candidates',type=int,default=96)
    p.add_argument('--num-trials',type=int,default=1)
    p.add_argument('--support',choices=['none','plane'],default='none')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/ycb'))
    p.add_argument('--render',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    for name in OBJECTS:
        output=a.output_dir/(name+'.json')
        main(['full','--scene',str(root/'assets/ycb'/name/'object.yaml'),'--support',a.support,
            '--num-candidates',str(a.num_candidates),'--num-trials',str(a.num_trials),'--output',str(output),'--resume'])
        data=load(output);g=data['grasps']
        print(f"YCB {name}: success={sum(v['physical_success'] is True for v in g)} failure={sum(v['physical_success'] is False for v in g)} unevaluated={sum(v['physical_success'] is None for v in g)}",flush=True)
        if a.render:
            main(['visualize','--output',str(output),'--all','--render',str(output.with_suffix('.png'))])
