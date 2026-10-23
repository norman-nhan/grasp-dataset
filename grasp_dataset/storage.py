"""Versioned portable JSON, with atomic replacement for interrupted-run recovery."""
import json
import os
from pathlib import Path

SCHEMA = 'openarm-grasp-dataset/v1'

def save(path, dataset):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(dataset, f, allow_nan=False, separators=(',', ':'))
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)

def load(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError('Unsupported dataset schema')
    return data

def aggregate(record, expected_trials=None):
    trials = record['trial_results']
    measured = [t for t in trials if t['status'] == 'completed']
    record['collision_valid'] = any(t['status'] == 'completed' for t in trials) if trials else None
    record['evaluation_status'] = 'completed' if measured else ('rejected_geometry' if trials else 'pending')
    if expected_trials is not None and len(trials) < expected_trials:
        record['evaluation_status'] = 'partial'
    record['rejection_reasons'] = sorted({r for t in trials for r in t['rejection_reasons']})
    record['physical_success_rate'] = sum(t['success'] for t in measured)/len(measured) if measured else None
    record['physical_success'] = any(t['success'] for t in measured) if measured else None
    slips=[t['slip_distance'] for t in measured if t.get('slip_distance') is not None]
    record['slip_distance'] = max(slips) if slips else None
    record['accessibility_rate'] = len(measured)/len(trials) if trials else None
    record['num_trials_measured'] = len(measured)
    record['num_trials_attempted'] = len(trials)
    record['execution_success_rate'] = sum(bool(t['success']) for t in trials)/len(trials) if trials else None


def externalize_trace(dataset_path, grasp_id, trial):
    """Store large traces independently, before atomically checkpointing their references."""
    import gzip
    import hashlib
    if 'trace' not in trial:
        return
    path = Path(dataset_path)
    directory = path.parent/(path.stem+'.traces')
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory/f'grasp_{int(grasp_id):06d}_trial_{int(trial["trial_id"]):04d}.json.gz'
    payload = json.dumps(trial['trace'], allow_nan=False, separators=(',', ':')).encode()
    temporary = trace_path.with_name(trace_path.name+'.tmp')
    with temporary.open('wb') as f:
        f.write(gzip.compress(payload, compresslevel=3, mtime=0)); f.flush(); os.fsync(f.fileno())
    temporary.replace(trace_path)
    trial['trace_file'] = str(trace_path.relative_to(path.parent))
    trial['trace_sha256'] = hashlib.sha256(payload).hexdigest()
    trial['trace_samples'] = len(trial.pop('trace'))


def load_trace(dataset_path, trial):
    """Load inline or compressed sidecar traces, verifying external content hashes."""
    import gzip
    import hashlib
    if 'trace' in trial:
        return trial['trace']
    if 'trace_file' not in trial:
        return []
    payload = gzip.decompress((Path(dataset_path).parent/trial['trace_file']).read_bytes())
    if hashlib.sha256(payload).hexdigest() != trial['trace_sha256']:
        raise ValueError('Trial trace checksum mismatch')
    return json.loads(payload)
