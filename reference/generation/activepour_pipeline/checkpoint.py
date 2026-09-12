"""Full optimizer-boundary state, atomic checkpoints, explicit warm-start lineage."""
import contextlib
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from .io import atomic_json,read,sha,fingerprint


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s):
    random.setstate(s['python']);n=s['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch_cpu'])
    if s['torch_cuda']:torch.cuda.set_rng_state_all(s['torch_cuda'])


@contextlib.contextmanager
def isolated_rng():
    state=rng_state()
    try:yield
    finally:restore_rng(state)


def save(run,state):
    # Not called mid-accumulation. .tmp files are never used as checkpoints.
    path=Path(run)/'checkpoints'/f"step_{state['step']:08d}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('immutable checkpoint '+str(path))
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    pointer={'path':path.relative_to(run).as_posix(),'sha256':sha(path),'step':state['step']}
    atomic_json(path.with_suffix('.json'),pointer)
    atomic_json(Path(run)/'latest.json',pointer)
    return path


def load(run,identity,*,which='latest'):
    if which not in ('latest','best'):raise ValueError('checkpoint selector must be latest/best')
    pointer=read(Path(run)/(which+'.json'));path=Path(run)/pointer['path']
    if sha(path)!=pointer['sha256']:raise ValueError('checkpoint hash mismatch')
    s=torch.load(path,map_location='cpu',weights_only=True)
    if s['format']!='activepour_resume_v1' or s['identity']!=identity:raise ValueError('resume identity mismatch')
    if s['step']!=pointer['step']:raise ValueError('checkpoint step mismatch')
    return s,path


def named_optimizer_signature(transformer,conditioner):
    return [n for n,p in transformer.named_parameters() if p.requires_grad]+['conditioner.'+n for n,p in conditioner.named_parameters() if p.requires_grad]


def recover_stale_run(run,token):
    """Explicit crash recovery; never clear another live process's lock."""
    import socket
    run=Path(run);lock=run/'running.lock';owner=read(lock)
    if owner['token']!=token or owner['host']!=socket.gethostname():raise ValueError('owner/host mismatch')
    try:os.kill(owner['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('owner PID still exists; refusing recovery')
    dest=run/'abandoned_locks';dest.mkdir(exist_ok=True)
    os.rename(lock,dest/(token+'.json'))
