"""One reusable DEM task path for legacy import and future incremental generation."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .dataset import DatasetStore
from .io import read,sha,fingerprint

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_geotaichi_reference import sample_guards,command
from render_geotaichi_training import raw_particles,assemble,KEYS


def parameters(store,sid):
    row=store.samples[sid];ep=store.episodes[row['episode_id']]
    reg=store.catalog['registry']
    m=next(x for x in reg['materials'] if x['id']==ep['material_id'])
    r=next(x for x in reg['realizations'] if x['id']==ep['realization_id'])
    a=next(x for x in reg['actions'] if x['id']==row['action_id'])
    return row,ep,m,r,a


def audit_run(store,sid,run,probe_source):
    row,ep,m,r,a=parameters(store,sid);report=read(run/'report.json')
    probe_report=probe_source/('report.json' if (probe_source/'report.json').exists() else 'source_report.json')
    probe_history=probe_source/('history.json' if (probe_source/'history.json').exists() else 'source_history.json')
    pp=read(probe_report);c=store.protocol['config']
    expected={'dt':c['dt_s'],'radius':c['radius_m'],'kn':c['kn_N_m'],'seed':r['seed'],
        'mu':m['mu'],'rolling':m['rolling'],'arch':'cpu','canonical_order':True,
        'probe':c['probe_deg'],'prep':c['prep_s'],'rest':c['rest_s'],
        'verlet':c['verlet'],'jitter_fraction':c['jitter_fraction'],'rebuild_trigger_scale':1.0}
    for source in [report,pp]:
        if any(source['args'].get(k)!=v for k,v in expected.items()):raise ValueError('simulation protocol mismatch')
        if source['source_hashes']!=store.protocol['geotaichi_source_hashes']:raise ValueError('GeoTaichi source mismatch')
        if source['n']!=612:raise ValueError('particle count mismatch')
    if [report['args'][k] for k in ('theta','rotation','hold')]!=a['values']:raise ValueError('action mismatch')
    guards=sample_guards(report,read(run/'history.json'))
    if not all(guards.values()):raise ValueError('unsafe/incomplete sample')
    if not all(sample_guards(pp,read(probe_history)).values()):raise ValueError('unsafe source probe run')
    with np.load(probe_source/'post_state.npz') as snap:
        metadata=json.loads(str(snap['metadata']))
        if len(metadata['fields'])<20 or not all(k in snap or any(n.startswith(k+'::') for n in snap.files) for k in metadata['fields']):
            raise ValueError('incomplete contact snapshot')
        if metadata['contract']['seed']!=r['seed'] or metadata['contract']['mu']!=m['mu']:
            raise ValueError('probe snapshot contract mismatch')
        if metadata['time']!=pp['frames']['I_post']['t']:raise ValueError('snapshot time mismatch')
    for key in KEYS[:3]:
        with np.load(run/(key+'.npz')) as actual,np.load(probe_source/(key+'.npz')) as ref:
            if not all(np.array_equal(actual[k],ref[k]) for k in ('x','v','w','r','m','captured')):
                raise ValueError('probe branch differs')
    mass0=report['mass0'];eta=report['frames']['future_rest']['eta']
    with np.load(run/'future_rest.npz') as s:
        measured=float(s['m'][s['captured'].astype(bool)].sum()/s['m'].sum())
        if abs(measured-eta)>1e-12 or abs(float(s['m'].sum())-mass0)>1e-12:raise ValueError('mass label mismatch')
    return report,guards


def commit_run(store,sid,run,probe_source,*,legacy=None,claim=None,probe_claim=None):
    row,ep,m,r,a=parameters(store,sid)
    report,guards=audit_run(store,sid,Path(run),Path(probe_source))
    claim=claim or store.claim(sid)
    if claim is None:return 'skipped'
    try:
        accepted_probe=store.accepted(ep['episode_id'])
        if accepted_probe is None:
            probe_claim=probe_claim or store.claim(ep['episode_id'])
            rendered=probe_claim['attempt']/'render';rendered.mkdir()
            files={k+'.npz':probe_source/(k+'.npz') for k in KEYS[:3]}
            files.update({'post_state.npz':probe_source/'post_state.npz','source_report.json':probe_source/'report.json',
                          'source_history.json':probe_source/'history.json'})
            for key in KEYS[:3]:
                path=rendered/(key+'.png');raw_particles(probe_source/(key+'.npz'),size=256).save(path)
                if legacy and not np.array_equal(np.asarray(Image.open(legacy['root']/legacy['row']['inputs']['probe_paths'][KEYS.index(key)])),np.asarray(Image.open(path))):
                    raise ValueError('legacy probe rendering mismatch')
                files[key+'.png']=path
            store.commit(probe_claim,files,{'split':ep['split'],'guards':guards,'mass0':report['mass0'],
                'material':m,'realization':r,'source_run':str(probe_source),'snapshot_sha256':sha(probe_source/'post_state.npz')})
            probe_claim=None
        else:
            oldsnapshot=next(k for k in accepted_probe['files'] if k.endswith('/post_state.npz'))
            if sha(probe_source/'post_state.npz')!=accepted_probe['files'][oldsnapshot]:raise ValueError('different complete probe snapshot')
        render=claim['attempt']/'render';render.mkdir()
        target=render/'future_grid.png'
        assemble({k:raw_particles(run/(k+'.npz'),size=256) for k in KEYS[3:]},KEYS[3:],2).save(target)
        if legacy:
            old=legacy['row']
            if not np.array_equal(np.asarray(Image.open(target)),np.asarray(Image.open(legacy['root']/old['targets']['future_grid']))):raise ValueError('legacy target rendering mismatch')
            if abs(old['targets']['eta_final']-report['frames']['future_rest']['eta'])>1e-12:raise ValueError('legacy eta mismatch')
        files={k+'.npz':run/(k+'.npz') for k in KEYS[3:]}
        files.update({'future_grid.png':target,'report.json':run/'report.json','history.json':run/'history.json'})
        store.commit(claim,files,{'split':row['split'],'guards':guards,'action_deg_s_s':a['values'],
            'eta_final':report['frames']['future_rest']['eta'],'initial_mass':report['mass0'],
            'accepted_mass':report['mass0']*report['frames']['future_rest']['eta'],
            'frames':{k:report['frames'][k] for k in KEYS[3:]},'source_run':str(run),
            'legacy_id':legacy['row']['id'] if legacy else None})
        return 'accepted'
    except Exception as exc:
        if (store.directory(sid)/'claim.json').exists():store.fail(claim,exc)
        if probe_claim and (store.directory(ep['episode_id'])/'claim.json').exists():store.fail(probe_claim,exc)
        raise


def generate(store,sid,*,dry_run=False):
    if store.accepted(sid):return {'sample_id':sid,'state':'skipped_accepted','simulation_started':False}
    row,ep,m,r,a=parameters(store,sid);probe=store.accepted(ep['episode_id'])
    if dry_run:return {'sample_id':sid,'state':'planned','reuse_probe':bool(probe),'simulation_started':False}
    claim=store.claim(sid);pc=None
    try:
        if probe:
            snapshot=store.root/next(k for k in probe['files'] if k.endswith('/post_state.npz'))
            # Original report/hist have standard names in the source raw run. The
            # reusable snapshot itself lives alongside I_pre/peak/post NPZ files.
            probe_source=snapshot.parent
        else:pc=store.claim(ep['episode_id']);snapshot=None
        raw=claim['attempt']/'raw'
        cmd=command(store.protocol['config'],raw,r['seed'],m['mu'],m['rolling'],*a['values'])
        if snapshot:cmd+=['--post-state',str(snapshot)]
        import os
        with (claim['attempt']/'simulation.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        return {'sample_id':sid,'state':commit_run(store,sid,raw,probe_source if probe else raw,claim=claim,probe_claim=pc),
                'simulation_started':True,'reuse_probe':bool(probe)}
    except Exception as exc:
        if (store.directory(sid)/'claim.json').exists():store.fail(claim,exc)
        if pc and (store.directory(ep['episode_id'])/'claim.json').exists():store.fail(pc,exc)
        raise
