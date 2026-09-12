"""Immutable IDs/releases, per-task claims, validated imports, no simulator imports."""
import os
from pathlib import Path
import shutil
import socket
import time
import uuid
from .io import read,sha,fingerprint,atomic_json,inside


class DatasetStore:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.catalog=read(self.root/'registry/catalog.json')
        self.protocol=read(self.root/'registry/protocol.json')
        self.episodes={r['episode_id']:r for r in self.catalog['episodes']}
        self.samples={r['sample_id']:r for r in self.catalog['samples']}

    @classmethod
    def create(cls,root,registry,protocol,episodes,samples):
        root=Path(root);root.mkdir(parents=True,exist_ok=False)
        atomic_json(root/'registry/catalog.json',{'registry':registry,'episodes':episodes,'samples':samples})
        atomic_json(root/'registry/protocol.json',protocol)
        catalog=root/'registry/catalog.json'
        shutil.copy2(catalog,root/'registry'/('catalog_'+sha(catalog)+'.json'))
        return cls(root)

    def extend(self,registry,episodes,samples):
        """Append IDs only; archive the old catalog so old releases remain readable."""
        for key,value in self.catalog['registry'].items():
            if key in ('materials','realizations','actions'):
                incoming={x['id']:x for x in registry[key]}
                if any(incoming.get(x['id'])!=x for x in value):raise ValueError('cannot redefine existing registry IDs')
            elif registry.get(key)!=value:raise ValueError('non-append registry change')
        for old,new,key in [(self.catalog['episodes'],episodes,'episode_id'),(self.catalog['samples'],samples,'sample_id')]:
            incoming={x[key]:x for x in new}
            if len(incoming)!=len(new) or any(incoming.get(x[key])!=x for x in old):raise ValueError('cannot reassign existing samples/splits')
        path=self.root/'registry/catalog.json';archive=path.with_name('catalog_'+sha(path)+'.json')
        if not archive.exists():shutil.copy2(path,archive)
        atomic_json(path,{'registry':registry,'episodes':episodes,'samples':samples})
        archive=path.with_name('catalog_'+sha(path)+'.json')
        if not archive.exists():shutil.copy2(path,archive)
        self.__init__(self.root)

    def directory(self,key):
        row=self.samples.get(key) or self.episodes.get(key)
        if row is None:raise KeyError(key)
        return inside(self.root,row['relative_directory'])

    def accepted(self,key,verify=True):
        pointer=self.directory(key)/'accepted.json'
        if not pointer.exists():return None
        p=read(pointer);manifest=inside(self.root,p['manifest'])
        if sha(manifest)!=p['sha256']:raise ValueError('accepted manifest corrupted')
        result=read(manifest)
        if result['key']!=key or result['protocol_sha256']!=fingerprint(self.protocol):
            raise ValueError('accepted identity/protocol mismatch')
        if verify:
            for relative,digest in result['files'].items():
                if sha(inside(self.root,relative))!=digest:raise ValueError('corrupt artifact: '+relative)
        return result

    def claim(self,key):
        # Existing accepted data are hashed before skipping. A stale claim is NEVER
        # automatically stolen: an operator must explicitly abandon it after checking.
        if self.accepted(key):return None
        path=self.directory(key);path.mkdir(parents=True,exist_ok=True)
        owner={'token':uuid.uuid4().hex,'pid':os.getpid(),'host':socket.gethostname(),'time':time.time()}
        lock=path/'claim.json'
        try:
            with lock.open('x') as f:
                import json
                json.dump(owner,f);f.flush();os.fsync(f.fileno())
        except FileExistsError:raise RuntimeError('task already claimed: '+key)
        attempt=path/'attempts'/('run_'+owner['token']);attempt.mkdir(parents=True)
        atomic_json(attempt/'status.json',{'state':'running','owner':owner})
        return {'key':key,'owner':owner,'attempt':attempt}

    def _owned(self,claim):
        if read(self.directory(claim['key'])/'claim.json')!=claim['owner']:raise ValueError('claim owner mismatch')

    def fail(self,claim,reason):
        self._owned(claim)
        atomic_json(claim['attempt']/'status.json',{'state':'failed','reason':str(reason),'owner':claim['owner']})
        (self.directory(claim['key'])/'claim.json').unlink()

    def abandon(self,key,token,reason):
        owner=read(self.directory(key)/'claim.json')
        if owner['token']!=token or not reason:raise ValueError('explicit owner token/reason required')
        self.fail({'key':key,'owner':owner,'attempt':self.directory(key)/'attempts'/('run_'+token)},reason)

    def commit(self,claim,files,metadata):
        self._owned(claim)
        if not metadata.get('guards') or not all(metadata['guards'].values()):raise ValueError('guards not passed')
        key=claim['key']; row=self.samples.get(key) or self.episodes[key]
        if metadata['split']!=row['split']:raise ValueError('split mismatch')
        # Artifacts live in one immutable attempt; no overwriting old accepted outputs.
        hashes={}
        for name,source in files.items():
            dest=inside(claim['attempt'],name)
            if dest.exists():raise FileExistsError(dest)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
            if sha(dest)!=sha(source):raise ValueError('copy mismatch')
            hashes[dest.relative_to(self.root).as_posix()]=sha(dest)
        manifest={'key':key,'protocol_sha256':fingerprint(self.protocol),'files':hashes,'metadata':metadata}
        mp=claim['attempt']/'manifest.json';atomic_json(mp,manifest)
        self._owned(claim)
        atomic_json(self.directory(key)/'accepted.json',{'manifest':mp.relative_to(self.root).as_posix(),'sha256':sha(mp)})
        atomic_json(claim['attempt']/'status.json',{'state':'accepted','owner':claim['owner']})
        (self.directory(key)/'claim.json').unlink()
        return manifest

    def publish(self,name,sample_ids=None):
        if not name.replace('_','').isalnum():raise ValueError('invalid release name')
        rows=[]
        for sid in sorted(self.samples if sample_ids is None else sample_ids):
            accepted=self.accepted(sid)
            if accepted is None:
                if sample_ids is not None:raise ValueError('requested sample not accepted')
                continue
            plan=self.samples[sid];ep=self.accepted(plan['episode_id'])
            if ep is None:raise ValueError('missing accepted probe')
            def artifact(manifest,name):
                matches=[p for p in manifest['files'] if Path(p).name==name]
                if len(matches)!=1:raise ValueError('artifact ambiguous/missing '+name)
                return matches[0]
            meta=accepted['metadata']
            rows.append({'id':sid,'episode_id':plan['episode_id'],'split':plan['split'],
                'evaluation_axis':plan['evaluation_axis'],
                'inputs':{'probe_paths':[artifact(ep,n+'.png') for n in ['I_pre','I_probe_peak','I_post']],
                          'action_deg_s_s':meta['action_deg_s_s']},
                'targets':{'future_grid':artifact(accepted,'future_grid.png'),'eta_final':meta['eta_final']},
                'artifact_hashes':dict(ep['files'],**accepted['files'])})
        if not rows:raise ValueError('empty release')
        release={'format':'activepour_release_v1','dataset_id':self.catalog['registry']['dataset_id'],
                 'protocol_sha256':fingerprint(self.protocol),'catalog_sha256':sha(self.root/'registry/catalog.json'),
                 'rows':rows}
        folder=self.root/'manifests'/name;folder.mkdir(parents=True,exist_ok=False)
        atomic_json(folder/'release.json',release)
        for split in ['train','validation','test']:
            atomic_json(folder/(split+'.json'),[r for r in rows if r['split']==split])
        return folder/'release.json'


def load_release(root,release,split,*,verify=True):
    if split not in ('train','validation','test'):raise ValueError('explicit split required')
    root=Path(root);data=read(release)
    if data['format']!='activepour_release_v1':raise ValueError('release format')
    if data['protocol_sha256']!=fingerprint(read(root/'registry/protocol.json')):raise ValueError('protocol changed')
    snapshot=root/'registry'/('catalog_'+data['catalog_sha256']+'.json')
    if not snapshot.exists():snapshot=root/'registry/catalog.json'  # First v1 imports before catalog archiving.
    if data['catalog_sha256']!=sha(snapshot):raise ValueError('catalog changed')
    rows=data['rows'];seen={};ids=set()
    for r in rows:
        if r['id'] in ids:raise ValueError('duplicate sample')
        ids.add(r['id']);prev=seen.setdefault(r['episode_id'],r['split'])
        if prev!=r['split']:raise ValueError('episode split leakage')
        if set(r['inputs'])!={'probe_paths','action_deg_s_s'}:raise ValueError('unexpected model inputs')
    selected=[r for r in rows if r['split']==split]
    if verify:
        hashes={p:h for r in selected for p,h in r['artifact_hashes'].items()}
        for p,h in hashes.items():
            if sha(inside(root,p))!=h:raise ValueError('release artifact corrupted: '+p)
    return selected
