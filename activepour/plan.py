"""Score 32 global + 5 x 6 local + 4 extra actions; never read simulation outcomes."""
import argparse
from pathlib import Path
import torch
import numpy as np
from .runtime import load_model,sample
from .readout import Head,frozen_features,normalize
from .search import coarse,refine,rank
from .io import sha,fingerprint,read,save,atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--generator',required=True)
    p.add_argument('--heads',nargs=3,required=True);p.add_argument('--bank',required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--target',type=float,required=True)
    p.add_argument('--seed',type=int,default=25000);p.add_argument('--out',required=True)
    a=p.parse_args()
    if not 0<=a.target<=1:raise ValueError('Target outside [0, 1]')
    data=torch.load(a.bank,map_location='cpu',weights_only=True)
    # Only observed images and fixed text are selected from the bank.
    probe=data['probe'][a.index:a.index+1].cuda();text=data['text_tokens'].cuda();pooled=data['pooled_text'].cuda()
    if probe.shape!=(1,5,256,256):raise ValueError('Invalid observation index')
    net,cond=load_model(a.model,checkpoint=a.generator)
    if cond.frame_mode!='full5':raise ValueError('Planning requires full5')
    net.eval().requires_grad_(False);cond.eval().requires_grad_(False);heads=[]
    for path in a.heads:
        st=torch.load(path,map_location='cpu',weights_only=True)
        if st['identity']['arm']!='latent':raise ValueError('Only latent heads may score actions')
        if st['identity']['parent_sha256']!=sha(a.generator):raise ValueError('Head belongs to another generator')
        h=Head().cuda().eval().requires_grad_(False);h.load_state_dict(st['model']);heads.append((h,st['normalization']))
    if len({torch.load(p,map_location='cpu',weights_only=True)['identity']['seed'] for p in a.heads})!=3:
        raise ValueError('Expected three independently trained seeds')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    identity=dict(search_protocol='global32_local5x6_extra4_v1',generator=sha(a.generator),heads=[sha(p) for p in a.heads],bank=sha(a.bank),index=a.index,target=a.target,seed=a.seed)
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Search identity changed')
    atomic_json(out/'identity.json',identity)
    @torch.inference_mode()
    def evaluate(candidates):
        records=[]
        for c in candidates:
            path=out/'candidates'/(c['id']+'.pt')
            if path.exists():
                obj=torch.load(path,map_location='cpu',weights_only=True)
                if obj['identity']!=identity or obj['action']!=c['action']:raise ValueError('Cache identity mismatch')
            else:
                action=torch.tensor([c['action']],device='cuda',dtype=torch.float32)
                z=sample(net,cond,a.model,probe,action,text,pooled)
                x=frozen_features(cond,probe,action,z)
                values=[float(h(normalize(x,n)).item()) for h,n in heads]
                obj=dict(identity=identity,action=c['action'],latent=z.cpu(),features=x.cpu(),values=values);save(path,obj)
            records.append(dict(c,eta_pred=float(np.mean(obj['values'])),eta_seeds=obj['values']))
        return records
    first=evaluate(coarse(a.seed));extra=refine(first,a.target,a.seed+int(a.target*100))
    records=first+evaluate(extra['candidates'])
    assert len(records)==66 and len({r['id'] for r in records})==66
    winner=rank(records,a.target)[0]
    atomic_json(out/'selection.json',dict(identity=identity,candidates=records,winner=winner,
                note='Select once before simulation. This file contains no simulated discharge labels.'))
    print('Selected action [degrees, rotation seconds, hold seconds]:',winner['action'])
    print('Predicted discharge:',winner['eta_pred'])

if __name__=='__main__':main()
