"""Evaluate a validation-selected readout without refitting normalization."""
import argparse
import torch
from .readout import Head,Direct,normalize
from .io import atomic_json,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--bank',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();st=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    bank=torch.load(a.bank,map_location='cpu',weights_only=True);arm=st['identity']['arm']
    if arm!='direct' and bank['parent_sha256']!=st['identity']['parent_sha256']:
        raise ValueError('Feature bank belongs to another generator')
    model=(Direct() if arm=='direct' else Head()).to(a.device).eval();model.load_state_dict(st['model']);pred=[]
    with torch.inference_mode():
        for i in range(0,len(bank['eta']),32):
            s=slice(i,i+32)
            if arm=='direct':y=model(bank['probe'][s].to(a.device),bank['action'][s].to(a.device))
            else:y=model(normalize(bank['x'][s].to(a.device),st['normalization'],arm=='zero'))
            pred.extend(y.flatten().cpu().tolist())
    err=torch.tensor(pred)-bank['eta']
    atomic_json(a.out,dict(arm=arm,split=bank['split'],checkpoint_sha256=sha(a.checkpoint),step=st['step'],
                mae=float(err.abs().mean()),rmse=float(err.square().mean().sqrt()),
                predictions=[dict(id=s,prediction=p,truth=float(y)) for s,p,y in zip(bank['ids'],pred,bank['eta'])]))

if __name__=='__main__':main()
