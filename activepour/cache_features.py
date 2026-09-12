"""Cache predicted final latent features once for all readout seeds."""
import argparse
from pathlib import Path
import torch
from .runtime import load_model,sample
from .readout import frozen_features
from .io import save,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--bank',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    net,cond=load_model(a.model,checkpoint=a.checkpoint)
    if cond.frame_mode!='full5':raise ValueError('The final readout uses the five-frame model')
    net.eval().requires_grad_(False);cond.eval().requires_grad_(False);parent=sha(a.checkpoint)
    for split in ['train','validation','test']:
        path=Path(a.bank)/(split+'.pt')
        if not path.exists():continue
        data=torch.load(path,map_location='cpu',weights_only=True);xs=[]
        identity={'parent_sha256':parent,'bank_sha256':sha(path),'seed':20260919,'steps':28}
        for i,sid in enumerate(data['ids']):
            cache=out/'samples'/split/(str(i)+'.pt')
            if cache.exists():
                obj=torch.load(cache,map_location='cpu',weights_only=True)
                if obj['identity']!=identity or obj['id']!=sid:raise ValueError('Incompatible cache')
            else:
                probe=data['probe'][i:i+1].cuda();action=data['action'][i:i+1].cuda()
                z=sample(net,cond,a.model,probe,action,data['text_tokens'].cuda(),data['pooled_text'].cuda())
                with torch.inference_mode():x=frozen_features(cond,probe,action,z).cpu()
                obj=dict(identity=identity,id=sid,latent=z.cpu(),x=x);save(cache,obj)
            xs.append(obj['x'])
        # The direct arm can use the original bank; future arms use this bank.
        save(out/(split+'.pt'),dict(split=split,ids=data['ids'],episode_ids=data['episode_ids'],eta=data['eta'],
                                   x=torch.cat(xs),parent_sha256=parent))
        print(split,len(xs),flush=True)

if __name__=='__main__':main()
