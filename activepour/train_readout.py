"""Train a readout with identical head initialization, batches and regularization."""
import argparse
import math
from pathlib import Path
import torch
from .io import seed_all,sha,save,read,atomic_json,rng_state,restore_rng
from .readout import Head,Direct,normalize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True,help='Directory containing train.pt and validation.pt')
    p.add_argument('--arm',choices=['latent','zero','direct'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=101)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--stop-after',type=int)
    a=p.parse_args();device=a.device;seed_all(a.seed);torch.set_num_threads(4)
    if a.steps<1:raise ValueError('Steps must be positive')
    train=torch.load(Path(a.bank)/'train.pt',map_location='cpu',weights_only=True)
    val=torch.load(Path(a.bank)/'validation.pt',map_location='cpu',weights_only=True)
    if train['split']!='train' or val['split']!='validation':raise ValueError('Wrong data split')
    if set(train['episode_ids'])&set(val['episode_ids']):raise ValueError('Episode leakage')
    # Initialize the same head before constructing the trainable direct encoder.
    head=Head();model=(Direct(head) if a.arm=='direct' else head).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001,eps=1e-8)
    norm=None
    if a.arm!='direct':
        if train['parent_sha256']!=val['parent_sha256']:raise ValueError('Mixed generator checkpoints')
        norm=dict(mean=train['x'].mean(0),std=train['x'].std(0).clamp_min(.01),fit_split='train')
    identity=dict(arm=a.arm,seed=a.seed,steps=a.steps,train_sha256=sha(Path(a.bank)/'train.pt'),
                  validation_sha256=sha(Path(a.bank)/'validation.pt'),dropout=.15,weight_decay=.001,
                  batch_size=32,learning_rate=3e-4,
                  parent_sha256=train.get('parent_sha256') if a.arm!='direct' else None)
    out=Path(a.out);start=0;best=float('inf');history=[]
    batches=torch.Generator().manual_seed(a.seed+10000)
    if a.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        if state['identity']!=identity:raise ValueError('Resume identity changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['step'];best=state['best'];history=state['history'];restore_rng(state['rng']);batches.set_state(state['batch_rng'])
    else:out.mkdir(parents=True,exist_ok=False);atomic_json(out/'identity.json',identity)
    def predict(bank,ids):
        if a.arm=='direct':return model(bank['probe'][ids].to(device),bank['action'][ids].to(device)).flatten()
        return model(normalize(bank['x'][ids].to(device),norm,a.arm=='zero')).flatten()
    @torch.no_grad()
    def evaluate(bank):
        model.eval();pred=torch.cat([predict(bank,list(range(i,min(i+32,len(bank['eta']))))).cpu() for i in range(0,len(bank['eta']),32)])
        errors=pred-bank['eta'];return dict(mae=float(errors.abs().mean()),mse=float(errors.square().mean())),pred
    end=min(a.steps,a.stop_after or a.steps)
    for step in range(start+1,end+1):
        model.train();ids=torch.randint(len(train['eta']),(32,),generator=batches)
        factor=step/100 if step<=100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/max(1,a.steps-100)))
        for g in optimizer.param_groups:g['lr']=3e-4*factor
        optimizer.zero_grad(set_to_none=True)
        loss=(predict(train,ids)-train['eta'][ids].to(device)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite readout loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        row=dict(step=step,loss=float(loss.detach()),lr=optimizer.param_groups[0]['lr']);history.append(row)
        new_best=False
        if step%100==0 or step==end:
            metrics,pred=evaluate(val);row['validation']=metrics;new_best=metrics['mae']<best
            if new_best:best=metrics['mae']
            save(out/'validation_latest.pt',dict(ids=val['ids'],prediction=pred,truth=val['eta'],step=step))
            if step%500==0:row['train_eval']=evaluate(train)[0]
            state=dict(format='activepour_public_readout_v1',identity=identity,step=step,model=model.state_dict(),
                       optimizer=optimizer.state_dict(),normalization=norm,best=best,history=history,
                       rng=rng_state(),batch_rng=batches.get_state())
            save(out/'latest.pt',state);save(out/f'checkpoints/step_{step:08d}.pt',state)
            if new_best:save(out/'best.pt',state)
            atomic_json(out/'history.json',history)
            print(step,metrics,flush=True)

if __name__=='__main__':main()
