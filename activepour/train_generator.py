"""Portable paired generation trainer; no archived optimizer state is overwritten."""
import argparse
import math
from pathlib import Path
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from .runtime import load_model,velocity
from .io import seed_all,sha,save,atomic_json,rng_state,restore_rng
from activepour_model.lora import lora_state,load_lora_state
from activepour_model.objectives import paired_flow_batch,fixed_spatial_weights,flow_mse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--bank',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['full5','pre1'],required=True);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=916);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();torch.set_num_threads(4);seed_all(a.seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    banks={s:torch.load(Path(a.bank)/(s+'.pt'),weights_only=True,map_location='cpu') for s in ['train','validation']}
    train,val=banks['train'],banks['validation']
    if train['split']!='train' or val['split']!='validation':raise ValueError('Wrong split')
    if set(train['episode_ids'])&set(val['episode_ids']):raise ValueError('Episode leakage')
    net,cond=load_model(a.model,a.mode);net.enable_gradient_checkpointing()
    optimizer=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.requires_grad],'lr':3e-5},
                               {'params':cond.parameters(),'lr':2e-4}],weight_decay=.01)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(a.model,subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas[:len(scheduler.timesteps)].cuda();times=scheduler.timesteps.cuda()
    weights=fixed_spatial_weights(train['foreground_union'],4).cuda()
    identity=dict(mode=a.mode,steps=a.steps,seed=a.seed,bank={s:sha(Path(a.bank)/(s+'.pt')) for s in banks},
                  base=sha(Path(a.model)/'transformer/diffusion_pytorch_model.safetensors'))
    compatible=dict(conditioner_spec=cond.spec(),base=identity['base'],vae=sha(Path(a.model)/'vae/diffusion_pytorch_model.safetensors'))
    out=Path(a.out);history=[];start=0;best=float('inf');batch_rng=torch.Generator().manual_seed(a.seed+10000)
    if a.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        if state['identity']!=identity:raise ValueError('Resume identity changed')
        load_lora_state(net,state['lora']);cond.load_state_dict(state['conditioner']);optimizer.load_state_dict(state['optimizer'])
        start=state['step'];best=state['best'];history=state['history'];restore_rng(state['rng']);batch_rng.set_state(state['batch_rng'])
    else:out.mkdir(parents=True,exist_ok=False);atomic_json(out/'identity.json',identity)
    def forward(bank,ids,z,t):
        return velocity(net,cond,bank['probe'][ids].cuda(),bank['action'][ids].cuda(),
                        bank['text_tokens'].cuda(),bank['pooled_text'].cuda(),z,t)
    @torch.no_grad()
    def validation():
        # Fixed panel and local RNG; evaluation must not disturb training randomness.
        net.eval();cond.eval();g=torch.Generator(device='cuda').manual_seed(20260920);losses=[]
        for i in range(min(96,len(val['ids']))):
            target=val['target'][i:i+1].cuda();noise=torch.randn(target.shape,device='cuda',generator=g)
            for sigma in [.2,.5,.8]:
                pred=forward(val,[i],(1-sigma)*target+sigma*noise,torch.tensor([sigma*1000],device='cuda'))
                losses.append(float(flow_mse(pred,noise-target)))
        return sum(losses)/len(losses)
    end=min(a.steps,a.stop_after or a.steps)
    for step in range(start+1,end+1):
        net.train();cond.train();ids=torch.randint(len(train['ids']),(8,),generator=batch_rng)
        factor=step/200 if step<=200 else .1+.9*.5*(1+math.cos(math.pi*(step-200)/max(1,a.steps-200)))
        for g,lr in zip(optimizer.param_groups,[3e-5,2e-4]):g['lr']=lr*factor
        target=train['target'][ids].cuda();z,v,k=paired_flow_batch(target,sigmas,boundary_probability=.5)
        optimizer.zero_grad(set_to_none=True);loss=flow_mse(forward(train,ids,z,times[k]),v,weights)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite flow loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],1);optimizer.step()
        row=dict(step=step,loss=float(loss.detach()),sigma_index=int(k[0]));history.append(row);improved=False
        if step%500==0 or step==a.steps:
            row['validation_flow']=validation();improved=row['validation_flow']<best;best=min(best,row['validation_flow'])
        if step%250==0 or step==end:
            state=dict(format='activepour_public_generation_v1',identity=identity,compatible=compatible,step=step,
                       lora=lora_state(net),conditioner=cond.state_dict(),optimizer=optimizer.state_dict(),
                       rng=rng_state(),batch_rng=batch_rng.get_state(),history=history,best=best)
            save(out/'latest.pt',state);save(out/f'checkpoints/step_{step:08d}.pt',state)
            if improved:save(out/'best.pt',state)
            atomic_json(out/'history.json',history);print(row,flush=True)

if __name__=='__main__':main()
