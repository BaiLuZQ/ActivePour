"""Full SD3.5 Medium loading and serial 28-step generation from noise."""
import torch
from pathlib import Path
from .io import sha
from diffusers import SD3Transformer2DModel,FlowMatchEulerDiscreteScheduler
from activepour_model.conditioning import PhysicsConditioner
from activepour_model.lora import attach_lora,load_lora_state

def load_model(model,mode='full5',checkpoint=None):
    net=SD3Transformer2DModel.from_pretrained(model,subfolder='transformer',local_files_only=True,torch_dtype=torch.bfloat16).cuda()
    cfg=net.config
    if (cfg.num_layers,cfg.num_attention_heads,cfg.attention_head_dim,cfg.patch_size)!=(24,24,64,2):
        raise ValueError('This implementation requires full SD3.5 Medium')
    attach_lora(net,16,16)
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['compatible']['base']!=sha(Path(model)/'transformer/diffusion_pytorch_model.safetensors'):
            raise ValueError('Checkpoint belongs to another base transformer')
        if state['compatible']['vae']!=sha(Path(model)/'vae/diffusion_pytorch_model.safetensors'):
            raise ValueError('Checkpoint belongs to another VAE')
        spec=state['compatible']['conditioner_spec']
    else:
        spec=dict(hidden_dim=1536,context_dim=4096,feature_width=128,num_layers=24,
                  action_low=[45.,.6,.1],action_high=[100.,2.,1.5],architecture='phase_action_v2',
                  frame_mode=mode,condition_dropout=.05)
    cond=PhysicsConditioner(**spec).cuda()
    if checkpoint:
        load_lora_state(net,state['lora']);cond.load_state_dict(state['conditioner'])
    return net,cond

def velocity(net,cond,probe,action,text,pooled,z,t):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        residuals,context=cond(probe,action,text)
        return net(hidden_states=z.to(torch.bfloat16),timestep=t,
                   encoder_hidden_states=context,pooled_projections=pooled.expand(len(z),-1),
                   block_controlnet_hidden_states=residuals,return_dict=False)[0]

@torch.inference_mode()
def sample(net,cond,model,probe,action,text,pooled,seed=20260919,steps=28):
    if net.training or cond.training:raise ValueError('Sampling requires eval mode')
    if len(probe)!=1:raise ValueError('Use serial reference sampling for each candidate')
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(steps,device='cuda')
    z=torch.randn((1,16,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        residuals,context=cond(probe,action,text)
        for t in scheduler.timesteps:
            v=net(hidden_states=z.to(torch.bfloat16),timestep=t.expand(1),encoder_hidden_states=context,
                  pooled_projections=pooled,block_controlnet_hidden_states=residuals,return_dict=False)[0]
            z=scheduler.step(v.float(),t,z,return_dict=False)[0]
    return z
