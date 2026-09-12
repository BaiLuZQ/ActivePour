"""Production-oriented optimizer loop, also exercised by short Node8A real-SD tests.

No new physics is run. Old Node6 scripts/checkpoints remain untouched.
"""
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from diffusers import SD3Transformer2DModel,FlowMatchEulerDiscreteScheduler
from activepour_model.conditioning import PhysicsConditioner
from activepour_model.lora import attach_lora,lora_state,load_lora_state
from activepour_model.training import prediction,gradient_audit
from activepour_model.objectives import paired_flow_batch,flow_mse,fixed_spatial_weights
from activepour_model.data import image_tensor
from .io import sha,read,atomic_json,fingerprint
from .model_data import prepare_cache
from . import checkpoint


def batch_prediction(transformer,conditioner,cache,z,t,ids):
    # Keep large target/probe banks on CPU. Only the current batch enters VRAM.
    batch={k:cache[k][ids].cuda() for k in ['probe_features','actions']}
    batch.update(text_tokens=cache['text_tokens'].cuda(),pooled_text=cache['pooled_text'].cuda())
    return prediction(transformer,conditioner,batch,z,t,list(range(len(ids))))


@torch.no_grad()
def validation_flow(transformer,conditioner,cache,seed):
    with checkpoint.isolated_rng():
        transformer.eval();conditioner.eval();g=torch.Generator(device='cuda').manual_seed(seed)
        scores=[]
        for i,sid in enumerate(cache['ids']):
            target=cache['target_latents'][[i]].cuda().float()
            noise=torch.randn(target.shape,device='cuda',generator=g);values=[]
            for sigma in [.2,.5,.8]:
                z=(1-sigma)*target+sigma*noise
                out=batch_prediction(transformer,conditioner,cache,z.to(torch.bfloat16),torch.tensor([sigma*1000],device='cuda'),[i])
                values.append(float(flow_mse(out,noise-target)))
            scores.append({'sample_id':sid,'flow_mse':float(np.mean(values))})
        return {'mean_flow_mse':float(np.mean([r['flow_mse'] for r in scores])),'samples':scores}


def train(config,run,*,resume=False,stop_after=None,warm_start=None):
    if resume and warm_start:raise ValueError('resume and warm_start are distinct')
    run=Path(run);started=time.perf_counter()
    torch.set_num_threads(4)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if resume:
        manifest=read(run/'run_manifest.json')
        if manifest['config']!=config:raise ValueError('resume configuration changed')
    else:
        run.mkdir(parents=True,exist_ok=False)
    token=uuid.uuid4().hex;lock=run/'running.lock'
    # Only this process may advance a run. Crash recovery requires explicit stale
    # lock clearance by the CLI after the operator checks the old PID/session.
    with lock.open('x') as f:json.dump({'host':__import__('socket').gethostname(),'pid':os.getpid(),'token':token},f)
    attempt=run/'attempts'/token;attempt.mkdir(parents=True)
    try:
        legacy_report=read(Path(config['legacy_cache'])/'cache_report.json')
        if sha(Path(config['model'])/'vae/diffusion_pytorch_model.safetensors')!=legacy_report['vae_weight_sha256']:
            raise ValueError('VAE identity before cache generation')
        vae_holder=[]
        @torch.no_grad()
        def encode_new(image):
            if config.get('read_only_cache'):raise RuntimeError('Expected existing cache; no re-encoding allowed')
            from diffusers import AutoencoderKL
            from activepour_model.data import encode_latent
            if not vae_holder:
                vae_holder.append(AutoencoderKL.from_pretrained(config['model'],subfolder='vae',
                    local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False))
            return encode_latent(vae_holder[0],image_tensor(image).cuda()).cpu()
        cache,rows,cache_stats=prepare_cache(config['dataset'],config['release'],config['target_cache'],
            config['legacy_cache'],config['legacy_dataset'],config['condition_mode'],encode=encode_new)
        from .dataset import load_release
        validation=None
        if load_release(config['dataset'],config['release'],'validation',verify=False):
            validation,_,_=prepare_cache(config['dataset'],config['release'],config['target_cache'],
                config['legacy_cache'],config['legacy_dataset'],config['condition_mode'],split='validation',encode=encode_new,
                row_ids=config.get('validation_ids'))
        vae_holder.clear()
        import gc
        gc.collect();torch.cuda.empty_cache()
        if cache_stats['prompt']!=config['prompt'] or cache_stats['max_text_length']!=config['max_text_length']:
            raise ValueError('fixed text cache differs from config')
        if sha(Path(config['model'])/'transformer/diffusion_pytorch_model.safetensors')!=cache_stats['base_transformer_sha256']:
            raise ValueError('base model identity')
        if sha(Path(config['model'])/'vae/diffusion_pytorch_model.safetensors')!=cache_stats['vae_sha256']:
            raise ValueError('VAE identity')
        atomic_json(attempt/'cache_report.json',cache_stats)
        # Cache misses may instantiate a VAE and consume RNG. Re-seed AFTER cache
        # work so full3/post1 start with identical trainable weights regardless of
        # which one populated the shared cache first.
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
        transformer=SD3Transformer2DModel.from_pretrained(config['model'],subfolder='transformer',
            local_files_only=True,torch_dtype=torch.bfloat16).cuda()
        cfg=transformer.config
        if (cfg.num_layers,cfg.num_attention_heads,cfg.attention_head_dim,cfg.patch_size)!=(24,24,64,2):
            raise ValueError('not full SD3.5 Medium')
        modules=attach_lora(transformer,config['lora_rank'],config['lora_alpha'])
        transformer.enable_gradient_checkpointing()
        conditioner=PhysicsConditioner(hidden_dim=transformer.inner_dim,context_dim=cfg.joint_attention_dim,
            num_layers=cfg.num_layers,feature_width=config['feature_width'],injection_layers=config['injection_layers'],
            action_low=config['action_low'],action_high=config['action_high'],probe_mode='raw',
            probe_transform=config['probe_transform'],response_gain=config['response_gain'],
            full_precision_conditions=True,architecture=config.get('architecture','legacy'),frame_mode=config['condition_mode'],
            condition_dropout=config.get('condition_dropout',.05)).cuda()
        compatible={'base':cache_stats['base_transformer_sha256'],'vae':cache_stats['vae_sha256'],
            'prompt':config['prompt'],'max_text_length':config['max_text_length'],
            'condition_mode':config['condition_mode'],'conditioner_spec':conditioner.spec(),
            'lora_rank':config['lora_rank'],'lora_alpha':config['lora_alpha'],'lora_modules':modules}
        source={str(p.relative_to(Path(__file__).resolve().parents[1])):sha(p)
                for folder in ['activepour_pipeline','activepour_model']
                for p in (Path(__file__).resolve().parents[1]/folder).glob('*.py')}
        identity={'config':config,'release_sha256':sha(config['release']),'compatible':compatible,
            'source_hashes':source,'versions':{k:md.version(k) for k in ['torch','diffusers','transformers']},
            'optimizer_names':checkpoint.named_optimizer_signature(transformer,conditioner)}
        parent=None
        if not resume:
            if warm_start:
                state=torch.load(warm_start,map_location='cpu',weights_only=True)
                migrated=False
                if config.get('migration_parent_sha256'):
                    from .migration import migrate_post_parent
                    migrate_post_parent(state,warm_start,config['migration_parent_sha256'],compatible,conditioner)
                    parent_identity=fingerprint(state['identity']);migrated=True
                elif state.get('format')=='activepour_native_lora_v1':
                    old=state['config'];oldreport=read(Path(old['cache'])/'cache_report.json')
                    if (config['condition_mode']!='full3' or state['conditioner_spec']!=conditioner.spec()
                        or state['base_transformer_sha256']!=compatible['base']
                        or oldreport['vae_weight_sha256']!=compatible['vae']
                        or any(old[k]!=config[k] for k in ['prompt','max_text_length','lora_rank','lora_alpha'])):
                        raise ValueError('legacy warm-start contract mismatch')
                    parent_identity=fingerprint({'legacy_cache_sha256':state['cache_sha256'],
                        'base_sha256':state['base_transformer_sha256'],'format':state['format']})
                else:
                    if state['compatible']!=compatible:raise ValueError('warm start input/model contract mismatch')
                    parent_identity=fingerprint(state['identity'])
                load_lora_state(transformer,state['lora'])
                if not migrated:conditioner.load_state_dict(state['conditioner'],strict=True)
                parent={'path':str(warm_start),'sha256':sha(warm_start),'step':state['step'],'optimizer_reset':True,
                        'parent_identity_sha256':parent_identity,'source_format':state['format']}
            import hashlib
            init_hash=hashlib.sha256()
            for prefix,mapping in [('lora',lora_state(transformer)),('conditioner',conditioner.state_dict())]:
                for name,value in sorted(mapping.items()):
                    init_hash.update((prefix+'.'+name+str(tuple(value.shape))).encode())
                    init_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
            atomic_json(run/'run_manifest.json',{'run_id':run.name,'config':config,'identity':identity,
                'parent':parent,'initial_trainable_sha256':init_hash.hexdigest(),'scope':config['scope'],'created':time.time()})
        elif manifest['identity']!=identity:raise ValueError('resume data/source/environment identity changed')
        lp=[p for p in transformer.parameters() if p.requires_grad];cp=list(conditioner.parameters())
        optimizer=torch.optim.AdamW([{'params':lp,'lr':config['learning_rate_lora']},
            {'params':cp,'lr':config['learning_rate_condition']}],weight_decay=config['weight_decay'],eps=1e-8)
        params=lp+cp;order=[];history=[];step0=0;best=None
        schedule=FlowMatchEulerDiscreteScheduler.from_pretrained(config['model'],subfolder='scheduler',local_files_only=True)
        # A freshly loaded schedule can have N sigmas, not N+1. Drop a terminal
        # sigma only when it really exists; all three modes train the same full grid.
        sigmas=schedule.sigmas;times=schedule.timesteps
        if len(sigmas)==len(times)+1:sigmas=sigmas[:-1]
        if len(sigmas)!=len(times):raise ValueError('sigma/timestep grid mismatch')
        sigmas=sigmas.cuda();times=times.cuda()
        # Streaming union avoids stacking every 512px training target in memory.
        union=None
        for r in rows:
            mask=image_tensor(Path(config['dataset'])/r['targets']['future_grid']).mean(1,keepdim=True)<.9
            union=mask if union is None else union|mask
        weights=fixed_spatial_weights(union,config['spatial_loss_boost']).cuda()
        restore=None
        if resume:
            restore,path=checkpoint.load(run,identity)
            load_lora_state(transformer,restore['lora']);conditioner.load_state_dict(restore['conditioner'],strict=True)
            optimizer.load_state_dict(restore['optimizer'])
            order=restore['order'];history=restore['history'];step0=restore['step'];best=restore['best']
            checkpoint.restore_rng(restore['rng'])
        atomic_json(attempt/'start.json',{'resume':resume,'start_step':step0,'optimizer_state_entries':len(optimizer.state),
            'parent':parent,'train_sample_ids':cache['ids'],'condition_mode':config['condition_mode']})
        end=min(config['max_steps'],stop_after or config['max_steps'])
        if end<=step0:raise ValueError('no new steps requested; run already at requested step')
        last_saved=step0;last_saved_time=time.monotonic()
        # Persist accumulated time across tmux/restarts; never splice a new data
        # release into this run. Wall time includes setup/validation but not pauses.
        prior_wall=restore.get('timing',{}).get('active_wall_s',0.) if restore else 0.
        optimizer_s=restore.get('timing',{}).get('optimizer_s',0.) if restore else 0.
        seen=sum(len(x['sample_ids']) for x in history)
        if not resume and config.get('save_step_zero'):
            metrics=validation_flow(transformer,conditioner,validation,config['evaluation_seed'])
            best={'step':0,'value':metrics['mean_flow_mse'],'metric':'validation_flow_mse'}
            atomic_json(attempt/'evaluations'/'step_00000000.json',{'step':0,'metrics':metrics})
            initial={'format':'activepour_resume_v1','identity':identity,'compatible':compatible,
                'step':0,'lora':{k:v.detach().cpu().clone() for k,v in lora_state(transformer).items()},
                'conditioner':{k:v.detach().cpu().clone() for k,v in conditioner.state_dict().items()},
                'optimizer':optimizer.state_dict(),'order':[],'history':[],'best':best,
                'rng':checkpoint.rng_state(),'scaler':None,'schedule':{'config':config,'completed_step':0}}
            checkpoint.save(run,initial);atomic_json(run/'best.json',dict(read(run/'latest.json'),metric=best))
        for step in range(step0+1,end+1):
            torch.cuda.synchronize();step_started=time.perf_counter()
            transformer.train();conditioner.train();optimizer.zero_grad(set_to_none=True)
            scale=min(step/max(config['warmup_steps'],1),1.)
            if config.get('cosine_decay'):
                progress=max(0,(step-config['warmup_steps'])/max(config['max_steps']-config['warmup_steps'],1))
                floor=config.get('cosine_floor',.1)
                scale*=floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
            for g,lr in zip(optimizer.param_groups,[config['learning_rate_lora'],config['learning_rate_condition']]):g['lr']=lr*scale
            losses=[];used=[];noise_times=[];noise_hashes=[]
            for _ in range(config['gradient_accumulation']):
                ids=[]
                for _ in range(config['train_batch_size']):
                    if not order:order=torch.randperm(len(rows)).tolist()
                    ids.append(order.pop())
                used.extend(cache['ids'][i] for i in ids)
                target=cache['target_latents'][ids].cuda().float()
                noisy,velocity,k=paired_flow_batch(target,sigmas,boundary_probability=config['pure_noise_probability'])
                noise_times.append(k.detach().cpu().tolist())
                import hashlib
                noise_hashes.append(hashlib.sha256(velocity.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                predicted=batch_prediction(transformer,conditioner,cache,noisy.to(torch.bfloat16),times[k],ids)
                loss=flow_mse(predicted,velocity,weights)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                (loss/config['gradient_accumulation']).backward();losses.append(float(loss.detach()))
            audit=gradient_audit(transformer,conditioner)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            record={'step':step,'sample_ids':used,'loss':float(np.mean(losses)),
                    'lr':[g['lr'] for g in optimizer.param_groups],'grad_norm':float(norm),'gradients':audit,
                    'noise_time_indices':noise_times,'paired_velocity_sha256':noise_hashes}
            record['response_gradient_l2']={prefix:float(sum((p.grad.detach().float().square().sum()
                for n,p in conditioner.named_parameters() if n.startswith(prefix) and p.grad is not None),
                torch.zeros((),device='cuda')).sqrt())
                for prefix in ['probe_encoder','spatial_refine','projections','action_encoder','history_','query_','delta_projection','delta_alpha','phase_match_bias','stage_action','stage_query','stage_film']}
            history.append(record)
            torch.cuda.synchronize();optimizer_s+=time.perf_counter()-step_started;seen+=len(used)
            record.update(samples_seen=seen,optimizer_s=optimizer_s,
                          active_wall_s=prior_wall+time.perf_counter()-started)
            with (attempt/'steps.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('TRAIN',run.name,json.dumps(record),flush=True)
            improved=False
            if validation is not None and (step%config.get('eval_every',200)==0 or step==end):
                metrics=validation_flow(transformer,conditioner,validation,config['evaluation_seed'])
                ef=attempt/'evaluations'/f'step_{step:08d}.json'
                atomic_json(ef,{'run_id':run.name,'step':step,'release_sha256':sha(config['release']),
                    'condition_mode':config['condition_mode'],'noise_seed':config['evaluation_seed'],'metrics':metrics,
                    'samples_seen':seen,'optimizer_s':optimizer_s,'active_wall_s':prior_wall+time.perf_counter()-started})
                eligible=step in config.get('selection_steps',[step])
                if eligible and (best is None or metrics['mean_flow_mse']<best['value']):
                    best={'step':step,'value':metrics['mean_flow_mse'],'metric':'validation_flow_mse'};improved=True
            # No training RNG consumption in optional validation/callbacks.
            if improved or step%config['checkpoint_every']==0 or time.monotonic()-last_saved_time>=300 or step==end:
                state={'format':'activepour_resume_v1','identity':identity,'compatible':compatible,
                    'step':step,'lora':{k:v.detach().cpu().clone() for k,v in lora_state(transformer).items()},
                    'conditioner':{k:v.detach().cpu().clone() for k,v in conditioner.state_dict().items()},
                    'optimizer':optimizer.state_dict(),'order':list(order),'history':history,'best':best,
                    'rng':checkpoint.rng_state(),'scaler':None,'schedule':{'config':config,'completed_step':step},
                    'timing':{'optimizer_s':optimizer_s,'active_wall_s':prior_wall+time.perf_counter()-started}}
                checkpoint.save(run,state);last_saved=step;last_saved_time=time.monotonic()
                if config.get('shared_progress_dir'):
                    report=Path(config['shared_progress_dir'])/config['condition_mode']
                    atomic_json(report/'latest.json',read(run/'latest.json'))
                    atomic_json(report/'status.json',{'step':step,'max_steps':config['max_steps'],'loss':record['loss'],'best':best})
                if improved:atomic_json(run/'best.json',dict(read(run/'latest.json'),metric=best))
            if config.get('deadline_unix') and time.time()>=config['deadline_unix']:
                # Pause only at a persisted optimizer boundary; full resume is retained.
                if last_saved==step:
                    end=step;break
        atomic_json(run/'progress.json',{'step':end,'complete':end==config['max_steps'],'history':history,
                    'last_checkpoint_step':last_saved,'elapsed_this_attempt_s':time.perf_counter()-started})
        atomic_json(attempt/'exit.json',{'state':'complete' if end==config['max_steps'] else 'paused','step':end})
    except BaseException as exc:
        atomic_json(attempt/'exit.json',{'state':'failed','reason':repr(exc)})
        raise
    finally:
        if lock.exists() and read(lock)['token']==token:lock.unlink()
