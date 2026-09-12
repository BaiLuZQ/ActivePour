"""Flow-matching overfit test using the entire pretrained SD3.5 Medium.

Training: z_sigma=(1-sigma)*z_true + sigma*epsilon, v*=epsilon-z_true.
Sampling: start from independent noise and integrate dz/dsigma=v toward zero.
Neither generation nor its conditioning reads a true future image or eta.
"""
from __future__ import annotations
import gc
import importlib.metadata as md
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from .conditioning import PhysicsConditioner, convert_first_conv_to_response_basis
from .lora import attach_lora, lora_state, load_lora_state
from .data import (sha256, write_json, image_tensor, to_pixels, decode_latent,
                   image_metrics, intervention_partners)
from .objectives import paired_flow_batch, fixed_spatial_weights, flow_mse


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def metrics_for(config, truth, predicted):
    return image_metrics(truth,predicted,foreground=config.get('foreground','bright'),
                         macro_sigma=config.get('macro_metric_sigma',0.))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prediction(transformer, conditioner, cache, z, timesteps, ids,
               *, probe_ids=None, action_ids=None, disable_probe=False):
    # Index choices are diagnostic switches; model input remains only images/actions/text.
    with autocast():
        residuals, context = conditioner(cache['probe_features'][ids if probe_ids is None else probe_ids],
            cache['actions'][ids if action_ids is None else action_ids], cache['text_tokens'],
            disable_probe=disable_probe)
        return transformer(hidden_states=z, timestep=timesteps,
            encoder_hidden_states=context,
            pooled_projections=cache['pooled_text'].expand(len(ids), -1),
            block_controlnet_hidden_states=residuals, return_dict=False)[0]


@torch.no_grad()
def sample(transformer, conditioner, cache, config, index, seed, *, probe_index=None, action_index=None,
           disable_probe=False, steps=None):
    # This function deliberately never accesses cache['target_latents'].
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config['model'], subfolder='scheduler', local_files_only=True)
    scheduler.set_timesteps(steps or config['sample_steps'], device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(seed)
    z = torch.randn((1, 16, 64, 64), device='cuda', generator=generator, dtype=torch.float32)
    with autocast():
        residuals, context = conditioner(cache['probe_features'][[index if probe_index is None else probe_index]],
            cache['actions'][[index if action_index is None else action_index]], cache['text_tokens'],
            disable_probe=disable_probe)
        for t in scheduler.timesteps:
            velocity = transformer(hidden_states=z.to(torch.bfloat16), timestep=t.expand(1),
                encoder_hidden_states=context, pooled_projections=cache['pooled_text'],
                block_controlnet_hidden_states=residuals, return_dict=False)[0]
            z = scheduler.step(velocity.float(), t, z, return_dict=False)[0]
    return z


def truth_pixels(config, row):
    return to_pixels(image_tensor(Path(config['dataset'])/row['targets']['future_grid']))[0]


def comparison_image(truth, generated, label, path):
    canvas = Image.new('RGB', (1024, 544), 'white')
    canvas.paste(Image.fromarray(np.rint(truth*255).astype('uint8')), (0, 32))
    canvas.paste(Image.fromarray(np.rint(generated*255).astype('uint8')), (512, 32))
    ImageDraw.Draw(canvas).text((8, 8), 'SIMULATION | SD3.5 FROM NOISE  ' + label, fill='black')
    canvas.save(path)


@torch.no_grad()
def generate_eval(transformer, conditioner, vae, cache, config, rows, output, tag, indices):
    folder = output/tag
    folder.mkdir(exist_ok=False)
    transformer.eval()
    conditioner.eval()
    report = []
    for index in indices:
        z = sample(transformer, conditioner, cache, config, index, config['evaluation_seed'])
        y = to_pixels(decode_latent(vae, z.float()))[0]
        truth = truth_pixels(config, rows[index])
        metrics = metrics_for(config, truth, y)
        metrics.update({'index': index, 'id': rows[index]['id'], 'seed': config['evaluation_seed']})
        report.append(metrics)
        comparison_image(truth, y, f'{index}: IoU={metrics["mean_iou"]:.3f}', folder/f'{index:02d}_comparison.png')
        torch.save(z.cpu(), folder/f'{index:02d}_predicted_latent.pt')
        print(f'{tag} sample {index}: IoU={metrics["mean_iou"]:.4f} MAE={metrics["pixel_mae"]:.5f}', flush=True)
    summary = {'scope': 'training-set generation from noise, no generalization estimate', 'samples': report,
               'mean_iou': float(np.mean([r['mean_iou'] for r in report])),
               'pixel_mae': float(np.mean([r['pixel_mae'] for r in report])),
               'sampling_steps': config['sample_steps'], 'guidance_scale': 1.0,
               'sampler': 'FlowMatchEulerDiscreteScheduler from pretrained config, no CFG/SLG'}
    write_json(folder/'metrics.json', summary)
    return summary


@torch.no_grad()
def fixed_flow_eval(transformer, conditioner, cache, config):
    losses = []
    generator = torch.Generator(device='cuda').manual_seed(config['evaluation_seed'] + 1)
    transformer.eval()
    conditioner.eval()
    for i in range(len(cache['ids'])):
        target = cache['target_latents'][[i]].float()
        noise = torch.randn(target.shape, device='cuda', generator=generator)
        for sigma in (.2, .5, .8):
            z = (1-sigma)*target + sigma*noise
            v = prediction(transformer, conditioner, cache, z.to(torch.bfloat16),
                           torch.tensor([sigma*1000], device='cuda'), [i])
            losses.append(float(F.mse_loss(v.float(), noise-target)))
    return float(np.mean(losses))


def gradient_audit(transformer, conditioner):
    groups = {'lora': [], 'probe_encoder': [], 'stage_embeddings': [], 'projections': [], 'action_encoder': []}
    for name, p in transformer.named_parameters():
        if p.requires_grad and p.grad is not None:
            groups['lora'].append(p.grad.detach().float().square().sum())
        elif not p.requires_grad and p.grad is not None:
            raise RuntimeError('frozen original weight received a gradient: ' + name)
    for name, p in conditioner.named_parameters():
        if p.grad is not None:
            groups.setdefault(name.split('.')[0], []).append(p.grad.detach().float().square().sum())
    return {k: float(torch.stack(values).sum().sqrt()) if values else 0. for k, values in groups.items()}


def save_checkpoint(path, transformer, conditioner, config, step, cache_report):
    torch.save({'format': 'activepour_native_lora_v1', 'step': step, 'config': config,
                'conditioner_spec': conditioner.spec(),
                'conditioner': {n: t.detach().cpu().clone() for n, t in conditioner.state_dict().items()},
                'lora': lora_state(transformer), 'cache_sha256': cache_report['cache_sha256'],
                'base_transformer_sha256': cache_report['transformer_weight_sha256']}, path)


def load_checkpoint(path, transformer, conditioner, cache_report, *, allow_response_conversion=False):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'activepour_native_lora_v1' or state['base_transformer_sha256'] != cache_report['transformer_weight_sha256']:
        raise ValueError('checkpoint base identity mismatch')
    old_spec = dict(probe_transform='identity', response_gain=32., full_precision_conditions=False,
                    **{k: v for k, v in state['conditioner_spec'].items()
                       if k not in ('probe_transform', 'response_gain', 'full_precision_conditions')})
    old_spec.update(state['conditioner_spec'])
    new_spec = conditioner.spec()
    condition_state = state['conditioner']
    if allow_response_conversion:
        if old_spec['probe_mode'] != 'raw' or old_spec['probe_transform'] != 'identity' or new_spec['probe_transform'] != 'post_response':
            raise ValueError('only explicit identity -> post_response conversion supported')
        condition_state = convert_first_conv_to_response_basis(condition_state, new_spec['response_gain'])
        for field in ('probe_transform', 'response_gain', 'full_precision_conditions'):
            old_spec[field] = new_spec[field]
    if state['cache_sha256'] != cache_report['cache_sha256'] or old_spec != new_spec:
        raise ValueError('checkpoint data/architecture mismatch')
    conditioner.load_state_dict(condition_state, strict=True)
    load_lora_state(transformer, state['lora'])
    return state


@torch.no_grad()
def interventions(transformer, conditioner, vae, cache, config, rows, output):
    folder = output/'condition_interventions'
    folder.mkdir(exist_ok=False)
    results = []
    # Every crossed combination below also exists among the eight TRAIN examples.
    # Paired noise holds stochastic sampling fixed so changes are conditional.
    for i in range(len(rows)):
        p, a = intervention_partners(rows, i)
        truth = truth_pixels(config, rows[i])
        outputs = {}
        for label, kwargs in [('correct', {}), ('swap_probe', {'probe_index': p}),
                              ('swap_action', {'action_index': a})]:
            z = sample(transformer, conditioner, cache, config, i, config['evaluation_seed'], **kwargs)
            y = to_pixels(decode_latent(vae, z.float()))[0]
            outputs[label] = y
            reference_index = p if label == 'swap_probe' else a if label == 'swap_action' else i
            results.append({'index': i, 'id': rows[i]['id'], 'condition': label,
                'matching_reference_index': reference_index,
                'against_original': metrics_for(config, truth, y),
                'against_matching_condition': metrics_for(config, truth_pixels(config, rows[reference_index]), y)})
        for label in ('swap_probe', 'swap_action'):
            results[-2 if label == 'swap_probe' else -1]['pixel_change_vs_correct'] = float(np.abs(outputs[label]-outputs['correct']).mean())
        canvas = Image.new('RGB', (2048, 540), 'white')
        for column, (label, img) in enumerate([('truth', truth), *outputs.items()]):
            canvas.paste(Image.fromarray(np.rint(img*255).astype('uint8')), (column*512, 28))
            ImageDraw.Draw(canvas).text((column*512+8, 7), label, fill='black')
        canvas.save(folder/f'{i:02d}_interventions.png')
        print('condition interventions', i, flush=True)
    correct = [r for r in results if r['condition'] == 'correct']
    summary = {'scope': 'TRAIN-set paired-condition sensitivity, not proof of material identification', 'results': results}
    for label in ('swap_probe', 'swap_action'):
        swapped = [r for r in results if r['condition'] == label]
        summary[label] = {
            'mean_pixel_change': float(np.mean([r['pixel_change_vs_correct'] for r in swapped])),
            'mean_original_iou_drop': float(np.mean([a['against_original']['mean_iou']-b['against_original']['mean_iou'] for a, b in zip(correct, swapped)])),
            'mean_original_mae_increase': float(np.mean([b['against_original']['pixel_mae']-a['against_original']['pixel_mae'] for a, b in zip(correct, swapped)])),
            'matching_reference_iou': float(np.mean([r['against_matching_condition']['mean_iou'] for r in swapped]))}
    write_json(folder/'report.json', summary)
    return summary


def run(config, output, *, mode, trained_run=None):
    started = time.perf_counter()
    seed_all(config['seed'])
    cache_dir = Path(config['cache'])
    cache_report = json.loads((cache_dir/'cache_report.json').read_text())
    if sha256(cache_dir/'cache.pt') != cache_report['cache_sha256']:
        raise ValueError('latent/text cache hash mismatch')
    if config['prompt'] != cache_report['prompt'] or config['max_text_length'] != cache_report['max_text_length']:
        raise ValueError('cached text differs from the requested condition')
    if sha256(Path(config['model'])/'transformer/diffusion_pytorch_model.safetensors') != cache_report['transformer_weight_sha256']:
        raise ValueError('pretrained transformer differs from the cached identity')
    for relative, expected in json.loads((cache_dir/'input_hashes.json').read_text()).items():
        if sha256(Path(config['dataset'])/relative) != expected:
            raise ValueError('source data changed after caching: ' + relative)
    cache = torch.load(cache_dir/'cache.pt', map_location='cpu', weights_only=True)
    rows = json.loads((cache_dir/'selection.json').read_text())
    if any(r['split'] != 'train' for r in rows) or cache['ids'] != [r['id'] for r in rows]:
        raise ValueError('training split/order mismatch')
    if 'untrained_baseline_iou' in config:
        baseline = json.loads(Path(config['untrained_baseline_source']).read_text())['generation']['mean_iou']
        if baseline != config['untrained_baseline_iou']:
            raise ValueError('untrained baseline metric does not match recorded evidence')
    for key, value in cache.items():
        if torch.is_tensor(value):
            cache[key] = value.to('cuda')
    spatial_weights = None
    if config.get('spatial_loss_boost', 0) > 0:
        pictures=[((image_tensor(Path(config['dataset'])/r['targets']['future_grid'])+1)/2).mean(1,keepdim=True) for r in rows]
        masks=[x<.95 if config.get('foreground','bright')=='dark' else x>.05 for x in pictures]
        union = torch.cat(masks).any(0,keepdim=True)
        spatial_weights = fixed_spatial_weights(union,config['spatial_loss_boost']).cuda()
        torch.save(spatial_weights.cpu(),output/'fixed_training_loss_weights.pt')
    transformer = SD3Transformer2DModel.from_pretrained(config['model'], subfolder='transformer',
        local_files_only=True, torch_dtype=torch.bfloat16).to('cuda')
    cfg = transformer.config
    # Fail closed: a reduced demo model cannot accidentally pass as SD3.5 Medium.
    if (cfg.num_layers, cfg.num_attention_heads, cfg.attention_head_dim, cfg.patch_size) != (24, 24, 64, 2):
        raise ValueError('not the expected full SD3.5 Medium configuration')
    base_count = sum(p.numel() for p in transformer.parameters())
    lora_names = attach_lora(transformer, config['lora_rank'], config['lora_alpha'])
    transformer.enable_gradient_checkpointing()
    conditioner = PhysicsConditioner(hidden_dim=transformer.inner_dim, context_dim=cfg.joint_attention_dim,
        feature_width=config['feature_width'], num_layers=cfg.num_layers,
        injection_layers=config['injection_layers'], action_low=config['action_low'], action_high=config['action_high'],
        probe_mode=cache['probe_mode'], probe_transform=config.get('probe_transform', 'identity'),
        response_gain=config.get('response_gain', 32.),
        full_precision_conditions=config.get('full_precision_conditions', False)).to('cuda')
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    condition_params = list(conditioner.parameters())
    all_params = lora_params + condition_params
    vae = AutoencoderKL.from_pretrained(config['model'], subfolder='vae', local_files_only=True,
        torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    runtime = {'model': config['model'], 'base_transformer_parameters': base_count,
        'lora_parameters': sum(p.numel() for p in lora_params),
        'condition_parameters': sum(p.numel() for p in condition_params),
        'lora_modules': lora_names, 'conditioner_spec': conditioner.spec(),
        'versions': {n: md.version(n) for n in ('torch', 'diffusers', 'transformers')},
        'base_dtype': 'bfloat16', 'trainable_dtype': 'float32', 'vae_dtype': 'float32',
        'base_transformer_sha256': cache_report['transformer_weight_sha256'],
        'cache_sha256': cache_report['cache_sha256'], 'gpu': torch.cuda.get_device_name(),
        'scope': 'all SD3.5 layers active; original weights frozen; no quantization, no PEFT dependency'}
    write_json(output/'runtime.json', runtime)
    print(json.dumps({k: v for k, v in runtime.items() if k not in ('lora_modules',)}, indent=2), flush=True)
    if mode == 'evaluate':
        checkpoint = trained_run/'checkpoint_final.pt'
        load_checkpoint(checkpoint, transformer, conditioner, cache_report)
        metrics = generate_eval(transformer, conditioner, vae, cache, config, rows, output, 'reloaded_generation', range(len(rows)))
        intervention = interventions(transformer, conditioner, vae, cache, config, rows, output)
        # New noise checks that the memorization did not depend on one evaluation seed.
        second_config = dict(config, evaluation_seed=config['evaluation_seed']+101)
        second = generate_eval(transformer, conditioner, vae, cache, second_config, rows, output, 'second_noise', config['preview_indices'])
        write_json(output/'evaluation_report.json', {'checkpoint_sha256': sha256(checkpoint),
            'generation': metrics, 'second_noise': second,
            'interventions': {k: v for k, v in intervention.items() if k != 'results'},
            'elapsed_s': time.perf_counter()-started})
        return
    initial_state = None
    if config.get('initial_checkpoint'):
        initial_state = load_checkpoint(Path(config['initial_checkpoint']), transformer, conditioner, cache_report,
            allow_response_conversion=config.get('convert_response_basis', False))
        print('initialized from saved weights; optimizer is newly initialized', flush=True)
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': config['learning_rate_lora']},
        {'params': condition_params, 'lr': config['learning_rate_condition']}],
        weight_decay=config['weight_decay'], eps=1e-8)
    # Witnesses supplement the stronger structural check: optimizer contains only adapters.
    frozen = [(n, p, p.detach().flatten()[:16].clone()) for n, p in transformer.named_parameters() if not p.requires_grad]
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config['model'], subfolder='scheduler', local_files_only=True)
    sigmas = scheduler.sigmas[:-1].to('cuda')
    timesteps = scheduler.timesteps.to('cuda')
    before_flow = fixed_flow_eval(transformer, conditioner, cache, config)
    before = generate_eval(transformer, conditioner, vae, cache, config, rows, output, 'step_0000',
                           config['preview_indices'] if mode == 'smoke' else range(len(rows)))
    write_json(output/'baseline.json', {'fixed_flow_mse': before_flow, 'generation': before})
    max_steps = 3 if mode == 'smoke' else config['max_steps']
    audits, history = [], []
    update_start = time.perf_counter()
    # Balanced repeated passes through all selected examples; noise/time freshly sampled.
    order = []
    for step in range(1, max_steps+1):
        transformer.train()
        conditioner.train()
        optimizer.zero_grad(set_to_none=True)
        warmup = min(step / max(config['warmup_steps'], 1), 1.)
        if config.get('cosine_decay', False):
            progress = max(0., (step-config['warmup_steps']) / max(max_steps-config['warmup_steps'], 1))
            warmup *= .2 + .8 * .5 * (1+math.cos(math.pi*progress))
        for group, lr in zip(optimizer.param_groups, [config['learning_rate_lora'], config['learning_rate_condition']]):
            group['lr'] = lr * warmup
        losses = []
        for _ in range(config['gradient_accumulation']):
            batch_size = config.get('train_batch_size', 1)
            if batch_size < 1 or batch_size > len(rows):
                raise ValueError('batch size must fit selected TRAIN set')
            ids = []
            for _ in range(batch_size):
                if not order:
                    order = torch.randperm(len(rows)).tolist()
                ids.append(order.pop())
            target = cache['target_latents'][ids].float()
            noise = torch.randn_like(target)
            if config.get('paired_condition_noise', False):
                noisy, velocity_target, k = paired_flow_batch(target,sigmas,
                    boundary_probability=config.get('pure_noise_probability',.5))
            elif config.get('stratified_sigma', False):
                # Stratify time coverage within each batch, still uniformly sampled
                # over scheduler indices marginally, independently of condition IDs.
                k = ((torch.arange(batch_size, device='cuda') + torch.rand(batch_size, device='cuda'))
                     * len(sigmas) / batch_size).long()[torch.randperm(batch_size, device='cuda')]
            else:
                k = torch.randint(len(sigmas), (batch_size,), device='cuda')
            if not config.get('paired_condition_noise', False):
                sigma = sigmas[k].reshape(batch_size, 1, 1, 1)
                noisy = (1-sigma)*target + sigma*noise
                velocity_target = noise-target
            velocity = prediction(transformer, conditioner, cache, noisy.to(torch.bfloat16), timesteps[k], ids)
            loss = flow_mse(velocity,velocity_target,spatial_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite flow loss')
            (loss/config['gradient_accumulation']).backward()
            losses.append(float(loss.detach()))
        if step <= 3 or step % 100 == 0:
            audit = {'step': step, **gradient_audit(transformer, conditioner)}
            audits.append(audit)
            print('gradient', json.dumps(audit), flush=True)
        norm = torch.nn.utils.clip_grad_norm_(all_params, 1., error_if_nonfinite=True)
        optimizer.step()
        record = {'step': step, 'loss': float(np.mean(losses)), 'grad_norm': float(norm),
                  'elapsed_s': time.perf_counter()-update_start,
                  'peak_allocated_gb': torch.cuda.max_memory_allocated()/1e9}
        history.append(record)
        with (output/'train_log.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if step <= 3 or step % 10 == 0:
            print('train', json.dumps(record), flush=True)
        if mode == 'train' and step % config['eval_every'] == 0:
            generate_eval(transformer, conditioner, vae, cache, config, rows, output, f'step_{step:04d}', config['preview_indices'])
        if mode == 'train' and step % config['checkpoint_every'] == 0:
            save_checkpoint(output/f'checkpoint_{step:04d}.pt', transformer, conditioner, config, step, cache_report)
    after_flow = fixed_flow_eval(transformer, conditioner, cache, config)
    final = generate_eval(transformer, conditioner, vae, cache, config, rows, output, 'final',
                          config['preview_indices'] if mode == 'smoke' else range(len(rows)))
    save_checkpoint(output/'checkpoint_final.pt', transformer, conditioner, config, max_steps, cache_report)
    frozen_ok = all(p.grad is None and torch.equal(p.detach().flatten()[:16], witness) for _, p, witness in frozen)
    gradient_ok = all(any(a[key] > 0 and math.isfinite(a[key]) for a in audits[1:])
                      for key in ('lora', 'probe_encoder', 'stage_embeddings', 'projections', 'action_encoder'))
    gate = config['memorization_gate']
    report = {'mode': mode, 'steps': max_steps, 'sample_count': len(rows),
        'train_batch_size': config.get('train_batch_size', 1),
        'initial_checkpoint': config.get('initial_checkpoint'),
        'initial_checkpoint_step': initial_state['step'] if initial_state else 0,
        'fixed_flow_before': before_flow, 'fixed_flow_after': after_flow,
        'baseline_generation': before, 'final_generation': final,
        'frozen_original_weights_gradient_and_witness_check': frozen_ok,
        'all_trainable_branches_receive_finite_nonzero_gradients': gradient_ok,
        'gradient_audits': audits, 'elapsed_s': time.perf_counter()-started,
        'peak_gpu_allocated_gb': torch.cuda.max_memory_allocated()/1e9,
        'peak_gpu_reserved_gb': torch.cuda.max_memory_reserved()/1e9,
        'memorization_gate': gate,
        'memorization_passed': mode == 'train' and final['mean_iou'] >= gate['mean_iou'] and
            final['pixel_mae'] <= gate['max_pixel_mae'] and
            final['mean_iou']-config.get('untrained_baseline_iou', before['mean_iou']) >= gate['min_iou_gain'],
        'interpretation': 'Memorizing TRAIN images is an integration gate only. No claim about held-out accuracy, probe benefit, or superiority over direct regression.'}
    write_json(output/'training_report.json', report)
    print('DONE', json.dumps({k: v for k, v in report.items() if k not in ('baseline_generation', 'final_generation', 'gradient_audits')}, indent=2), flush=True)
    if not frozen_ok or not gradient_ok:
        raise RuntimeError('gradient/freeze acceptance failed; see report')
