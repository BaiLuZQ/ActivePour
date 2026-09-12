"""Export and run a future generator with no access to true-future caches."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from diffusers import SD3Transformer2DModel, AutoencoderKL
from .conditioning import PhysicsConditioner
from .lora import attach_lora, load_lora_state
from .data import sha256, write_json, decode_latent, to_pixels
from .training import sample


def export_bundle(config, trained_run, output):
    state = torch.load(trained_run/'checkpoint_final.pt', map_location='cpu', weights_only=True)
    source_config = state['config']
    cache_dir = Path(source_config['cache'])
    cache = torch.load(cache_dir/'cache.pt', map_location='cpu', weights_only=True)
    if sha256(cache_dir/'cache.pt') != state['cache_sha256']:
        raise ValueError('cannot export: text cache identity changed')
    # Deliberately omit target_latents, training images, labels, IDs and source paths.
    bundle = {'format': 'activepour_generator_bundle_v1',
        'conditioner_spec': state['conditioner_spec'], 'conditioner': state['conditioner'],
        'lora': state['lora'], 'text_tokens': cache['text_tokens'], 'pooled_text': cache['pooled_text'],
        'base_transformer_sha256': state['base_transformer_sha256'],
        'source_checkpoint_sha256': sha256(trained_run/'checkpoint_final.pt'),
        'generator_config': {k: source_config[k] for k in ('model', 'sample_steps', 'lora_rank', 'lora_alpha')},
        'prompt': source_config['prompt'], 'training_scope': 'eight-sample TRAIN memorization only'}
    torch.save(bundle, output/'generator_bundle.pt')
    write_json(output/'bundle_manifest.json', {
        'format': bundle['format'], 'bundle_sha256': sha256(output/'generator_bundle.pt'),
        'source_checkpoint_sha256': bundle['source_checkpoint_sha256'],
        'base_transformer_sha256': bundle['base_transformer_sha256'],
        'contains_true_future_latents': False, 'contains_training_probes': False,
        'contains_eta_or_material_labels': False, 'scope': bundle['training_scope']})
    print('exported standalone conditioning/LoRA/text bundle; no true future data', flush=True)


@torch.no_grad()
def predict_bundle(bundle_path, probe_paths, action, output, *, seed=20260907, steps=None, model_path=None):
    started = time.perf_counter()
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    if bundle['format'] != 'activepour_generator_bundle_v1':
        raise ValueError('not a standalone generator bundle')
    forbidden = {'target_latents', 'eta_final', 'targets', 'probe_features'}
    if forbidden.intersection(bundle):
        raise ValueError('unexpected dataset fields in inference bundle')
    config = dict(bundle['generator_config'])
    if model_path is not None:
        config['model'] = str(model_path)
    if steps is not None:
        config['sample_steps'] = steps
    if sha256(Path(config['model'])/'transformer/diffusion_pytorch_model.safetensors') != bundle['base_transformer_sha256']:
        raise ValueError('pretrained base hash mismatch')
    arrays = [np.asarray(Image.open(path).convert('L'), dtype=np.float32) for path in probe_paths]
    if any(a.shape != (256,256) for a in arrays):
        raise ValueError('three aligned 256x256 grayscale probe images required')
    if np.asarray(action).shape != (3,) or not np.isfinite(action).all():
        raise ValueError('three finite action numbers required')
    probe = torch.from_numpy(np.stack(arrays)/127.5-1).unsqueeze(0).cuda()
    action_tensor = torch.tensor([action], device='cuda', dtype=torch.float32)
    transformer = SD3Transformer2DModel.from_pretrained(config['model'], subfolder='transformer',
        local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
    attach_lora(transformer, config['lora_rank'], config['lora_alpha'])
    load_lora_state(transformer, bundle['lora'])
    transformer.requires_grad_(False)
    conditioner = PhysicsConditioner(**bundle['conditioner_spec']).to('cuda').eval()
    conditioner.load_state_dict(bundle['conditioner'], strict=True)
    conditioner.requires_grad_(False)
    if torch.any(action_tensor < conditioner.action_low-1e-5) or torch.any(action_tensor > conditioner.action_high+1e-5):
        raise ValueError('action outside current calibrated domain')
    vae = AutoencoderKL.from_pretrained(config['model'], subfolder='vae',
        local_files_only=True, torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    if conditioner.probe_mode == 'vae':
        from .data import encode_latent
        probe = encode_latent(vae, probe)
    conditions = {'probe_features': probe, 'actions': action_tensor,
                  'text_tokens': bundle['text_tokens'].cuda(), 'pooled_text': bundle['pooled_text'].cuda()}
    torch.cuda.synchronize()
    sampling_started = time.perf_counter()
    z = sample(transformer, conditioner, conditions, config, 0, seed)
    torch.cuda.synchronize()
    sampling_elapsed = time.perf_counter()-sampling_started
    decoding_started = time.perf_counter()
    pixels = to_pixels(decode_latent(vae, z.float()))[0]
    decoding_elapsed = time.perf_counter()-decoding_started
    Image.fromarray(np.rint(pixels*255).astype('uint8')).save(output/'future_grid.png')
    torch.save(z.cpu(), output/'predicted_future_latent.pt')
    write_json(output/'prediction.json', {'bundle_sha256': sha256(bundle_path),
        'probe_sha256': [sha256(path) for path in probe_paths], 'action_deg_s_s': list(action),
        'seed': seed, 'sample_steps': config['sample_steps'], 'guidance_scale': 1.,
        'latent_shape': list(z.shape), 'elapsed_s': time.perf_counter()-started,
        'sampling_elapsed_s':sampling_elapsed,'vae_decode_elapsed_s':decoding_elapsed,
        'peak_gpu_allocated_gb': torch.cuda.max_memory_allocated()/1e9,
        'scope': 'prediction from three observations/action/noise; no ground truth loaded; no eta head'})
    print('saved', output/'future_grid.png', flush=True)
