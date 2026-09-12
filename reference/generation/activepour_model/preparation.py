"""Cache frozen inputs/targets and test the three-time-channel VAE bottleneck."""
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from diffusers import AutoencoderKL, StableDiffusion3Pipeline
from .data import (select_rows, sha256, write_json, image_tensor, probe_tensor,
                   encode_latent, decode_latent, to_pixels)


@torch.no_grad()
def prepare(config, output):
    start = time.perf_counter()
    dataset, model = Path(config['dataset']), Path(config['model'])
    rows = select_rows(dataset, config['episodes'])
    write_json(output/'selection.json', rows)
    hashes = {str(p): sha256(dataset/p) for r in rows
              for p in [*r['inputs']['probe_paths'], r['targets']['future_grid']]}
    write_json(output/'input_hashes.json', hashes)
    vae = AutoencoderKL.from_pretrained(model, subfolder='vae', local_files_only=True,
                                       torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    probes, diagnostics, targets = {}, [], []
    for row in rows:
        ep = row['episode_id']
        if ep not in probes:
            x = probe_tensor(dataset, row).cuda()
            z = encode_latent(vae, x)
            rec = decode_latent(vae, z)
            original, reconstructed = to_pixels(x)[0], to_pixels(rec)[0]
            ious = []
            for channel in range(3):
                if config.get('foreground','bright')=='dark':
                    a,b=original[...,channel]<.5,reconstructed[...,channel]<.5
                else:
                    a, b = original[..., channel] > .05, reconstructed[..., channel] > .05
                ious.append(float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1)))
            # Assess the *temporal responses*, not only each frame's silhouette.
            aa = F.avg_pool2d((x+1)/2, 16)
            bb = F.avg_pool2d(((rec+1)/2).clamp(0, 1), 16)
            delta_a = aa[:, 1:] - aa[:, :1]
            delta_b = bb[:, 1:] - bb[:, :1]
            relative = float((delta_a-delta_b).abs().sum() / delta_a.abs().sum().clamp_min(1e-8))
            probes[ep] = {'raw': x.cpu(), 'vae': z.cpu()}
            diagnostics.append({'episode': ep, 'channel_iou': ious,
                                'coarse_response_relative_l1': relative,
                                'pixel_mae': float(np.abs(original-reconstructed).mean())})
            Image.fromarray(np.rint(np.concatenate([original, reconstructed], 1)*255).astype('uint8')).save(output/f'{ep}_probe_vae.png')
        target = image_tensor(dataset/row['targets']['future_grid']).cuda()
        targets.append(encode_latent(vae, target).cpu())
        print('cached image', row['id'], flush=True)
    gate = config['probe_vae_gate']
    passed = all(min(d['channel_iou']) >= gate['min_channel_iou'] and
                 d['coarse_response_relative_l1'] <= gate['max_coarse_response_relative_l1'] for d in diagnostics)
    mode = 'vae' if passed else 'raw'
    if config.get('force_probe_mode'):
        if config['force_probe_mode'] not in ('raw','vae'):
            raise ValueError('invalid explicit probe mode')
        mode=config['force_probe_mode']
    write_json(output/'probe_vae_report.json', {'diagnostics': diagnostics, 'gate': gate,
        'passed': passed, 'selected_probe_mode': mode,'forced_mode':config.get('force_probe_mode'),
        'scope': 'train-only representation diagnostic, not a probe-benefit experiment'})
    del vae, x, z, rec, target
    gc.collect()
    torch.cuda.empty_cache()
    print('probe mode:', mode, '; encoding fixed text with all three original text encoders', flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(model, transformer=None, vae=None,
        local_files_only=True, torch_dtype=torch.bfloat16).to('cuda')
    prompt, _, pooled, _ = pipe.encode_prompt(prompt=config['prompt'], prompt_2=None, prompt_3=None,
        device=torch.device('cuda'), num_images_per_prompt=1, do_classifier_free_guidance=False,
        max_sequence_length=config['max_text_length'])
    cache = {'target_latents': torch.cat(targets),
             'probe_features': torch.cat([probes[r['episode_id']][mode] for r in rows]),
             'actions': torch.tensor([r['inputs']['action_deg_s_s'] for r in rows], dtype=torch.float32),
             'text_tokens': prompt.cpu(), 'pooled_text': pooled.cpu(),
             'ids': [r['id'] for r in rows], 'probe_mode': mode,
             'target_role': 'true_future_latents_for_stage_A_supervision_ONLY'}
    torch.save(cache, output/'cache.pt')
    report = {'sample_count': len(rows), 'probe_mode': mode,
              'shapes': {k: list(v.shape) for k, v in cache.items() if torch.is_tensor(v)},
              'elapsed_s': time.perf_counter()-start, 'cache_sha256': sha256(output/'cache.pt'),
              'transformer_weight_sha256': sha256(model/'transformer/diffusion_pytorch_model.safetensors'),
              'vae_weight_sha256': sha256(model/'vae/diffusion_pytorch_model.safetensors'),
              'dataset_index_sha256': sha256(dataset/'training_index.json'),
              'prompt': config['prompt'], 'max_text_length': config['max_text_length']}
    write_json(output/'cache_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
