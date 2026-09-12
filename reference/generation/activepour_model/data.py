"""Strict train-only Node-6 selection and image/latent conventions."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageFilter
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def select_rows(dataset, episodes):
    rows = json.loads((Path(dataset)/'training_index.json').read_text())
    selected = [r for episode in episodes for r in rows if r['episode_id'] == episode]
    if len(set(episodes)) != len(episodes) or set(r['episode_id'] for r in selected) != set(episodes):
        raise ValueError('duplicate or missing episode')
    if not 8 <= len(selected) <= 16 or any(r['split'] != 'train' for r in selected):
        raise ValueError('Node-6 requires 8-16 TRAIN rows; held-out rows forbidden')
    for episode in episodes:
        group = [r for r in selected if r['episode_id'] == episode]
        if len(group) != 4 or len({tuple(r['inputs']['probe_paths']) for r in group}) != 1:
            raise ValueError('each selected episode must contain all four action branches')
    for r in selected:
        if set(r['inputs']) != {'action_deg_s_s', 'probe_paths'}:
            raise ValueError('unexpected input fields: possible physical-label leakage')
        a=np.asarray(r['inputs']['action_deg_s_s'])
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError('expected three finite continuous action values')
    return selected


def intervention_partners(rows, index):
    """Find real matched examples, never manufacture a label for an unseen pairing."""
    row = rows[index]
    probe_choices = [i for i, other in enumerate(rows)
                     if other['episode_id'] != row['episode_id'] and
                     other['inputs']['action_deg_s_s'] == row['inputs']['action_deg_s_s']]
    action_choices = [i for i, other in enumerate(rows)
                      if other['episode_id'] == row['episode_id'] and
                      other['inputs']['action_deg_s_s'] != row['inputs']['action_deg_s_s']]
    if not probe_choices or not action_choices:
        raise ValueError('condition intervention requires matched actions across episodes')
    # Largest normalized action distance gives a useful, reproducible contrast.
    widths = np.array([20., .6, .7])
    original = np.array(row['inputs']['action_deg_s_s'])
    action = max(action_choices, key=lambda i: np.linalg.norm(
        (np.array(rows[i]['inputs']['action_deg_s_s']) - original) / widths))
    return probe_choices[0], action


def image_tensor(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def probe_tensor(dataset, row):
    # Exactly three grayscale time channels, not colored overlays or labels.
    arrays = [np.asarray(Image.open(Path(dataset)/p).convert('L'), dtype=np.float32)
              for p in row['inputs']['probe_paths']]
    return torch.from_numpy(np.stack(arrays) / 127.5 - 1).unsqueeze(0)


def to_pixels(tensor):
    return (tensor.detach().float().cpu().permute(0, 2, 3, 1).numpy() / 2 + .5).clip(0, 1)


def encode_latent(vae, x):
    z = vae.encode(x).latent_dist.mode()
    return (z - vae.config.shift_factor) * vae.config.scaling_factor


def decode_latent(vae, z):
    return vae.decode(z / vae.config.scaling_factor + vae.config.shift_factor).sample


def image_metrics(truth, predicted, *, foreground='bright', macro_sigma=0.):
    a, b = truth.mean(-1), predicted.mean(-1)
    if foreground not in ('bright','dark'):
        raise ValueError('unknown foreground polarity')
    if foreground == 'dark':
        a,b=1-a,1-b
    stages = []
    for y, x in ((0, 0), (0, 256), (256, 0), (256, 256)):
        aa, bb = a[y:y+256, x:x+256], b[y:y+256, x:x+256]
        if foreground == 'dark':
            # White background must NEVER count as correctly predicted material.
            def smooth(v):
                return np.asarray(Image.fromarray(np.uint8(np.round(np.clip(v,0,1)*255))).filter(
                    ImageFilter.GaussianBlur(macro_sigma)))/255.
            ma,mb=smooth(aa)>.2,smooth(bb)>.2
        else:
            ma, mb = aa > .05, bb > .05
        union = np.count_nonzero(ma | mb)
        stage={'iou': float(np.count_nonzero(ma & mb) / max(union, 1)),
               'pixel_mae': float(np.abs(aa-bb).mean())}
        if foreground=='dark':
            pa,pb=aa>.5,bb>.5
            stage['particle_iou']=float((pa & pb).sum()/max(1,(pa | pb).sum()))
            stage['foreground_roi_mae']=float(abs(aa-bb)[ma | mb].mean()) if union else 0.
        stages.append(stage)
    result={'mean_iou': float(np.mean([s['iou'] for s in stages])),
            'pixel_mae': float(np.abs(truth-predicted).mean()), 'stages': stages}
    if foreground=='dark':
        result.update(particle_iou=float(np.mean([s['particle_iou'] for s in stages])),
                      foreground_roi_mae=float(np.mean([s['foreground_roi_mae'] for s in stages])),
                      foreground=foreground,macro_sigma_pixels=macro_sigma)
    return result
