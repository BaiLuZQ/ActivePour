"""Image and latent conventions used by the frozen experiments."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch

def image_tensor(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


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


