"""Optional overfit diagnostics: paired noise and a fixed spatial loss metric.

Neither changes inference inputs. These are optimization choices, not an
extra predictor or an observation of the true future at inference time.
"""
import torch
from torch.nn import functional as F


def paired_flow_batch(target, sigmas, *, boundary_probability=.5):
    """Use one random noise/time for all conditions; each row keeps its own label.

    Shared randomness makes between-condition differences less noisy. A
    deliberate atom at sigma=1 emphasizes predicting from pure noise, where
    the network cannot infer the answer from a partially visible true future.
    The remaining half still trains the entire denoising trajectory.
    """
    n = len(target)
    noise = torch.randn_like(target[:1]).expand_as(target)
    k = torch.randint(len(sigmas), (1,), device=target.device)
    if torch.rand((), device=target.device) < boundary_probability:
        k.zero_()
    k = k.expand(n)
    sigma = sigmas[k].reshape(n,1,1,1)
    return (1-sigma)*target+sigma*noise, noise-target, k


def fixed_spatial_weights(foreground_union, boost=9.):
    """One shared TRAIN-derived spatial mask; no sample-specific future mask.

    A positive, fixed spatial metric preserves the population FM optimum at
    each output coordinate. The same map is used for all examples/times.
    Max-pool maps 512px -> 64 latent cells and adds one-cell VAE-boundary margin.
    """
    mask = F.max_pool2d(foreground_union.float(),8)
    mask = F.max_pool2d(mask,3,stride=1,padding=1)
    weights = 1 + boost*mask
    return weights / weights.mean()


def flow_mse(prediction, target, spatial_weights=None):
    error = (prediction.float()-target.float()).square()
    if spatial_weights is not None:
        error = error*spatial_weights
    return error.mean()
