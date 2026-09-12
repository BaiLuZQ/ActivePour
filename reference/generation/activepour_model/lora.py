"""Minimal, explicit LoRA for existing attention Linear layers.

The original full weight W stays frozen. Forward is W*x + (alpha/r)*B*A*x.
This small implementation avoids changing the established SD environment just
to install PEFT; checkpoints are our own named tensors, not PEFT-format files.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base_output = self.base(x)
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        # AMP may produce BF16 even when the residual stream is FP32. Match the
        # actual Linear output, not x.dtype, to retain efficient Q/K/V attention.
        return base_output + delta.to(base_output.dtype) * (self.alpha / self.rank)


def attach_lora(transformer, rank=16, alpha=16):
    transformer.requires_grad_(False)
    names = []
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(('.to_q', '.to_k', '.to_v', '.to_out.0')):
            parent_name, leaf = name.rsplit('.', 1)
            parent = transformer.get_submodule(parent_name)
            setattr(parent, leaf, LoRALinear(module, rank, alpha))
            names.append(name)
    if not names:
        raise ValueError('no attention Linear layers matched')
    return names


def lora_state(transformer):
    return {name: p.detach().cpu().clone() for name, p in transformer.named_parameters()
            if name.endswith(('lora_A', 'lora_B'))}


def load_lora_state(transformer, state):
    expected = {n for n, p in transformer.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('LoRA checkpoint keys do not match the installed adapters')
    with torch.no_grad():
        for name, p in transformer.named_parameters():
            if name in state:
                p.copy_(state[name])
