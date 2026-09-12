"""Stage-aligned action conditioning for the paired ActivePour world models.

The five-frame model starts its spatial base and attention query at F_post;
the pre-only model starts at F_pre. Both still measure probe changes from F_pre.
The parameterization is identical; available observations are not identical.
No target future image, material identity or discharge label enters this module.
"""

import torch
from torch import nn


def stage_grid(stages):
    """Tile TL/TR/BL/BR in 2-D before flattening into SD's image-token order."""
    return torch.cat(
        [torch.cat([stages[:, 0], stages[:, 1]], 3),
         torch.cat([stages[:, 2], stages[:, 3]], 3)],
        2,
    )


class PhysicsConditioner(nn.Module):
    VALID = {'pre1': (0,), 'full5': (0, 1, 2, 3, 4)}

    def __init__(self, *, hidden_dim=1536, context_dim=4096, feature_width=128,
                 num_layers=24, injection_layers=(0, 4, 8, 12, 16, 20),
                 action_low=(60., 1., .3), action_high=(80., 1.3, 1.),
                 probe_mode='raw', probe_transform='identity', response_gain=1.,
                 full_precision_conditions=True, architecture='phase_action_v2',
                 frame_mode='full5', condition_dropout=.05):
        super().__init__()
        if architecture != 'phase_action_v2' or probe_mode != 'raw':
            raise ValueError('phase_action_v2 raw encoder required')
        if frame_mode not in self.VALID:
            raise ValueError(frame_mode)
        if (feature_width % 8 or num_layers < 1 or
                any(i < 0 or i >= num_layers - 1 for i in injection_layers) or
                len(set(injection_layers)) != len(injection_layers)):
            raise ValueError('invalid dimensions/layers')
        if not 0. <= condition_dropout < 1.:
            raise ValueError('condition_dropout must be in [0, 1)')
        self.hidden_dim = hidden_dim
        self.frame_mode = frame_mode
        self.feature_width = feature_width
        self.num_layers = num_layers
        self.injection_layers = tuple(injection_layers)
        self.heads = 4
        self.probe_mode = probe_mode
        self.probe_transform = probe_transform
        self.response_gain = response_gain
        self.full_precision_conditions = full_precision_conditions
        self.architecture = architecture
        self.condition_dropout = float(condition_dropout)

        layers = []
        before = 1
        for after in (32, 64, 96, feature_width):
            layers += [nn.Conv2d(before, after, 3, stride=2, padding=1),
                       nn.GroupNorm(8, after), nn.SiLU()]
            before = after
        self.probe_encoder = nn.Sequential(*layers)
        self.stage_embeddings = nn.Parameter(torch.randn(4, feature_width) * .02)
        self.observation_phase = nn.Parameter(torch.randn(5, feature_width) * .02)
        self.observation_time = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, feature_width))
        self.history_input = nn.Linear(2 * feature_width, feature_width)
        self.history_norm = nn.LayerNorm(feature_width)
        self.query_norm = nn.LayerNorm(feature_width)
        # Keep both original global action paths: query embedding and text token.
        self.query_action = nn.Sequential(
            nn.Linear(3, feature_width), nn.SiLU(),
            nn.Linear(feature_width, feature_width))
        self.action_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, context_dim))

        # Five physical descriptors distinguish stages with identical endpoint
        # angle/velocity. One shared MLP is used, NOT four separate encoders.
        self.stage_action_encoder = nn.Sequential(
            nn.Linear(5, feature_width), nn.SiLU(),
            nn.Linear(feature_width, feature_width))
        self.action_stage_embeddings = nn.Parameter(torch.randn(4, feature_width) * .02)
        self.stage_query = nn.Linear(feature_width, feature_width)
        self.stage_film = nn.Linear(feature_width, 2 * feature_width)
        # No learned normalization parameters are needed for the FiLM residual.
        self.film_norm = nn.LayerNorm(feature_width, elementwise_affine=False)
        for projection in (self.stage_query, self.stage_film):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

        self.history_q = nn.Linear(feature_width, feature_width)
        self.history_k = nn.Linear(feature_width, feature_width)
        self.history_v = nn.Linear(feature_width, feature_width)
        self.history_out = nn.Linear(feature_width, feature_width)
        self.phase_match_bias = nn.Parameter(torch.full((4,), .5))
        # Bias-free W_delta ensures zero changes remain exactly zero changes.
        self.delta_projection = nn.Linear(feature_width, feature_width, bias=False)
        self.delta_alpha = nn.Parameter(torch.tensor(.1))
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(feature_width, feature_width, 3, padding=1),
            nn.GroupNorm(8, feature_width), nn.SiLU(),
            nn.Conv2d(feature_width, feature_width, 3, padding=1))
        self.projections = nn.ModuleDict({
            str(i): nn.Linear(feature_width, hidden_dim) for i in injection_layers})
        for projection in self.projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

        low = torch.tensor(action_low, dtype=torch.float32)
        high = torch.tensor(action_high, dtype=torch.float32)
        if low.shape != (3,) or high.shape != (3,) or not bool((high > low).all()):
            raise ValueError('action bounds must be three strictly ordered pairs')
        self.register_buffer('action_low', low)
        self.register_buffer('action_high', high)
        self.register_buffer('observation_seconds', torch.tensor([0., .4, 1., 1.4, 4.4]))
        # These fixed physical scales are independent of data/split statistics.
        self.register_buffer('stage_descriptor_scale', torch.tensor([100., 3., 3., 3., 3.]))
        self.register_buffer('formal_rest_seconds', torch.tensor(3.))
        valid = torch.zeros(5, dtype=torch.bool)
        valid[list(self.VALID[frame_mode])] = True
        self.register_buffer('valid_frames', valid, persistent=False)
        self.register_buffer('matched_slots', torch.eye(5)[1:], persistent=False)

    def normalize_action(self, action):
        return 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1

    def stage_descriptors(self, action, *, normalized=True):
        """B,4,5: angle and elapsed out/hold/return/rest durations at each frame.

        The formal action is [degrees, rotation seconds, hold seconds]. Return
        rotation has the same duration; fixed 3 s rest is part of the protocol.
        This is known action history, never a measurement from a future label.
        """
        if action.ndim != 2 or action.shape[-1] != 3:
            raise ValueError('expected B x 3 physical action values')
        angle, rotation, hold = action.unbind(-1)
        zero = torch.zeros_like(angle)
        rest = torch.ones_like(angle) * self.formal_rest_seconds
        descriptors = torch.stack([
            torch.stack([angle, rotation, zero, zero, zero], -1),
            torch.stack([angle, rotation, hold, zero, zero], -1),
            torch.stack([angle, rotation, hold, rotation, zero], -1),
            torch.stack([angle, rotation, hold, rotation, rest], -1),
        ], 1)
        return descriptors / self.stage_descriptor_scale if normalized else descriptors

    def stage_action_features(self, action):
        return (self.stage_action_encoder(self.stage_descriptors(action)) +
                self.action_stage_embeddings[None])

    def shared_stage_dropout(self, stages):
        """One inverted channel mask per sample, shared over all stages/pixels.

        Never independently corrupt the frames before computing differences.
        Evaluation, sampling and downstream feature caching must call eval().
        """
        if not self.training or self.condition_dropout == 0.:
            return stages
        keep = 1. - self.condition_dropout
        mask = stages.new_empty((len(stages), 1, self.feature_width, 1, 1))
        mask.bernoulli_(keep).div_(keep)
        return stages * mask

    def temporal_features(self, probe, action, *, disable_delta=False):
        """Return post-FiLM, PRE-dropout B,4,C,16,16 features plus diagnostics."""
        if probe.ndim != 4 or probe.shape[1:] != (5, 256, 256):
            raise ValueError('expected five-slot bank')
        if len(probe) != len(action):
            raise ValueError('probe/action batch mismatch')
        # Sanitize hidden images BEFORE CNN/GroupNorm, even if they contain NaNs.
        # Thus pre1 cannot read F_post through any nominally masked side path.
        x = torch.where(self.valid_frames[None, :, None, None], probe, torch.zeros_like(probe))
        b = len(x)
        c = self.feature_width
        d = c // self.heads
        f = self.probe_encoder(x.reshape(b * 5, 1, 256, 256)).reshape(b, 5, c, 16, 16)
        f = f.permute(0, 3, 4, 1, 2).reshape(b, 256, 5, c)
        pre = f[:, :, 0]
        base = f[:, :, 4] if self.frame_mode == 'full5' else pre
        # K/V retain BOTH absolute state and response relative to the PRE frame.
        tokens = self.history_input(torch.cat([f, f - pre[:, :, None]], -1))
        tokens = (tokens + self.observation_phase[None, None] +
                  self.observation_time((self.observation_seconds / 4.4)[:, None])[None, None])
        tokens = self.history_norm(tokens)
        z = self.stage_action_features(action)
        queries = (base[:, :, None] + self.stage_embeddings[None, None] +
                   self.query_action(self.normalize_action(action))[:, None, None] +
                   self.stage_query(z)[:, None])
        q = self.history_q(self.query_norm(queries)).reshape(b, 256, 4, 4, d).transpose(2, 3)
        k = self.history_k(tokens).reshape(b, 256, 5, 4, d).transpose(2, 3)
        v = self.history_v(tokens).reshape(b, 256, 5, 4, d).transpose(2, 3)
        # Explicit products retain compatibility with the frozen Torch runtime.
        score = (q[:, :, :, :, None, :] * k[:, :, :, None, :, :]).sum(-1) / d ** .5
        score = (score + self.phase_match_bias[None, None, :, None, None] *
                 self.matched_slots[None, None, None])
        weights = score.masked_fill(
            ~self.valid_frames[None, None, None, None, :], float('-inf')).softmax(-1)
        fused = (weights[..., None] * v[:, :, :, None, :, :]).sum(-2)
        fused = fused.transpose(2, 3).reshape(b, 256, 4, c)
        attention = self.history_out(fused)
        # The direct bypass stays relative to PRE, independently of base choice.
        difference = (f[:, :, 1:] - pre[:, :, None]) * self.valid_frames[None, None, 1:, None]
        direct = self.delta_alpha * self.delta_projection(difference)
        if disable_delta:
            direct = direct * 0
        h = base[:, :, None] + attention + direct + self.stage_embeddings[None, None]
        stages = h.permute(0, 2, 3, 1).reshape(b, 4, c, 16, 16)
        refined = self.spatial_refine(stages.reshape(b * 4, c, 16, 16)).reshape_as(stages)
        h = stages + refined
        # Channel-only LN at each pixel; stage action is broadcast over space.
        norm = self.film_norm(h.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        gamma, beta = self.stage_film(z).chunk(2, -1)
        h_prime = h + gamma[..., None, None] * norm + beta[..., None, None]
        return h_prime, weights, direct

    def forward(self, probe, action, text_tokens, *, disable_probe=False):
        with torch.autocast(device_type=probe.device.type, enabled=False):
            stages, _, _ = self.temporal_features(probe.float(), action.float())
            stages = self.shared_stage_dropout(stages)
            tokens = stage_grid(stages).flatten(2).transpose(1, 2)
            residuals = [
                self.projections[str(i)](tokens)
                if i in self.injection_layers and not disable_probe else tokens.new_zeros(())
                for i in range(self.num_layers)]
            action_token = self.action_encoder(self.normalize_action(action.float()))[:, None]
            context = torch.cat([
                text_tokens.expand(len(probe), -1, -1), action_token.to(text_tokens.dtype)], 1)
        return residuals, context

    def spec(self):
        # Also valid for the direct-regression encoder with zero SD projections.
        return dict(
            hidden_dim=self.hidden_dim, context_dim=self.action_encoder[-1].out_features,
            feature_width=self.feature_width, num_layers=self.num_layers,
            injection_layers=list(self.injection_layers), action_low=self.action_low.tolist(),
            action_high=self.action_high.tolist(), probe_mode=self.probe_mode,
            probe_transform=self.probe_transform, response_gain=self.response_gain,
            full_precision_conditions=self.full_precision_conditions,
            architecture=self.architecture, frame_mode=self.frame_mode,
            condition_dropout=self.condition_dropout)


def convert_first_conv_to_response_basis(*args, **kwargs):
    raise ValueError('No migration from previous architecture; fresh paired initialization required')
