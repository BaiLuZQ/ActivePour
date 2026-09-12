# Final architecture

All shapes below omit the batch dimension. All source comments and documentation in this release are in English.

## Observations and action

The five observation slots are `I_pre`, maximum-probe-angle arrival, end of probe hold, upright return arrival, and `I_post`. Their times relative to `I_pre` are `[0, 0.4, 1.0, 1.4, 4.4]` seconds. The formal pour begins at `I_post` for **both** generation arms.

The action is `[maximum angle in degrees, outbound rotation time, hold time]`. Return time equals outbound time. The four future panels correspond to angle arrival, hold end, return arrival, and return plus 3 seconds.

## Probe conditioning

Each grayscale 256×256 frame is encoded by a shared CNN with channels `1→32→64→96→128`. Each stage uses a 3×3 stride-2 convolution, GroupNorm with 8 groups, and SiLU. Each output is `128×16×16`.

Let `F_0 = F_pre`, `F_4 = F_post`, and `k=1..4` index future stages. At each spatial position:

```text
history_i = LN(W_history concat(F_i, F_i - F_pre) + observation_phase_i + time_embedding_i)
base      = F_post for full5, F_pre for pre1
query_k   = W_Q LN(base + stage_k + global_action_query + W_stage z_k)
A_k       = four-head attention(query_k, history K/V), with a matched-phase score bias
H_tilde_k = base + A_k + alpha W_delta(F_k - F_pre) + stage_k
H_k       = H_tilde_k + spatial_refinement(H_tilde_k)
H_prime_k = H_k + gamma(z_k) * LN_channel(H_k) + beta(z_k)
```

The spatial refinement is a two-convolution residual branch. `W_delta` has no bias, so a zero difference remains zero. `alpha` starts at 0.1. Attention sees **both absolute states and differences**, not differences alone.

For `pre1`, the other slots are sanitized before the CNN, masked in attention, and removed from the difference bypass. A NaN-valued hidden frame cannot change its output. Both arms have the same named parameterization, but masking changes effective information and gradient usage; this is not a claim of identical effective capacity.

## Stage-aligned action conditioning

The global action is scaled to `[-1,1]`, then encoded as `3→128→SiLU→4096` and appended to the fixed text-token sequence. The text sequence is not a single token. A separate small action path informs the history queries.

Stage descriptors are:

```text
[theta, T_rot, 0,      0,     0]
[theta, T_rot, T_hold, 0,     0]
[theta, T_rot, T_hold, T_rot, 0]
[theta, T_rot, T_hold, T_rot, 3]
```

Divide by fixed scales `[100,3,3,3,3]`, apply a shared `5→128→SiLU→128` MLP, then add a stage embedding to obtain `z_k`. These are known control descriptors, not measured future labels. Stage-query and FiLM output projections start at zero.

Condition dropout is 0.05: one inverted channel mask per sample, shared across all four stages and spatial positions. It is applied after FiLM, before tiling. It is disabled for evaluation and frozen feature extraction.

## MMDiT injection and LoRA

Tile `4×128×16×16` stage maps in 2×2 order, yielding `128×32×32`. Flatten in image-token order to `1024×128`. Six independent linear projections create `1024×1536` residuals, added after MMDiT blocks `[0,4,8,12,16,20]` through Diffusers' block-control residual interface.

The SD image latent is `16×64×64`; a 2×2 latent patch embedding yields 1,024 tokens with width 1,536. The original model has 24 MMDiT blocks.

Every adapted image-side attention Q/K/V/O matrix receives its own rank-16 LoRA. This includes the extra image attention modules where present, not text-specific added Q/K/V projections. Base matrices remain frozen. The custom LoRA checkpoint format is **not PEFT format**. Native pooled-text/time modulation, text encoders, and VAE remain frozen; the new stage FiLM is an additional condition path, not a rewrite of native AdaLN.

## Final-latent readout

```text
post-FiLM/pre-dropout condition maps: 4×128×16×16 → global average → 512
global action Linear/SiLU intermediate:                              128
final sampled latent: 16×64×64 → four 16×32×32 quadrants
                     → average pool each to 16×4×4 → flatten        1024
total:                                                              1664
```

No learned pooling or random projection is used. The final latent is generated from noise, not encoded from the true future image. Frozen features are standardized using training-only mean/std (std floor 0.01). The zero ablation clears the future slot **after** standardization.

All heads use `1664→128→SiLU→Dropout(.15)→64→SiLU→Dropout(.15)→1→Sigmoid`: 221,441 parameters each. The direct baseline treats the observed five frames as five CNN input channels, pools `96×16×16` to `96×4×4`, and appends a 128-dimensional action feature. It has no future-stage queries or FiLM and trains 397,441 parameters in total.

## What the architecture does not establish

Frozen conditions plus latent features support a scalar readout, but this does not establish that the readout reasons through a physically correct future. The zero-slot ablation is important for that distinction. In the reported tests, the latent's independent contribution is uncertain, and direct regression is more accurate.
