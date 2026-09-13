# Experimental protocol and reproduction boundaries

## Published artifacts versus retraining

This repository is a curated, portable source release. The reported experiments were run before the public packaging. The final conditioner, LoRA implementation, flow objective, feature pooling and search rules are preserved. The public orchestration is simplified and does not claim bitwise recovery of the archived sample order, optimizer state, validation-panel selection or interrupted jobs.

The complete dataset and trained checkpoints are not distributed here. `configs/manifest.example.json` is a schema example, **not an actual sample**. There is no automatic private-server access. Published scalar results can be inspected and planning metrics recomputed without those artifacts.

## Data and leakage controls

The accepted release contains 1,392 training, 456 validation and 448 test examples. All actions branching from one probe state belong to one split. The formal expansion adds 48 training material pairs covering friction 0.25–0.95 and rolling friction 0–0.18, plus 16 held-out pairs; compatible earlier material configurations are also reused. New starts are assigned eight actions from a broader action pool rather than a full Cartesian product.

Each manifest row identifies a sample, a probe episode, its split, five probe image paths, the continuous action, the four-panel future image and the true mass discharge fraction. Image paths are relative to a supplied dataset root and cannot escape it.

The public `prepare` CLI encodes true future images only for training/evaluation targets. `sample` and `plan` never use those targets or discharge labels. Feature caches are bound to the generator checkpoint. Train and validation episodes are checked for overlap; material-level split design remains the dataset creator's responsibility.

## Simulation

The DEM adapter represents approximately 612 polydisperse dry spheres in a quasi-2D setting. Radii are independently sampled between 4.25 and 5 mm. Centers start on a lattice with spacing `2.04 * maximum_radius`, with small position jitter controlled by the same seed. The resulting settled state is produced by dynamics, not manually assigned.

GeoTaichi uses linear rolling contacts, normal stiffness 20,000 N/m, tangential stiffness half of that value, density 2,500 kg/m³, and fixed normal viscous damping parameter 0.3. The varied material parameters are contact friction and rolling resistance, **not cohesion or liquid viscosity**.

The reference protocol uses CPU single-thread contact accumulation with canonical candidate order and a fixed `6.25e-6 s` time step. Independent scenes can run on different CPU cores. The GPU is used for SD3.5; the CPU choice is about this experiment's replay protocol, not a general CPU-versus-GPU speed claim.

Container coordinates remain fixed. Gravity is rotated as `[g sin(theta), -g cos(theta), 0]`. Exiting particles are irreversibly captured and removed from further contact. Discharge is captured mass divided by original mass. This omits complete moving-frame inertial effects, re-entry and a receiving vessel. No claim of full time-step convergence or real-world physical certification is made.

The adapter uses GeoTaichi's internal scene/contact APIs. A new upstream checkout may require compatibility review. Snapshot metadata records source hashes and complete state fields; do not disable mismatch checks to load snapshots from another build. The public source changes only interface/defaults and documentation around the retained adapter, not the frozen experimental run.

### Timing

Probe: tilt to 35 degrees in 0.4 s, hold 0.6 s, return in 0.4 s, wait 3 s. Five observations relative to `I_pre` are taken at 0, 0.4, 1.0, 1.4 and 4.4 s.

The raw adapter filenames are slightly historical:

| Model slot | Raw file |
|---|---|
| Pre-probe | `I_pre.npz` |
| Probe angle arrival | `peak_arrival.npz` |
| Probe hold end | `I_probe_peak.npz` |
| Probe upright arrival | `return_arrival.npz` |
| Probe return + 3 s | `I_post.npz` |

In particular, `I_probe_peak.npz` means **hold end**, not first angle arrival. The frame-export script uses the correct order.

Formal action: angle 45–100 degrees, rotation time 0.6–2 s, hold time 0.1–1.5 s, equal return duration and fixed 3 s post-return rest. Rotation uses `s(u)=10u^3-15u^4+6u^5`. Future panels are angle arrival, hold end, upright arrival and upright plus 3 s. Fixed waiting time does not guarantee every material is motionless.

Some starts require longer initial settling before `I_pre`; this does not change relative probe timing. Every formal action must restore the matching complete `I_post`, including contact history, not just positions and velocities.

### Numerical acceptance

Completed output is checked for finite state, mass error below `1e-10`, planarity error below `1e-10`, particle overlap/diameter and wall overlap/diameter no greater than 0.02, and no probe spill. These are screening guards, not a convergence certificate. The public adapter writes histories; dataset builders must apply and preserve these checks before admitting data. Existing failure flags must not be rewritten to make a dataset or graph look better.

## Generation training

The backbone is full SD3.5 Medium. Each arm trains for 10k optimizer steps, batch 8, seed 916. LoRA LR is `3e-5`; conditioner LR is `2e-4`; AdamW weight decay is 0.01. Learning rate uses 200-step warmup and cosine decay to 0.1 of its initial value. Condition dropout is 0.05.

For target latent `z_true` and Gaussian noise `epsilon`, training uses:

```text
z_sigma = (1 - sigma) z_true + sigma epsilon
velocity_target = epsilon - z_true
```

The batch shares one noise/time draw. With probability 0.5, the time index is the pure-noise endpoint; otherwise it is drawn from the pretrained scheduler grid. A train-derived, fixed spatial weight map emphasizes the foreground region (boost 4, mean-normalized). This is not a per-example future mask provided at inference.

The archived run evaluated a fixed 96-example validation panel every 500 steps at noise levels 0.2, 0.5 and 0.8. This selects full5 step 10000 and pre1 step 9500. Public training uses the first up-to-96 validation entries as its fixed panel, so arrange your manifest deliberately; it is not automatically the historical panel.

All image evaluations sample from noise with 28 FlowMatch Euler steps and seed 20260919, without additional CFG/SLG. Macro IoU uses dark-foreground coverage blurred with a 2-pixel Gaussian and thresholded at 0.2. Particle IoU uses raw coverage threshold 0.5. Four panel scores are averaged. White background does not count as correctly predicted material.

The test macro-IoU increase is 0.006655. A probe-episode grouped bootstrap gave a 95% interval `[0.001691, 0.011769]`. This is conditional on one trained generator pair and does not measure variation across generator-training seeds.

## Readout training

Three arms, three seeds `[101,202,303]`, 10k steps each, batch 32. The MLP is identical in all arms: `1664→128→64→1`, SiLU activations, two dropout layers at 0.15, final sigmoid. AdamW uses LR `3e-4`, weight decay 0.001, eps `1e-8`, 100-step warmup and a 0.1 cosine floor. Gradient norm clipping is 1.

Features are 512 post-FiLM/pre-dropout condition averages, 128 pre-4096-projection action features, and 1024 final sampled latent features. Average pooling introduces no parameters. Train-only feature std is floored at 0.01. Zero-slot ablation clears the standardized future features. The ordinary direct CNN is initialized from scratch, not inherited from the generator.

Best validation-MAE steps for seeds 101/202/303 are:

| Arm | Best steps |
|---|---|
| Latent | 4400 / 5400 / 6300 |
| Zero slot | 5300 / 3300 / 3400 |
| Direct CNN | 4400 / 3500 / 1100 |

Test latent-minus-direct MAE is 0.005393, grouped bootstrap interval `[0.002367, 0.008740]`. Test latent-minus-zero MAE is -0.000570, interval `[-0.002740, 0.001515]`. The latter does not establish an independent latent advantage. No equivalence margin was prespecified. MLP training is small, but generating the frozen representation has pretraining and inference costs.

Training curves contain MSE, while model selection uses validation MAE. Their absolute values must not be compared as if they were the same metric. Evaluation-mode training MAE is the appropriate comparison for diagnosing a generalization gap. Three-seed plot ranges/standard deviations are not confidence intervals.

## Decision protocol

The three latent-head seeds are averaged for scoring. This differs from the phase-one table, which averages independently evaluated seed metrics. The direct and zero heads do not participate in planning.

For each material, the first 32 actions are 8 box corners plus 24 Latin-hypercube samples. For each target, five good centers are greedily selected with normalized separation 0.20, relaxed to 0.15/0.10/0 only if required. Three local points use Gaussian scale 0.05 and three use 0.12 per center, with boundary reflection and duplicate rejection. Four extra global points are independent per target. This gives 32 + 5 x 6 + 4 = 66 unique candidates per task. Rank by absolute predicted target error; ties prefer shorter actions, then stable ID.

The 66-candidate follow-up preserves all 16 materials and the frozen models. Its 2,144 candidate/state pairs are not 2,144 newly generated samples: compatible old predictions were reused. The workstation cache also retains old candidates outside the new search sets; cache-file count is not the candidate budget. Published `planning.json` and `planning_summary.json` describe this follow-up; `planning_52_candidates.json` preserves the earlier experiment. Numerical failure flags are retained in both.

The original 48-task experiment used one common noise seed, serial sampling after a batch-equivalence check, and one selected-action simulation per task. Predictions/candidates were committed before simulation. No target-specific retraining, true-result reselection or threshold relaxation occurred.

All 48 task records are published. `t000_r006_target_020` and `t000_r006_target_080` failed wall-overlap guards but reached discharge fractions 0.230064 and 0.825965 respectively. Both remain visible. The all-case error view is descriptive and must not be described as 48 numerically accepted cases.

## Frozen identities

- Five-frame generator: `5f365da168c248eedf5afd9614ef6fcdc8147337913ec13fa3ff469f1dbd8d88`
- Accepted data release: `d7be851e37361ce11f78051be120bfa6981df74743b87023e60f759482b2b232`
- Current 66-candidate decision-plan fingerprint: `dc1848dff1dd5cd5446ba7d6834703072eb2298953bd898b3b916093abdc7bb3`
- Earlier 52-candidate decision-plan fingerprint: `4e9871454e0797c7fed61b95e13e04f5b230f57049645234a3ba23ad12244589`

These identify the archived results, not a public checkpoint download. `results/source_hashes.json` records the upstream first-party inputs used in the export; files with portable interface edits need not match those original hashes.
