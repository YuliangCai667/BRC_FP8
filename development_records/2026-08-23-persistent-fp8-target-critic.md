# Persistent FP8 target Critic

## Motivation

The existing `fp8_direct` mode accelerates the online Critic's four residual Dense layers but keeps all persistent parameters FP32. This change introduces the simplest uncorrected target-state experiment: store only the matching target-Critic residual kernels in E4M3 and apply the original per-step `tau=0.005` EMA. Its purpose is to measure whether a small RL target-network update is swallowed by the FP8 lattice before adding Kahan summation, stochastic rounding, error feedback, block scaling, or a modified update cadence.

The four logical kernels correspond to eight physical `width x width` matrices across the two ensemble members. Input/output projections, LayerNorm, biases, task embeddings, the online parameter master, Adam state, Actor, and temperature remain FP32.

## Changes

- Added `target_critic_precision={fp32,fp8_resident}`, defaulting to `fp32`. C is online FP32 plus resident-FP8 target; D is online `fp8_direct` plus the same resident-FP8 target.
- Added a small `ResidentFp8Dense`. It dynamically quantizes the current activation to E4M3, consumes the stored E4M3 kernel directly, requests FP32 accumulation/output from `lax.dot_general`, restores the activation/kernel scales, and adds the FP32 bias.
- Each target kernel and ensemble member has an independent FP32 per-tensor scale. Every update computes `old=codes*scale`, `candidate=old+0.005*(online-old)` transiently in FP32, then selects `new_scale=amax(candidate)/448` and immediately writes E4M3 codes. There is no persistent FP32 target master.
- The normal training path does not compute target-wide diagnostic reductions. Only when tensor statistics are due, the final real EMA inside that learner update returns per-layer/member intended-versus-applied measurements from its exact pre-write state.
- Tensor records split target codes, scales, activations, actual EMA measurements, and target forward error by ensemble. The low-frequency FP32 reference forward uses the same stored target after transient dequantization; it is not a shadow network or learning state.
- Checkpoints save resident codes and scales, and manifests now include Actor, online Critic, target Critic, and temperature dtypes. Resident targets restore only into the same mode; FP32-target-to-resident conversion is deliberately unsupported.

## Native Blackwell evidence

The validation environment used JAX/JAXlib `0.6.0`, Flax `0.10.4`, Optax `0.2.5`, CUDA `12.8`, and `/usr/local/cuda-12.8/bin/ptxas` `12.8.61` on an NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability `12.0`).

For a width-512, batch-256 target forward, optimized HLO exposed all four resident kernels as entry parameters of type `f8e4m3fn[2,512,512]`. Each residual GEMM was lowered as `f8e4m3fn x f8e4m3fn -> f32`; the RHS path contained only a transpose/bitcast of the E4M3 parameter, with no preceding FP32-to-FP8 kernel conversion. XLA emitted Triton GEMM TTIR and PTX artifacts. The compiled output was finite FP32.

## Validation

- CPU regression: all 27 unit tests passed in `67.542 s`. Coverage includes unchanged default FP32 behavior, exact four-kernel resident scope, per-ensemble initialization, FP8 RHS JAXPR, FP32 non-hot EMA, resident EMA formula, constructed swallowed updates, online/optimizer dtypes, same-mode checkpoint round-trip and deterministic next update, manifest dtypes, incompatible direct load rejection, and historical online FP32-to-FP8 behavior.
- Final GPU3 smoke run: `brc_cheetah_run_d_target_fp8_resident_s0_smoke_final`, seed 0, width 512, batch 256, two updates/step, 200 env steps and 203 learner updates. It finished normally with five post-update samples, `update_nan_count=0`, `update_inf_count=0`, finite critic loss `8.344`–`12.118`, critic gradient norm `1.125`–`21.042`, actor loss `-1.327`–`4.565`, and actor gradient norm `2.508`–`12.004`.
- At step 150, all 394 tensor-stat rows were finite. The eight actual resident-matrix EMA measurements had code-unchanged fraction `0.4316`–`0.5135`, swallowed-update fraction `8.01e-5`–`1.18e-4`, applied/intended L2 ratio `0.999999`–`1.000005`, relative update error `1.25e-4`–`2.23e-4`, and sign agreement at least `0.999958`.
- Against a transient dequantized-weight FP32 forward, the BRC bootstrap aggregate (ensemble-mean probability) expected-Q MAE was `0.07494`, signed bias `-0.01306`, and mean 101-bin probability JS divergence `0.000651`; ensemble-0/1 MAE was `0.08925`/`0.12916`.

## Interpretation and limits

- The smoke result confirms the mechanism and record path, not long-horizon stability. At this width-512 point, many raw FP8 codes remain unchanged while dynamic scale movement keeps the physically applied update norm very close to the intended norm; code-unchanged fraction must not be mislabeled as swallowed-update fraction.
- The EMA diagnostic is one actual final learner update at each tensor-stat interval, not a per-step time series. Formal C/D runs are needed to determine whether the mechanism changes with width 4,096, training phase, task distribution, or return.
- The target forward is mixed precision: only the residual kernels and their matrix operands are FP8; FP32 boundaries and accumulation remain. This is not an end-to-end FP8 claim.
- GPU3 also hosted an unrelated approximately 8 GiB process during the reviewed smoke. C/D intentionally share GPU3 as well, so wall clock, throughput, utilization, and power from these runs are invalid performance comparisons.
- No correction technique or performance optimization was added. A resolution-matched target cadence is considered only if formal C/D show material update loss or learning instability.
