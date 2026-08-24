# FP8 Performance Optimization Backlog

Date started: 2026-08-23

## Purpose

This is a living backlog of possible performance optimizations for BRC FP8 training. Entries are candidates, not commitments. Numerical-stability work should be evaluated first; performance techniques can then be selected using exclusive-GPU profiles. Keep failed or rejected ideas with their evidence so that later decisions remain traceable.

The performance reference baseline remains `fp8_direct`: E4M3 online-Critic residual-block operands, E5M2 output gradients, FP32 accumulation/output, per-tensor historical-`amax` scaling, FP32 master parameters, FP32 Adam state, and an FP32 target Critic. The current stability experiment additionally makes the target residual kernels E4M3-resident, but shared-GPU C/D runs must not be used as performance measurements.

## Measurement discipline

- Compare FP32 and FP8 sequentially on the same otherwise-idle GPU.
- Exclude initialization and the first JIT-compiled update from steady-state latency.
- Use a fixed batch and synchronized `agent.update()` timing for kernel-focused tests.
- Separately report GEMM/kernel latency, `agent.update()` latency, environment-step throughput, and complete run wall clock.
- Disable evaluation, tensor statistics, checkpointing, videos, and W&B for short performance-only tests; measure their costs separately for end-to-end accounting.
- Record GPU model, clocks, power, utilization, memory, CUDA/JAX/XLA versions, and competing processes.
- Use width and batch sweeps. FP8 overhead can dominate small matrices even when it helps `4096 x 4096` GEMMs.
- Integrate `gpu_power_w` only for exclusive-GPU runs; device-level power is not attributable to one process on a shared GPU.

## Scaling and quantization

### Delayed-scaling implementation and fusion

Flax `Fp8DirectDotGeneralOp` already chooses the current scale from its stored `amax` history and records the current tensor's `amax` for later use. The current implementation is therefore delayed scaling algorithmically. It still has to reduce the current tensor for the history update and cast/clip the operand before each GEMM; the open performance question is whether these operations are fused efficiently and whether a repeatedly used weight must be rescanned and recast on every call.

- profile current-`amax` reduction, clipping/cast, history update, and GEMM kernels separately;
- inspect optimized HLO/profile traces to determine which operations are fused rather than inferring traffic from source code;
- measure step-to-step `current_amax / history_amax` and actual saturation;
- compare history lengths and scale update policies without changing FP8 coverage;
- compare the current Flax path with Transformer Engine or a small prequantized-RHS dot wrapper only if profiling identifies material unfused overhead.

Reference: NVIDIA Transformer Engine, "FP8 Delayed Scaling": https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html

### Block, row, column, or task-aware scaling

- Use only when per-tensor telemetry shows outliers or poor effective resolution.
- For MetaWorld-50, record per-task `amax / global_amax`; a small set of tasks may control a shared tensor scale.
- Compare accuracy gain against added reductions, scale storage, casts, and loss of kernel efficiency.
- Layer-selective block scaling is preferable to converting every layer when only one layer is problematic.

### Scale margins and history length

- A scale margin may reduce rare saturation at the cost of effective mantissa use.
- A 1,024-entry max window may be conservative when distributions change quickly or contain one old outlier.
- Candidate comparisons include max history, short-window max, percentile/history decay, and separate input/kernel policies.
- Treat these as performance/accuracy trade-offs, not automatic protections.

## Dense/GEMM path

### Extend FP8 to the target-Critic residual core

- This is the largest remaining FP32 GEMM path.
- It adds only forward FP8 GEMMs, not backward GEMMs, so expected end-to-end gain is modest.
- A target-FP32-master plus FP8-forward shadow remains a useful forward-only baseline. The accepted first stability experiment instead deliberately removes that master for the four residual kernels so the `tau=0.005` EMA write is exposed to E4M3 storage resolution.
- Target current scaling has relatively high overhead because the target has no backward path over which to amortize it.

### Cache or shadow quantized weights

- Current direct FP8 starts from FP32 master weights and quantizes operands for use.
- Investigate an FP8 shadow copy for large kernels so repeated forward/backward uses avoid redundant weight scans and casts.
- Online weights change after Adam; target weights change slowly through EMA. Cache invalidation/refresh cadence differs for the two networks.
- Measure stale-shadow numerical error separately from scale error. Do not silently reuse stale weights.

Disposition as of 2026-08-23: `implementation / stability test`. The target-Critic residual kernels are being made truly E4M3-resident and passed directly to the FP8 dot, without an FP32 master or shadow. The shared-GPU C/D experiments are not a performance benchmark; conversion savings require a later exclusive-GPU profile and optimized-HLO evidence.

Lag-coded follow-up, 2026-08-24: `fp8_lag` intentionally persists only the
E4M3 lag. Every target forward currently reconstructs
`FP32 online + dequantized lag` and requantizes that temporary kernel for the
native FP8 GEMM. A derived E4M3 target-kernel cache could move this reconstruction
and cast to the learner-update boundary, but would add another FP8 matrix state
and cache-write traffic. Treat it as a later exclusive-GPU optimization; do not
add it until closed-loop stability is established and profiling shows repeated
RHS conversion is material.

### Reuse or precompute transposed FP8 weights

- Backward input-gradient GEMMs need a transposed logical view of weights.
- Determine whether XLA/Triton materializes transposes or uses an efficient layout.
- A cached quantized transpose may help but doubles FP8 shadow storage and refresh traffic.

### Kernel backend and autotuning

- Native FP8 HLO/Triton lowering proves hardware FP8 use but not optimal kernel selection.
- Compare Triton lowering with cuBLASLt or Transformer Engine where compatible.
- Inspect tile sizes, occupancy, tensor-core utilization, ensemble layout, transpose handling, and fprop/dgrad/wgrad separately.
- Preserve a simple fallback and avoid backend complexity unless profiles show a meaningful gap.

### Ensemble batching/layout

- The Critic has two ensemble members via `vmap`.
- Check whether XLA emits efficient strided/batched GEMMs or separate kernels with unfavorable layouts.
- Compare fused batch dimensions, explicit grouped GEMM, and current `vmap` without changing model semantics.

### Width and batch alignment

- Benchmark widths 512, 1024, 2048, and 4096 and representative batch sizes.
- Identify the compute-intensity crossover at which FP8 becomes faster than FP32/TF32 after quantization overhead.
- Keep tensor dimensions aligned to efficient Blackwell tile sizes when architecture changes are allowed.

## Target-network update path

### Fuse Polyak/EMA updates

- Polyak currently walks all FP32 online and target parameters and is memory-bandwidth heavy.
- Consider fusing the target update with the online optimizer update so online parameters already resident in registers/cache are reused.
- This requires careful optimizer integration and should follow a profile proving Polyak traffic is material.

### Less-frequent resolution-matched target updates

- If target weights are eventually stored in FP8, per-step `tau=0.005` changes may round away.
- Candidate: update every `K` steps with `tau_K = 1 - (1 - tau)^K`, making each representable update larger while approximately preserving the EMA time constant.
- This avoids a target-sized Kahan compensation buffer but introduces approximation from online-weight motion within the window.
- Choose `K` using measured update-to-ULP and effective-update survival, not task reward tuning.
- Compare with naive FP8 EMA, stochastic rounding, Kahan/error feedback, FP32-master target, and periodic hard copies.

### Quantized target shadow refresh

- Keep target EMA in FP32 but refresh an FP8 forward shadow less frequently than every optimizer update.
- This reduces quantization/cast traffic but introduces explicit target-forward staleness.
- Track parameter drift and target-output divergence as a function of refresh interval.

## Optimizer and persistent-state path

### Fused optimizer kernels

- FP32 Adam reads/writes parameters, gradients, first moments, and second moments for roughly 134M large-layer parameters.
- Profile whether Adam is the post-GEMM memory-bandwidth bottleneck.
- A fused AdamW implementation can reduce intermediate traffic even before state quantization.

### BF16 optimizer-state baseline

- BF16 retains FP32-like exponent range and halves moment-state bytes, but has much less mantissa precision.
- Use it as a diagnostic bridge between FP32 state and 8-bit state, not as the final FP8 claim.

### Block-wise 8-bit optimizer states

- Per-tensor state quantization is not justified: recorded Critic Adam states contain extreme outliers.
- Block-wise scaling/nonlinear quantization can isolate outliers and reduce moment storage/traffic.
- Dequantize state in registers, perform updates in FP32, and requantize without materializing full FP32 temporary state.
- Evaluate first- and second-moment formats separately; the positive second moment has different range/precision needs.

Reference: Dettmers et al., "8-bit Optimizers via Block-wise Quantization": https://arxiv.org/abs/2110.02861

### Stochastic rounding and error feedback

- Stochastic rounding can preserve small updates in expectation when round-to-nearest freezes them.
- Error-feedback/residual accumulation can retain discarded update components but adds persistent state.
- Measure RNG/kernel cost and additional memory; use as comparison baselines rather than assuming novelty.

## Training-loop and host/device overlap

### Environment and learner pipelining

- The current loop serializes action sampling, environment stepping, replay sampling, and learner updates.
- Once learner latency falls, environment/CPU work becomes a larger Amdahl-law limit.
- Investigate double-buffered replay batches and asynchronous environment stepping only after correctness and determinism requirements are defined.

### Replay sampling and transfer

- Profile NumPy sampling, normalization, host-to-device transfer, and allocation.
- Consider pinned buffers, prefetching, device-resident batches, or buffer reuse when these become visible bottlenecks.

### Reduce synchronization and logging overhead

- JAX dispatch is asynchronous; unnecessary `block_until_ready`, host conversions, and frequent logging serialize the loop.
- Retain synchronization required for correct performance measurements.
- Aggregate low-frequency diagnostics outside the hot path and keep FP32/FP8 paired forwards at recorder intervals only.

### Compile/update-loop structure

- Verify the `fori_loop` over `updates_per_step` remains one compiled computation and does not introduce repeated host dispatch.
- Compare compilation size and steady-state performance when increasing `updates_per_step` to improve environment/update amortization.

## Checkpoint and end-to-end overhead

- Recovery checkpoints include the replay buffer and can dominate periodic stalls.
- Measure analysis checkpoint, recovery checkpoint, tensor statistics, evaluation, W&B, and video time independently.
- Consider asynchronous/reduced-frequency persistence for production throughput, while keeping the formal experiment protocol identical across precisions.
- FP8 target/optimizer storage may reduce checkpoint size only if persistent states are actually stored in low precision; FP8 compute alone does not.

## Current evidence and open measurements

- The four online-Critic residual Dense layers account for about 98.94% of MetaWorld Critic Dense multiply-adds.
- Humanoids preliminary component profiles: FP8 `agent.update()` averages about 27.55 ms versus 31.62 ms for the still-running FP32 comparison; the comparison is preliminary and must be repeated on the same exclusive GPU.
- Dogs wall clock is confounded because its FP8 run shared GPU1 with another width-4096 training job.
- Dogs/Humanoids online FP8 input saturation-risk maxima are below 0.98; output-gradient maxima are below 0.88. This supports testing delayed scaling but does not prove it will preserve accuracy.
- The highest-priority missing profile is a kernel-level breakdown of FP32/TF32 GEMM, FP8 `amax`, cast, FP8 GEMM, Adam, Polyak, and non-GEMM operations.

## References

- Micikevicius et al., "FP8 Formats for Deep Learning": https://arxiv.org/abs/2209.05433
- Björck et al., "Low-Precision Reinforcement Learning: Running Soft Actor-Critic in Half Precision": https://proceedings.mlr.press/v139/bjorck21a.html
- NVIDIA, "Run High-Throughput Reinforcement Learning Training with End-to-End FP8 Precision": https://developer.nvidia.com/blog/?p=115945
- NVIDIA Transformer Engine, FP8 scaling and speed documentation: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/
