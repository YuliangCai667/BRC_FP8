# BRC FP8 Research Idea Backlog

Started: 2026-08-23

## Purpose

This is a living record of research ideas for extending BRC toward end-to-end low-precision training. It is separate from `fp8-performance-optimization-backlog.md`: this document records hypotheses, mechanisms, experimental tests, and possible paper narratives, while the performance backlog records engineering techniques that may improve throughput.

An idea being listed here does not mean that it will be implemented or claimed as novel. Each idea should remain marked as a hypothesis until experiments support its mechanism, and its novelty must be checked against prior work before paper submission.

Suggested status labels:

- `hypothesis`: mechanism is plausible but not yet demonstrated;
- `diagnosed`: measurements confirm the proposed failure mechanism;
- `testing`: implementation or experiments are in progress;
- `supported`: experiments support both the mechanism and the intervention;
- `rejected`: evidence does not support the idea or its trade-off;
- `baseline-only`: useful comparison, but not a candidate contribution.

## Idea 001: Resolution-Matched Target Updates

Status: `testing`

Updated: 2026-08-24. The accepted first test stores only the four logical residual-Dense kernels of the target Critic in E4M3, with one dynamic per-tensor scale per ensemble member. It applies the original per-step `tau=0.005` EMA and deliberately excludes Kahan summation, error feedback, stochastic rounding, block scaling, and delayed target updates. C keeps the online Critic FP32; D uses the existing online `fp8_direct` path. Both use the same resident-FP8 target core. The final width-512 D smoke run completed 203 updates without NaN/Inf; at step 150, the final actual EMA write swallowed only `8.01e-5`–`1.18e-4` of nonzero element updates and retained an applied/intended L2 ratio of `0.999999`–`1.000005`. This is a mechanism check, not enough evidence to accept or reject the idea at width 4,096 and 500k steps.

Formal width-4,096 Dogs seed-42 C/D runs started from clean commit `f88b662` on 2026-08-24. C remains on GPU3 as W&B `trzg1mjw`. The original D segment (`fb02bf23`) completed its first update without NaN/Inf, then was deliberately stopped at env step 10,057 / update 10,115 to move GPUs; it had no checkpoint and is not treated as a complete formal result. D-R1 restarted from step 0 on GPU1 as W&B `iwlomjbu` from clean docs-only successor `1ff6625` (identical training code) and also passed its first update without NaN/Inf. GPU1 has another training process, so only stability/learning evidence is admissible; no performance comparison is drawn from these runs.

Working names:

- resolution-matched target update;
- quantization-aware target cadence;
- temporal-resolution-matched EMA.

### Motivation

BRC updates the target Critic with a soft/Polyak update:

```text
target <- target + tau * (online - target),  tau = 0.005
```

If target parameters remain FP32 and only target Dense operands are converted to FP8 for matrix multiplication, this update is still FP32 and does not suffer FP8 update loss. The problem considered by this idea begins only when target parameters are persistently stored in FP8, or when the updated target is requantized and written back to FP8 after every step.

E4M3 has only three explicit mantissa bits. Once online and target parameters are close, the intended increment

```text
delta = tau * (online - target)
```

may be smaller than the distance from the current target value to the next representable FP8 value. Requantization can then return the unchanged target. Repeated soft updates may become a staircase: many updates are lost, followed by an occasional large jump after online-target drift grows enough.

Per-tensor or block scaling improves range allocation but does not remove this relative-precision limit for normal values.

### Core proposal

Update the low-precision target less frequently, every `K` online-Critic updates, and use an adjusted coefficient:

```text
tau_K = 1 - (1 - tau)^K
target <- target + tau_K * (online - target)
```

For `tau = 0.005`:

| K | tau_K |
|---:|---:|
| 1 | 0.0050 |
| 8 | 0.0393 |
| 16 | 0.0771 |
| 32 | 0.1482 |
| 64 | 0.2744 |

The larger update is more likely to cross an FP8 representable-value boundary. Unlike Kahan summation or error feedback, this proposal does not require a compensation tensor as large as the target network.

If the online parameters are constant during the `K`-step interval, the adjusted update is exactly equivalent to applying the original EMA `K` times. When online parameters move during the interval, it is an approximation; online-parameter drift must therefore be measured rather than assumed negligible.

### Minimal experimental sequence

1. **Target-forward FP8 only.** Keep target storage and Polyak arithmetic FP32. This remains a possible separate ablation and is not called C/D in the current experiment table. It cannot diagnose small EMA update loss.
2. **Offline naive-state simulation.** At matched online/target FP32 checkpoints, simulate per-step FP8 target storage and requantization without Kahan or other corrections.
3. **Naive persistent-FP8 target run.** Store the target in FP8 and apply the original per-step `tau=0.005` update. This is the current C/D experiment: C compares directly with FP32/FP32 A, while D compares with online-FP8/target-FP32 B. Verify whether the predicted update-loss mechanism occurs and whether it harms training.
4. **Resolution-matched cadence.** Only if stage 3 exposes a material problem, test a small set of `K` values selected from representability measurements rather than return tuning.

### Required diagnostics

Record per layer, ensemble member, and recorder step:

- intended FP32 update `delta = tau * (online - target)`;
- fraction of nonzero intended updates that leave the stored FP8 target unchanged;
- actual-update norm divided by intended-update norm;
- actual/intended update sign agreement;
- distance to the next representable FP8 value, using the actual scale;
- online-target parameter drift over each candidate `K`-step interval;
- FP32-versus-FP8 target expected-Q MAE and signed bias;
- FP32-versus-FP8 101-bin probability and Bellman-target distribution divergence;
- task-wise errors and task `amax / global_amax` for MetaWorld-50.

The implementation samples the final real EMA inside the learner update only when tensor statistics are due. It computes the intended update and the immediately requantized/applied update from the exact pre-write online and target states; it does not infer a past update from a later probe state. Raw codes, scales, activations, and forward errors are split by ensemble before the generic tensor summarizer.

### Baselines

- FP32 master target with FP8 target forward;
- naive per-step persistent-FP8 EMA;
- Kahan momentum;
- stochastic rounding;
- FP32/BF16 error-feedback residual;
- periodic hard target copy.

Kahan is prior art and should be treated as a strong baseline rather than the proposed contribution. If its compensation is itself stored in FP8, the compensation can be rounded away; if it is stored in FP32, it adds target-sized high-precision state.

### Candidate variants

- one global `K` for the whole target network, preferred as the simplest first version;
- layer-wise `K` chosen from measured update-survival rates;
- event-triggered updates when the intended change is large enough to cross the FP8 lattice;
- BF16 target storage as an intermediate diagnostic between FP32 and FP8;
- FP8 target storage with FP32 arithmetic performed transiently before requantization.

Variants should not be pre-implemented. Add one only when the naive measurements identify the failure it addresses.

### Paper narrative

Existing FP8 scaling mainly allocates representable values across a tensor at one moment; this can be described as **spatial scaling**. Persistent RL states introduce a different problem: their changes across training steps can be smaller than the low-precision lattice spacing. This is a **temporal-resolution mismatch**.

The possible story is:

1. native FP8 handles the dominant BRC Critic GEMMs without measurable return loss;
2. low-precision target forward introduces bootstrap feedback and requires distribution-level validation;
3. persistent FP8 target storage exposes update loss caused by the mismatch between EMA timescale and FP8 resolution;
4. resolution-matched target cadence restores effective updates without a target-sized compensation buffer;
5. the same temporal-resolution viewpoint may later inform low-precision Adam states.

This narrative is conditional on stages 2 and 3 actually demonstrating the mechanism. If naive persistent-FP8 target storage is stable and its updates survive sufficiently, the idea should be rejected rather than protected with unnecessary machinery.

### Open questions

- Does dynamic scale movement materially change the observed update-survival rate?
- Is one global `K` sufficient, or do output/input layers behave differently from the large residual kernels?
- How much does the online Critic move within an interval, and when does the constant-online approximation fail?
- Does staircase target motion affect expected Q, the complete 101-bin distribution, or only parameters with no behavioral consequence?
- Is adjusted low-frequency EMA already established under another name in quantized training or RL target-network literature?
- Does reduced target-update frequency save measurable memory bandwidth or kernel-launch overhead in addition to improving representability?

### References to investigate

- Björck et al., "Low-Precision Reinforcement Learning: Running Soft Actor-Critic in Half Precision," ICML 2021: https://proceedings.mlr.press/v139/bjorck21a.html
- Micikevicius et al., "FP8 Formats for Deep Learning": https://arxiv.org/abs/2209.05433

## Idea 002: Resident Low-Precision Critic Weights

Status: `testing`

Updated: 2026-08-23. The first resident-state implementation is limited to the target Critic residual core. It consumes stored E4M3 kernels directly in target forward GEMMs while input/output projections, LayerNorm, biases, task embeddings, and all online learning state retain their selected existing precision. Performance is explicitly deferred; the first C/D runs test numerical stability and EMA update survival.

### Motivation

The current `fp8_direct` mode stores online-Critic parameters and Adam states in FP32. On every covered Dense invocation, Flax obtains/scans scaling metadata, clips and casts both the activation and FP32 kernel to E4M3, performs a native FP8 GEMM, and dequantizes the accumulated result to FP32. The target Critic is currently entirely FP32.

Persistently storing a quantized kernel and its scale on device could avoid repeatedly scanning and casting the large kernel operand. It cannot automatically remove activation quantization, because activations depend on the current batch and are regenerated on every forward pass. Merely writing an FP8 checkpoint and restoring it to FP32 would save disk space but would not remove runtime quantization; the low-precision representation must remain device-resident and be consumed directly by the dot operation.

Across two ensemble members, the four logical `4096 x 4096` residual Dense kernels contain about 134.2 million weights. Their approximate persistent storage is:

| Stored format | Online residual kernels | Online + target residual kernels |
|---|---:|---:|
| FP32 | 512 MiB | 1,024 MiB |
| BF16 | 256 MiB | 512 MiB |
| FP8 | 128 MiB | 256 MiB |

The online Critic's two FP32 Adam moment tensors add roughly another 1,024 MiB for these kernels. Therefore low-precision parameter storage is valuable, but optimizer-state quantization is still required for the final end-to-end memory claim.

### Important format distinction

- **BF16-resident kernel + FP8 GEMM:** halves parameter traffic/storage, but still requires BF16-to-E4M3 conversion before a native FP8 GEMM. It is a useful stability bridge, not a complete solution to repeated FP8 quantization.
- **FP8-resident kernel + scale:** can remove repeated kernel conversion only if the dot wrapper accepts a prequantized right-hand operand directly. Passing an FP8 parameter through the current Flax wrapper may still invoke its quantization path and must not be assumed efficient without HLO/profile evidence.
- **FP32 master + FP8 shadow:** can test the maximum compute-side benefit safely, but adds approximately 25% parameter storage and does not constitute low-precision parameter storage.
- **FP8 parameter without FP32 master:** provides the strongest storage/bandwidth benefit but exposes every optimizer or EMA update to FP8 representability limits.

### Target versus online Critic

The target Critic is the cleaner first persistent-storage experiment:

- it has no backward pass or Adam state;
- its only state transition is the known `tau=0.005` EMA;
- an FP8 target can be reused for forward passes if it is refreshed less frequently;
- its main new risk is quantized bootstrap output plus lost/staircase EMA updates.

The online Critic is harder:

- Adam changes the weights after every learner update;
- small optimizer updates may round away when written directly to BF16 or FP8;
- an FP8 shadow must be refreshed after each optimizer update, although it can still be reused by subsequent Critic/Actor paths before the next update;
- removing the FP32 master should be evaluated together with, or after, the optimizer-state design.

### Staged experiment

1. **Profile current conversion cost.** Separate weight-`amax` reduction/cast, activation cast, GEMM, and optimizer/EMA traffic on an exclusive GPU. Inspect optimized HLO because source-level operations may be fused.
2. **FP32 master + FP8 compute shadow.** Quantize each large kernel once after it changes and consume the shadow directly. This isolates performance benefit without introducing persistent-weight error.
3. **Persistent target storage.** Compare BF16 and naive FP8 target storage, first without stabilizing techniques. Combine with Idea 001 only if measured target-update loss is material.
4. **Persistent online BF16.** Test whether Adam updates survive when parameters are written to BF16 while moment states remain FP32.
5. **Persistent online FP8.** Attempt only after measuring optimizer update-to-lattice ratios and selecting an explicit update strategy. Do not silently retain an FP32 master while calling this full-chain FP8 storage.

### Required measurements

- time and bytes attributable to kernel `amax`, cast, GEMM, optimizer, and Polyak update;
- number of times each unchanged kernel is quantized per learner update;
- peak live memory and checkpoint size by state category;
- parameter-update survival and actual/intended update-norm ratio;
- FP32-master versus resident-weight output MAE, signed bias, and distribution divergence;
- return/success parity across seeds;
- whether stored FP8 kernels remain directly visible as E4M3 operands in optimized HLO.

### Connection to the paper story

This idea separates three increasingly difficult claims:

1. **FP8 compute:** native tensor-core GEMMs with FP32 persistent learning state;
2. **FP8 compute representation:** device-resident FP8 shadows that remove repeated weight conversion but retain FP32 masters;
3. **FP8 learning state:** parameters themselves persist in FP8 and must absorb EMA/optimizer updates.

The separation prevents a speed optimization from being confused with a numerical-stability contribution. Idea 001 addresses the target-state transition in stage 3; a later optimizer idea must address the online-state transition.
