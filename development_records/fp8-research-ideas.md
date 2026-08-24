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

Status: `diagnosed` for the temporal-resolution failure; interventions remain `hypothesis`

Updated: 2026-08-24. The accepted first test stores only the four logical residual-Dense kernels of the target Critic in E4M3, with one dynamic per-tensor scale per ensemble member. It applies the original per-step `tau=0.005` EMA and deliberately excludes Kahan summation, error feedback, stochastic rounding, block scaling, and delayed target updates. C keeps the online Critic FP32; D uses the existing online `fp8_direct` path. Both use the same resident-FP8 target core. The width-512 smoke was numerically finite, but its single sparsely sampled EMA write was not representative of the width-4,096 target state and must not be used as evidence that updates generally survive.

Formal width-4,096 Dogs seed-42 C/D runs started from clean commit `f88b662` on 2026-08-24. They were deliberately stopped on 2026-08-24 after the failure was decisive: C ended at env step 150k / update 290003 with its last complete 125k eval return `8.16`; D-R1 ended at env step 137438 / update 264877 with 125k eval `7.23`. Their matched A/B controls were already at `427.68/411.66` by 125k. Both resident runs remained finite with no NaN/Inf, so the failure is learning stagnation rather than arithmetic explosion. The original short D segment (`fb02bf23`) remains a separate migration-interrupted run.

The target-forward-only arm is complete: online and target residual GEMMs use `fp8_direct`, but all target parameters and EMA arithmetic remain FP32. This is a control rather than a proposed low-precision state method. B → target-forward-only isolates target bootstrap compute error; target-forward-only → D-R1 isolates persistent E4M3 target storage plus per-step requantized EMA. The formal Dogs seed-42 run `EXP-FP8-TARGET-FWD-S42` (W&B `gij6jhgd`) completed 500k steps with final mean return `757.81`, versus `816.37` for B and `7.23` at 125k for D-R1. This single-seed result does not establish equivalence to B, but it rules out target FP8 forward alone as an explanation for the resident target's learning stagnation. GPU2 was shared, so the run does not support performance claims.

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

### Diagnosed evidence

The formal controls isolate persistent storage and requantized EMA as the dominant failure:

- At 50k eval, online-FP8/target-FP32 B scored `124.50`, and target-FP8-forward/FP32-storage scored `120.40`; C/D resident targets scored only `15.16/14.93`.
- The bootstrap value diverged before return: at env step 6k, B and target-forward-only had `q_mean=5.85/5.93`, while C/D were still `-5.70/-5.81`.
- At 50k, the mean absolute online-target residual-kernel gap was `7.48e-4` for FP32 target storage, versus `8.53e-2` for C and `5.76e-2` for D: approximately `114x/77x` larger.
- A direct one-step probe from the saved 100k C/D checkpoints found `99.9947%`–`99.9994%` of codes unchanged. The applied/intended update L2 ratio was only `0.195%`–`0.992%` for C and `0.240%`–`0.519%` for D; relative update error was approximately one and several layer/member update directions were near-orthogonal or negative.
- The existing exact-zero swallowed fraction is insufficient: a changing per-tensor scale can make the physical update nonzero while preserving almost none of the intended vector. Sparse `last_applied_ema` samples also conflict with checkpoint-next-step probes, so consecutive-window/cumulative diagnostics are required before using those recorder fields again.
- FP8 target-forward error relative to the same dequantized resident weights stayed small even while return collapsed; this reference does not measure drift from the counterfactual FP32 EMA target.

The supported mechanism is therefore a temporal-resolution mismatch followed by bootstrap amplification. The intervention remains open.

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

### Candidate A: dynamically triggered resolution-matched cadence

A fixed `K` is useful as an ablation but is not the preferred final method because representability changes by layer, training phase, model width, and benchmark. A benchmark-adaptive variant keeps a scalar target age `a` and tests the update corresponding to the elapsed time:

```text
a <- a + 1
alpha_a = 1 - (1 - tau)^a
delta_a = alpha_a * (online - target)
applied_a = Q(target + delta_a) - target
```

The target is written only when `applied_a` retains enough of `delta_a`; then `a` resets to zero. Otherwise the state remains unchanged and elapsed updates accumulate through `alpha_a`. The trigger should be derived from the current FP8 lattice rather than a benchmark-tuned return threshold. Candidate signals are projected retention `<applied,delta>/||delta||^2`, cosine alignment, and the quantization-error floor implied by the E4M3 unit roundoff. The exact universal rule is an open design decision, not yet frozen.

Prefer one global age across the four residual kernels first so the target remains a coherent network. Layer-wise age is a later variant only if measurements show persistent layer differences. Freezing each kernel scale between committed writes is worth testing so skipped steps are true no-ops rather than global scale drift.

Benefits: no target-sized master or residual; frequency adapts automatically across benchmarks. Risks: using the latest online network to approximate several elapsed EMA steps is not exact when online parameters move quickly, and probing representability scans the target matrices. Online drift and wall-clock cost must be measured.

### Candidate A2: probability-corrected representable EMA

Instead of waiting a deterministic number of steps, first find a larger coefficient `alpha >= tau` whose quantized write is genuinely representable:

```text
gap = online - target
applied_alpha = Q(target + alpha * gap) - target
```

Then apply that representable write only with a probability chosen to preserve the original EMA rate. In the simple well-aligned case, `p = tau / alpha`; for example, if FP8 needs an `alpha=0.08` write, execute it with probability `0.005/0.08=0.0625`. The conditional expected coefficient remains `tau`, so the method does not introduce a benchmark-specific cadence.

A more quantization-aware version uses the actual projected progress:

```text
progress = <applied_alpha, gap>
p = tau * ||gap||^2 / progress
```

Only candidates with `progress >= tau * ||gap||^2` are eligible, so `p <= 1`. Among eligible candidates, select the one minimizing `p * ||applied_alpha||^2`, a proxy for stochastic update variance. This gives a parameter-free selection principle: use the lowest-variance representable jump whose expected progress along the desired EMA direction matches the original update. A powers-of-two search from `tau` to `1` is a simple first implementation; `alpha=1` provides a fallback equivalent to a probabilistic hard copy.

This avoids the constant-online approximation made by deterministic `K`-step accumulation because each decision conditions on the current online/target pair. Its main risk is rare target jumps entering the bootstrap chain. A deterministic scalar phase accumulator and a block-balanced randomized version should be tested as variance-reduction variants; neither requires a target-sized FP32 residual.

Decision note, 2026-08-24: whole-network probability correction is no longer the preferred first method. Matching the EMA coefficient only in expectation does not preserve the path followed by one nonlinear, bootstrapped RL run. It introduces two coupled risks: a stale target during the hold interval and a larger Bellman-target jump at the event. Keep it as a mechanism branch, but require a continuous/global-smoothness variant before promoting it.

### Candidate A3: lattice-matched interleaved block EMA

Status: `testing` through the open-loop teacher simulation registered on
2026-08-24.

Partition each large residual kernel into many blocks. Each block independently determines the smallest FP8-representable coefficient `alpha_b >= tau`, but its events are phase-shifted or balanced across blocks. At one learner step, only a subset of blocks makes a larger representable move; at the next step, a different subset moves. With thousands of blocks, the target network changes on every learner step even though each individual block updates less frequently.

For block `b`, use an event rate approximately `p_b = tau / alpha_b`, or the actual projection-corrected form from Candidate A2. A deterministic block-level phase accumulator can schedule these events without per-weight residuals or random whole-network jumps. The block phases are tiny metadata compared with a target-sized Kahan buffer.

This candidate directly addresses both objections to whole-network cadence: there is no long interval in which the entire target is frozen, and a single event changes only a small spatial fraction of the network. Its main risks are hybrid-aged blocks, interaction with a shared per-tensor scale, and extra block bookkeeping. The first design should freeze the tensor scale between explicit rebase events; otherwise changing one shared scale moves every physical weight and defeats the meaning of a block-local update. This is a candidate, not yet an accepted implementation.

Accepted first-screen definition: split each physical residual kernel into
`128x128` blocks, give every block its own dynamic FP32 scale and deterministic
phase, and stagger phases by the global block index. At every learner update,
search `alpha` over `{tau, 2*tau, 4*tau, ..., 1}`. A candidate is eligible only
when its dequantized write has projection at least `tau * ||gap||^2` along the
current online-target gap. Among eligible candidates, minimize
`p_alpha * ||applied_alpha||^2`, where
`p_alpha = tau * ||gap||^2 / <applied_alpha, gap>`. Advance the block phase by
`p_alpha`; commit codes and the new scale only when the phase crosses one.
Blocks without an eligible positive-direction candidate stay bitwise unchanged
and are counted, with no compensation buffer or fallback. This is deliberately
an open-loop screen, not yet a training precision mode.

### Candidate B: dual-scale FP8 residual accumulator

Store the visible target as E4M3 plus a second FP8 residual at a much finer scale:

```text
desired = tau * (online - target)
accumulated = dequant(residual_fp8) + desired
new_target = Q_target(target + accumulated)
residual_fp8 = Q_residual(accumulated - (new_target - target))
```

The residual is not used in the target forward; it accumulates sub-lattice updates until they can be folded into the visible FP8 target. Both states remain FP8, so storage is about 16 bits per quantized target weight rather than a 32-bit master. A residual-specific per-block scale is likely necessary because its magnitude is much smaller than the weight magnitude.

Benefits: deterministic, preserves per-element direction, and directly addresses discarded increments. Risks: doubles FP8 target-state storage, adds update bandwidth, and is related to error-feedback/Kahan-style prior art. It is a strong method/baseline branch, but novelty must not be assumed.

### Candidate B2: sub-ULP phase accumulator

Disposition: `rejected` as a separate research branch on 2026-08-24. Its compensation principle is too close to Kahan/error feedback for the current novelty goal. Retain the note for provenance, but use Kahan as the representative compensated-arithmetic baseline instead of implementing this variant.

Replace a general floating residual by an 8-bit per-weight phase measured in fractions of the local FP8 spacing. Each intended update advances that phase; when it crosses one FP8 boundary, the visible target code advances and the remaining fractional phase is retained. The unit is derived from the current code, block scale, and neighboring E4M3 values, so it adapts to tensor magnitude without a benchmark-specific `K`.

This is a deterministic way to retain sub-ULP updates with approximately 8 extra bits per stored target weight rather than a 32-bit master. It is still conceptually related to compensated/error-feedback arithmetic, and exponent-boundary plus scale-change behavior must be defined carefully. It is attractive as a robust engineering branch even if it is not the main novelty claim.

### Candidate B3: scaled Kahan-momentum baseline

Status: `baseline-only`; the 30k open-loop screen is complete and rejects this
FP8 adaptation as an effective replacement for lag-coded state.

Follow the FP16 SAC paper's Kahan-momentum ordering and its state scale
`C=1e4`. For each residual kernel, persist an E4M3 representation of
`C * target` and a second E4M3 Kahan compensation tensor; the two buffers have
independent current-amax per-tensor scales for each ensemble member. One update
is:

```text
value = C * tau * (online - target)
y = value - compensation
new_scaled_target = Q_E4M3(scaled_target + y)
compensation = Q_E4M3((dequant(new_scaled_target) - scaled_target) - y)
target = dequant(new_scaled_target) / C
```

There is no FP32 target master. This preserves the paper's scaled-buffer and
compensated-addition semantics while avoiding immediate overflow from storing a
literal `1e4`-scaled tensor without an FP8 scale. Because current-amax scaling
is homogeneous, multiplying by `C` does not by itself create extra E4M3
mantissa resolution; any gain must come from the compensation buffer. That is a
material difference from fixed-range FP16 and must be stated when interpreting
the baseline.

The first screen replays 30,000 learner updates from the same fixed healthy
Dogs checkpoint and reports the same parameter, expected-Q, categorical-target,
and non-finite diagnostics as lag-coded. A useful result for the paper is not
merely “Kahan is worse”: lag-coded should remain closer to the FP32 teacher
while using one FP8 matrix state per kernel, whereas Kahan uses two.

Observed result, `EXP-FP8-TARGET-KAHAN-OFFLINE-30K`: all 30,000 updates and 60
diagnostic points were finite. At the endpoint, lag-coded versus Kahan target
relative error was `0.007725/0.258575`, displacement cosine
`0.999540/0.030080`, expected-Q MAE `0.001216/0.141415`, and categorical
probability JS `3.26e-6/0.012536`. Kahan was effectively indistinguishable
from naive per-tensor resident EMA across the entire window (maximum target
relative-error difference `6.26e-7`). The compensation was nonzero, so this is
not a dead-state implementation failure. Under dynamic FP8 scaling, the
paper's scaled Kahan buffer does not repair the absolute target lattice; keep
it as the strongest directly relevant prior-art baseline rather than a method
branch.

### Candidate C: stochastic FP8 EMA rounding

Disposition: `baseline-only`, deferred until a proposed method exists.

Round each candidate weight to adjacent FP8 values with probability chosen so the expected stored value equals the FP32 candidate. Sub-ULP EMA updates then survive in expectation without a master or residual tensor.

Benefits: no additional parameter state and no cadence hyperparameter. Risks: target-network noise enters the bootstrap feedback chain, efficient adjacent-FP8 sampling is nontrivial, and stochastic rounding is established prior art. Treat it as an important baseline and possible practical solution, not the default novelty claim.

### Candidate D: projection-preserving block rounding

Within a block, choose which weights advance one FP8 lattice step so that the aggregate applied update preserves the projection and approximate norm of the intended EMA vector. This is a deterministic, update-aware alternative to independent stochastic rounding. A small block-level carry could preserve leftover update mass without a full residual tensor.

Benefits: can guarantee useful directional motion with little persistent metadata. Risks: selecting elements may require ranking or histograms, repeated deterministic choices can bias particular weights, and implementation is substantially more complex. Keep it exploratory until simpler alternatives are measured.

### Candidate E: bootstrap-aware functional update guard

Weight representability alone does not measure the error that actually feeds RL. On a shared probe or current batch, measure the target distribution change caused by a candidate write. Commit only when the update is above the measured FP8 forward-noise floor, while preventing a single quantized jump from producing an excessive expected-Q or 101-bin JS change. This can be combined with Candidate A, or evaluated independently with adaptive hard target refreshes.

Benefits: directly addresses bootstrap amplification and may adapt across tasks based on function rather than weight scale. Risks: adds forward work, needs a principled non-tuned functional boundary, and can hide rather than solve parameter-state error. Multi-step or TD-lambda targets are related bootstrap-reduction controls but change the RL algorithm more substantially.

Decision note, 2026-08-24: do not modify the bootstrap rule in the first intervention. Retain expected-Q error, signed bias, and 101-bin divergence as diagnostics to determine whether target-state error accumulates functionally. Only consider an active guard or a different Bellman target if those measurements show an additional failure after the EMA write problem is repaired.

### Candidate F: lag-coded target state

Status: `supported` by the 50k and Kahan-controlled 30k open-loop screens;
closed-loop training remains pending.

Do not store the target as an absolute copy of a relatively large weight. Store its lag relative to the online Critic:

```text
d_t = target_t - online_t
d_(t+1) = (1 - tau) * (d_t - (online_(t+1) - online_t))
target_(t+1) = online_(t+1) + d_(t+1)
```

If `d` is exact, this is algebraically the same EMA. In a healthy BRC run the online-target gap is much smaller than the absolute weights, so an FP8 representation scaled to `d` offers much finer absolute resolution than storing the target itself in FP8. This removes the operation “add a tiny number to a large FP8 weight,” which is the diagnosed failure mode.

The target kernel must be reconstructed and quantized for the FP8 GEMM, so this route may sacrifice the resident-kernel conversion saving. It nevertheless removes the separate FP32 target state and is directly relevant to full-chain low-precision storage. The key risks are whether quantized `d` can cancel online updates smoothly enough for bootstrap stability, and overlap with delta-coding literature; both require offline screening and a novelty search.

Here `d` is the FP8 object, not the target kernel itself. For example, storing `online=0.100` and `d=-0.003` does not directly store the target value `0.097`. The forward path must dequantize/add `online + d`, then cast the reconstructed kernel to E4M3 for the native GEMM. That final cast is the “re-quantization.” A cached reconstructed FP8 kernel can move this work to the learner update rather than every target forward, but it adds another FP8 state; alternatively, computing `X @ online + X @ d` avoids reconstruction at the cost of a second GEMM.

Accepted first-screen definition: only the four residual Dense kernels use the
lag representation; each ensemble member keeps E4M3 lag codes and one current-
amax FP32 scale. The recurrence consumes the exact same consecutive
`online_old/online_new` pair as every other shadow. Non-hotspot target leaves are
borrowed from the healthy FP32 teacher only when a diagnostic forward is built.
There is no FP32 target master in the lag shadow, and the temporary reconstructed
FP32 kernel is never saved.

### Diagnostic and intermediate controls

- BF16 target storage with FP8 target forward: fast test of how much mantissa precision is required; not an end-to-end FP8 solution.
- Per-row or block target scales: spatial-resolution control; may reduce outlier damage but does not itself accumulate sub-ULP temporal updates.
- One-time target quantization followed by FP32 EMA: separates initialization perturbation from repeated requantization.
- Fixed `K` cadence: cheap ablation that validates the cadence mechanism before the dynamic trigger is finalized.

These branches should first be screened with checkpoint-based consecutive-window simulation. Promote only the small subset that restores cumulative EMA displacement and keeps bootstrap distribution error bounded.

### Minimal experimental sequence

1. **Target-forward FP8 only.** Keep target storage and Polyak arithmetic FP32. This control is implemented and smoke-validated as `EXP-FP8-TARGET-FWD-S42`; it cannot diagnose small EMA update loss, but it separates bootstrap compute error from persistent-storage/EMA error.
2. **Offline naive-state simulation.** At matched online/target FP32 checkpoints, simulate per-step FP8 target storage and requantization without Kahan or other corrections.
3. **Naive persistent-FP8 target run.** Completed and stopped early: C/D show decisive learning stagnation, and checkpoint probes confirm that the intended EMA vector is mostly not represented.
4. **Consecutive-window offline diagnosis.** From stored FP32/resident checkpoints, compare cumulative target displacement against an FP32 shadow EMA and audit why sparse in-training diagnostics conflict with checkpoint-next-step probes.
5. **Open-loop teacher simulation.** Resume a healthy FP32-target checkpoint, let its online Critic generate one shared parameter trajectory, and feed that same sequence to naive FP8 plus every candidate target-update operator. This isolates update fidelity before candidate bootstrap errors are allowed to change the online trajectory.
6. **Primary-method screens.** First compare lag-coded target state and lattice-matched interleaved block EMA on the shared sequence. Whole-network dynamic/probabilistic cadence remains a mechanism reference. Do not spend the initial iteration implementing sub-ULP compensation or stochastic-rounding baselines.
7. **Short closed-loop environment runs.** Promote only candidates that restore cumulative update direction and norm while keeping target distribution error bounded; then allow their bootstrap targets to affect learning before launching new 500k runs.

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

The current implementation samples one real EMA only when tensor statistics are due. Although its formula uses exact pre-write online and target states, those sparse samples conflict with immediate checkpoint-next-step probes and the exact-zero metric is blind to scale-induced nonzero drift. Before further method claims, replace single-point interpretation with a consecutive-window cumulative comparison against an FP32 shadow EMA. Raw codes, scales, activations, and forward errors remain split by ensemble before the generic tensor summarizer.

### Baselines

- FP32 master target with FP8 target forward;
- naive per-step persistent-FP8 EMA;
- Kahan momentum;
- stochastic rounding;
- FP32/BF16 error-feedback residual;
- periodic hard target copy.

Kahan is prior art and should be treated as a strong baseline rather than the proposed contribution. If its compensation is itself stored in FP8, the compensation can be rounded away; if it is stored in FP32, it adds target-sized high-precision state.

Stochastic rounding and error feedback are also established low-precision techniques. The current differentiated research questions are therefore the FP8 target-network temporal mismatch itself, an automatically selected representable EMA transition, and controlling the resulting error in BRC's bootstrap distribution. Novelty is not assumed until a dedicated literature search is completed.

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
- Gupta et al., "Deep Learning with Limited Numerical Precision," ICML 2015: https://proceedings.mlr.press/v37/gupta15.html
- Zhao et al., "Direct Quantized Training of Language Models with Stochastic Rounding," ACML 2025: https://proceedings.mlr.press/v304/zhao26b.html
- Micikevicius et al., "FP8 Formats for Deep Learning": https://arxiv.org/abs/2209.05433

## Idea 002: Resident Low-Precision Critic Weights

Status: `rejected` for naive per-step E4M3 target storage; corrected representations remain `hypothesis`

Updated: 2026-08-24. The first resident-state implementation is limited to the target Critic residual core. It consumes stored E4M3 kernels directly in target forward GEMMs while input/output projections, LayerNorm, biases, task embeddings, and all online learning state retain their selected existing precision. Formal C/D runs decisively rejected naive per-step requantized EMA: arithmetic remained finite, but target tracking and return stagnated. Resident low-precision storage remains an active direction only with an explicit temporal-update mechanism such as Idea 001.

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
