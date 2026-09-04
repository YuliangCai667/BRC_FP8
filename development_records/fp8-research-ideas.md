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

Status: `testing`; supported by the 50k and Kahan-controlled 30k open-loop
screens, with the first closed-loop D-lag implementation smoke-validated.

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

Accepted first closed-loop definition: `target_critic_precision=fp8_lag`
keeps the online Critic parameters and Adam state FP32 while its GEMMs use the
existing `fp8_direct` path. The target's four residual-kernel leaves persist
E4M3 lag codes rather than absolute target weights; each learner update consumes
the exact pre/post-update online kernels and writes
`(1-tau)*(lag-online_delta)` back with an ensemble-specific current-amax scale.
All target forward consumers reconstruct through one shared entry. Only the
training bootstrap advances target input/kernel delayed-scaling metadata;
diagnostics, time-limit queries, and evaluation are read-only. The first
width-512 closed-loop smoke completed 203 learner updates without NaN/Inf.
Formal Dogs seed-42 D-lag started from clean commit `1c135af` on GPU1 as
W&B `nu2d5b90` and passed its first learner update without NaN/Inf. It is the
next decision gate; no closed-loop learning claim is made before its return and
functional diagnostics mature.

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

Status: `supported` for lag-coded target state; naive target storage `rejected`; naive online backward `rejected`; repaired scale-aware online resident operator `testing`; fixed anchor supports its norm invariant but is `rejected as a sufficient online return repair at 100k`; matched current-amax FP32-master isolation `supports masterless-write attribution at 100k`, run continuing

Updated: 2026-09-03. The first resident-state implementation was limited to the target Critic residual core. It consumes stored E4M3 kernels directly in target forward GEMMs while input/output projections, LayerNorm, biases, task embeddings, and all online learning state retain their selected existing precision. Formal C/D runs decisively rejected naive per-step requantized EMA: arithmetic remained finite, but target tracking and return stagnated. Lag-coded target state subsequently completed the Dogs seed-42 closed loop with final return `806.10`, supporting the representation correction for the target update. The online-resident runs also showed real scale-dominated norm growth, while a retrospective gradient audit found every recorded resident-kernel gradient exactly zero (`68/68` fixed-anchor and `80/80` naive seed42), versus `80/80` nonzero in the matched FP8-direct control. The missing scale-aware backward was a real implementation failure and is now repaired. However, the fresh repaired-backward + fixed-initial-anchor arm reached only `134.44` return at 100k versus `157.98` for repaired-only and `340.61` for matched FP8-direct, even though all four resident gradients are nonzero and all eight anchors pass independent checkpoint audit. The new causal evidence rejects both backward repair and fixed norm, alone or together, as sufficient explanations of the remaining return gap. The common unresolved mechanism is now the masterless online E4M3 parameter transition and its optimization trajectory. A strict current-amax FP8-compute/FP32-master control is running to remove the remaining current-amax-versus-delayed-amax mismatch from that attribution.

Accepted scope definition: full persistent-state FP8 applies to large,
long-lived payloads such as online/target matrix state and later optimizer
state. FP32 per-tensor/per-block scales, amax histories, counters, and other
small quantization metadata are allowed, as are FP32/BF16 reductions,
normalization, softmax, and accumulation. A same-size FP32 weight master,
shadow, or residual is forbidden. State-byte reporting must separate FP8
payload from high-precision metadata and report the metadata fraction.

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

### 2026-08-28 online-resident diagnostic phase

Research question: with the existing native FP8 GEMMs unchanged, can the four online residual-Dense kernels persist in E4M3 and absorb AdamW updates without an FP32 weight master, while target parameters/EMA and Adam moments remain FP32 controls?

The first baseline intentionally adds no correction mechanism. For each resident kernel, the optimizer forms the intended post-Adam candidate transiently in FP32 from the dequantized current weight, then immediately requantizes it to E4M3. The transient candidate is permitted as update arithmetic but is not a persistent master. Target storage and EMA remain FP32, while target residual GEMMs retain the already validated `fp8_direct` compute path; Actor, non-residual Critic leaves, and optimizer moments also remain FP32 in this phase. This isolates the online parameter write while keeping both online and target residual GEMMs FP8; it is not yet a full-chain claim.

Falsifiable hypothesis: naive online E4M3 residency causes a material learning deficit relative to the existing online `fp8_direct` control, and the deficit can be assigned to at least one measured mechanism rather than merely to non-finite arithmetic. Failure is not assumed in advance.

The diagnostic must distinguish:

1. **Update-resolution loss.** Measure intended and applied update norms, cosine/projection, sign agreement, coordinate-wise swallowed fraction, and code-change fraction. High swallowed/projection loss with preserved direction supports a lattice-resolution diagnosis.
2. **Scale coupling.** Measure `log2(new_scale/old_scale)`, scale turnover, code churn among coordinates with negligible intended updates, and a same-candidate old-scale counterfactual. Correlated global churn or dequantized movement during scale jumps supports per-tensor scale coupling; raw code churn alone is insufficient.
3. **Weight-to-function sensitivity.** On the same diagnostic batch, compare the ephemeral FP32 post-Adam candidate with its resident E4M3 write using logits, expected-Q MAE/signed bias, 101-bin JS divergence, and critic loss. Small weight error with large functional error supports Q-function sensitivity.
4. **Bootstrap feedback.** A growing correlation between one-step error and later return is only suggestive. A causal claim requires a matched open-loop/teacher-target control against the normal closed loop: stability with fixed FP32 teacher targets but failure when quantized errors feed the target/bootstrapping path supports RL-specific amplification.

The primary control for the first closed-loop run is the completed target-forward-only arm: online `fp8_direct` + target `fp8_direct`, with FP32 target storage/EMA. The new arm changes only online storage to `fp8_resident`; target remains `fp8_direct`. The online-FP8/fully-FP32-target B run remains a secondary ceiling rather than the isolating control. Lag-coded and resident target storage remain disabled. The established Dogs seed-42, width-4,096 protocol is the proposed formal comparison; smoke length, diagnostic cadence, early-stop rule, and the later open-loop construction remain decision gates rather than frozen protocol.

Implementation warning: the existing `ResidentFp8Dense` cannot simply be enabled for the online Critic under the generic `Model.apply_gradient`. Its parameter is an FP8 code tensor with a separate scale; AdamW must update the physical dequantized weight and explicitly write new codes/scale. Treating codes as ordinary parameters would optimize in code space, may promote the stored dtype, and would not define the intended resident-weight algorithm.

Implementation status on 2026-09-01: the naive per-tensor baseline and sampled
real-update diagnostics are implemented in the working tree. The physical FP32
tree and post-Adam candidate exist only inside one compiled update; neither is
returned as persistent model state nor checkpointed. Diagnostics return scalar
summaries only. The current retry additionally records optimizer-update radial
cosine against the old physical weight and quantization-error radial cosine
against the pre-write candidate. Earlier CPU validation is recorded in
[2026-08-28-online-fp8-resident-critic.md](2026-08-28-online-fp8-resident-critic.md);
the radial-cosine addition was launched without smoke or extra validation at the
user's explicit request. GPU runs and IDs are recorded in
[experiment-runs.md](experiment-runs.md).

The 2026-09-02 diagnostic-only extension adds exact intended/actual angular
motion, first-/second-order norm-growth decomposition, the installed-Optax
AdamW adaptive/decay split, and an exact scale-then-code write decomposition.
It is designed to distinguish norm-denominator effects, FP8 directional-motion
loss, second-order tangential accumulation, decay attenuation, and long-run
scale domination without introducing an intervention. Definitions, validation,
and the smoke record are in
[2026-08-28-online-fp8-resident-critic.md](2026-08-28-online-fp8-resident-critic.md).

The two enhanced Dogs runs (`EXP-FP8-ONLINE-RESIDENT-MECH-S42/S1`) completed
500k. Their resident norms grew `21.7x/23.2x` from 25k to 500k and intended
effective angular steps fell `19.9x/33.4x`. Actual/intended angular retention
remained within roughly `1e-4` of one, so the failure is not loss of intended
direction at the FP8 write. The always-positive second-order update-squared
term is three to four orders of magnitude too small to account for observed
norm-square growth. AdamW decay retention is mildly reduced (`0.829/0.937`
when weighted by intended radial shrinkage), but full decay would itself shrink
only about 2.93% over the run. Code freezing and scale domination reproduce:
late code-unchanged fractions reach `0.870/0.991`, code L2 is nearly fixed,
and runaway scales grow as much as `39.9x/71.5x`. The supported mechanism is
therefore scale-gauge norm runaway causing intended angular-step collapse;
the exact origin of the accumulated first-order radial drift remains unresolved
because diagnostics sample only one update every 25k env steps.

### 2026-09-03 failed intervention: Moving-Anchor FP8 Residency

Status: previous-norm recurrent formulation `mechanism-failed` and its two
formal runs were stopped on 2026-09-03; fixed-reference radius revision remains
a candidate.

The tested intervention was a LayerNorm-aware, per-update canonicalized
resident write. For each kernel/member, the AdamW candidate is quantized exactly
once with the unchanged current-amax E4M3 quantizer. If `C_next` and `q_next` are
that raw code and scale, it computes
`alpha = ||W_old||_F / max(||q_next C_next||_F, 1e-12)`, persists the same
`C_next` with scale `alpha q_next`, and writes the paired FP32 candidate bias as
`alpha b_candidate`. Scaling kernel and bias together preserves the complete
pre-LayerNorm affine output up to a positive global factor. AdamW moments remain
FP32 and unmodified.

This differs from the earlier candidate wording: it uses the previous physical
kernel norm as the current radius after raw quantization and needs no separate
reference-norm tensor. Recursively applying the rule keeps the initialized norm
while preserving the quantizer-selected code/direction. The implementation and
validation are recorded in
[2026-09-03-gauge-fixed-fp8-residency.md](2026-09-03-gauge-fixed-fp8-residency.md),
and the matched formal runs are `EXP-FP8-ONLINE-CANON-S42/S1` in the
[experiment ledger](experiment-runs.md). The next decision gate is whether fixed
norm prevents scale runaway, restores effective angular motion, and closes the
naive-resident return gap; tangent projection remains deferred.

The first mechanism readout on 2026-09-03 falsified the assumption that the
previous physical norm can serve as a numerically stable recursive anchor.
Although each sampled write reports `post/old` within about `1.2e-7` of one,
seed42's four-kernel norm grows `804.1→1833.9` from 25k to 50k, while code L2 is
nearly unchanged and stored scale carries the growth. The matched naive seed42
grows `441.1→794.1` over the same interval. The discrepancy identifies
accumulated FP32 reduction/scale recurrence and a self-correlated single-write
diagnostic, not successful gauge fixing. A fixed initialization-radius scalar
per kernel/member would remove the recursive anchor and costs only eight FP32
scalars. The formal seed42/seed1 runs were stopped at env step `55152/47628`
after writing `run_interrupted` and `run_finished`; this revision is not yet
implemented or accepted for launch.

### 2026-09-03 revision: Fixed-Initialization-Anchor FP8 Residency

Status: norm mechanism implemented and validated; seed42 25k mechanism gate
passed, but the 425k reward readout exposed an independent resident-backward
failure, so this run cannot establish fixed-anchor return efficacy.

For each of the four resident kernels and two ensemble members, initialization
now persists `rho0 = ||dequantize(initial_code, initial_scale)||_F` as one FP32
scalar. Every AdamW candidate still undergoes exactly one current-amax E4M3
quantization. The write keeps `C_next`, sets
`scale_next = rho0 / ||C_next.astype(float32)||_F`, and scales the paired bias by
`scale_next / raw_scale`. No previous physical norm enters the enabled path, so
rounding error cannot become the next reference radius.

The checkpoint gate independently reloads serialized code, scale, and anchor
and computes `scale * ||code.astype(float64)|| / rho0` with NumPy float64 for all
eight members. Implementation and validation are recorded in
[2026-09-03-fixed-anchor-fp8-residency.md](2026-09-03-fixed-anchor-fp8-residency.md).

The resumed seed42 run reached `572.84` at 400k and `499.04` at 425k, versus
`766.69/764.57` for the matched target-forward-only control. This is not a
renewed anchor drift: the 400k independent ratios are
`0.9996764–1.0035631`, and the 425k combined resident norm is still about
`256.0`. Instead, all 68 fixed-anchor resident-kernel gradient records are
exactly zero. The same defect is present in all 80 records of the completed
naive seed42 run, while all 80 matched FP8-direct records are nonzero.

The backward-invalid `ResidentFp8Dense` differentiated a raw E4M3
`lax.dot_general` whose
scale multiplication is outside the dot. Unlike Flax FP8 Direct, it has no
custom VJP that dynamically scales the output gradient to E5M2 and dequantizes
the operand gradients. The unscaled cotangent is therefore cast to FP8 before
reciprocal-scale recovery and usually underflows. This explains both partial
learning through FP32 projections/biases/LayerNorm/residual skips and the
failure to match the control. Future fixed-anchor return tests require a
scale-aware resident backward first; further anchor tolerance changes do not
address this failure.

### 2026-09-03 repair: Scale-Aware Resident FP8 Backward

Status: implementation and short numerical gates `passed`; formal Dogs seed42
500k run `running` on GPU3.

The resident forward still uses current-amax E4M3 activations, persistent E4M3
kernel codes and FP32 accumulation.  Its backward now reuses Flax's
`quantized_dot` custom VJP: FP32 output cotangents are dynamically scaled with
the same delayed-amax policy as FP8-direct, represented in E5M2, and consumed by
native FP8 transpose GEMMs.  The VJP returns physical FP32 kernel gradients
scaled by activation/output-gradient scales and physical FP32 input gradients
scaled by kernel/output-gradient scales.  Autodiff no longer traverses the
physical-kernel-to-E4M3 cast.

The output-gradient scale/history is persistent checkpoint metadata for all
four layers and both members.  Actor updates advance the same metadata while
propagating through the critic, so `dQ/da` uses the repaired input gradient.
AdamW moments and candidate/writeback behavior are unchanged.  The first
formal repair run deliberately disables canonicalization to isolate this
intervention.

The separate 25k mechanism gate was subsequently cancelled by the user.  The
first formal seed42 run therefore starts directly with `max_steps=500000` and
the normal 25k eval/tensor cadence; only startup is accepted here, with manual
stopping left to the user.  Seed1 remains sequential after seed42 completion.

Observed pre-run gates: the independent GPU probe gives repaired-resident vs
FP8-direct kernel cosine/L2 ratio `1.0/1.0` at output-gradient sigma
`1e-3/1e-2`, with no zero coordinates.  On a retained real Dogs batch, all four
kernel gradients have cosine `0.99927–0.99994` and L2 ratio
`0.99977–1.00081` versus FP8-direct; `dQ/da` has cosine `0.98988`, L2 ratio
`1.00182`, and no zero coordinates.  The 203-update end-to-end GPU smoke is
finite and all four kernel gradients are nonzero.  Details and limitations are
recorded in
[2026-09-03-scale-aware-resident-fp8-backward.md](2026-09-03-scale-aware-resident-fp8-backward.md).

Observed 94k readout: the repaired kernel gradients remain nonzero and the
candidate-to-resident write remains geometrically faithful, but the unanchored
resident norm runaway has returned more rapidly.  The independently decoded
combined resident-kernel norm is `691.53` at 25k and `1905.03` at 50k; the 75k
sample is `3022.31`, about `11.8x` the approximately `256` initialization norm.
Critic pnorm at 75k is `3046.30` versus `958.63` for the matched FP8-direct
control.  Eval return is `129.44` versus `220.44`; all four control tasks are
higher at that point.  The sampled intended effective angular step mean falls
about `5x` from 25k to 75k.

Inference: repairing the backward is necessary but does not remove the
scale-gauge degree of freedom; once physical gradients are restored, radial
drift can be faster than in the backward-invalid runs.  The evidence supports,
but does not yet prove, norm runaway as the remaining cause of the reward gap.
The clean next causal arm is a fresh same-seed run combining the validated
scale-aware backward with the fixed-initial anchor.  The old fixed-anchor run
cannot answer this because its resident kernels received zero gradients.  The
user accepted this arm on 2026-09-03; it is planned from random initialization
on GPU0 as `EXP-FP8-ONLINE-SCALED-BWD-FIXED-ANCHOR-S42-500K`.  It started at
2026-09-03 17:16:25 Asia/Shanghai and passed startup acceptance; status is
`running`.

Observed 100k fixed-anchor causal readout: the combined intervention does not
recover the control curve.  Mean eval return is `134.44`, versus `157.98` for
the repaired unanchored arm and `340.61` for the matched FP8-direct/FP32-weight
control.  The corresponding critic pnorm values are
`519.08/4171.12/1129.21`; update NaN/Inf counts remain `0/0`.  NumPy float64
decoding of the serialized 100k checkpoint gives all eight
`scale * ||code|| / rho0` ratios in `0.9999815–1.0004089`, and all four sampled
resident-kernel gradients are nonzero.  This is therefore neither recurrence
of moving-anchor drift nor recurrence of the old all-zero backward.

The immediate write is also functionally close: at 100k its aggregate
expected-Q MAE is `1.80e-5`, critic-loss relative error `9.27e-8`, and JS
divergence about `1.03e-8`.  Nevertheless, an average `80.69%` of stored E4M3
codes are unchanged in that write.  Fixed anchoring preserves the norm of the
tangential update almost exactly, but it is not optimization-neutral: it pins
each member at the initialization norm `90.51`, while the matched direct
control's four combined-member residual-kernel norms have already grown to
`241.87–669.69` at 100k.  The layer is functionally close to scale-invariant
because each affine is followed by LayerNorm, but AdamW and its coordinatewise
moments are not invariant to this reparameterization.  Consistently, several
fixed-run kernel directions rotate strongly between 50k and 100k (cosine as
low as `0.0458–0.3246`), whereas the runaway unanchored arm is nearly frozen
in code direction between 100k and 150k (most cosine values
`0.999999–1.0`).  Thus the two existing resident arms occupy opposite bad
regimes: unanchored scale growth suppresses code/directional motion, while an
initial-radius hard projection removes the norm evolution that normally sets
the effective angular learning-rate scale.

Revised inference: norm runaway is a valid symptom and fixed anchoring solves
its stated invariant, but it is not the primary causal explanation for the
return deficit.  The next isolating control should preserve FP32 optimizer
parameter evolution while matching the resident current-amax forward/backward
operator, or equivalently add a non-persistent experimental shadow only for
causal diagnosis.  That separates per-step masterless E4M3 projection from
the current-amax versus delayed-amax compute-policy mismatch before designing
an error-feedback or scale-decoupled resident update.  This is a proposed next
gate, not an implemented method or an authorization to launch another run.

Continued 100k exclusion checks narrow the fixed-arm mechanism further.  On
the exact serialized physical weights and retained probe batch, resident FP8
versus a full-FP32 Critic gives loss relative error `6.54e-6`, residual-kernel
gradient cosine `0.98384–0.98892` with L2 ratio `0.88758–0.92540`, and
actor-facing `dQ/da` cosine/L2 ratio `0.9999979/0.9999878`.  The E5M2 history
is not at the old underflow cliff: fixed-layer current/history amax ratios are
roughly `0.36–0.44` at 100k, while a healthy direct layer can be as low as
`0.07`.  All `134,217,728` resident code elements round-trip exactly through
the critic-update `code -> float32 physical -> code` path.  Online-target
kernel cosine is `0.9999933–0.9999999` with relative gap
`0.000552–0.003675`, so the FP32 target EMA is tracking rather than collapsing.

The optimizer-state check also rejects the simple claim that stale radial
momentum is being thrown away wholesale.  At fixed100, first-moment cosine to
the weight is only about `6.9e-5–1.3e-3`; coordinatewise Adam normalization
creates at most a `0.171` radial cosine in the sampled next direction, and the
recorded write removes that radial component while preserving tangential
direction to numerical precision.  The more specific mechanism is the scale
dependence of the *effective angular step*: for a LayerNorm-preceding affine,
rescaling kernel and bias leaves the function nearly unchanged and rescales
the raw gradient inversely, while Adam's normalized absolute step is
approximately scale-invariant.  Hence angular motion scales approximately as
`1 / ||W||`.  Direct training allows the four combined-member norms to grow
from about `128` initially to `241.87–669.69` by 100k; fixed anchoring holds
them at `128`.  The hard anchor therefore removes the implicit angular
learning-rate decay that the healthy control actually uses.  The unanchored
resident arm overshoots in the opposite direction, with several norms in the
thousands and near-frozen code direction.  This explains why both extremes
can underperform without requiring a local forward, backward, write, or target
tracking failure.

### 2026-09-03 causal isolation: Current-Amax FP8 with FP32 Master

Status: implementation/unit gates `passed`; Dogs seed42 150k control `running`
on GPU1.

This control changes only the online residual-kernel persistent state relative
to the repaired resident operator. Each forward independently quantizes the
FP32 activation and persistent FP32 kernel with current-amax E4M3 scaling, then
uses the same `quantized_dot`, FP32 accumulation, output dequantization and
scale-aware delayed-amax E5M2 custom backward as the repaired resident path.
AdamW updates FP32 parameters normally. There is no resident code/scale write,
canonicalization, persistent FP8 shadow, or delayed-amax weight/input history.
Target precision and every RL hyperparameter remain matched.

The initial current-master and resident logits are elementwise identical in the
new unit gate, so this is not merely another FP32 or delayed-scaling baseline.
All current-master parameters and floating optimizer leaves are FP32; after
three learner updates the parameters move and each of the four output-gradient
histories advances. The formal run is
`EXP-FP8-ONLINE-CURRENT-MASTER-S42-150K`, W&B `vgd8apty`. At 25k its return is
`49.53`, versus repaired/fixed/direct `45.94/34.78/19.92`; therefore it does not
reproduce an early resident deficit, although this first point precedes curve
separation. Its critic pnorm `472.85` is already much closer to direct `495.37`
than repaired `733.80` or fixed `359.29`. The four serialized combined-member
kernel norms show the same current/direct proximity, and all four current-master
kernel gradients are nonzero. The 50k/100k return trajectory remains the
decision evidence.

At 50k, the behavior and geometry now agree with the causal prediction.
Current-master return is `113.17`, or `94.0%` of direct `120.40`, versus
repaired/fixed resident `101.41/87.12`; all four task returns individually
track direct. Current/direct critic pnorm is `720.93/756.40`, while
repaired/fixed is `1930.97/412.72`. The four physical kernel norms likewise
match current to direct rather than either resident regime. This supports the
loss of the off-lattice FP32 parameter position during every resident write as
the main trajectory-separation cause. It remains a single-seed 50k result;
75k/100k must confirm it after the control curve normally accelerates.

At 75k the separation is larger: current-master return `178.43` is
`37.9%/38.0%` above repaired/fixed resident `129.44/129.26`, and every task is
individually higher. It remains `19.1%` below direct `220.44`, so the evidence
supports masterless writeback as the main cause rather than the sole source of
all single-seed trajectory variance. The 100k point is retained as the final
decision gate.

The 100k gate confirms a large but not exclusive masterless-write effect.
Current-master return is `244.89`, versus repaired/fixed/direct
`157.98/134.44/340.61`: retaining the off-lattice FP32 parameter position gains
`55.0%/82.2%` over the two resident arms, and every task improves. Its internal
trajectory is much closer to direct than reward alone suggests: current/direct
critic pnorm is `1090.48/1129.21`, four layer norms are close, and the 5k–100k
critic-pnorm SMAPE is only `3.63%`, versus `77.11%/47.97%` against
repaired/fixed. Q-prediction and actor-pnorm trajectories also assign current
to direct. Therefore the resident state replacement, not a broken current-amax
GEMM or backward, causes the abnormal optimization geometry and a large return
loss. The remaining `28.1%` current-to-direct return gap means the small
current/delayed scaling-policy difference is amplified materially by the RL
closed loop; it must remain a second causal factor rather than being dismissed.

The remaining current-versus-delayed compute mismatch is independently small
at a healthy evolved state. Using the direct 500k FP32 parameters and exact
saved probe input for both operators, current-amax versus saved delayed-amax
has logits relative L2 `0.004814`, expected-Q MAE `0.003916`, and four kernel
VJP cosine/L2-ratio ranges `0.97565–0.99818` / `0.99079–1.00024`; gradient zero
fractions are nearly identical. This mismatch can still perturb a sensitive RL
trajectory and may contribute to the residual single-seed return difference,
but it is not a catastrophic local forward/backward failure.

An additional deterministic width-16 probe removes even the initialization
state mismatch: current-master online physical parameters and target state are
replaced by the resident physical parameters and target state before update 0,
while both Adam states are identically zero. Step 0 parameters and logits match
exactly. After one update, the resident write creates only `0.000897` maximum
relative kernel error and the quantized logits still match exactly; after ten
otherwise matched closed updates, kernel error is `0.008423` while logits
relative L2 reaches `0.027712`. This directly demonstrates that discarding the
off-lattice FP32 position can accumulate into a function-trajectory difference
even when the immediate quantized forward initially hides the write error.

### 2026-09-04 method under test: CARRY-FP8 trajectory preservation

Status: `corrective main+carry barriers validated; formal dual-seed running`. The
matched Dogs seed42/seed1 formal runs failed the 25k/50k/75k/100k
persistent-carry gate. A minimal probe shows
that the current GPU-JIT graph bypasses the lossy value of a newly-created FP8
code when it is immediately widened, so the residual used for carry is compiled
to zero. Those runs were gracefully stopped and retained only as invalid
negative controls.

CARRY-FP8 directly targets the diagnosed off-lattice state loss without
introducing an FP32 master or FP32 residual. Each resident kernel persists
`(C,s,R)` and represents the logical optimizer parameter as
`Theta=s(C+R/16)`, with both `C` and `R` in E4M3 and a shared per-member
current-amax FP32 scale. Initialization first writes the ordinary main code and
then quantizes `16 * (W_fp32/s-C)` into the carry, so the original FP32
initialization residual is not discarded.

The carry is deliberately absent from forward/backward: native FP8 compute
continues to consume only `sC`, preserving the repaired scale-aware custom VJP
and matched current-amax policy. At update time AdamW receives the transient
logical tree, including parameter-dependent weight decay, and the continuous
candidate is quantized once into the next main code/scale. The normalized
write residual is then quantized into the next carry. Target EMA likewise reads
the reconstructed logical online kernel, matching the FP32-master reference's
continuous parameter semantics while keeping target precision unchanged.

Observed unit evidence: main codes/scales and physical weights are identical to
ordinary current-amax quantization; carry reconstruction reduces random and
mixed-range matrix error; a two-element fixed-amax probe accumulates 32 updates
of `4e-4`, crosses the small element's main-code grid while naive resident RTN
remains frozen, and is over `4x` closer to the FP32 reference. Seven carry
CPU/GPU tests and the full `35/35` FP8 plus `63/63` repository suites pass.

Observed smoke evidence, reclassified after root-cause isolation: width-4096 Dogs completed at env 5001/update 5 with
all four covered kernel gradients nonzero, no parameter/update NaN or Inf,
E4M3 main/carry storage, zero saturation, and successful analysis/recovery
checkpoints. The initial four real learner updates were a degenerate boundary
case: their reported main-only write error was only `2.33e-8–6.53e-8`, carry
was zero, and the reduction ratio was approximately one. Those values are now
known to be false in-graph fidelity readings caused by the GPU JIT conversion
bypass, not genuine lattice-aligned candidates. The recovery probe that produced
main/logical relative error `0.008859–0.012317/0.000212–0.000311` and
`39.65x–43.32x` reduction validated arithmetic across an eager/materialization
boundary, not the production graph. The storage accounting remains valid: carry
adds one `134,217,728`-byte E4M3 payload and no full-size FP32 tensor.

Formal counter-evidence: in both seed42 and seed1, every one of the eight
layer/member carry tensors is 100% zero at both 25k and 50k;
`carry_prequant_absmax=0`, main/logical error is identical, reduction ratio is
approximately one, and physical/logical norms coincide. Direct decoding of
both 50k checkpoints confirms all four `(2,4096,4096)` carry arrays contain no
nonzero values, so this is not a tensor-logger artifact. Gradients remain
nonzero and update NaN/Inf remain zero, distinguishing the failure from the old
unscaled-backward bug. Seed42 pnorm at 25k/50k/68k is
`782.90/1839.92/2605.80`, close to repaired resident
`733.80/1930.97/2608.39` and far above current-master
`472.85/720.93/about 851@65k`. Thus the currently launched treatment does not
implement effective long-horizon carry trajectory preservation, despite the
isolated checkpoint-next-update probe. Root cause must be resolved before any
formal return result is attributed to CARRY-FP8; no carry-gain variant is
accepted in this cycle.

Root cause evidence: on JAX/JAXLIB `0.6.0`, Flax `0.10.4`, driver
`580.95.05`, and RTX PRO 6000 Blackwell, CPU JIT and GPU eager correctly expose
the E4M3 round-trip error. GPU JIT instead reports a false near-identity
round-trip and zero residual even though the returned FP8 code is genuinely
quantized. A `4096x4096` plain probe reports main relative error `2.684e-8`,
zero carry, and zero prequant residual. Placing `jax.lax.optimization_barrier`
on the E4M3 code before any widening changes those values to `0.0264938`,
`16,777,205/16,777,216` nonzero carry entries, and prequant absmax `255.995`.
Stop-gradient, a bitcast detour, and a barrier on the FP32 candidate do not fix
it. The next implementation gate must therefore validate returned-code and
same-graph dequantization equality under GPU JIT; CPU-only arithmetic tests are
insufficient.

Corrective evidence: the shared quantizer now applies the main code-side
barrier, and the carry cast has its own barrier as well. The actual vmapped
carry kernel's GPU-JIT regression passes and makes in-graph main/logical
reconstruction exactly equal to a second dispatch over both returned FP8
buffers. A 4096² probe gives main/persisted-carry relative error
`0.00122739/3.2394553e-5`, `16,767,853/16,777,216` nonzero carry entries, and
zero saturation. The earlier `3.598577e-8` carry value was another unbarriered
same-graph false reading. The complete CPU suite passes `64/64`. Corrected
checkpoints record a new materialization semantic and refuse legacy resident/
lag resumes. A dedicated-GPU width-4096 consecutive update and fresh 5001-step
Dogs smoke both pass. The smoke and its actual recovery-next-update show
`35.0x–48.1x` persisted reconstruction improvement, nonzero carry, zero
saturation, finite resident gradients, and no NaN/Inf. Fresh tagged seed42/
seed1 formal runs are therefore authorized; pre-barrier runs cannot be resumed.
The fresh corrected runs started on GPU1/GPU2 at 04:48 with W&B
`7mdayzva/hxf61n84`. Their 25k/50k/75k formal gates all show persistent nonzero
carry, zero carry saturation, and zero tensor/update NaN or Inf. At 75k, carry
reduces main-write error by `42.85x–668.98x`; eval return rises monotonically
`72.14→125.16→185.73` for seed42 and `32.39→68.43→189.56` for seed1. The
seed42 75k critic pnorm is `1212.55`, far below repaired resident without carry
(`3046.30`) and its return is already in the same early-learning range as the
target-FP8-direct/FP32-storage and current-master controls (`220.44/178.43`).
The later evidence establishes more than an early trajectory: seed42 reached
450k eval/best `832.74` before an intentional replacement stop at env 454409,
while seed1 reached 450k `807.60` and then 500k final/best/tail-three
`782.86/807.60/793.28`, with zero update NaN/Inf. Seed42's 450k carry remained
nonzero and reduced reconstruction error `66.64x–1130.49x`. Across two seeds,
corrected CARRY-FP8 is therefore established as an effective learning method
under this Dogs protocol; exact final parity and cross-suite generality remain
separate questions.

The next accepted experiment combines online CARRY-FP8 with the existing
lag-coded FP8 target so both online and target residual-Dense weight states avoid
a full-size FP32 master. A width-4096 5001-step compatibility smoke passed and
the fresh seed42 formal run started on GPU1 at 2026-09-04 10:32 Asia/Shanghai
(W&B `5o2266m9`). The smoke showed nonzero online carry and target lag codes,
finite training, and complete checkpoints. It also measured `2.30%–2.61%` lag
quantization error, which is `4.58–5.20` times the tiny intended first-step EMA
update; whether that startup noise remains benign is the explicit 25k mechanism
and learning gate, not a pre-assumed conclusion.

The formal run passed that 25k gate with eval return `58.07` and no NaN/Inf in
925 tensor-stat records. Online carry stayed nonzero in 8/8 leaves and improved
logical reconstruction by `199.80x–430.67x`; target lag also stayed nonzero in
8/8 leaves with `0.845%–2.728%` lag quantization error. However, the error
relative to the tiny intended EMA update remained `1.68–5.43`, and update
cosine ranged from `-0.830` to `0.857`. The experiment therefore continues to
50k/75k: the representation is operational and learning has started, while the
long-horizon effect of noisy target increments is still an open empirical risk.

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

Under the revised direction, the conditional paper story is: different persistent RL states fail for different temporal reasons. Slow target EMA needs a lag-relative representation; online Adam writes may instead require update-residual, scale-decoupled, or function-aware handling; optimizer moments are a subsequent state class. This becomes an RL training-state framework only if the staged evidence shows distinct failure mechanisms and the combined system trains stably. Merely assembling known quantizers remains insufficient.
