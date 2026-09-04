# Current-Amax FP8 / FP32-Master Causal Control

## Motivation

The repaired resident and fixed-anchor runs both remain below the matched
FP8-direct curve. Existing probes reject recurring anchor drift, zero kernel
gradients, target lag, code round-trip corruption, non-finite arithmetic, and
a catastrophic one-step function-write error. The remaining comparison still
mixes masterless E4M3 parameter writes with current-amax compute, while the
healthy FP8-direct control uses FP32 parameters and delayed scaling. A stricter
single-factor control is required before assigning the reward gap.

## Changes

Added online precision mode `fp8_current_master`. Its four residual Dense
layers keep FP32 kernels and FP32 AdamW state, quantize activation and kernel
with independent current-amax E4M3 scales on every forward, and reuse the same
Flax `quantized_dot` / delayed-amax E5M2 custom backward as repaired resident
training. It has no resident code/scale write, fixed anchor, persistent FP8
shadow, or input/kernel amax history. The target remains the existing
FP8-direct compute path with FP32 parameters and EMA.

A dedicated Dogs launcher records the mode explicitly. Training metadata now
distinguishes current-amax FP32-master compute from resident storage, and
checkpoint loading requires its output-gradient FP8 metadata.

## Evaluation

The causal comparison is Dogs seed42 with the same width, batch, update ratio,
target mode, paper-alignment settings, evaluation seeds, and cadence as the
resident arms and completed FP8-direct control. Primary readouts are eval return
and critic parameter norm at 25k/50k/100k; task-wise return and gradient
metadata are secondary checks.

Run: `brc_dmc_dogs_online_fp8_current_master_s42-20260903-184747`, GPU1,
W&B `vgd8apty`, planned through 150k.

## Pitfalls

This is a causal diagnostic, not a low-memory method: its full online kernel
master remains FP32 and the weights are re-quantized on every covered forward.
Agreement with FP8-direct would implicate masterless writeback but would not by
itself choose an error-feedback or alternate resident optimizer design. A
single seed can strongly separate large early curves but cannot establish final
multi-seed parity.

## Validation and limits

- Targeted unittest passed: initial current-master and resident logits are
  elementwise identical, all parameters/floating AdamW leaves are FP32, three
  learner updates change parameters, and all four output-gradient histories
  advance.
- Python compilation, launcher `bash -n`, and `git diff --check` passed.
- tmux process, local run directory, and W&B run were present after launch.
- The 25k evaluation completed at return `49.53`, versus repaired/fixed/direct
  `45.94/34.78/19.92`; all four kernel gradients are nonzero and update
  NaN/Inf is `0/0`. The serialized FP32 layer norms and whole-critic pnorm track
  direct much more closely than either resident intervention. This early point
  is encouraging but precedes clear curve separation; 50k/100k remain pending.
- In a deterministic width-16 paired probe with physical online parameters,
  target state, and zero Adam state exactly aligned at step 0, the first
  resident write leaves logits identical but creates `0.000897` maximum kernel
  relative L2. By update 10, kernel/logit relative L2 reaches
  `0.008423/0.027712`. Raw values are preserved in
  `current_master_aligned_pair_probe.json`.
- At 50k, current-master return is `113.17` versus direct/repaired/fixed
  `120.40/101.41/87.12`; all four task returns individually follow direct.
  Current/direct critic pnorm is `720.93/756.40`, versus
  `1930.97/412.72` for repaired/fixed. Four per-layer physical norms give the
  same assignment, and update NaN/Inf remains `0/0`. This is the first closed-
  loop evidence supporting masterless writeback as the main divergence source;
  later points remain pending.
- On the healthy direct 500k FP32 weights and saved 256-sample probe,
  current-amax versus delayed-amax gives logits relative L2 `0.004814`,
  expected-Q MAE `0.003916`, kernel-gradient cosine `0.97565–0.99818`, and
  L2 ratio `0.99079–1.00024`. Thus compute-policy mismatch remains a plausible
  secondary cumulative effect, not a local operator collapse. Raw values are
  in `current_vs_delayed_direct500_probe.json`.
- At 75k, current-master return is `178.43`, versus repaired/fixed/direct
  `129.44/129.26/220.44`. It improves over both resident arms by about 38%, and
  every task is individually higher, while a 19.1% residual direct gap remains.
  This supports masterless writeback as the dominant cause without claiming it
  is the only source of single-seed trajectory variation.
- At the 100k decision gate, current-master return is `244.89`, versus
  repaired/fixed/direct `157.98/134.44/340.61`; all four tasks improve over
  both resident arms. Current/direct critic pnorm is `1090.48/1129.21`, versus
  `4171.12/519.08` for repaired/fixed. Across every 1k sample from 5k–100k,
  current-to-direct critic-pnorm SMAPE is `3.63%`, against `77.11%/47.97%` to
  repaired/fixed; Q-prediction and actor-pnorm trajectories are also closest to
  direct. The causal control therefore validates masterless writeback as a
  large source of the failed resident trajectory, while its remaining `28.1%`
  direct return gap validates current/delayed scaling as a secondary cumulative
  factor. Analysis and full replay recovery checkpoints saved successfully;
  the process continues normally to 150k.
