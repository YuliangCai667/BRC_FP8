# FP8 Target Forward with FP32 Storage

## Motivation

The resident-target C/D runs combine two effects: FP8 target forward computation and persistent E4M3 target storage with requantized EMA. This change adds the missing compute-only target arm so those effects can be separated. The online Critic remains `fp8_direct`; the target Critic uses native FP8 only for the same four residual Dense GEMMs, while every target parameter and the original `tau=0.005` EMA remain FP32.

## Changes

- Added `target_critic_precision=fp8_direct` alongside `fp32` and `fp8_resident`. Target input/output projections, LayerNorm, bias, embedding, all parameter storage, and EMA arithmetic remain FP32.
- Reused Flax's direct E4M3 forward path and delayed per-tensor scaling. Because the target has no backward pass, the bootstrap training forward explicitly writes only input/kernel scale and amax history; output-gradient metadata stays unchanged. Diagnostics, evaluation, and reward-normalizer queries are read-only.
- Threaded the updated target metadata through every learner update before the FP32 EMA. Resident-target detection now uses the explicit precision mode instead of treating every FP8 metadata tree as resident storage.
- Added target-forward FP8 diagnostics, an FP32-reference expected-Q/JS comparison, same-mode checkpoint recovery, resolved storage/compute metadata, and tests for FP32 storage, metadata progression, EMA, checkpoint replay, and mode compatibility.

## Evaluation

This arm is compared in two directions:

- B → target-forward-only isolates target bootstrap FP8 computation while online FP8 and target FP32 storage are fixed.
- target-forward-only → D isolates persistent target E4M3 storage and requantized EMA while online/target FP8 computation is fixed.

The formal run uses the established DMC Dogs seed-42, width-4096, 500k-step protocol. Learning/eval and numerical diagnostics are admissible; performance is not, because GPU2 is shared with another training process.

## Pitfalls

- `fp8_direct` target parameters are FP32 and are requantized for every residual GEMM. It must not be described as FP8 target storage or as removing RHS quantization cost.
- Target output-gradient E5M2 metadata is intentionally not advanced because there is no target backward pass. Only bootstrap training forwards update input/kernel delayed-scaling state; changing logging/eval frequency does not change training state.
- Target precision conversion on checkpoint resume remains unsupported. Only same-mode target recovery is accepted.

## Validation and limits

- CPU regression: all 30 repository tests passed; the strengthened target-metadata and target-only checkpoint checks also passed independently. Checkpoint round-trip reproduces the next update.
- Blackwell GPU2, CUDA 12.8, compute capability 12.0: target-only StableHLO contained six Dense dots, with exactly four residual `E4M3 x E4M3 -> FP32` dots and FP32 input/head dots. Target parameter leaves were all FP32; optimized HLO contained E4M3 Triton GEMM paths and no E5M2 target-backward path. Outputs were finite, input/kernel histories advanced, and output-gradient histories stayed zero.
- GPU2 smoke: `cheetah-run`, seed 0, width 512, batch 256, 200 steps and 203 learner updates completed with `run_finished`; every sampled update had zero NaN/Inf. Step-150 tensor recording produced 298 entries. The aggregate FP8-vs-FP32 target expected-Q MAE was `0.0704`, signed bias `-0.0441`, and probability JS divergence `9.09e-4`; these are smoke diagnostics, not a learning conclusion.
- The formal width-4096 Dogs result is still required before judging target-forward accuracy or the storage/EMA mechanism.
