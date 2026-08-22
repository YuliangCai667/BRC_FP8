# Blackwell FP8-direct critic training

## Motivation

For the paper-aligned MetaWorld critic, the four 4096-by-4096 Dense layers in the two residual blocks account for approximately 98.94% of Dense multiply-adds. They are therefore the highest-value FP8 target, while the observation/action interface, categorical value head, and TD bootstrap path remain more useful as FP32 numerical boundaries.

## Changes

- Added `critic_precision={fp32,fp8_direct}` with FP32 as the compatibility default. FP8-direct applies only to the online critic's four residual-block Dense layers; parameters, Adam state, LayerNorm, residual additions, input/output projections, actor, and target critic remain FP32.
- Integrated Flax `Fp8DirectDotGeneralOp`: E4M3 forward operands, E5M2 output gradients, FP32 accumulation, per-tensor scaling, and a configurable 1,024-entry amax history. Each critic ensemble member owns separate scale/history state.
- Critic and actor losses both update FP8 metadata. The actor path differentiates only actor variables and critic metadata, so it does not perform a second critic weight update.
- Checkpoints now preserve FP8 metadata and still decode the historical FP32 model schema. FP32-to-FP8 resume restores training state but starts fresh scales, appends to the original run, and records a `precision_transition` boundary.
- Fixed lifted `intermediates` collection for critic activations. Tensor diagnostics now include scale, current/history amax, and saturation-risk ratio for input, kernel, and output-gradient quantization.
- Added a CUDA 12.8 MetaWorld runner and pinned JAX 0.6.0, Flax 0.10.4, and Optax 0.2.5.

## Evaluation

- Confirm `critic_precision=fp8_direct`, `fp8_amax_history_length=1024`, CUDA paths, and `ptxas 12.8.61` in run metadata.
- Use `fp8/*/{scale,current_amax,history_amax,saturation_risk_ratio}` together with `activations/critic/*`, actor/critic gradient norms, NaN/Inf counts, throughput, memory, and return.
- Treat a saturation-risk ratio above one as evidence that the current tensor exceeds the range selected from the preceding history. Use layer-local evidence before considering row/column block scaling.

## Pitfalls

- `fp8_direct` is a mixed-precision configuration, not end-to-end FP8. The target critic is deliberately FP32.
- An FP32-to-FP8 resumed run contains two precision phases. The event boundary and `fp8_direct_enabled` metric must be used when reading the curve; it is not equivalent to FP8 training from initialization.
- The runner sets `CUDA_ROOT`, `CUDA_HOME`, and `PATH` only. A parent shell retaining an older CUDA directory in `LD_LIBRARY_PATH` can still mix runtimes; the Codex validation shell's inherited CUDA 12.4 entry was removed outside the script.
- The 200-step smoke run completed no full episode, so episode-return NaNs there indicate missing episodes rather than numerical failure.

## Validation and limits

- CPU regression: 23 tests passed in 44.7 seconds. FP8 checkpoint round-trip, deterministic next update, historical FP32 checkpoint conversion, activation capture, and both metadata-update paths are covered.
- Blackwell GPU: 5 FP8 tests passed in 92.7 seconds on an RTX PRO 6000 (compute capability 12.0). Optimized HLO contained 14 E4M3 references, 4 E5M2 references, one `__triton_gemm`, and an FP32 matrix output; forward/backward values and metadata were finite.
- FP32 compatibility: the initialized critic parameter SHA-256 matched the paper-alignment source branch exactly.
- A 200-step `cheetah-run` smoke test completed 203 updates with `update_nan_count=0` and `update_inf_count=0`. It wrote 6 critic activation tensors and 48 FP8 diagnostic rows; steady post-JIT throughput reached approximately 334-380 updates per second at width 512 and batch 256.
- No 50-task, width-4096, 500k-step quality or speed comparison was run. Final return, variance, and production-scale speedup remain unverified.
