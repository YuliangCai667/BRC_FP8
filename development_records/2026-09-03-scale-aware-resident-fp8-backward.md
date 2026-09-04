# Scale-Aware Resident FP8 Backward

## Motivation

The stopped fixed-anchor run kept the four resident-kernel norms near their
initial radii, but all `68/68` recorded resident-kernel gradients were exactly
zero.  The naive resident run had the same `80/80` pattern, while all `80/80`
matched FP8-direct gradients were nonzero.  The preceding resident operator was
therefore forward-correct but backward-invalid: generic autodiff exposed a
scale-suppressed cotangent to FP8 before analytical scale recovery.

## Changes

- `ResidentFp8Dense` retains current-amax E4M3 activation quantization and the
  resident E4M3 kernel code, but now calls Flax's `quantized_dot` custom VJP.
  That existing primitive dynamically scales output cotangents with delayed
  amax metadata, casts them to E5M2, performs native FP8 transpose GEMMs with
  FP32 accumulation, and returns physical FP32 kernel/input gradients.
- Each of the four logical resident Dense layers stores
  `output_grad_scale` and `output_grad_amax_history` per ensemble member.  The
  history length is the same `fp8_amax_history_length` used by FP8-direct.
- Critic and actor updates differentiate the resident forward with respect to
  both physical FP32 parameters and FP8 metadata.  Only output-gradient scale
  and history updates are merged; resident kernel scales, optional fixed
  anchors, AdamW moments, target critic behavior, and writeback semantics are
  unchanged.
- Checkpoint state includes the new backward metadata.  Resume validation
  records `resolved_online_fp8_backward` and rejects a legacy
  `generic_autodiff_unscaled_fp8` resident checkpoint, preventing silent
  continuation across different backward algorithms.
- The first repair experiment explicitly disables fixed-anchor
  canonicalization.  Its only intervention is the scale-aware backward on top
  of naive resident persistence.

## Validation

- Complete CPU `tests.test_fp8` suite: 27 tests executed; the only initial
  error was an over-strict check on the already supported FP32-to-FP8-direct
  checkpoint migration.  The validator was narrowed to skip the new backward
  field only for that transition, and its focused regression then passed.
  Three resident-focused tests, including persistent output-gradient metadata
  and nonzero kernel/input-gradient checks, passed together.
- Independent GPU probe at activation scale `0.5`, kernel scale `2e-4`, width
  512 and output-gradient sigma `1e-3/1e-2`: repaired resident versus
  FP8-direct kernel-gradient cosine/L2 ratio is exactly `1.0/1.0`; input-gradient
  cosine is effectively one and L2 ratio differs by less than `1.1e-7`.
  Both repaired gradients have zero fraction `0.0`.  Relative to FP32, cosine
  is about `0.9986` and L2 ratio about `0.9985–0.9988`.
- A matched batch of 256 transitions from the completed Dogs seed42 replay was
  evaluated at common physical weights.  All four repaired resident kernel
  gradients are nonzero; versus FP8-direct their cosine is
  `0.99927–0.99994`, L2 ratio `0.99977–1.00081`, and zero fractions are the same
  order.  `dQ/da` has cosine `0.98988`, L2 ratio `1.00182`, and zero fraction
  `0.0`.
- End-to-end GPU smoke completed 200 env steps / 203 learner updates without
  update NaN/Inf.  At step 150 the four resident-kernel gradient L2 norms are
  `0.36380/0.41178/0.29192/0.33431`, with zero fractions
  `0/0.00883/0/0.02454` and no non-finite entries; actor gradient telemetry is
  also finite and nonzero.
- Machine-readable probe results are retained in
  `scale_aware_backward_independent_probe.json` and
  `scale_aware_backward_real_batch_probe.json`.

## Formal experiment and limits

The first formal run is Dogs seed42 directly to 500k, width 4096, batch 1024,
two updates per env step, online `fp8_resident`, target `fp8_direct`, and
`fp8_resident_canonicalization=false`.  The initially planned separate 25k gate
was cancelled by the user; normal 25k eval/tensor diagnostics remain in the
formal run.  After short startup acceptance, the run is not automatically
stopped or continuously monitored; the user will stop it manually if needed.
Seed1 starts only after seed42 completes.

This repair is required FP8 training correctness, not a novelty claim.  It
does not establish long-horizon reward parity or decide whether fixed-anchor
canonicalization is useful once the backward operator is valid.
