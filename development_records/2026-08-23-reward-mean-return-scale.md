# Reward-mean return scale

## Motivation

The paper-aligned preset allowed the target critic's time-limit bootstrap to update the reward normalizer's `Gbar`. A Dogs diagnostic run showed a feedback path: normalized Q saturated near 10 by step 7,000 while `Gbar` expanded much faster than observed episode returns. The goal is to break only this feedback path without adding clipping thresholds, windows, EMA state, or FP8-facing quantities.

## Changes

- `paper_alignment=true` now resolves `return_bootstrap=reward_mean` while retaining L1 task embeddings and per-task empirical entropy correction.
- Explicit `--return_bootstrap=critic` remains available for reproducing the earlier behavior and one-factor ablations.
- The normalizer equation, critic TD targets, checkpoint schema, and recorder are unchanged.

## Evaluation

- Check `resolved_return_bootstrap` in `config.yaml`; the expected value for the updated preset is `reward_mean`.
- Compare `return_range_by_task`, `reward_denominator_by_task`, normalized reward statistics, critic Q saturation, and train/eval return against the critic-bootstrap run.
- A short DMC Dogs validation should confirm that the run starts cleanly and that `Gbar` no longer follows the critic's saturated boundary prediction.

## Pitfalls

- This changes only time-limit return-scale estimation; it does not remove the target critic bootstrap used by ordinary TD learning.
- DMC `success=0` is not the evaluation criterion; use return.
- A single short run validates the feedback mechanism, not final multi-seed performance.

## Validation and limits

- `JAX_PLATFORMS=cpu python -m unittest -v tests.test_paper_alignment`: 11 tests passed.
- GPU1 DMC Dogs seed-42 validation completed 10,000 steps with the recorder enabled; W&B run [`7egh3qkh`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/7egh3qkh). Evaluation, checkpoints, profiling, and tensor statistics were disabled for this mechanism check.
- At step 10,000, normalized critic Q was still saturated near 10, but per-task `Gbar` remained `[19.1, 16.8, 11.0, 5.7]`. The earlier critic-bootstrap plus empirical-entropy run reached approximately `[294.6, 284.8, 287.3, 287.2]` at the same step.
- This confirms scale-path decoupling in one short seed-42 run. Final return quality and variance still require the normal full-length, multi-seed comparison.
