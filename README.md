# BRC: pure training with CARRY, LAG and compressed AdamW

This branch, `codex/pure-training`, provides a training entry without the research
pipeline's tensor diagnostics. It is based on Adam commit `9904a05` from
[`codex/blackwell-fp8-direct`](https://github.com/YuliangCai667/BRC_FP8/tree/codex/blackwell-fp8-direct).
The original BRC algorithm is described in [Bigger, Regularized, Categorical](https://arxiv.org/abs/2505.23150).

## Run Dogs

```bash
bash scripts/run_dogs_pure_training.sh GPU_ID SEED [MAX_STEPS] [RESUME_FROM]
```

Defaults: 500k environment steps, width 4096, batch 1024, two learner updates
per step, 5k warmup, residual-only CARRY-FP8 online weights, lag-coded target,
block128 CARRY-AdamW moments, high-precision input/head, and the existing
paper-aligned reward normalization. GPU selection is explicit. For local-only
logging, prefix the command with `BRC_LOG_TO_WANDB=false`.

The training entry retains:

- Actual actor/critic/temperature/target updates and FP8 scaling metadata.
- Actor/critic/temperature losses, entropy, mean target Q, episode returns,
  and lightweight JAX allocator counters every 1,000 environment steps.
- Deterministic 10-episode-per-task evaluation every 25k steps.
- Analysis checkpoints every 50k and recovery checkpoints every 100k, with
  optimizer, RNG, reward normalization, replay, and episode bookkeeping.
- Local config, source provenance, scalar JSONL logs, and optional W&B mirroring.

`train.py` has no tensor-stat, probe-batch, or synchronized component-profile
branch and starts no background GPU polling thread. `TrainingBRC` disables tensor
diagnostics and logs only the basic scalars. It keeps the original compiled update
boundary, including its scalar reductions, to preserve GPU FP8 rounding behavior.
No diagnostic tree or old optimizer snapshot is retained by the training loop.
FP8 amax/scales, logical CARRY reconstruction, and lag reconstruction remain part
of the numerical method.

Existing research helpers are available for offline validation and the separate
memory benchmark; they are not invoked by the training entry. Older diagnostic
launchers are historical tools; use the launcher above for this branch.

## Resume

Same-precision research and pure-training checkpoints share the same model and
optimizer serialization format. Pass the recovery checkpoint as the fourth
launcher argument. The existing resume policy continues writing the source run
directory and records the changed runtime/config. Environment simulator state
is reset, while model, optimizer, RNG, replay, and normalization state restore.
For independent experiments use fresh run IDs rather than resuming a completed
research run.

## Validation and memory measurement

```bash
JAX_PLATFORMS=cpu python -m unittest tests.test_training_learner
```

The tests compare required metrics, parameters, optimizer state, target/scaling
state, and entropy bookkeeping against the research learner across consecutive
updates. They also check research-to-pure checkpoint loading and the next update.

Run each memory mode in a fresh process with identical GPU/allocator settings:

```bash
CUDA_VISIBLE_DEVICES=GPU_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/benchmark_training_memory.py --mode=pure \
  --output=runs/pure_training_validation/pure.json
```

Repeat with `--mode=research` to retain full diagnostic tensors as the original
loop does, and `--mode=release` to release them immediately after summarization.
The benchmark uses fixed synthetic Dogs-shaped inputs (4 tasks, 223 observation
features, 38 actions), width 4096, batch 1024, and 16 actual learner updates.
It measures memory, not return or a full RL experiment. Compare `bytes_in_use`
separately from `peak_bytes_in_use`: releasing tensors can lower active usage
while the allocator retains its pool. The benchmark is never started by training.

Measured results and validation details are recorded in
[the branch change record](development_records/2026-09-07-pure-training.md).

## Citation

If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@article{nauman2025bigger,
  title={Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners},
  author={Nauman, Michal and Cygan, Marek and Sferrazza, Carmelo and Kumar, Aviral and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2505.23150},
  year={2025}
}
```
