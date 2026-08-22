# Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners

https://arxiv.org/pdf/2505.23150

This branch contains the implementation of the BRC algorithm.

## Example usage

To run the BRC algorithm in a single task mode, just pass a single task name to the `env_names` variable:

`python3 train.py --env_names=dog-run`

By passing a list of task names, multi-task mode will be enabled. 

## Experiment records and checkpoints

Training now writes a local, append-only run directory under
`runs/<env_names>/<run_id>/`. It contains per-episode JSONL, periodic training and
evaluation metrics, tensor summaries, system metrics, reproducibility metadata,
and atomic checkpoints. W&B is an optional mirror; local recording continues if
W&B or GPU monitoring is unavailable.

The low-overhead defaults sample training scalars every 1,000 steps, run tensor
diagnostics every 25,000 steps, save parameter-only analysis checkpoints every
50,000 steps, and save recovery checkpoints every 100,000 steps. Set an interval
to `0` to disable that category. Recovery checkpoints include only the valid
portion of the replay buffer and may be resumed from either the checkpoint itself
or the run directory:

```bash
python train.py \
  --env_names=METAWORLD_ALL \
  --resume_from=runs/METAWORLD_ALL/<run_id>
```

Environment simulator state is intentionally not serialized. On resume, all
environments and unfinished episodes are reset while model, optimizer, RNG,
normalizer history, episode counts, and replay data are restored.

Resource logs distinguish physical-device allocation from live JAX allocations:
`system_metrics.csv` reports device-wide NVML usage, while `train_metrics.jsonl`
and W&B report JAX `bytes_in_use`, peak usage, reserved allocator memory, and the
device limit. These allocator counters are sampled only at the existing metric
synchronization boundary. For FP32/FP8 memory comparisons, use the same allocator
settings for both runs; `XLA_PYTHON_CLIENT_PREALLOCATE=false` makes the physical
device metric easier to interpret.

## Blackwell FP8 critic training

The online critic can run the four Dense layers inside its two residual blocks
with native FP8 matrix multiplication. Input/output projections, LayerNorm,
the target critic, master parameters, and optimizer state remain FP32. Use
`--critic_precision=fp8_direct`; per-tensor scaling uses a 1,024-entry amax
history by default.

For the paper-aligned MetaWorld configuration, run:

```bash
scripts/run_metaworld_fp8_direct.sh GPU_ID SEED
```

The script selects CUDA 12.8 through `CUDA_ROOT`, `CUDA_HOME`, and `PATH`. Start
it from a shell that does not retain an older CUDA directory in
`LD_LIBRARY_PATH`. Resuming an FP32 recovery checkpoint in `fp8_direct` mode is
supported for exploration: optimizer and replay state are restored while FP8
scales start fresh, and a `precision_transition` event marks the boundary. Such
a run is not equivalent to FP8 training from initialization.

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
