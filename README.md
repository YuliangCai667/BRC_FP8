# Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners

https://arxiv.org/pdf/2505.23150

This branch contains the implementation of the BRC algorithm.

The current actor deployment experiment uses per-tensor E4M3 forward / E5M2
backward with true CARRY contributions in both critic and actor bodies.
`bash scripts/run_actor_qat_fp8_export.sh` runs the fixed seed42 protocol and
automatically aligns/exports the actor as W8A16. For the FP32-actor baseline or
isolated critic change, use `bash scripts/run_hybrid_critic.sh legacy` or
`bash scripts/run_hybrid_critic.sh hybrid`. See the
[implementation and validation record](development_records/2026-09-09-hybrid-carry-actor-export.md)
and [protocol configuration](configs/actor_qat_fp8_export_v1.yaml).

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
master parameters, and optimizer state remain FP32. Use
`--critic_precision=fp8_direct`; per-tensor scaling uses a 1,024-entry amax
history by default.

For the naive online-resident baseline, use
`--critic_precision=fp8_resident`. The same four residual kernels persist as
E4M3 codes with one FP32 current-amax scale per ensemble member and are consumed
directly by the FP8 GEMMs. Each AdamW step transiently dequantizes the current
physical weight, forms the FP32 candidate using FP32 moments, and immediately
writes new E4M3 codes and scales. The candidate is not returned as model state,
saved in checkpoints, or retained for the next update; there is no full-size
FP32 master or residual. Other Critic parameters remain FP32 in this first
isolation stage.

At tensor-stat intervals, the last real update reports only scalar write
diagnostics: update norm retention and cosine, swallowed/code-unchanged
fractions, weight-relative error, scale movement, a fixed-old-scale
counterfactual, and same-batch expected-Q/JS/loss error. Raw resident codes are
not copied into a second tensor-stat category; checkpoints retain the payload,
while code-change fraction captures the write behavior needed for diagnosis.
Checkpoint manifests report FP8 payload bytes, FP8 metadata bytes, optimizer
bytes, total persistent-state bytes, and the metadata fraction.

The same four logical Dense kernels in the target critic can instead remain
device-resident in E4M3 with `--target_critic_precision=fp8_resident`. Each
ensemble member has an independent FP32 per-tensor scale. Target activations are
dynamically quantized to E4M3, GEMMs accumulate and return FP32, and the original
`tau=0.005` EMA is computed transiently in FP32 before the result is immediately
requantized. There is no persistent FP32 target master, Kahan compensation,
stochastic rounding, block scaling, or delayed target update in this mode.

For a compute-only target ablation, use
`--target_critic_precision=fp8_direct`. All target parameters and the EMA remain
FP32, while the same four residual Dense operands are quantized to E4M3 for
native FP8 GEMMs. Input and kernel per-tensor delayed-scaling histories advance
once per training bootstrap forward; target diagnostics and evaluation are
read-only, and no target backward pass is introduced.

For lag-coded persistent target state, use
`--target_critic_precision=fp8_lag`. The four residual kernels store only
`target - online` as E4M3 codes with one current-amax FP32 scale per ensemble
member. Their target value is reconstructed transiently as FP32
`online + dequantized_lag`, then passed through the same delayed-scaling native
FP8 Direct forward. Other target leaves retain the original FP32 EMA. There is
no FP32 master for the four target kernels and no derived FP8 target cache.

The controlled target experiments are:

- C: `--critic_precision=fp32 --target_critic_precision=fp8_resident`
- D: `--critic_precision=fp8_direct --target_critic_precision=fp8_resident`
- Target-forward-only: `--critic_precision=fp8_direct --target_critic_precision=fp8_direct`
- D-lag: `--critic_precision=fp8_direct --target_critic_precision=fp8_lag`

The first online-resident Dogs arm keeps target parameters and EMA FP32 while
retaining target FP8 GEMMs through `target_critic_precision=fp8_direct`. Run it
manually as follows; the script does not queue or detach the job:

```bash
scripts/run_dogs_online_fp8_resident.sh GPU_ID 42
```

FP8-target checkpoints support same-mode recovery only. Converting an existing
FP32 target checkpoint to direct, resident, or lag-coded FP8 is intentionally
unsupported in these mechanism tests.

### Offline target-state simulation

The offline simulator replays a complete healthy recovery checkpoint without
creating an environment. One teacher follows the original BRC learner update,
while lag-coded, scaled Kahan-momentum, naive per-tensor, naive block-scale, and
lattice-matched interleaved block targets consume the same online-Critic
trajectory without feeding back into it. The Kahan baseline follows the FP16
SAC formulation with a `C=1e4` scaled target buffer, but stores both that
buffer and its compensation in E4M3 with independent current-amax scales:

```bash
python scripts/simulate_fp8_target_updates.py \
  --checkpoint=runs/offline_inputs/dogs_target_fwd_s42_step100000 \
  --num_updates=50000 \
  --block_size=128 \
  --diagnostic_interval=500 \
  --output_root=runs/offline_target_simulations \
  --run_id=dogs_s42_step100k_lag_interleaved
```

Batch size, task order, seed, network width, update ratio, and Critic precision
come from the checkpoint manifest. The replay buffer and reward-normalizer
statistics remain frozen. Results, the fixed 256-sample probe, and JSONL metrics
are written below `runs/` and are not version controlled.

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
