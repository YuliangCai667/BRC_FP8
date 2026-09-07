# Pure training branch

Base: `9904a05`, the CARRY-AdamW commit on `codex/blackwell-fp8-direct`.
Branch: `codex/pure-training`.

## Change

The research loop computes forward/gradient/weight/moment diagnostic trees at
tensor-stat intervals and keeps `diagnostic_trees` in `main` until the next
assignment. The dictionary includes reconstructed FP32 weights and references
to optimizer state that later training updates replace. Reducing those tensors
to scalar logs does not release the original dictionary. The loop's `tree`
variable can also retain the last category after iteration.

This branch removes the tensor-stat/probe and synchronized component-profile
branches from `train.py`, and disables the system-monitor thread. It retains
training updates, basic losses/entropy/mean target Q, episode returns, evaluation,
checkpoints, source provenance and lightweight allocator counters at the normal
scalar sampling boundary. The original algorithm citation remains in README.

`TrainingBRC` uses the same BRC model, optimizer, scaling/lag state and checkpoint
format. It fixes `collect_update_diagnostics=False`, uses the original compiled
update, and filters the returned scalar dictionary in Python. Per-task entropy
and count statistics used by normalization follow the original update unchanged.
The original scalar norm/Q-range reductions still run within that compilation;
the full tensor diagnostic paths are absent from the training entry.
No buffer donation, new quantizer, fused optimizer,
activation rematerialization, or changed batch/evaluation protocol is introduced.

The historical diagnostic helpers remain available to offline tools and the
standalone memory benchmark. They are not called by `train.py`. The canonical
launcher is `bash scripts/run_dogs_pure_training.sh GPU_ID SEED [MAX_STEPS]
[RESUME_FROM]`; it selects an independent fresh run identity by default.

## Validation

- The extracted Adam base passed its independent CPU suite: 74/74 tests.
- Two new tests compare FP32-moment and
  CARRY-moment learners across consecutive UTD2 updates, checking all model,
  optimizer, FP8/target, RNG and entropy state plus retained metrics against the
  research implementation. They also check research-to-pure checkpoint loading,
  the next update, and a pure checkpoint round trip.
  The final implementation passed both CPU tests in 91.069 s.
  Both GPU2 tests also passed in 305.737 s with `JAX_PLATFORMS=cuda`, checking
  FP8 codes exactly and other floating values at the original tolerances.
  Logs: `runs/pure_training_validation/{cpu,gpu}_state_equivalence_final.log`.
- A real `cheetah-run` training-entry smoke used width16/batch8 and completed
  10 learner updates, evaluation, and complete analysis/recovery checkpoints.
  Restoring that recovery completed another two updates; the final implementation
  restored again and completed two more, through environment step14/update step15.
  Tensor-stat JSONL stayed
  empty; no probe batch or system-monitor CSV was created. These are integration
  checks, not learning-quality results.
- Python compilation, launcher syntax and diff whitespace checks passed.

An initial implementation filtered the scalar outputs inside a new outer JIT.
Although CPU comparison and GPU checkpoint restoration passed, GPU consecutive
updates exposed one differing FP8 code (1/512 elements in the failing leaf).
The final implementation therefore filters scalars after the original JIT has
returned. It keeps the original numerical compilation boundary; no equality
tolerance was relaxed. The initial attempt and its pure-memory measurement are
superseded by validation of this final implementation.

## Memory protocol

`scripts/benchmark_training_memory.py` runs outside the training entry. Each arm
uses a fresh process on shared GPU2, no preallocation, and the same 0.15 allocator
fraction (limit approximately 14.25 GiB). Inputs are fixed synthetic Dogs-shaped
batches: 4 tasks, 223 observation features, 38 actions, width4096, batch1024,
UTD2, CARRY weights, LAG target, CARRY-AdamW, and 16 actual learner updates.

The `research` arm collects full diagnostics after eight updates, summarizes them
and retains the tree across subsequent updates, matching the original loop's
lifetime. The `release` arm instead deletes the tree after scalar reduction. The
`pure` arm never creates it. Counters come from the process's JAX allocator after
synchronization, rather than device-wide NVML memory. Timing is descriptive
because GPU2 is shared. These short measurements do not cover a 500k run,
evaluation or checkpoint peaks, and do not measure RL return.

Raw reports and logs: `runs/pure_training_validation/`. CPU entry smoke and
recovery: `runs/cheetah-run/pure_training_entry_smoke/`.

## Measured results

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 580.95.05;
JAX/jaxlib 0.6.0, Flax 0.10.4, Optax 0.2.5. Each counter was sampled after
synchronizing model state. The original counter names end in `_mb` but use MiB.

| Mode | Active after 16 updates (MiB) | Allocator peak (MiB) | Active / peak (GiB) |
| --- | ---: | ---: | ---: |
| Research, retain diagnostic tree | 4597.805 | 6746.773 | 4.490 / 6.589 |
| Research, delete tree after summarizing | 998.485 | 5057.491 | 0.975 / 4.939 |
| Pure training, no diagnostic tree | 998.562 | 3177.333 | 0.975 / 3.103 |

In the release arm, deleting the diagnostic tree immediately reduced active
allocation from 4017.490 to 992.909 MiB: 3024.581 MiB (2.954 GiB) had been kept
alive by that reference. Later training in the retained arm also keeps old model
and optimizer buffers alive through the tree. Relative to that arm, pure training
reduces final active allocation by 3599.243 MiB (78.28%) and the observed peak by
3569.440 MiB (52.91%). These percentages apply to this short diagnostic-lifetime
comparison, not to the 500k experiment or Adam compression alone.

The release arm still reaches the diagnostic-computation peak. Allocation samples
also depend on buffer lifetimes, so peak and active usage are reported separately.
`bytes_in_use` and its peak are JAX allocator accounting, not NVML process totals
or just the sum of model tensors. A cached allocator pool may remain visible in
`nvidia-smi` after a buffer is released. Adam's separate persistent-state saving
remains 504 MiB; removing diagnostics does not increase that algorithmic saving.

The [machine-readable evidence](evidence/2026-09-07-pure-training-memory.json)
contains every sampled point, allocator settings, versions, and source hashes.
To reproduce the measured environment, run the benchmark command in README with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`, `CUDA_ROOT=/usr/local/cuda-12.8`,
`CUDA_HOME=/usr/local/cuda-12.8`, CUDA 12.8 on `PATH`, and `LD_LIBRARY_PATH` unset.
