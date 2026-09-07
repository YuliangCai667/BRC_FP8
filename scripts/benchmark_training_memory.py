#!/usr/bin/env python3
"""Isolated full-width learner memory comparison, outside the training entry.

Run each mode in a fresh process with identical allocator settings. This uses
fixed synthetic batches to measure allocation, not to evaluate RL quality.
Research mode reproduces the training loop's retained diagnostic tree; release
mode deletes that tree after summarization. Neither changes model arithmetic.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.training_learner import TrainingBRC
from jaxrl.experiment import collect_jax_memory_stats, summarize_tree
from jaxrl.utils import Batch


def run(args):
    rng = np.random.RandomState(42)
    cls = TrainingBRC if args.mode == 'pure' else BRC
    agent = cls(42, np.zeros((1, 223), np.float32), np.zeros((1, 38), np.float32),
                num_tasks=4, width_critic=args.width, updates_per_step=2,
                task_embedding_norm='l1', critic_precision='fp8_resident',
                fp8_resident_carry=True, target_critic_precision='fp8_lag',
                critic_optimizer_state='fp8_carry')
    shape = (2, args.batch_size)
    batch = Batch(rng.normal(size=shape + (223,)).astype(np.float32),
                  rng.uniform(-1, 1, size=shape + (38,)).astype(np.float32),
                  rng.normal(size=shape).astype(np.float32),
                  np.ones(shape, np.float32),
                  rng.normal(size=shape + (223,)).astype(np.float32),
                  np.tile(np.arange(args.batch_size, dtype=np.int32) % 4, (2, 1)))
    points = []

    def snapshot(stage):
        jax.block_until_ready((agent.actor, agent.critic, agent.target_critic, agent.temp))
        row = {'stage': stage, 'actual_updates': int(agent.step) - 1,
               **collect_jax_memory_stats(jax.devices())}
        points.append(row)
        print(json.dumps(row), flush=True)

    snapshot('initialized')
    for i in range(3):
        jax.block_until_ready(agent.update(batch, 2, i))
    snapshot('warm_training')
    if args.mode != 'pure':
        jax.block_until_ready(agent.update(batch, 2, 3, collect_update_diagnostics=True))
        probe = Batch(*(v[0, :min(256, args.batch_size)] for v in batch))
        trees = agent.get_tensor_diagnostics(probe)
        stats = summarize_tree(trees)
        assert all(r['nan_count'] == r['inf_count'] == 0 for r in stats.values())
        snapshot('diagnostics_retained')
        if args.mode == 'release':
            del trees
            gc.collect()
            snapshot('diagnostics_released')
    else:
        jax.block_until_ready(agent.update(batch, 2, 3))
    samples = []
    for i in range(4):
        start = time.perf_counter()
        metrics = agent.update(batch, 2, i + 4)
        jax.block_until_ready(metrics)
        samples.append((time.perf_counter() - start) * 1000 / 2)
    snapshot('final_training')
    result = {'mode': args.mode, 'width': args.width, 'batch_size': args.batch_size,
              'input': 'fixed synthetic Dogs-shaped batch, not a return experiment',
              'timing': 'shared GPU; descriptive only',
              'ms_per_update': samples, 'points': points}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pure', 'research', 'release'), required=True)
    parser.add_argument('--width', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
