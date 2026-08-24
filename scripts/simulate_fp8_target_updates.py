#!/usr/bin/env python3
"""Replay a healthy BRC checkpoint through offline FP8 target-state shadows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from jaxrl.agent.brc_learner import BRC
from jaxrl.checkpoint import CheckpointManager
from jaxrl.normalizer import RewardNormalizer
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.target_simulation import (
    functional_diagnostics,
    initialize_shadow_states,
    teacher_shadow_update,
)
from jaxrl.utils import Batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_updates', type=int, default=50000)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--diagnostic_interval', type=int, default=500)
    parser.add_argument('--output_root', default='runs/offline_target_simulations')
    parser.add_argument('--run_id', required=True)
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        return scalar if not isinstance(scalar, bytes) else scalar.decode()
    return array.tolist()


def append_jsonl(path, record):
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(jsonable(record), sort_keys=True) + '\n')


def load_probe(path):
    with np.load(path, allow_pickle=False) as data:
        return Batch(**{field: data[field] for field in Batch._fields})


def save_probe(path, batch):
    np.savez(path, **{field: getattr(batch, field) for field in Batch._fields})


def infer_replay_shapes(checkpoint):
    replay_dir = checkpoint / 'replay_buffer'
    with (replay_dir / 'manifest.json').open(encoding='utf-8') as file:
        manifest = json.load(file)
    first = manifest['chunks'][0]['files']
    observations = np.load(replay_dir / first['observations'], mmap_mode='r')
    actions = np.load(replay_dir / first['actions'], mmap_mode='r')
    return manifest, observations.shape[-1], observations.dtype, actions.shape[-1]


def flatten_numbers(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f'{prefix}/{key}' if prefix else str(key)
            result.update(flatten_numbers(item, child))
        return result
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return result
    if array.ndim == 0:
        result[prefix] = float(array)
    else:
        for index in np.ndindex(array.shape):
            suffix = ','.join(map(str, index))
            result[f'{prefix}[{suffix}]'] = float(array[index])
    return result


def summarize(records):
    flattened = [flatten_numbers(record['metrics']) for record in records]
    keys = sorted(set().union(*(record.keys() for record in flattened)))
    windows = {}
    for key in keys:
        values = np.asarray(
            [record[key] for record in flattened if key in record], np.float64
        )
        windows[key] = {
            'mean': float(values.mean()),
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return {
        'diagnostic_points': len(records),
        'final': records[-1] if records else None,
        'full_window': windows,
    }


def main():
    args = parse_args()
    if args.num_updates <= 0 or args.diagnostic_interval <= 0:
        raise ValueError('num_updates and diagnostic_interval must be positive')

    checkpoint = CheckpointManager.resolve_recovery_checkpoint(args.checkpoint)
    manifest = CheckpointManager.read_manifest(checkpoint)
    config = manifest['config']
    if config.get('critic_precision') != 'fp8_direct':
        raise ValueError('teacher checkpoint must use online fp8_direct')
    if config.get('target_critic_precision') != 'fp8_direct':
        raise ValueError('teacher checkpoint must use target fp8_direct/FP32 storage')
    updates_per_step = int(config['updates_per_step'])
    if args.num_updates % updates_per_step:
        raise ValueError('num_updates must be divisible by checkpoint updates_per_step')

    replay_manifest, observation_dim, observation_dtype, action_dim = (
        infer_replay_shapes(checkpoint)
    )
    width = int(config['width_critic'])
    if width % args.block_size:
        raise ValueError('block_size must divide the target residual kernel width')

    output_dir = Path(args.output_root).resolve() / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    events_path = output_dir / 'events.jsonl'
    metrics_path = output_dir / 'metrics.jsonl'
    started = time.time()
    append_jsonl(events_path, {
        'event': 'simulation_started', 'time': started,
        'checkpoint': str(checkpoint), 'num_updates': args.num_updates,
    })

    try:
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,),
            dtype=observation_dtype,
        )
        replay = ParallelReplayBuffer(
            observation_space, action_dim, int(replay_manifest['capacity']),
            len(manifest['task_names']),
        )
        replay.load(str(checkpoint / 'replay_buffer'))

        seed = int(config['seed'])
        agent = BRC(
            seed,
            np.zeros((1, observation_dim), observation_dtype),
            np.zeros((1, action_dim), np.float32),
            num_tasks=len(manifest['task_names']),
            updates_per_step=updates_per_step,
            width_critic=width,
            task_embedding_norm=config['resolved_task_embedding_norm'],
            critic_precision=config['critic_precision'],
            target_critic_precision=config['target_critic_precision'],
            fp8_amax_history_length=int(config['fp8_amax_history_length']),
        )
        agent.load(str(checkpoint))
        normalizer = RewardNormalizer(
            len(manifest['task_names']), target_entropy=agent.target_entropy,
            discount=agent.discount,
            return_bootstrap=config['resolved_return_bootstrap'],
            entropy_correction=config['resolved_entropy_correction'],
        )
        with (checkpoint / 'training_state.pkl').open('rb') as file:
            training_state = pickle.load(file)
        normalizer.load_state_dict(training_state['reward_normalizer'])
        random.setstate(training_state['python_random_state'])
        np.random.set_state(training_state['numpy_random_state'])

        source_probe = checkpoint / 'probe_batch.npz'
        if source_probe.exists():
            shutil.copyfile(source_probe, output_dir / 'probe_batch.npz')
            probe = load_probe(source_probe)
        else:
            probe = replay.make_probe_batch(256, seed=seed + 9173)
            save_probe(output_dir / 'probe_batch.npz', probe)

        initial_target_params = agent.target_critic.params
        shadows = initialize_shadow_states(
            agent.critic.params, agent.target_critic.params, args.block_size
        )
        diagnostic_rng = jax.random.PRNGKey(seed + 130363)
        current_commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        run_config = {
            'schema_version': 1,
            'run_id': args.run_id,
            'source_checkpoint': str(checkpoint),
            'source_manifest': manifest,
            'implementation_commit': current_commit,
            'num_updates': args.num_updates,
            'updates_per_step': updates_per_step,
            'batch_size': int(config['batch_size']),
            'tau': agent.tau,
            'block_size': args.block_size,
            'diagnostic_interval': args.diagnostic_interval,
            'probe_size': int(probe.observations.shape[0]),
            'methods': {
                'lag_coded': 'per-ensemble E4M3 lag with current-amax scale',
                'kahan_momentum': (
                    'scaled Kahan EMA (C=1e4) with E4M3 target and '
                    'compensation buffers, each dynamically scaled per ensemble'
                ),
                'naive_per_tensor': 'per-step current-amax per-tensor E4M3 EMA',
                'naive_block_scale': (
                    f'per-step dynamic {args.block_size}x{args.block_size} E4M3 EMA'
                ),
                'interleaved_block': 'lattice-matched dynamic block EMA with staggered phase',
            },
            'cuda': {
                key: os.environ.get(key) for key in (
                    'CUDA_ROOT', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES'
                )
            },
            'jax_devices': [str(device) for device in jax.devices()],
        }
        (output_dir / 'config.json').write_text(
            json.dumps(jsonable(run_config), indent=2), encoding='utf-8'
        )

        rng, actor, critic = agent.rng, agent.actor, agent.critic
        target_critic, temp = agent.target_critic, agent.temp
        task_entropies = agent.task_entropies
        task_entropy_counts = agent.task_entropy_counts
        step = int(agent.step)
        records = []
        first_update_started = time.perf_counter()
        last_info = None
        for group_start in range(0, args.num_updates, updates_per_step):
            batches = replay.sample(int(config['batch_size']), updates_per_step)
            batches = normalizer.normalize(
                batches, temp(), task_entropies=task_entropies
            )
            for batch_index in range(updates_per_step):
                batch = jax.tree.map(
                    lambda value: value[batch_index], batches
                )
                (
                    rng, actor, critic, target_critic, temp, shadows, last_info,
                ) = teacher_shadow_update(
                    rng, actor, critic, target_critic, temp, batch, shadows,
                    agent.discount, agent.tau, agent.target_entropy,
                    agent.num_bins, agent.v_max, agent.multitask,
                    agent.num_tasks, args.block_size,
                )
                step += 1
                completed = group_start + batch_index + 1
                if completed == 1:
                    jax.block_until_ready(last_info)
                    append_jsonl(events_path, {
                        'event': 'first_update_compiled',
                        'learner_updates': completed,
                        'elapsed_sec': time.perf_counter() - first_update_started,
                    })
                if completed % args.diagnostic_interval == 0:
                    diagnostic_rng, diagnostic_key = jax.random.split(
                        diagnostic_rng
                    )
                    normalized_probe = normalizer.normalize(
                        probe, temp(), task_entropies=task_entropies
                    )
                    metrics = functional_diagnostics(
                        diagnostic_key, actor, critic, target_critic,
                        agent.target_critic_reference_def, temp,
                        normalized_probe, shadows, initial_target_params,
                        args.block_size, agent.multitask, agent.num_tasks,
                        agent.num_bins, agent.v_max, agent.discount,
                    )
                    host_metrics = {
                        'functional': jax.device_get(metrics),
                        'teacher_update': jax.device_get(last_info),
                    }
                    flattened = flatten_numbers(host_metrics)
                    nan_count = sum(np.isnan(value) for value in flattened.values())
                    inf_count = sum(np.isinf(value) for value in flattened.values())
                    record = {
                        'learner_updates': completed,
                        'teacher_step': step,
                        'elapsed_sec': time.time() - started,
                        'nan_count': int(nan_count),
                        'inf_count': int(inf_count),
                        'metrics': host_metrics,
                    }
                    append_jsonl(metrics_path, record)
                    records.append(jsonable(record))

            entropy_by_task = last_info['_entropy_by_task']
            entropy_counts = last_info['_entropy_counts_by_task']
            task_entropies = jnp.where(
                entropy_counts > 0, entropy_by_task, task_entropies
            )
            task_entropy_counts = entropy_counts.astype(jnp.int32)

        jax.block_until_ready(shadows)
        summary = summarize(records)
        summary.update({
            'status': 'completed',
            'num_updates': args.num_updates,
            'teacher_final_step': step,
            'elapsed_sec': time.time() - started,
            'teacher_metric_nan_count': int(sum(
                np.isnan(value) for value in flatten_numbers(
                    jax.device_get(last_info)
                ).values()
            )),
            'teacher_metric_inf_count': int(sum(
                np.isinf(value) for value in flatten_numbers(
                    jax.device_get(last_info)
                ).values()
            )),
        })
        (output_dir / 'summary.json').write_text(
            json.dumps(jsonable(summary), indent=2), encoding='utf-8'
        )
        append_jsonl(events_path, {
            'event': 'simulation_completed',
            'learner_updates': args.num_updates,
            'teacher_step': step,
            'elapsed_sec': time.time() - started,
        })
    except Exception as error:
        append_jsonl(events_path, {
            'event': 'simulation_failed', 'time': time.time(),
            'error_type': type(error).__name__, 'error': str(error),
        })
        raise


if __name__ == '__main__':
    main()
