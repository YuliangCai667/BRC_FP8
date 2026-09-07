#!/usr/bin/env python3
"""Matched moment-storage probe on real frozen-checkpoint learner gradients.

Sample intact 128-element blocks across all four residual matrices and both
members. Every arm receives the same replay batches/gradients, starts from the
same saved moments/count, and makes 256 FP32 AdamW updates. This is a local
optimizer probe, not a closed-loop RL return experiment.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import get_critic_gradients, reconstruct_carry_logical_critic_params
from jaxrl.normalizer import RewardNormalizer
from jaxrl.optimizers import (
    MOMENT_MODES, decode_moment_tree, encode_moment_tree, is_residual_kernel,
    moment_diagnostics, optimizer_state_inventory, scale_by_stored_adam,
)
from jaxrl.utils import Batch


class ReplaySample:
    def __init__(self, path):
        self.path = path
        self.manifest = json.loads((path / 'manifest.json').read_text())
        self.cache = {}

    def sample(self, rng, size):
        tasks = rng.randint(self.manifest['num_tasks'], size=size).astype(np.int32)
        indices = rng.randint(self.manifest['size'], size=size)
        result = {}
        for i, chunk in enumerate(self.manifest['chunks']):
            selected = (indices >= chunk['start']) & (indices < chunk['end'])
            if not selected.any():
                continue
            for field, filename in chunk['files'].items():
                if filename not in self.cache:
                    self.cache[filename] = np.load(self.path / filename, mmap_mode='r')
                array = self.cache[filename]
                if field not in result:
                    result[field] = np.empty((size,) + array.shape[2:], array.dtype)
                result[field][selected] = array[tasks[selected], indices[selected] - chunk['start']]
        return Batch(**result, task_ids=tasks)


def select_blocks(tree, blocks):
    selected = {}
    for path, value in traverse_util.flatten_dict(tree).items():
        if is_residual_kernel(path):
            flat = value.reshape(value.shape[0], -1, 128)
            indices = jnp.asarray(np.linspace(0, flat.shape[1] - 1, blocks, dtype=np.int32))
            selected[path] = flat[:, indices].reshape(value.shape[0], blocks * 128)
    return traverse_util.unflatten_dict(selected)


def vector(tree):
    return jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(tree)])


@jax.jit
def compare(value, reference):
    a, b = vector(value), vector(reference)
    norm = jnp.linalg.norm(b)
    error = a - b
    return {
        'relative_l2': jnp.linalg.norm(error) / jnp.maximum(norm, 1e-30),
        'cosine': jnp.vdot(a, b) / jnp.maximum(jnp.linalg.norm(a) * norm, 1e-30),
        'max_abs_error': jnp.max(jnp.abs(error)),
        'max_abs_update': jnp.max(jnp.abs(a)),
        'reference_max_abs_update': jnp.max(jnp.abs(b)),
        # Coordinate-level error normalized by reference RMS does not hide
        # rare spikes behind a near-one aggregate cosine.
        'max_error_over_reference_rms': jnp.max(jnp.abs(error)) /
            jnp.maximum(jnp.sqrt(jnp.mean(b * b)), 1e-30),
        'nonfinite_count': jnp.sum(~jnp.isfinite(a)),
    }


def host(tree):
    return jax.tree.map(lambda value: np.asarray(value).item(), tree)


def run(args):
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    config = manifest['config']
    if not manifest.get('includes_optimizer'):
        raise ValueError('Probe requires saved FP32 moments from a recovery checkpoint')
    if config.get('resolved_online_optimizer_state') != 'fp32_adamw':
        raise ValueError('Probe reference must have FP32 AdamW moments')
    if config.get('target_critic_precision') != 'fp8_lag' or any(
        config.get(key, False) for key in
        ('fp8_all_dense_kernels', 'fp8_input_dense_kernel', 'fp8_output_dense_kernel')
    ):
        raise ValueError('Use the healthy residual-only CARRY + lag baseline')
    replay = ReplaySample(checkpoint / 'replay_buffer')
    rng = np.random.RandomState(42)
    initial_batch = replay.sample(rng, args.batch_size)
    agent = BRC(config['seed'], initial_batch.observations[:1], initial_batch.actions[:1],
                num_tasks=replay.manifest['num_tasks'], width_critic=config['width_critic'],
                critic_precision='fp8_resident', fp8_resident_carry=True,
                target_critic_precision='fp8_lag',
                fp8_amax_history_length=config['fp8_amax_history_length'],
                task_embedding_norm=config['resolved_task_embedding_norm'])
    agent.load(str(checkpoint))
    # Checkpoints deserialize to host arrays; frozen models should be transferred
    # once, not streamed back to the device for every probe gradient.
    for name in ('actor', 'critic', 'target_critic', 'temp'):
        setattr(agent, name, jax.device_put(getattr(agent, name)))
    normalizer = RewardNormalizer(agent.num_tasks, agent.target_entropy,
        return_bootstrap=config['resolved_return_bootstrap'],
        entropy_correction=config['resolved_entropy_correction'])
    with (checkpoint / 'training_state.pkl').open('rb') as file:
        normalizer.load_state_dict(pickle.load(file)['reward_normalizer'])
    adam = agent.critic.opt_state[0]
    mu, nu = select_blocks(adam.mu, args.blocks), select_blocks(adam.nu, args.blocks)
    params = select_blocks(reconstruct_carry_logical_critic_params(agent.critic), args.blocks)
    transforms = {mode: scale_by_stored_adam(mode) for mode in MOMENT_MODES}
    states = {mode: optax.ScaleByAdamState(adam.count, encode_moment_tree(mu, mode),
                                         encode_moment_tree(nu, mode)) for mode in MOMENT_MODES}
    positions = {mode: params for mode in MOMENT_MODES}
    # Closed-over modules are static; model buffers remain runtime arguments.
    @jax.jit
    def gradients(key, actor, critic, target, temp, batch):
        return select_blocks(get_critic_gradients(key, actor, critic, target, temp,
            batch, agent.discount, agent.num_bins, agent.v_max, agent.multitask), args.blocks)

    update = {mode: jax.jit(tx.update) for mode, tx in transforms.items()}
    @jax.jit
    def move(position, direction):
        delta = jax.tree.map(lambda p, u: -3e-4 * (u + 1e-4 * p), position, direction)
        return optax.apply_updates(position, delta), delta

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'checkpoint': str(checkpoint), 'env_step': manifest['env_step'],
        'adam_count': int(adam.count), 'updates': args.updates,
        'blocks_per_matrix_per_member': args.blocks, 'block_size': 128,
        'batch_size': args.batch_size, 'seed': 42,
        'gradient_protocol': 'frozen healthy checkpoint; independent replay batches and action RNG; fixed FP8 metadata',
        'parameter_protocol': 'sampled logical FP32 positions plus AdamW deltas; weight writeback covered by learner smoke',
        'devices': [str(device) for device in jax.devices()],
        'inventory': {mode: optimizer_state_inventory(state) for mode, state in states.items()},
    }
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'event': 'initialized', 'checkpoint': str(checkpoint), 'output': str(output)}), flush=True)
    records = []
    start = time.perf_counter()
    with (output / 'updates.jsonl').open('w', buffering=1) as file:
        for t in range(args.updates):
            batch = replay.sample(rng, args.batch_size)
            batch = normalizer.normalize(batch, agent.get_temperature(), agent.get_task_entropies())
            grads = gradients(jax.random.fold_in(agent.rng, t), agent.actor,
                              agent.critic, agent.target_critic, agent.temp, batch)
            directions, deltas = {}, {}
            for mode in MOMENT_MODES:
                directions[mode], states[mode] = update[mode](grads, states[mode])
                positions[mode], deltas[mode] = move(positions[mode], directions[mode])
            for mode in MOMENT_MODES:
                decoded_v = decode_moment_tree(states[mode].nu)
                reference_v = states['fp32'].nu
                v, ref_v = vector(decoded_v), vector(reference_v)
                row = {'step': t + 1, 'mode': mode,
                       'u': host(compare(directions[mode], directions['fp32'])),
                       'delta': host(compare(deltas[mode], deltas['fp32'])),
                       'position': host(compare(positions[mode], positions['fp32'])),
                       'v_min': float(jnp.min(v)),
                       'v_negative_count': int(jnp.sum(v < 0)),
                       'v_below_half_reference_count': int(jnp.sum((ref_v > 0) & (v < 0.5 * ref_v)))}
                if t == args.updates - 1 or t % 32 == 0:
                    row['moments'] = host(moment_diagnostics((states[mode],)))
                records.append(row)
                file.write(json.dumps(row) + '\n')
                if row['u']['nonfinite_count'] or row['v_negative_count']:
                    raise FloatingPointError(f'Invalid optimizer update: step={t+1}, mode={mode}')
            if (t + 1) % 32 == 0:
                print(json.dumps({'step': t + 1, 'seconds': time.perf_counter() - start,
                    'carry_u_relative_l2': records[-1]['u']['relative_l2']}), flush=True)
    summary = {'protocol': metadata, 'seconds': time.perf_counter() - start, 'arms': {}}
    for mode in MOMENT_MODES:
        rows = [r for r in records if r['mode'] == mode]
        summary['arms'][mode] = {
            'last': rows[-1], 'max_u_relative_l2': max(r['u']['relative_l2'] for r in rows),
            'min_u_cosine': min(r['u']['cosine'] for r in rows),
            'max_coordinate_error_over_rms': max(r['u']['max_error_over_reference_rms'] for r in rows),
            'v_negative_count_max': max(r['v_negative_count'] for r in rows),
            'v_below_half_reference_count_max': max(r['v_below_half_reference_count'] for r in rows),
        }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({'event': 'finished', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=1024)
    run(parser.parse_args())
