#!/usr/bin/env python3
"""GPU checks for the scale-aware online-resident FP8 backward path."""

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from flax.linen import fp8_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import (
    _is_resident_kernel,
    _merge_resident_backward_metadata,
    dequantize_critic_params,
)


DIMENSION_NUMBERS = (((1,), (0,)), ((), ()))
DEFAULT_REPLAY = Path(
    'runs/DMC_DOGS/'
    'brc_dmc_dogs_target_fp8_direct_fp32_storage_s42_gpu2/'
    'checkpoints/recovery_step_000000500000/replay_buffer'
)


def _metrics(value, reference):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    value_norm = np.linalg.norm(value)
    reference_norm = np.linalg.norm(reference)
    denominator = max(value_norm * reference_norm, np.finfo(np.float64).tiny)
    return {
        'l2': float(value_norm),
        'zero_fraction': float(np.mean(value == 0)),
        'cosine_to_reference': float(np.dot(value, reference) / denominator),
        'l2_ratio_to_reference': float(
            value_norm / max(reference_norm, np.finfo(np.float64).tiny)
        ),
    }


def _scale_and_history(value, dtype, history_length):
    value = jnp.asarray(value, dtype=jnp.float32)
    amax = jnp.max(jnp.abs(value))
    dtype_max = fp8_ops.get_fp8_max(dtype, jnp.float32)
    scale = jnp.where(amax > 0, amax / dtype_max, jnp.float32(1.0))
    return scale, jnp.full((history_length,), amax, dtype=jnp.float32)


def independent_probe(batch_size, width, history_length):
    key_x, key_w, key_g1, key_g2 = jax.random.split(
        jax.random.PRNGKey(314159), 4
    )
    activation_scale = jnp.float32(0.5)
    kernel_scale = jnp.float32(2e-4)
    activation_codes = jax.random.uniform(
        key_x, (batch_size, width), minval=-448, maxval=448
    ).astype(jnp.float8_e4m3fn)
    kernel_codes = jax.random.uniform(
        key_w, (width, width), minval=-448, maxval=448
    ).astype(jnp.float8_e4m3fn)
    activation_codes = activation_codes.at[0, 0].set(
        jnp.asarray(448, dtype=jnp.float8_e4m3fn)
    )
    kernel_codes = kernel_codes.at[0, 0].set(
        jnp.asarray(448, dtype=jnp.float8_e4m3fn)
    )
    inputs = activation_codes.astype(jnp.float32) * activation_scale
    kernel = kernel_codes.astype(jnp.float32) * kernel_scale
    input_history = jnp.full(
        (history_length,), jnp.max(jnp.abs(inputs)), dtype=jnp.float32
    )
    kernel_history = jnp.full(
        (history_length,), jnp.max(jnp.abs(kernel)), dtype=jnp.float32
    )

    def fp32_loss(lhs, rhs, cotangent):
        output = jax.lax.dot_general(
            lhs,
            rhs,
            DIMENSION_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        return jnp.vdot(output, cotangent)

    def direct_loss(lhs, rhs, cotangent, output_scale, output_history):
        output = fp8_ops.q_dot_dq(
            lhs,
            rhs,
            activation_scale,
            kernel_scale,
            output_scale,
            input_history,
            kernel_history,
            output_history,
            jnp.float32,
            DIMENSION_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        return jnp.vdot(output, cotangent)

    def resident_loss(lhs, rhs, cotangent, output_scale, output_history):
        output = fp8_ops.quantized_dot(
            lhs,
            activation_codes,
            activation_scale,
            rhs,
            kernel_codes,
            kernel_scale,
            output_scale,
            output_history,
            jnp.float32,
            DIMENSION_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        output = fp8_ops.out_dq(
            jnp.float32, activation_scale, kernel_scale, output
        )
        return jnp.vdot(output, cotangent)

    report = {
        'device': str(jax.devices()[0]),
        'activation_scale': float(activation_scale),
        'kernel_scale': float(kernel_scale),
        'batch_size': batch_size,
        'width': width,
        'cases': {},
    }
    for sigma, key in ((1e-3, key_g1), (1e-2, key_g2)):
        cotangent = sigma * jax.random.normal(
            key, (batch_size, width), dtype=jnp.float32
        )
        output_scale, output_history = _scale_and_history(
            cotangent, jnp.float8_e5m2, history_length
        )
        fp32_dx, fp32_dw = jax.grad(fp32_loss, argnums=(0, 1))(
            inputs, kernel, cotangent
        )
        direct_dx, direct_dw = jax.grad(direct_loss, argnums=(0, 1))(
            inputs, kernel, cotangent, output_scale, output_history
        )
        resident_dx, resident_dw = jax.grad(
            resident_loss, argnums=(0, 1)
        )(inputs, kernel, cotangent, output_scale, output_history)
        fp32_dx, fp32_dw, direct_dx, direct_dw, resident_dx, resident_dw = (
            jax.device_get(
                (fp32_dx, fp32_dw, direct_dx, direct_dw, resident_dx, resident_dw)
            )
        )
        report['cases'][str(sigma)] = {
            'kernel_gradient': {
                'fp32': _metrics(fp32_dw, fp32_dw),
                'fp8_direct': _metrics(direct_dw, fp32_dw),
                'repaired_resident': _metrics(resident_dw, fp32_dw),
                'resident_vs_direct': _metrics(resident_dw, direct_dw),
            },
            'input_gradient': {
                'fp32': _metrics(fp32_dx, fp32_dx),
                'fp8_direct': _metrics(direct_dx, fp32_dx),
                'repaired_resident': _metrics(resident_dx, fp32_dx),
                'resident_vs_direct': _metrics(resident_dx, direct_dx),
            },
        }
    return report


def _load_real_batch(replay_path, per_task):
    observations = np.load(
        replay_path / 'observations_000000.npy', mmap_mode='r'
    )[:, :per_task].astype(np.float32)
    actions = np.load(
        replay_path / 'actions_000000.npy', mmap_mode='r'
    )[:, :per_task].astype(np.float32)
    num_tasks = observations.shape[0]
    task_ids = np.broadcast_to(
        np.arange(num_tasks, dtype=np.int32)[:, None],
        observations.shape[:2],
    )
    return (
        observations.reshape((-1, observations.shape[-1])),
        actions.reshape((-1, actions.shape[-1])),
        task_ids.reshape(-1),
    )


def _loss_and_grad(model, params, metadata, observations, actions, task_ids):
    def loss_fn(inner_params, inner_metadata):
        logits = model.apply_fn.apply(
            model.variables(params=inner_params, fp8_meta=inner_metadata),
            observations,
            actions,
            task_ids,
        )
        target_bins = jnp.mod(task_ids * 17 + 43, logits.shape[-1])
        targets = jax.nn.one_hot(target_bins, logits.shape[-1])
        return -(targets[None] * jax.nn.log_softmax(logits, axis=-1)).sum(
            axis=-1
        ).mean()

    return jax.value_and_grad(loss_fn, argnums=(0, 1))(
        params, metadata
    )


def _advance_metadata(precision, current, update):
    if precision == 'fp8_resident':
        return _merge_resident_backward_metadata(current, update)
    return update


def matched_real_batch_probe(replay_path, per_task, width, history_length):
    observations, actions, task_ids = _load_real_batch(replay_path, per_task)
    num_tasks = int(np.max(task_ids)) + 1
    initial_observation = np.zeros((1, observations.shape[-1]), np.float32)
    initial_action = np.zeros((1, actions.shape[-1]), np.float32)
    common = dict(
        seed=42,
        observations=initial_observation,
        actions=initial_action,
        num_tasks=num_tasks,
        width_critic=width,
        width_actor=64,
        updates_per_step=1,
        target_critic_precision='fp32',
        fp8_amax_history_length=history_length,
    )
    direct = BRC(critic_precision='fp8_direct', **common).critic
    resident = BRC(
        critic_precision='fp8_resident',
        fp8_resident_canonicalization=False,
        **common,
    ).critic
    common_params = dequantize_critic_params(resident)
    direct_metadata = direct.fp8_meta
    resident_metadata = resident.fp8_meta
    observations = jnp.asarray(observations)
    actions = jnp.asarray(actions)
    task_ids = jnp.asarray(task_ids)

    # The first two passes fill delayed amax histories.  Parameters stay fixed.
    for _ in range(2):
        (_, (_, direct_update)) = _loss_and_grad(
            direct,
            common_params,
            direct_metadata,
            observations,
            actions,
            task_ids,
        )
        direct_metadata = _advance_metadata(
            'fp8_direct', direct_metadata, direct_update
        )
        (_, (_, resident_update)) = _loss_and_grad(
            resident,
            common_params,
            resident_metadata,
            observations,
            actions,
            task_ids,
        )
        resident_metadata = _advance_metadata(
            'fp8_resident', resident_metadata, resident_update
        )

    direct_loss, (direct_grads, _) = _loss_and_grad(
        direct,
        common_params,
        direct_metadata,
        observations,
        actions,
        task_ids,
    )
    resident_loss, (resident_grads, _) = _loss_and_grad(
        resident,
        common_params,
        resident_metadata,
        observations,
        actions,
        task_ids,
    )
    direct_flat = traverse_util.flatten_dict(direct_grads)
    resident_flat = traverse_util.flatten_dict(resident_grads)
    kernel_report = {}
    for path in sorted(path for path in resident_flat if _is_resident_kernel(path)):
        name = '/'.join(path)
        direct_value, resident_value = jax.device_get(
            (direct_flat[path], resident_flat[path])
        )
        kernel_report[name] = {
            'fp8_direct': _metrics(direct_value, direct_value),
            'repaired_resident': _metrics(resident_value, direct_value),
            'members': {
                str(member): _metrics(
                    resident_value[member], direct_value[member]
                )
                for member in range(resident_value.shape[0])
            },
        }

    support = jnp.linspace(-10.0, 10.0, 101)

    def q_value(model, metadata, inner_actions):
        logits = model.apply_fn.apply(
            model.variables(params=common_params, fp8_meta=metadata),
            observations,
            inner_actions,
            task_ids,
        )
        return (jax.nn.softmax(logits, axis=-1) * support).sum(-1).mean()

    direct_dqda = jax.grad(
        lambda inner_actions: q_value(direct, direct_metadata, inner_actions)
    )(actions)
    resident_dqda = jax.grad(
        lambda inner_actions: q_value(
            resident, resident_metadata, inner_actions
        )
    )(actions)
    direct_dqda, resident_dqda = jax.device_get(
        (direct_dqda, resident_dqda)
    )
    return {
        'device': str(jax.devices()[0]),
        'replay_path': str(replay_path),
        'examples': int(observations.shape[0]),
        'width': width,
        'loss': {
            'fp8_direct': float(direct_loss),
            'repaired_resident': float(resident_loss),
        },
        'kernel_gradients': kernel_report,
        'dQ_da': {
            'fp8_direct': _metrics(direct_dqda, direct_dqda),
            'repaired_resident': _metrics(resident_dqda, direct_dqda),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('independent', 'real', 'all'), default='all')
    parser.add_argument('--replay_path', type=Path, default=DEFAULT_REPLAY)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--per_task', type=int, default=64)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--history_length', type=int, default=1024)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {}
    if args.mode in ('independent', 'all'):
        report['independent'] = independent_probe(
            args.batch_size, args.width, args.history_length
        )
    if args.mode in ('real', 'all'):
        report['real_batch'] = matched_real_batch_probe(
            args.replay_path,
            args.per_task,
            args.width,
            args.history_length,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
