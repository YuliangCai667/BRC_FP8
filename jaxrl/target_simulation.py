"""Pure shadow-state updates for offline FP8 target-Critic experiments."""

from __future__ import annotations

import functools

import flax
from flax import traverse_util
import jax
import jax.numpy as jnp

from jaxrl.agent.brc_learner import _update
from jaxrl.agent.update import _is_resident_kernel, build_actor_input
from jaxrl.networks import E4M3_MAX


KAHAN_MOMENTUM_SCALE = 1e4
SHADOW_METHODS = (
    'lag_coded', 'kahan_momentum', 'naive_per_tensor',
    'naive_block_scale', 'interleaved_block',
)


def _flatten(tree):
    return traverse_util.flatten_dict(tree)


def _unflatten(tree):
    return flax.core.freeze(traverse_util.unflatten_dict(tree))


def resident_kernel_paths(params):
    return tuple(path for path in sorted(_flatten(params)) if _is_resident_kernel(path))


def _quantize_ensemble(values):
    amax = jnp.max(jnp.abs(values), axis=(-2, -1))
    scales = jnp.where(amax > 0, amax / E4M3_MAX, jnp.float32(1.0))
    codes = (values / scales[:, None, None]).astype(jnp.float8_e4m3fn)
    return codes, scales


def _dequantize_ensemble(codes, scales):
    return codes.astype(jnp.float32) * scales[:, None, None]


def block_view(values, block_size: int):
    ensemble, rows, columns = values.shape
    return values.reshape(
        ensemble, rows // block_size, block_size,
        columns // block_size, block_size,
    ).transpose(0, 1, 3, 2, 4)


def unblock_view(blocks):
    ensemble, row_blocks, column_blocks, rows, columns = blocks.shape
    return blocks.transpose(0, 1, 3, 2, 4).reshape(
        ensemble, row_blocks * rows, column_blocks * columns
    )


def quantize_e4m3_blocks(values, block_size: int):
    blocks = block_view(jnp.asarray(values, jnp.float32), block_size)
    amax = jnp.max(jnp.abs(blocks), axis=(-2, -1))
    scales = jnp.where(amax > 0, amax / E4M3_MAX, jnp.float32(1.0))
    codes = (blocks / scales[..., None, None]).astype(jnp.float8_e4m3fn)
    return unblock_view(codes), scales


def dequantize_e4m3_blocks(codes, scales, block_size: int):
    blocks = block_view(codes, block_size).astype(jnp.float32)
    return unblock_view(blocks * scales[..., None, None])


def lag_recurrence_exact(lag, online_old, online_new, tau: float):
    """Unquantized lag recurrence; exposed for its algebraic unit test."""
    return (1.0 - tau) * (lag - (online_new - online_old))


@flax.struct.dataclass
class QuantizedMethodState:
    codes: flax.core.FrozenDict
    scales: flax.core.FrozenDict
    compensation_codes: flax.core.FrozenDict | None = None
    compensation_scales: flax.core.FrozenDict | None = None
    phases: flax.core.FrozenDict | None = None
    code_changes: flax.core.FrozenDict | None = None
    opportunities: flax.core.FrozenDict | None = None
    events: flax.core.FrozenDict | None = None
    unrepresentable: flax.core.FrozenDict | None = None


@flax.struct.dataclass
class TargetShadowStates:
    lag_coded: QuantizedMethodState
    kahan_momentum: QuantizedMethodState
    naive_per_tensor: QuantizedMethodState
    naive_block_scale: QuantizedMethodState
    interleaved_block: QuantizedMethodState


def _zero_counts(paths, params):
    flat = _flatten(params)
    return _unflatten({
        path: jnp.zeros((flat[path].shape[0],), jnp.float32) for path in paths
    })


def _initial_phases(paths, params, block_size):
    flat = _flatten(params)
    total = sum(
        value.shape[0] * (value.shape[1] // block_size)
        * (value.shape[2] // block_size)
        for path, value in flat.items() if path in paths
    )
    result = {}
    offset = 0
    for path in paths:
        value = flat[path]
        shape = (
            value.shape[0], value.shape[1] // block_size,
            value.shape[2] // block_size,
        )
        count = int(jnp.prod(jnp.asarray(shape)))
        result[path] = (
            jnp.arange(offset, offset + count, dtype=jnp.float32) + 0.5
        ).reshape(shape) / float(total)
        offset += count
    return _unflatten(result)


def initialize_shadow_states(online_params, target_params, block_size: int):
    paths = resident_kernel_paths(target_params)
    online = _flatten(online_params)
    target = _flatten(target_params)
    lag_codes, lag_scales = {}, {}
    kahan_codes, kahan_scales = {}, {}
    compensation_codes, compensation_scales = {}, {}
    tensor_codes, tensor_scales = {}, {}
    block_codes, block_scales = {}, {}
    for path in paths:
        lag_codes[path], lag_scales[path] = _quantize_ensemble(
            target[path] - online[path]
        )
        kahan_codes[path], kahan_scales[path] = _quantize_ensemble(
            KAHAN_MOMENTUM_SCALE * target[path]
        )
        compensation_codes[path], compensation_scales[path] = (
            _quantize_ensemble(jnp.zeros_like(target[path]))
        )
        tensor_codes[path], tensor_scales[path] = _quantize_ensemble(target[path])
        block_codes[path], block_scales[path] = quantize_e4m3_blocks(
            target[path], block_size
        )
    zeros = _zero_counts(paths, target_params)
    phases = _initial_phases(paths, target_params, block_size)
    common = dict(code_changes=zeros, opportunities=zeros)
    return TargetShadowStates(
        lag_coded=QuantizedMethodState(
            codes=_unflatten(lag_codes), scales=_unflatten(lag_scales), **common
        ),
        kahan_momentum=QuantizedMethodState(
            codes=_unflatten(kahan_codes), scales=_unflatten(kahan_scales),
            compensation_codes=_unflatten(compensation_codes),
            compensation_scales=_unflatten(compensation_scales), **common,
        ),
        naive_per_tensor=QuantizedMethodState(
            codes=_unflatten(tensor_codes), scales=_unflatten(tensor_scales), **common
        ),
        naive_block_scale=QuantizedMethodState(
            codes=_unflatten(block_codes), scales=_unflatten(block_scales), **common
        ),
        interleaved_block=QuantizedMethodState(
            codes=_unflatten(block_codes), scales=_unflatten(block_scales),
            phases=phases, code_changes=zeros, opportunities=zeros,
            events=zeros, unrepresentable=zeros,
        ),
    )


def _update_counts(state, changes, opportunities, events=None, unresolved=None):
    old_changes = _flatten(state.code_changes)
    old_opportunities = _flatten(state.opportunities)
    return state.replace(
        code_changes=_unflatten({
            path: old_changes[path] + value for path, value in changes.items()
        }),
        opportunities=_unflatten({
            path: old_opportunities[path] + value
            for path, value in opportunities.items()
        }),
        events=(
            _unflatten({
                path: _flatten(state.events)[path] + value
                for path, value in events.items()
            }) if events is not None else state.events
        ),
        unrepresentable=(
            _unflatten({
                path: _flatten(state.unrepresentable)[path] + value
                for path, value in unresolved.items()
            }) if unresolved is not None else state.unrepresentable
        ),
    )


def update_lag_coded(state, online_old, online_new, tau: float):
    old_online, new_online = _flatten(online_old), _flatten(online_new)
    old_codes, old_scales = _flatten(state.codes), _flatten(state.scales)
    codes, scales, changes, opportunities = {}, {}, {}, {}
    for path in old_codes:
        old_lag = _dequantize_ensemble(old_codes[path], old_scales[path])
        candidate = lag_recurrence_exact(
            old_lag, old_online[path], new_online[path], tau
        )
        codes[path], scales[path] = _quantize_ensemble(candidate)
        changes[path] = jnp.sum(
            codes[path] != old_codes[path], axis=(-2, -1), dtype=jnp.float32
        )
        opportunities[path] = jnp.full_like(
            changes[path], old_codes[path].shape[-2] * old_codes[path].shape[-1]
        )
    return _update_counts(
        state.replace(codes=_unflatten(codes), scales=_unflatten(scales)),
        changes, opportunities,
    )


def update_naive_per_tensor(state, online_new, tau: float):
    online = _flatten(online_new)
    old_codes, old_scales = _flatten(state.codes), _flatten(state.scales)
    codes, scales, changes, opportunities = {}, {}, {}, {}
    for path in old_codes:
        target = _dequantize_ensemble(old_codes[path], old_scales[path])
        codes[path], scales[path] = _quantize_ensemble(
            target + tau * (online[path] - target)
        )
        changes[path] = jnp.sum(
            codes[path] != old_codes[path], axis=(-2, -1), dtype=jnp.float32
        )
        opportunities[path] = jnp.full_like(
            changes[path], old_codes[path].shape[-2] * old_codes[path].shape[-1]
        )
    return _update_counts(
        state.replace(codes=_unflatten(codes), scales=_unflatten(scales)),
        changes, opportunities,
    )


def update_kahan_momentum(
    state, online_new, tau: float,
    momentum_scale: float = KAHAN_MOMENTUM_SCALE,
):
    """Paper-faithful scaled Kahan EMA with E4M3 persistent buffers."""
    online = _flatten(online_new)
    old_codes, old_scales = _flatten(state.codes), _flatten(state.scales)
    old_compensation_codes = _flatten(state.compensation_codes)
    old_compensation_scales = _flatten(state.compensation_scales)
    codes, scales = {}, {}
    compensation_codes, compensation_scales = {}, {}
    changes, opportunities = {}, {}
    for path in old_codes:
        scaled_target = _dequantize_ensemble(
            old_codes[path], old_scales[path]
        )
        target = scaled_target / momentum_scale
        compensation = _dequantize_ensemble(
            old_compensation_codes[path], old_compensation_scales[path]
        )
        value = momentum_scale * tau * (online[path] - target)
        corrected_value = value - compensation
        candidate = scaled_target + corrected_value
        codes[path], scales[path] = _quantize_ensemble(candidate)
        applied_sum = _dequantize_ensemble(codes[path], scales[path])
        compensation_candidate = (
            applied_sum - scaled_target
        ) - corrected_value
        (
            compensation_codes[path], compensation_scales[path]
        ) = _quantize_ensemble(compensation_candidate)
        changes[path] = jnp.sum(
            codes[path] != old_codes[path], axis=(-2, -1), dtype=jnp.float32
        )
        opportunities[path] = jnp.full_like(
            changes[path], old_codes[path].shape[-2] * old_codes[path].shape[-1]
        )
    return _update_counts(
        state.replace(
            codes=_unflatten(codes), scales=_unflatten(scales),
            compensation_codes=_unflatten(compensation_codes),
            compensation_scales=_unflatten(compensation_scales),
        ),
        changes, opportunities,
    )


def update_naive_block_scale(state, online_new, tau: float, block_size: int):
    online = _flatten(online_new)
    old_codes, old_scales = _flatten(state.codes), _flatten(state.scales)
    codes, scales, changes, opportunities = {}, {}, {}, {}
    for path in old_codes:
        target = dequantize_e4m3_blocks(
            old_codes[path], old_scales[path], block_size
        )
        codes[path], scales[path] = quantize_e4m3_blocks(
            target + tau * (online[path] - target), block_size
        )
        code_blocks = block_view(codes[path] != old_codes[path], block_size)
        changes[path] = jnp.sum(code_blocks, axis=(1, 2, 3, 4), dtype=jnp.float32)
        opportunities[path] = jnp.full_like(
            changes[path], old_codes[path].shape[-2] * old_codes[path].shape[-1]
        )
    return _update_counts(
        state.replace(codes=_unflatten(codes), scales=_unflatten(scales)),
        changes, opportunities,
    )


def candidate_alphas(tau: float):
    values = []
    value = float(tau)
    while value < 1.0:
        values.append(value)
        value *= 2.0
    values.append(1.0)
    return tuple(values)


def _interleaved_one(codes, scales, phases, online, tau, block_size):
    target_blocks = block_view(
        dequantize_e4m3_blocks(codes, scales, block_size), block_size
    )
    gap = block_view(online, block_size) - target_blocks
    gap_norm_sq = jnp.sum(gap * gap, axis=(-2, -1))
    required = tau * gap_norm_sq
    best_score = jnp.full_like(required, jnp.inf)
    best_p = jnp.zeros_like(required)
    best_codes = block_view(codes, block_size)
    best_scales = scales
    found = jnp.zeros_like(required, dtype=jnp.bool_)
    alphas = jnp.asarray(candidate_alphas(tau), jnp.float32)

    def consider_candidate(index, carry):
        best_score, best_p, best_codes, best_scales, found = carry
        alpha = alphas[index]
        candidate = target_blocks + alpha * gap
        candidate_values = unblock_view(candidate)
        candidate_codes, candidate_scales = quantize_e4m3_blocks(
            candidate_values, block_size
        )
        candidate_code_blocks = block_view(candidate_codes, block_size)
        applied = (
            candidate_code_blocks.astype(jnp.float32)
            * candidate_scales[..., None, None] - target_blocks
        )
        progress = jnp.sum(applied * gap, axis=(-2, -1))
        eligible = (gap_norm_sq > 0) & (progress >= required) & (progress > 0)
        p_alpha = jnp.where(eligible, required / progress, 0.0)
        score = jnp.where(
            eligible,
            p_alpha * jnp.sum(applied * applied, axis=(-2, -1)),
            jnp.inf,
        )
        take = score < best_score
        best_score = jnp.where(take, score, best_score)
        best_p = jnp.where(take, p_alpha, best_p)
        best_scales = jnp.where(take, candidate_scales, best_scales)
        best_codes = jnp.where(
            take[..., None, None], candidate_code_blocks, best_codes
        )
        found = found | eligible
        return best_score, best_p, best_codes, best_scales, found

    best_score, best_p, best_codes, best_scales, found = jax.lax.fori_loop(
        0, alphas.shape[0], consider_candidate,
        (best_score, best_p, best_codes, best_scales, found),
    )
    accumulated = phases + best_p
    commit = found & (accumulated >= 1.0)
    next_phases = jnp.where(commit, accumulated - 1.0, accumulated)
    old_code_blocks = block_view(codes, block_size)
    next_code_blocks = jnp.where(
        commit[..., None, None], best_codes, old_code_blocks
    )
    next_scales = jnp.where(commit, best_scales, scales)
    changed = jnp.sum(
        next_code_blocks != old_code_blocks,
        axis=(1, 2, 3, 4), dtype=jnp.float32,
    )
    events = jnp.sum(commit, axis=(1, 2), dtype=jnp.float32)
    unresolved = jnp.sum(
        (gap_norm_sq > 0) & ~found, axis=(1, 2), dtype=jnp.float32
    )
    block_count = phases.shape[1] * phases.shape[2]
    opportunities = jnp.full_like(events, block_count)
    return (
        unblock_view(next_code_blocks), next_scales, next_phases,
        changed, opportunities * block_size * block_size, events, unresolved,
    )


def update_interleaved_block(state, online_new, tau: float, block_size: int):
    online = _flatten(online_new)
    old_codes, old_scales = _flatten(state.codes), _flatten(state.scales)
    old_phases = _flatten(state.phases)
    codes, scales, phases = {}, {}, {}
    changes, opportunities, events, unresolved = {}, {}, {}, {}
    for path in old_codes:
        (
            codes[path], scales[path], phases[path], changes[path],
            opportunities[path], events[path], unresolved[path],
        ) = _interleaved_one(
            old_codes[path], old_scales[path], old_phases[path], online[path],
            tau, block_size,
        )
    return _update_counts(
        state.replace(
            codes=_unflatten(codes), scales=_unflatten(scales),
            phases=_unflatten(phases),
        ),
        changes, opportunities, events, unresolved,
    )


def update_shadow_states(states, online_old, online_new, tau, block_size):
    return TargetShadowStates(
        lag_coded=update_lag_coded(
            states.lag_coded, online_old, online_new, tau
        ),
        kahan_momentum=update_kahan_momentum(
            states.kahan_momentum, online_new, tau
        ),
        naive_per_tensor=update_naive_per_tensor(
            states.naive_per_tensor, online_new, tau
        ),
        naive_block_scale=update_naive_block_scale(
            states.naive_block_scale, online_new, tau, block_size
        ),
        interleaved_block=update_interleaved_block(
            states.interleaved_block, online_new, tau, block_size
        ),
    )


def reconstruct_shadow_params(method, state, online_params, teacher_target_params,
                              block_size: int):
    result = dict(_flatten(teacher_target_params))
    online = _flatten(online_params)
    codes, scales = _flatten(state.codes), _flatten(state.scales)
    for path in codes:
        if method == 'lag_coded':
            result[path] = online[path] + _dequantize_ensemble(
                codes[path], scales[path]
            )
        elif method == 'kahan_momentum':
            result[path] = (
                _dequantize_ensemble(codes[path], scales[path])
                / KAHAN_MOMENTUM_SCALE
            )
        elif method == 'naive_per_tensor':
            result[path] = _dequantize_ensemble(codes[path], scales[path])
        else:
            result[path] = dequantize_e4m3_blocks(
                codes[path], scales[path], block_size
            )
    return _unflatten(result)


def all_shadow_params(states, online_params, teacher_target_params, block_size):
    return {
        name: reconstruct_shadow_params(
            name, getattr(states, name), online_params, teacher_target_params,
            block_size,
        )
        for name in SHADOW_METHODS
    }


_teacher_update_jit = functools.partial(
    jax.jit,
    static_argnames=(
        'discount', 'tau', 'target_entropy', 'num_bins', 'v_max', 'multitask',
        'num_tasks', 'collect_target_ema_diagnostics',
    ),
)(_update)


_shadow_update_jit = functools.partial(
    jax.jit, static_argnames=('tau', 'block_size')
)(update_shadow_states)


def teacher_update_once(
    rng, actor, critic, target_critic, temp, batch,
    discount, tau, target_entropy, num_bins, v_max, multitask, num_tasks,
):
    return _teacher_update_jit(
        rng, actor, critic, target_critic, temp, batch, discount, tau,
        target_entropy, num_bins, v_max, multitask, num_tasks, False,
    )


def teacher_shadow_update(
    rng, actor, critic, target_critic, temp, batch, shadows,
    discount, tau, target_entropy, num_bins, v_max, multitask, num_tasks,
    block_size,
):
    online_old = critic.params
    rng, actor, critic, target_critic, temp, info, _ = teacher_update_once(
        rng, actor, critic, target_critic, temp, batch,
        discount, tau, target_entropy, num_bins, v_max, multitask, num_tasks,
    )
    shadows = _shadow_update_jit(
        shadows, online_old, critic.params, tau, block_size
    )
    return rng, actor, critic, target_critic, temp, shadows, info


def counter_metrics(state):
    metrics = {}
    changes = _flatten(state.code_changes)
    opportunities = _flatten(state.opportunities)
    events = _flatten(state.events) if state.events is not None else {}
    unresolved = (
        _flatten(state.unrepresentable)
        if state.unrepresentable is not None else {}
    )
    codes = _flatten(state.codes)
    phases = _flatten(state.phases) if state.phases is not None else {}
    for path in changes:
        label = '/'.join(path[:-1])
        for member in range(changes[path].shape[0]):
            denominator = jnp.maximum(opportunities[path][member], 1.0)
            item = {
                'code_changed_fraction': changes[path][member] / denominator,
            }
            if path in events:
                block_elements = (
                    codes[path].shape[-2] * codes[path].shape[-1]
                    / (phases[path].shape[-2] * phases[path].shape[-1])
                )
                block_opportunities = denominator / block_elements
                item['block_event_fraction'] = (
                    events[path][member] / jnp.maximum(block_opportunities, 1.0)
                )
                item['unrepresentable_block_fraction'] = (
                    unresolved[path][member]
                    / jnp.maximum(block_opportunities, 1.0)
                )
            metrics[f'{label}/ensemble_{member}'] = item
    return metrics


def _safe_cosine(left, right):
    denominator = jnp.linalg.norm(left) * jnp.linalg.norm(right)
    return jnp.where(
        denominator > 0, jnp.vdot(left, right) / denominator, 0.0
    )


def parameter_diagnostics(
    states, online_params, teacher_target_params, initial_target_params,
    block_size,
):
    shadows = all_shadow_params(
        states, online_params, teacher_target_params, block_size
    )
    teacher = _flatten(teacher_target_params)
    initial = _flatten(initial_target_params)
    online = _flatten(online_params)
    result = {}
    for method, params in shadows.items():
        flat = _flatten(params)
        method_state = getattr(states, method)
        method_scales = _flatten(method_state.scales)
        method_phases = (
            _flatten(method_state.phases)
            if method_state.phases is not None else {}
        )
        method_compensation = (
            _flatten(method_state.compensation_codes)
            if method_state.compensation_codes is not None else {}
        )
        method_compensation_scales = (
            _flatten(method_state.compensation_scales)
            if method_state.compensation_scales is not None else {}
        )
        method_result = {}
        aggregate = {
            'teacher_displacement_sq': jnp.float32(0.0),
            'shadow_displacement_sq': jnp.float32(0.0),
            'displacement_dot': jnp.float32(0.0),
            'error_sq': jnp.float32(0.0),
            'teacher_sq': jnp.float32(0.0),
            'nonfinite_count': jnp.float32(0.0),
        }
        for path in resident_kernel_paths(teacher_target_params):
            label = '/'.join(path[:-1])
            for member in range(flat[path].shape[0]):
                shadow_value = flat[path][member]
                teacher_value = teacher[path][member]
                teacher_displacement = teacher_value - initial[path][member]
                shadow_displacement = shadow_value - initial[path][member]
                error = shadow_value - teacher_value
                teacher_norm = jnp.linalg.norm(teacher_displacement)
                scale_values = method_scales[path][member]
                storage = {
                    'scale_min': jnp.min(scale_values),
                    'scale_mean': jnp.mean(scale_values),
                    'scale_max': jnp.max(scale_values),
                    'state_amax': jnp.max(jnp.abs(shadow_value)),
                }
                if method == 'lag_coded':
                    lag = shadow_value - online[path][member]
                    storage.update(
                        lag_abs_mean=jnp.mean(jnp.abs(lag)),
                        lag_amax=jnp.max(jnp.abs(lag)),
                    )
                if method == 'kahan_momentum':
                    compensation = _dequantize_ensemble(
                        method_compensation[path],
                        method_compensation_scales[path],
                    )[member]
                    compensation_scale = method_compensation_scales[path][member]
                    storage.update(
                        momentum_scale=jnp.float32(KAHAN_MOMENTUM_SCALE),
                        compensation_abs_mean=jnp.mean(jnp.abs(compensation)),
                        compensation_amax=jnp.max(jnp.abs(compensation)),
                        compensation_scale=compensation_scale,
                    )
                if path in method_phases:
                    phase = method_phases[path][member]
                    storage.update(
                        phase_min=jnp.min(phase),
                        phase_mean=jnp.mean(phase),
                        phase_max=jnp.max(phase),
                    )
                method_result[f'{label}/ensemble_{member}'] = {
                    'teacher_displacement_l2': teacher_norm,
                    'shadow_displacement_l2': jnp.linalg.norm(
                        shadow_displacement
                    ),
                    'displacement_l2_ratio': jnp.where(
                        teacher_norm > 0,
                        jnp.linalg.norm(shadow_displacement) / teacher_norm,
                        0.0,
                    ),
                    'displacement_cosine': _safe_cosine(
                        shadow_displacement, teacher_displacement
                    ),
                    'target_relative_error': (
                        jnp.linalg.norm(error)
                        / jnp.maximum(jnp.linalg.norm(teacher_value), 1e-30)
                    ),
                    'online_target_gap_l2': jnp.linalg.norm(
                        online[path][member] - shadow_value
                    ),
                    'storage': storage,
                }
                aggregate['teacher_displacement_sq'] += jnp.sum(
                    teacher_displacement ** 2
                )
                aggregate['shadow_displacement_sq'] += jnp.sum(
                    shadow_displacement ** 2
                )
                aggregate['displacement_dot'] += jnp.vdot(
                    shadow_displacement, teacher_displacement
                )
                aggregate['error_sq'] += jnp.sum(error ** 2)
                aggregate['teacher_sq'] += jnp.sum(teacher_value ** 2)
                aggregate['nonfinite_count'] += jnp.sum(
                    ~jnp.isfinite(shadow_value), dtype=jnp.float32
                )
        teacher_displacement_norm = jnp.sqrt(
            aggregate['teacher_displacement_sq']
        )
        shadow_displacement_norm = jnp.sqrt(
            aggregate['shadow_displacement_sq']
        )
        method_result['aggregate'] = {
            'teacher_displacement_l2': teacher_displacement_norm,
            'shadow_displacement_l2': shadow_displacement_norm,
            'displacement_l2_ratio': jnp.where(
                teacher_displacement_norm > 0,
                shadow_displacement_norm / teacher_displacement_norm,
                0.0,
            ),
            'displacement_cosine': jnp.where(
                teacher_displacement_norm * shadow_displacement_norm > 0,
                aggregate['displacement_dot']
                / (teacher_displacement_norm * shadow_displacement_norm),
                0.0,
            ),
            'target_relative_error': jnp.sqrt(aggregate['error_sq']) / jnp.maximum(
                jnp.sqrt(aggregate['teacher_sq']), 1e-30
            ),
            'nonfinite_count': aggregate['nonfinite_count'],
        }
        method_result['counters'] = counter_metrics(getattr(states, method))
        result[method] = method_result
    return result, shadows


def _js_divergence(left, right):
    midpoint = 0.5 * (left + right)
    left_term = jnp.where(
        left > 0, left * (jnp.log(left) - jnp.log(midpoint)), 0.0
    )
    right_term = jnp.where(
        right > 0, right * (jnp.log(right) - jnp.log(midpoint)), 0.0
    )
    return 0.5 * jnp.sum(left_term + right_term, axis=-1)


def _task_means(values, task_ids, num_tasks):
    totals = jnp.bincount(task_ids, weights=values, length=num_tasks)
    counts = jnp.bincount(task_ids, length=num_tasks)
    return totals / jnp.maximum(counts, 1)


def _categorical_target(probs, rewards, masks, log_probs, temp, discount,
                        num_bins, v_max):
    v_min = -v_max
    bins = jnp.linspace(v_min, v_max, num_bins)[None]
    delta = (v_max - v_min) / (num_bins - 1)
    target_bins = rewards[:, None] + discount * masks[:, None] * (
        bins - temp * log_probs[:, None]
    )
    target_bins = (jnp.clip(target_bins, v_min, v_max) - v_min) / delta
    lower, upper = jnp.floor(target_bins), jnp.ceil(target_bins)
    lower_mask = jax.nn.one_hot(
        lower.reshape(-1).astype(jnp.int32), num_bins
    ).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(
        upper.reshape(-1).astype(jnp.int32), num_bins
    ).reshape((-1, num_bins, num_bins))
    lower_values = (
        probs * (upper + (lower == upper).astype(jnp.float32) - target_bins)
    )[..., None]
    upper_values = (probs * (target_bins - lower))[..., None]
    return jnp.sum(
        lower_values * lower_mask + upper_values * upper_mask, axis=1
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        'reference_def', 'block_size', 'multitask', 'num_tasks', 'num_bins',
        'v_max', 'discount',
    ),
)
def functional_diagnostics(
    diagnostic_key, actor, critic, target_critic, reference_def, temp, batch,
    states, initial_target_params, block_size, multitask, num_tasks, num_bins,
    v_max, discount,
):
    parameter_metrics, shadows = parameter_diagnostics(
        states, critic.params, target_critic.params, initial_target_params,
        block_size,
    )
    actor_inputs = build_actor_input(
        critic, batch.next_observations, batch.task_ids, multitask
    )
    distribution = actor(actor_inputs)
    next_actions, next_log_probs = distribution.sample_and_log_prob(
        seed=diagnostic_key
    )
    teacher_logits = reference_def.apply(
        {'params': target_critic.params}, batch.next_observations,
        next_actions, batch.task_ids,
    )
    teacher_probs_members = jax.nn.softmax(teacher_logits, axis=-1)
    teacher_probs = teacher_probs_members.mean(axis=0)
    bins = jnp.linspace(-v_max, v_max, num_bins)
    teacher_q_members = jnp.sum(teacher_probs_members * bins, axis=-1)
    teacher_q = jnp.sum(teacher_probs * bins, axis=-1)
    teacher_target_probs = _categorical_target(
        teacher_probs, batch.rewards, batch.masks, next_log_probs, temp(),
        discount, num_bins, v_max,
    )
    teacher_target_q = jnp.sum(teacher_target_probs * bins, axis=-1)

    direct_logits = target_critic(
        batch.next_observations, next_actions, batch.task_ids
    )
    direct_probs = jax.nn.softmax(direct_logits, axis=-1).mean(axis=0)
    direct_q = jnp.sum(direct_probs * bins, axis=-1)
    result = {
        'parameters': parameter_metrics,
        'teacher_state_nonfinite_count': sum(
            jnp.sum(~jnp.isfinite(value), dtype=jnp.float32)
            for value in jax.tree.leaves(target_critic.params)
        ),
        'teacher_fp8_compute_noise_floor': {
            'expected_q_mae': jnp.mean(jnp.abs(direct_q - teacher_q)),
            'expected_q_signed_bias': jnp.mean(direct_q - teacher_q),
            'probability_js': jnp.mean(
                _js_divergence(direct_probs, teacher_probs)
            ),
        },
    }
    for method, params in shadows.items():
        logits = reference_def.apply(
            {'params': params}, batch.next_observations, next_actions,
            batch.task_ids,
        )
        probs_members = jax.nn.softmax(logits, axis=-1)
        probs = probs_members.mean(axis=0)
        q_members = jnp.sum(probs_members * bins, axis=-1)
        q = jnp.sum(probs * bins, axis=-1)
        q_error = q - teacher_q
        js = _js_divergence(probs, teacher_probs)
        target_probs = _categorical_target(
            probs, batch.rewards, batch.masks, next_log_probs, temp(),
            discount, num_bins, v_max,
        )
        target_q = jnp.sum(target_probs * bins, axis=-1)
        target_q_error = target_q - teacher_target_q
        member_metrics = {}
        for member in range(q_members.shape[0]):
            member_error = q_members[member] - teacher_q_members[member]
            member_metrics[f'ensemble_{member}'] = {
                'expected_q_mae': jnp.mean(jnp.abs(member_error)),
                'expected_q_signed_bias': jnp.mean(member_error),
                'probability_js': jnp.mean(_js_divergence(
                    probs_members[member], teacher_probs_members[member]
                )),
            }
        result[method] = {
            'expected_q_mae': jnp.mean(jnp.abs(q_error)),
            'expected_q_signed_bias': jnp.mean(q_error),
            'probability_js': jnp.mean(js),
            'bellman_expected_q_mae': jnp.mean(jnp.abs(target_q_error)),
            'bellman_expected_q_signed_bias': jnp.mean(target_q_error),
            'bellman_probability_js': jnp.mean(
                _js_divergence(target_probs, teacher_target_probs)
            ),
            'expected_q_mae_by_task': _task_means(
                jnp.abs(q_error), batch.task_ids, num_tasks
            ),
            'expected_q_signed_bias_by_task': _task_means(
                q_error, batch.task_ids, num_tasks
            ),
            'probability_js_by_task': _task_means(
                js, batch.task_ids, num_tasks
            ),
            'bellman_expected_q_mae_by_task': _task_means(
                jnp.abs(target_q_error), batch.task_ids, num_tasks
            ),
            'bellman_expected_q_signed_bias_by_task': _task_means(
                target_q_error, batch.task_ids, num_tasks
            ),
            'bellman_probability_js_by_task': _task_means(
                _js_divergence(target_probs, teacher_target_probs),
                batch.task_ids, num_tasks,
            ),
            'ensemble': member_metrics,
        }
    return result
