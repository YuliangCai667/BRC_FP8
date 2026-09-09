import functools
import jax.numpy as jnp
import jax
import optax
from flax import traverse_util
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

from jaxrl.networks import (
    _quantize_carry_resident_kernel,
    _reconstruct_carry_logical_kernel,
    dequantize_e4m3,
    quantize_e4m3_per_tensor,
)
from jaxrl.utils import Batch, Model, PRNGKey, tree_norm
from jaxrl.low_precision import actor_qat_dense as actor_qat


def _is_resident_kernel(path, model=None):
    # Offline residual-only callers predate model-dependent edge-layer scopes.
    apply_fn = getattr(model, 'apply_fn', None)
    all_dense = bool(
        getattr(apply_fn, 'fp8_all_dense_kernels', False)
    )
    input_dense = all_dense or bool(
        getattr(apply_fn, 'fp8_input_dense_kernel', False)
    )
    output_dense = all_dense or bool(
        getattr(apply_fn, 'fp8_output_dense_kernel', False)
    )
    in_residual_block = any(
        part.startswith('BronetBlock_') for part in path
    )
    return (
        path[-1] == 'kernel'
        and path[-2].startswith('Dense_')
        and (
            in_residual_block
            or (
                not in_residual_block
                and 'critic' in path
                and (
                    (input_dense and path[-2] == 'Dense_0')
                    or (output_dense and path[-2] == 'Dense_1')
                )
            )
        )
    )


def _scale_path(kernel_path):
    return kernel_path[:-1] + ('kernel_scale',)


def _carry_path(kernel_path):
    return kernel_path[:-1] + ('kernel_carry',)


def _anchor_norm_path(kernel_path):
    return kernel_path[:-1] + ('kernel_anchor_norm',)


def _bias_path(kernel_path):
    return kernel_path[:-1] + ('bias',)


def _lag_scale_path(kernel_path):
    return kernel_path[:-1] + ('lag_scale',)


def _quantize_ensemble(kernels):
    return jax.vmap(quantize_e4m3_per_tensor)(kernels)


def _dequantize_ensemble(codes, scales):
    return jax.vmap(dequantize_e4m3)(codes, scales)


def _reconstruct_carry_logical_ensemble(codes, scales, carry_codes):
    return jax.vmap(_reconstruct_carry_logical_kernel)(
        codes, scales, carry_codes
    )


def _quantize_carry_ensemble(kernels):
    return jax.vmap(_quantize_carry_resident_kernel)(kernels)


def _target_precision(target_critic: Model):
    return target_critic.apply_fn.critic_precision


def _critic_precision(critic: Model):
    return critic.apply_fn.critic_precision


def _resident_fixed_anchor_enabled(critic: Model):
    return bool(
        getattr(
            critic.apply_fn,
            'fp8_resident_canonicalization',
            False,
        )
    )


def _resident_carry_enabled(critic: Model):
    return bool(getattr(critic.apply_fn, 'fp8_resident_carry', False))


def _merge_resident_backward_metadata(current, backward_updates):
    """Apply only output-gradient metadata emitted by the resident custom VJP."""
    current_flat = traverse_util.flatten_dict(current)
    updates_flat = traverse_util.flatten_dict(backward_updates)
    for path in current_flat:
        if path[-1] in ('output_grad_scale', 'output_grad_amax_history'):
            current_flat[path] = updates_flat[path]
    return traverse_util.unflatten_dict(current_flat)


def _fixed_anchor_resident_affine(
    candidate_bias,
    next_codes,
    raw_next_scale,
    anchor_norm,
):
    """Anchor each resident member to its fixed initialization norm."""
    candidate_bias = jnp.asarray(candidate_bias, dtype=jnp.float32)
    raw_next_scale = jnp.asarray(raw_next_scale, dtype=jnp.float32)
    anchor_norm = jnp.asarray(anchor_norm, dtype=jnp.float32)
    raw_next_value = _dequantize_ensemble(next_codes, raw_next_scale)
    code_values = next_codes.astype(jnp.float32)
    reduce_axes = tuple(range(1, code_values.ndim))
    code_norm = jnp.sqrt(
        jnp.sum(jnp.square(code_values), axis=reduce_axes)
    )
    valid = (anchor_norm > 0) & (code_norm > 0)
    fixed_scale = jnp.where(
        valid,
        anchor_norm / jnp.maximum(code_norm, jnp.float32(1e-12)),
        raw_next_scale,
    )
    alpha = fixed_scale / raw_next_scale
    canonical_kernel = _dequantize_ensemble(next_codes, fixed_scale)
    bias_shape = alpha.shape + (1,) * (
        candidate_bias.ndim - alpha.ndim
    )
    canonical_bias = candidate_bias * jnp.reshape(alpha, bias_shape)
    return {
        'raw_next_value': raw_next_value,
        'canonical_kernel': canonical_kernel,
        'canonical_bias': canonical_bias,
        'canonical_scale': fixed_scale,
        'alpha': alpha,
        'anchor_norm': anchor_norm,
        'code_norm': code_norm,
    }


def dequantize_critic_params(critic: Model):
    """Build the transient physical-weight tree for an online resident Critic."""
    if _critic_precision(critic) != 'fp8_resident':
        return critic.params

    params = traverse_util.flatten_dict(critic.params)
    metadata = traverse_util.flatten_dict(critic.fp8_meta)
    physical = dict(params)
    for path, value in params.items():
        if _is_resident_kernel(path, critic):
            physical[path] = _dequantize_ensemble(
                value, metadata[_scale_path(path)]
            )
    return traverse_util.unflatten_dict(physical)


def reconstruct_carry_logical_critic_params(critic: Model):
    """Build the transient optimizer/EMA tree with s * (C + R / 16)."""
    if (
        _critic_precision(critic) != 'fp8_resident'
        or not _resident_carry_enabled(critic)
    ):
        return dequantize_critic_params(critic)

    params = traverse_util.flatten_dict(critic.params)
    metadata = traverse_util.flatten_dict(critic.fp8_meta)
    logical = dict(params)
    for path, value in params.items():
        if _is_resident_kernel(path, critic):
            logical[path] = _reconstruct_carry_logical_ensemble(
                value,
                metadata[_scale_path(path)],
                metadata[_carry_path(path)],
            )
    return traverse_util.unflatten_dict(logical)


def initialize_critic_optimizer(critic: Model):
    """Initialize Adam on the resident logical parameter tree."""
    if _critic_precision(critic) != 'fp8_resident':
        return critic
    return critic.replace(
        opt_state=critic.tx.init(reconstruct_carry_logical_critic_params(critic))
    )


def initialize_target_critic(critic: Model, target_critic: Model):
    """Initialize the target state from the online parameter tree."""
    precision = _target_precision(target_critic)
    online_params = reconstruct_carry_logical_critic_params(critic)
    if precision not in ('fp8_resident', 'fp8_lag'):
        return target_critic.replace(params=online_params)

    params = traverse_util.flatten_dict(online_params)
    metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
    target_params = dict(params)
    for path, value in params.items():
        if _is_resident_kernel(path, target_critic):
            if precision == 'fp8_resident':
                codes, scale = _quantize_ensemble(value)
                metadata_path = _scale_path(path)
            else:
                codes, scale = _quantize_ensemble(jnp.zeros_like(value))
                metadata_path = _lag_scale_path(path)
            target_params[path] = codes
            metadata[metadata_path] = scale
    return target_critic.replace(
        params=traverse_util.unflatten_dict(target_params),
        fp8_meta=traverse_util.unflatten_dict(metadata),
    )


@jax.jit
def dequantize_target_params(target_critic: Model):
    """Return an ephemeral FP32 parameter tree for low-frequency diagnostics."""
    if _target_precision(target_critic) != 'fp8_resident':
        return target_critic.params

    params = traverse_util.flatten_dict(target_critic.params)
    metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
    dequantized = dict(params)
    for path, value in params.items():
        if _is_resident_kernel(path, target_critic):
            dequantized[path] = _dequantize_ensemble(
                value, metadata[_scale_path(path)]
            )
    return traverse_util.unflatten_dict(dequantized)


@jax.jit
def reconstruct_target_params(critic: Model, target_critic: Model):
    """Build the ephemeral FP32 parameter tree used by target forwards."""
    precision = _target_precision(target_critic)
    if precision != 'fp8_lag':
        return dequantize_target_params(target_critic)

    online = traverse_util.flatten_dict(
        reconstruct_carry_logical_critic_params(critic)
    )
    target = traverse_util.flatten_dict(target_critic.params)
    metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
    reconstructed = dict(target)
    for path, value in target.items():
        if _is_resident_kernel(path, target_critic):
            lag = _dequantize_ensemble(
                value, metadata[_lag_scale_path(path)]
            )
            reconstructed[path] = online[path] + lag
    return traverse_util.unflatten_dict(reconstructed)


@functools.partial(jax.jit, static_argnames=('tau',))
def target_ema_diagnostics(
    critic: Model,
    target_critic: Model,
    tau: float,
    old_critic: Model = None,
):
    """Measure the EMA candidate for the supplied exact pre-update states."""
    precision = _target_precision(target_critic)
    if precision not in ('fp8_resident', 'fp8_lag'):
        return {}

    online = traverse_util.flatten_dict(
        reconstruct_carry_logical_critic_params(critic)
    )
    target = traverse_util.flatten_dict(target_critic.params)
    metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
    old_online = (
        traverse_util.flatten_dict(
            reconstruct_carry_logical_critic_params(old_critic)
        )
        if old_critic is not None
        else online
    )
    diagnostics = {}
    for path, online_value in online.items():
        if not _is_resident_kernel(path, target_critic):
            continue
        codes = target[path]
        if precision == 'fp8_resident':
            scale = metadata[_scale_path(path)]
            old_value = _dequantize_ensemble(codes, scale)
            intended = tau * (online_value - old_value)
            candidate = old_value + intended
            next_codes, next_scale = _quantize_ensemble(candidate)
            applied = _dequantize_ensemble(next_codes, next_scale) - old_value
        else:
            scale = metadata[_lag_scale_path(path)]
            old_lag = _dequantize_ensemble(codes, scale)
            online_delta = online_value - old_online[path]
            candidate = (1 - tau) * (old_lag - online_delta)
            next_codes, next_scale = _quantize_ensemble(candidate)
            next_lag = _dequantize_ensemble(next_codes, next_scale)
            old_value = old_online[path] + old_lag
            next_value = online_value + next_lag
            intended = tau * (online_value - old_value)
            applied = next_value - old_value
        layer = '/'.join(path[:-1])
        for member in range(codes.shape[0]):
            member_intended = intended[member]
            member_applied = applied[member]
            intended_norm = jnp.linalg.norm(member_intended)
            applied_norm = jnp.linalg.norm(member_applied)
            applied_nonzero = member_applied != 0
            intended_nonzero = member_intended != 0
            comparable = intended_nonzero & applied_nonzero
            row = {
                'code_unchanged_fraction': jnp.mean(
                    next_codes[member] == codes[member]
                ),
                'swallowed_update_fraction': (
                    jnp.sum(intended_nonzero & ~applied_nonzero)
                    / jnp.maximum(jnp.sum(intended_nonzero), 1)
                ),
                'intended_update_abs_mean': jnp.mean(jnp.abs(member_intended)),
                'applied_update_abs_mean': jnp.mean(jnp.abs(member_applied)),
                'applied_to_intended_l2_ratio': jnp.where(
                    intended_norm > 0,
                    applied_norm / intended_norm,
                    0.0,
                ),
                'relative_update_error': jnp.where(
                    intended_norm > 0,
                    jnp.linalg.norm(member_applied - member_intended)
                    / intended_norm,
                    0.0,
                ),
                'sign_agreement': (
                    jnp.sum(
                        comparable
                        & (jnp.sign(member_applied) == jnp.sign(member_intended))
                    )
                    / jnp.maximum(jnp.sum(comparable), 1)
                ),
                'online_target_gap_abs_mean': jnp.mean(
                    jnp.abs(online_value[member] - old_value[member])
                ),
            }
            if precision == 'fp8_resident':
                row.update({
                    'kernel_scale': scale[member],
                    'kernel_amax': jnp.max(jnp.abs(old_value[member])),
                    'next_kernel_scale': next_scale[member],
                })
            else:
                member_candidate = candidate[member]
                member_next_lag = next_lag[member]
                candidate_norm = jnp.linalg.norm(member_candidate)
                row.update({
                    'lag_scale': scale[member],
                    'next_lag_scale': next_scale[member],
                    'lag_amax': jnp.max(jnp.abs(old_lag[member])),
                    'lag_abs_mean': jnp.mean(jnp.abs(old_lag[member])),
                    'lag_rms': jnp.sqrt(jnp.mean(jnp.square(old_lag[member]))),
                    'target_kernel_amax': jnp.max(
                        jnp.abs(old_value[member])
                    ),
                    'lag_effective_min_subnormal': (
                        scale[member] * jnp.float32(0.001953125)
                    ),
                    'lag_candidate_underflow_fraction': (
                        jnp.sum(
                            (member_candidate != 0)
                            & (next_codes[member] == 0)
                        )
                        / jnp.maximum(jnp.sum(member_candidate != 0), 1)
                    ),
                    'lag_code_zero_fraction': jnp.mean(codes[member] == 0),
                    'lag_quantization_relative_error': jnp.where(
                        candidate_norm > 0,
                        jnp.linalg.norm(
                            member_next_lag - member_candidate
                        ) / candidate_norm,
                        0.0,
                    ),
                    'online_delta_abs_mean': jnp.mean(
                        jnp.abs(online_delta[member])
                    ),
                    'online_delta_amax': jnp.max(
                        jnp.abs(online_delta[member])
                    ),
                    'online_delta_l2': jnp.linalg.norm(
                        online_delta[member]
                    ),
                    'ema_update_cosine': jnp.where(
                        (intended_norm > 0) & (applied_norm > 0),
                        jnp.vdot(
                            member_intended, member_applied
                        ) / (intended_norm * applied_norm),
                        0.0,
                    ),
                })
            diagnostics[f'{layer}/ensemble_{member}'] = row
    return diagnostics

@functools.partial(jax.jit, static_argnames=('multitask'))
def build_actor_input(critic: Model, observations: jnp.ndarray, task_ids: jnp.ndarray, multitask: bool):
    inputs = observations
    if multitask:
        task_embeddings = critic(None, None, task_ids, True)
        inputs = jnp.concatenate((inputs, task_embeddings), axis=-1)
    return inputs

def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch, num_bins: int, v_max: float, multitask: bool, num_tasks: int):
    inputs = build_actor_input(critic, batch.observations, batch.task_ids, multitask)
    def actor_loss_fn(actor_variables, critic_fp8_meta=None):
        dist = actor.apply(actor_variables, inputs)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        critic_variables = critic.variables(fp8_meta=critic_fp8_meta)
        q_logits = critic.apply(
            critic_variables, batch.observations, actions, batch.task_ids
        )
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
        bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None]
        q_values = (bin_values * q_probs).sum(-1)    
        actor_loss = (log_probs * temp().mean() - q_values).mean()
        entropy_samples = -log_probs
        entropy_sums = jnp.bincount(
            batch.task_ids, weights=entropy_samples, length=num_tasks
        )
        entropy_counts = jnp.bincount(batch.task_ids, length=num_tasks)
        entropy_by_task = entropy_sums / jnp.maximum(entropy_counts, 1)
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': entropy_samples.mean(),
            '_entropy_by_task': entropy_by_task,
            '_entropy_counts_by_task': entropy_counts,
            'actor_pnorm': tree_norm(actor_variables['params']),
        }
    if actor_qat.enabled(actor):
        params = actor_qat.logical_params(actor)
        def logical_loss(params, critic_meta):
            return actor_loss_fn(actor_qat.logical_variables(actor, params), critic_meta)
        (grads, new_fp8_meta), info = jax.grad(logical_loss, argnums=(0, 1), has_aux=True)(params, critic.fp8_meta)
        new_actor, info = actor_qat.apply_gradients(actor, params, grads, info)
        if _critic_precision(critic) == 'fp8_resident':
            if getattr(critic.apply_fn, 'critic_residual_compute_format', 'legacy') == 'legacy':
                new_fp8_meta = _merge_resident_backward_metadata(critic.fp8_meta, new_fp8_meta)
            else:
                new_fp8_meta = critic.fp8_meta
        new_critic = critic.replace(fp8_meta=new_fp8_meta)
    elif critic.fp8_meta is None:
        new_actor, info = actor.apply_gradient(actor_loss_fn)
        new_critic = critic
    else:
        grad_fn = jax.grad(actor_loss_fn, argnums=(0, 1), has_aux=True)
        (actor_grads, new_fp8_meta), info = grad_fn(
            actor.variables(), critic.fp8_meta
        )
        new_actor, info = actor.apply_variable_gradients(actor_grads, info)
        if _critic_precision(critic) == 'fp8_resident':
            if getattr(critic.apply_fn, 'critic_residual_compute_format', 'legacy') == 'legacy':
                new_fp8_meta = _merge_resident_backward_metadata(critic.fp8_meta, new_fp8_meta)
            else:
                new_fp8_meta = critic.fp8_meta
        new_critic = critic.replace(fp8_meta=new_fp8_meta)
    info['actor_gnorm'] = info.pop('grad_norm')
    return new_actor, new_critic, info

def _update_geometry(weight, update):
    """Return exact radial/tangential and squared-norm update geometry."""
    weight_norm_sq = jnp.vdot(weight, weight)
    weight_norm = jnp.sqrt(weight_norm_sq)
    update_norm_sq = jnp.vdot(update, update)
    update_norm = jnp.sqrt(update_norm_sq)
    radial_dot = jnp.vdot(weight, update)
    radial_component = jnp.where(
        weight_norm_sq > 0,
        (radial_dot / weight_norm_sq) * weight,
        jnp.zeros_like(weight),
    )
    tangential = update - radial_component
    tangential_norm = jnp.linalg.norm(tangential)
    first_order_radial = 2.0 * radial_dot
    second_order_update_sq = update_norm_sq
    return {
        'update_l2': update_norm,
        'radial_dot': radial_dot,
        'radial_update': jnp.where(
            weight_norm > 0, radial_dot / weight_norm, 0.0
        ),
        'tangential_update_l2': tangential_norm,
        'effective_angular_step': jnp.where(
            weight_norm > 0, tangential_norm / weight_norm, 0.0
        ),
        'first_order_radial': first_order_radial,
        'second_order_update_sq': second_order_update_sq,
        'predicted_norm_sq_change': (
            first_order_radial + second_order_update_sq
        ),
        'actual_norm_sq_change': (
            jnp.vdot(weight + update, weight + update) - weight_norm_sq
        ),
    }


def _optimizer_update_components(
    tx, param_grads, opt_state, params, split_weight_decay
):
    """Split the current Optax update by replaying it with zero parameters."""
    updates, new_opt_state = tx.update(param_grads, opt_state, params)
    if not split_weight_decay:
        return updates, new_opt_state, None, None

    zero_params = jax.tree.map(jnp.zeros_like, params)
    adaptive_updates, _ = tx.update(param_grads, opt_state, zero_params)
    weight_decay_updates = jax.tree.map(
        lambda total, adaptive: total - adaptive,
        updates,
        adaptive_updates,
    )
    return (
        updates,
        new_opt_state,
        adaptive_updates,
        weight_decay_updates,
    )


def _scale_code_update_components(
    old_codes, old_scale, next_codes, next_scale
):
    """Decompose physical motion by changing scale first, then FP8 codes."""
    old_value = dequantize_e4m3(old_codes, old_scale)
    old_codes_at_next_scale = dequantize_e4m3(old_codes, next_scale)
    next_value = dequantize_e4m3(next_codes, next_scale)
    scale_only = old_codes_at_next_scale - old_value
    code_only = next_value - old_codes_at_next_scale
    return scale_only, code_only


def _resident_parameter_write(
    critic,
    param_grads,
    info,
    collect_diagnostics,
    backward_fp8_meta=None,
):
    canonicalization_enabled = _resident_fixed_anchor_enabled(critic)
    carry_enabled = _resident_carry_enabled(critic)
    if canonicalization_enabled and carry_enabled:
        raise ValueError(
            'resident carry and fixed-anchor canonicalization are mutually exclusive'
        )
    old_physical = dequantize_critic_params(critic)
    old_logical = reconstruct_carry_logical_critic_params(critic)
    grad_norm = tree_norm(param_grads)
    info['grad_norm'] = grad_norm
    (
        updates,
        new_opt_state,
        adaptive_updates,
        weight_decay_updates,
    ) = _optimizer_update_components(
        critic.tx,
        param_grads,
        critic.opt_state,
        old_logical,
        collect_diagnostics,
    )
    candidate = optax.apply_updates(old_logical, updates)

    stored = traverse_util.flatten_dict(critic.params)
    metadata = traverse_util.flatten_dict(
        critic.fp8_meta if backward_fp8_meta is None else backward_fp8_meta
    )
    old_flat = traverse_util.flatten_dict(old_physical)
    old_logical_flat = traverse_util.flatten_dict(old_logical)
    candidate_flat = traverse_util.flatten_dict(candidate)
    adaptive_flat = (
        traverse_util.flatten_dict(adaptive_updates)
        if collect_diagnostics
        else {}
    )
    weight_decay_flat = (
        traverse_util.flatten_dict(weight_decay_updates)
        if collect_diagnostics
        else {}
    )
    new_stored = dict(candidate_flat)
    new_metadata = dict(metadata)
    new_physical = dict(candidate_flat)
    diagnostics = {}

    for path, candidate_value in candidate_flat.items():
        if not _is_resident_kernel(path, critic):
            continue
        old_codes = stored[path]
        old_scale = metadata[_scale_path(path)]
        carry_quantization = (
            _quantize_carry_ensemble(candidate_value)
            if carry_enabled
            else None
        )
        if carry_enabled:
            next_codes = carry_quantization['main_codes']
            raw_next_scale = carry_quantization['kernel_scale']
            raw_next_value = carry_quantization['main_physical']
            next_carry = carry_quantization['carry_codes']
            logical_next_value = carry_quantization[
                'logical_reconstruction'
            ]
        else:
            next_codes, raw_next_scale = _quantize_ensemble(candidate_value)
            next_carry = None
            logical_next_value = None

        if canonicalization_enabled:
            bias_path = _bias_path(path)
            if bias_path not in candidate_flat:
                raise KeyError(
                    'missing paired bias for resident kernel '
                    f'{"/".join(path)}: expected {"/".join(bias_path)}'
                )
            canonical = _fixed_anchor_resident_affine(
                candidate_flat[bias_path],
                next_codes,
                raw_next_scale,
                metadata[_anchor_norm_path(path)],
            )
            raw_next_value = canonical['raw_next_value']
            next_scale = canonical['canonical_scale']
            next_value = canonical['canonical_kernel']
            canonicalization_factor = canonical['alpha']
            anchor_norms = canonical['anchor_norm']
            code_norms = canonical['code_norm']
            new_stored[bias_path] = canonical['canonical_bias']
            new_physical[bias_path] = canonical['canonical_bias']
        else:
            if not carry_enabled:
                raw_next_value = _dequantize_ensemble(
                    next_codes, raw_next_scale
                )
            next_scale = raw_next_scale
            next_value = raw_next_value
            canonicalization_factor = jnp.ones_like(
                raw_next_scale, dtype=jnp.float32
            )
            if collect_diagnostics:
                reduce_axes = tuple(range(1, raw_next_value.ndim))
                raw_next_norms = jnp.sqrt(
                    jnp.sum(
                        jnp.square(raw_next_value), axis=reduce_axes
                    )
                )
                anchor_norms = jnp.zeros_like(raw_next_scale)
                code_norms = jnp.sqrt(
                    jnp.sum(
                        jnp.square(next_codes.astype(jnp.float32)),
                        axis=reduce_axes,
                    )
                )

        if collect_diagnostics and canonicalization_enabled:
            raw_next_norms = jnp.sqrt(
                jnp.sum(
                    jnp.square(raw_next_value),
                    axis=tuple(range(1, raw_next_value.ndim)),
                )
            )

        new_stored[path] = next_codes
        new_metadata[_scale_path(path)] = next_scale
        if carry_enabled:
            new_metadata[_carry_path(path)] = next_carry
        new_physical[path] = next_value

        if not collect_diagnostics:
            continue
        scale_shape = old_scale.shape + (1,) * (
            candidate_value.ndim - old_scale.ndim
        )
        expanded_old_scale = jnp.reshape(old_scale, scale_shape)
        fixed_codes = jnp.clip(
            candidate_value / expanded_old_scale,
            -jnp.float32(448.0),
            jnp.float32(448.0),
        ).astype(jnp.float8_e4m3fn)
        fixed_value = fixed_codes.astype(jnp.float32) * expanded_old_scale
        intended = candidate_value - old_logical_flat[path]
        applied = next_value - old_flat[path]
        scale_only, code_only = jax.vmap(
            _scale_code_update_components
        )(old_codes, old_scale, next_codes, next_scale)
        layer = '/'.join(path[:-1])
        for member in range(old_codes.shape[0]):
            member_old = old_flat[path][member]
            member_logical_old = old_logical_flat[path][member]
            member_candidate = candidate_value[member]
            member_intended = intended[member]
            member_applied = applied[member]
            member_quantization_error = (
                raw_next_value[member] - member_candidate
            )
            member_adaptive = adaptive_flat[path][member]
            member_weight_decay = weight_decay_flat[path][member]
            member_scale_only = scale_only[member]
            member_code_only = code_only[member]
            old_norm = jnp.linalg.norm(member_old)
            intended_norm = jnp.linalg.norm(member_intended)
            applied_norm = jnp.linalg.norm(member_applied)
            candidate_norm = jnp.linalg.norm(member_candidate)
            quantization_error_norm = jnp.linalg.norm(
                member_quantization_error
            )
            intended_geometry = _update_geometry(
                member_logical_old, member_intended
            )
            actual_geometry = _update_geometry(member_old, member_applied)
            wd_intended_l2 = jnp.linalg.norm(member_weight_decay)
            wd_intended_radial = jnp.where(
                jnp.linalg.norm(member_logical_old) > 0,
                jnp.vdot(member_logical_old, member_weight_decay)
                / jnp.linalg.norm(member_logical_old),
                0.0,
            )
            # Signed radial motion left after subtracting the exact adaptive
            # Optax component from the real post-FP8 write. Any write-time
            # quantization loss therefore appears in the decay retention.
            wd_applied_radial = jnp.where(
                old_norm > 0,
                (
                    jnp.vdot(member_old, member_applied)
                    - jnp.vdot(member_old, member_adaptive)
                ) / old_norm,
                0.0,
            )
            scale_only_norm = jnp.linalg.norm(member_scale_only)
            code_only_norm = jnp.linalg.norm(member_code_only)
            old_code = old_codes[member].astype(jnp.float32)
            next_code = next_codes[member].astype(jnp.float32)
            old_code_norm = jnp.linalg.norm(old_code)
            next_code_norm = jnp.linalg.norm(next_code)
            row = {
                'intended_update_l2': intended_norm,
                'intended_update_radial_cosine': jnp.where(
                    (jnp.linalg.norm(member_logical_old) > 0)
                    & (intended_norm > 0),
                    jnp.vdot(member_logical_old, member_intended)
                    / (jnp.linalg.norm(member_logical_old) * intended_norm),
                    0.0,
                ),
                'intended_tangential_update_l2': (
                    intended_geometry['tangential_update_l2']
                ),
                'intended_effective_angular_step': (
                    intended_geometry['effective_angular_step']
                ),
                'actual_tangential_update_l2': (
                    actual_geometry['tangential_update_l2']
                ),
                'actual_effective_angular_step': (
                    actual_geometry['effective_angular_step']
                ),
                'angular_retention': jnp.where(
                    intended_geometry['effective_angular_step'] > 0,
                    actual_geometry['effective_angular_step']
                    / intended_geometry['effective_angular_step'],
                    0.0,
                ),
                'intended_first_order_radial': (
                    intended_geometry['first_order_radial']
                ),
                'intended_second_order_update_sq': (
                    intended_geometry['second_order_update_sq']
                ),
                'intended_predicted_norm_sq_change': (
                    intended_geometry['predicted_norm_sq_change']
                ),
                'intended_actual_norm_sq_change': (
                    intended_geometry['actual_norm_sq_change']
                ),
                'actual_first_order_radial': (
                    actual_geometry['first_order_radial']
                ),
                'actual_second_order_update_sq': (
                    actual_geometry['second_order_update_sq']
                ),
                'actual_predicted_norm_sq_change': (
                    actual_geometry['predicted_norm_sq_change']
                ),
                'actual_norm_sq_change': (
                    actual_geometry['actual_norm_sq_change']
                ),
                'wd_intended_l2': wd_intended_l2,
                'wd_intended_radial': wd_intended_radial,
                'wd_applied_radial': wd_applied_radial,
                'wd_radial_retention': jnp.where(
                    wd_intended_radial != 0,
                    wd_applied_radial / wd_intended_radial,
                    0.0,
                ),
                'applied_to_intended_l2_ratio': jnp.where(
                    intended_norm > 0, applied_norm / intended_norm, 0.0
                ),
                'update_cosine': jnp.where(
                    (intended_norm > 0) & (applied_norm > 0),
                    jnp.vdot(member_intended, member_applied)
                    / (intended_norm * applied_norm),
                    0.0,
                ),
                'swallowed_update_fraction': jnp.mean(
                    (member_intended != 0) & (member_applied == 0)
                ),
                'code_unchanged_fraction': jnp.mean(
                    next_codes[member] == old_codes[member]
                ),
                'code_l2': next_code_norm,
                'code_cosine': jnp.where(
                    (old_code_norm > 0) & (next_code_norm > 0),
                    jnp.vdot(old_code, next_code)
                    / (old_code_norm * next_code_norm),
                    0.0,
                ),
                'weight_relative_error': jnp.where(
                    candidate_norm > 0,
                    quantization_error_norm / candidate_norm,
                    0.0,
                ),
                'quantization_error_radial_cosine': jnp.where(
                    (candidate_norm > 0) & (quantization_error_norm > 0),
                    jnp.vdot(member_candidate, member_quantization_error)
                    / (candidate_norm * quantization_error_norm),
                    0.0,
                ),
                'scale_log2_ratio': jnp.log2(
                    next_scale[member] / old_scale[member]
                ),
                'relative_scale_change': (
                    next_scale[member] - old_scale[member]
                ) / old_scale[member],
                'scale_only_update_l2': scale_only_norm,
                'code_only_update_l2': code_only_norm,
                'scale_update_fraction_of_actual_l2': jnp.where(
                    applied_norm > 0,
                    scale_only_norm / applied_norm,
                    0.0,
                ),
                'dynamic_vs_fixed_relative_difference': jnp.where(
                    intended_norm > 0,
                    jnp.linalg.norm(
                        raw_next_value[member] - fixed_value[member]
                    )
                    / intended_norm,
                    0.0,
                ),
                'canonicalization_factor': (
                    canonicalization_factor[member]
                ),
                'fixed_anchor_kernel_norm': anchor_norms[member],
                'code_norm': code_norms[member],
                'pre_canonical_kernel_norm': raw_next_norms[member],
                'post_canonical_kernel_norm': jnp.linalg.norm(
                    next_value[member]
                ),
                'post_to_fixed_anchor_kernel_norm_ratio': jnp.where(
                    anchor_norms[member] > 0,
                    jnp.linalg.norm(next_value[member])
                    / anchor_norms[member],
                    0.0,
                ),
                'canonicalization_kernel_delta_l2': jnp.linalg.norm(
                    next_value[member] - raw_next_value[member]
                ),
            }
            if carry_enabled:
                member_logical_next = logical_next_value[member]
                member_main_error = carry_quantization['main_error'][member]
                member_carry_error = carry_quantization['carry_error'][member]
                main_error_l2 = jnp.linalg.norm(member_main_error)
                carry_error_l2 = jnp.linalg.norm(member_carry_error)
                logical_norm = jnp.linalg.norm(member_logical_next)
                member_carry = next_carry[member].astype(jnp.float32)
                row.update({
                    'carry_code_l2': jnp.linalg.norm(member_carry),
                    'carry_zero_fraction': jnp.mean(member_carry == 0),
                    'carry_absmax': jnp.max(jnp.abs(member_carry)),
                    'carry_prequant_absmax': carry_quantization[
                        'carry_prequant_absmax'
                    ][member],
                    'carry_saturation_fraction': carry_quantization[
                        'carry_saturation_fraction'
                    ][member],
                    'main_write_relative_l2': jnp.where(
                        candidate_norm > 0,
                        main_error_l2 / candidate_norm,
                        0.0,
                    ),
                    'logical_reconstruction_relative_l2': jnp.where(
                        candidate_norm > 0,
                        carry_error_l2 / candidate_norm,
                        0.0,
                    ),
                    'carry_error_reduction_ratio': main_error_l2
                    / jnp.maximum(carry_error_l2, jnp.float32(1e-12)),
                    'physical_kernel_norm': jnp.linalg.norm(next_value[member]),
                    'logical_kernel_norm': logical_norm,
                    'logical_to_physical_relative_l2': jnp.where(
                        logical_norm > 0,
                        jnp.linalg.norm(
                            member_logical_next - next_value[member]
                        ) / logical_norm,
                        0.0,
                    ),
                })
            diagnostics[f'{layer}/ensemble_{member}'] = row

    new_critic = critic.replace(
        step=critic.step + 1,
        params=traverse_util.unflatten_dict(new_stored),
        opt_state=new_opt_state,
        fp8_meta=traverse_util.unflatten_dict(new_metadata),
    )
    return (
        new_critic,
        diagnostics,
        candidate,
        traverse_util.unflatten_dict(new_physical),
    )


def _probability_js(left, right):
    midpoint = 0.5 * (left + right)
    tiny = jnp.finfo(left.dtype).tiny
    safe_left = jnp.maximum(left, tiny)
    safe_right = jnp.maximum(right, tiny)
    safe_midpoint = jnp.maximum(midpoint, tiny)
    return jnp.maximum(
        0.5 * jnp.sum(
            jnp.where(
                left > 0,
                left * (jnp.log(safe_left) - jnp.log(safe_midpoint)),
                0.0,
            )
            + jnp.where(
                right > 0,
                right * (jnp.log(safe_right) - jnp.log(safe_midpoint)),
                0.0,
            ),
            axis=-1,
        ),
        0.0,
    )


def _resident_function_write_diagnostics(
    critic, candidate, resident_physical, batch, target_probs, num_bins, v_max
):
    candidate_logits = critic.reference_apply_fn.apply(
        {'params': candidate}, batch.observations, batch.actions, batch.task_ids
    )
    resident_logits = critic.reference_apply_fn.apply(
        {'params': resident_physical},
        batch.observations,
        batch.actions,
        batch.task_ids,
    )
    candidate_probs = jax.nn.softmax(candidate_logits, axis=-1)
    resident_probs = jax.nn.softmax(resident_logits, axis=-1)
    support = jnp.linspace(-v_max, v_max, num_bins)
    candidate_q = (candidate_probs * support).sum(axis=-1)
    resident_q = (resident_probs * support).sum(axis=-1)

    def summarize(left_probs, right_probs, left_q, right_q):
        error = right_q - left_q
        return {
            'expected_q_mae': jnp.mean(jnp.abs(error)),
            'expected_q_signed_bias': jnp.mean(error),
            'probability_js_divergence_mean': jnp.mean(
                _probability_js(left_probs, right_probs)
            ),
        }

    candidate_loss = -(
        target_probs[None] * jax.nn.log_softmax(candidate_logits, axis=-1)
    ).sum(-1).mean(-1).sum(-1)
    resident_loss = -(
        target_probs[None] * jax.nn.log_softmax(resident_logits, axis=-1)
    ).sum(-1).mean(-1).sum(-1)
    aggregate = summarize(
        candidate_probs.mean(axis=0),
        resident_probs.mean(axis=0),
        candidate_q.mean(axis=0),
        resident_q.mean(axis=0),
    )
    aggregate['critic_loss_relative_error'] = jnp.abs(
        resident_loss - candidate_loss
    ) / jnp.maximum(jnp.abs(candidate_loss), jnp.float32(1e-12))
    return {
        'aggregate': aggregate,
        **{
            f'ensemble_{member}': summarize(
                candidate_probs[member],
                resident_probs[member],
                candidate_q[member],
                resident_q[member],
            )
            for member in range(candidate_logits.shape[0])
        },
    }


def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float,
           multitask: bool, collect_resident_diagnostics: bool = False):
    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    target_precision = _target_precision(target_critic)
    if target_precision in ('fp8_direct', 'fp8_lag'):
        target_params = reconstruct_target_params(critic, target_critic)
        next_q_logits, updated_variables = target_critic.apply_fn.apply(
            target_critic.variables(params=target_params),
            batch.next_observations,
            next_actions,
            batch.task_ids,
            mutable=[OVERWRITE_WITH_GRADIENT],
        )
        target_critic = target_critic.replace(
            fp8_meta=updated_variables[OVERWRITE_WITH_GRADIENT]
        )
    else:
        next_q_logits = target_critic(
            batch.next_observations, next_actions, batch.task_ids
        )
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]
    
    delta_z = ((v_max - v_min) / (num_bins - 1))
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (bin_values - temp() * next_log_probs[:, None])
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / delta_z
    
    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1).astype(jnp.int32), num_bins).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(upper.reshape(-1).astype(jnp.int32), num_bins).reshape((-1, num_bins, num_bins))
    
    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]        
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
    
    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)
    def critic_loss_fn(critic_variables):
        q_logits = critic.apply(
            critic_variables, batch.observations, batch.actions, batch.task_ids
        )
        q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
        q_prediction = (bin_values * jnp.exp(q_logprobs)).sum(-1)
        critic_loss = -(target_probs[None] * q_logprobs).sum(-1).mean(-1).sum(-1)
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q_value_target.mean(),
            "q_min": q_value_target.min(),
            "q_max": q_value_target.max(),
            "q_std": q_value_target.std(),
            "q_prediction_mean": q_prediction.mean(),
            "q_prediction_min": q_prediction.min(),
            "q_prediction_max": q_prediction.max(),
            "q_prediction_std": q_prediction.std(),
            "r": batch.rewards.mean(),
            "critic_pnorm": tree_norm(critic_variables['params']),
        }
    resident_diagnostics = {}
    if _critic_precision(critic) == 'fp8_resident':
        physical_params = dequantize_critic_params(critic)

        def physical_loss_fn(params, fp8_meta):
            return critic_loss_fn(
                critic.variables(params=params, fp8_meta=fp8_meta)
            )

        (param_grads, backward_fp8_meta), info = jax.grad(
            physical_loss_fn, argnums=(0, 1), has_aux=True
        )(physical_params, critic.fp8_meta)
        if getattr(critic.apply_fn, 'critic_residual_compute_format', 'legacy') == 'legacy':
            backward_fp8_meta = _merge_resident_backward_metadata(critic.fp8_meta, backward_fp8_meta)
        else:
            backward_fp8_meta = critic.fp8_meta
        (
            new_critic,
            parameter_write,
            candidate,
            resident_physical,
        ) = _resident_parameter_write(
            critic,
            param_grads,
            info,
            collect_resident_diagnostics,
            backward_fp8_meta=backward_fp8_meta,
        )
        if collect_resident_diagnostics:
            resident_diagnostics = {
                'parameter_write': parameter_write,
                'function_write': _resident_function_write_diagnostics(
                    critic,
                    candidate,
                    resident_physical,
                    batch,
                    target_probs,
                    num_bins,
                    v_max,
                ),
            }
    else:
        new_critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, target_critic, info, resident_diagnostics

def update_target_critic(
    critic: Model,
    target_critic: Model,
    tau: float,
    old_critic: Model = None,
):
    precision = _target_precision(target_critic)
    if precision in ('fp8_resident', 'fp8_lag'):
        online = traverse_util.flatten_dict(
            reconstruct_carry_logical_critic_params(critic)
        )
        target = traverse_util.flatten_dict(target_critic.params)
        metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
        old_online = (
            traverse_util.flatten_dict(
                reconstruct_carry_logical_critic_params(old_critic)
            )
            if old_critic is not None
            else online
        )
        new_params = {}
        new_metadata = dict(metadata)
        for path, online_value in online.items():
            target_value = target[path]
            if _is_resident_kernel(path, target_critic):
                if precision == 'fp8_resident':
                    old_value = _dequantize_ensemble(
                        target_value, metadata[_scale_path(path)]
                    )
                    candidate = old_value + tau * (online_value - old_value)
                    scale_path = _scale_path(path)
                else:
                    if old_critic is None:
                        raise ValueError(
                            'fp8_lag target updates require old online parameters'
                        )
                    old_lag = _dequantize_ensemble(
                        target_value, metadata[_lag_scale_path(path)]
                    )
                    candidate = (1 - tau) * (
                        old_lag - (online_value - old_online[path])
                    )
                    scale_path = _lag_scale_path(path)
                codes, scale = _quantize_ensemble(candidate)
                new_params[path] = codes
                new_metadata[scale_path] = scale
            else:
                new_params[path] = (
                    online_value * tau + target_value * (1 - tau)
                )
        return target_critic.replace(
            params=traverse_util.unflatten_dict(new_params),
            fp8_meta=traverse_util.unflatten_dict(new_metadata),
        )

    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau),
        reconstruct_carry_logical_critic_params(critic),
        target_critic.params)
    return target_critic.replace(params=new_target_params)

def update_temperature(temp: Model, entropy: float, target_entropy: float):
    def temperature_loss_fn(temp_variables):
        temperature = temp.apply(temp_variables)
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')
    return new_temp, info


def get_actor_gradients(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    temp: Model,
    batch: Batch,
    num_bins: int,
    v_max: float,
    multitask: bool,
    q_only: bool = False,
):
    """Compute actor gradients for low-frequency numerical diagnostics."""
    inputs = build_actor_input(critic, batch.observations, batch.task_ids, multitask)

    def loss_fn(actor_variables):
        dist = actor.apply(actor_variables, inputs)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q_logits = critic(batch.observations, actions, batch.task_ids)
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
        bin_values = jnp.linspace(-v_max, v_max, num_bins)[None]
        q_values = (bin_values * q_probs).sum(-1)
        return -q_values.mean() if q_only else (log_probs * temp().mean() - q_values).mean()

    if actor_qat.enabled(actor):
        return jax.grad(lambda p: loss_fn(actor_qat.logical_variables(actor, p)))(actor_qat.logical_params(actor))
    return jax.grad(loss_fn)(actor.variables())['params']


def get_critic_gradients(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    num_bins: int,
    v_max: float,
    multitask: bool,
):
    """Compute critic gradients for low-frequency numerical diagnostics."""
    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    target_params = reconstruct_target_params(critic, target_critic)
    next_q_logits = target_critic.apply_fn.apply(
        target_critic.variables(params=target_params),
        batch.next_observations,
        next_actions,
        batch.task_ids,
    )
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = -v_max
    bin_values = jnp.linspace(v_min, v_max, num_bins)[None]
    delta_z = (v_max - v_min) / (num_bins - 1)
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (
        bin_values - temp() * next_log_probs[:, None]
    )
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / delta_z
    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1).astype(jnp.int32), num_bins).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(upper.reshape(-1).astype(jnp.int32), num_bins).reshape((-1, num_bins, num_bins))
    lower_values = (
        next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values)
    )[..., None]
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
    target_probs = jax.lax.stop_gradient(
        jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1)
    )

    def loss_fn(critic_variables):
        q_logits = critic.apply(
            critic_variables, batch.observations, batch.actions, batch.task_ids
        )
        q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
        return -(target_probs[None] * q_logprobs).sum(-1).mean(-1).sum(-1)

    if _critic_precision(critic) == 'fp8_resident':
        physical_params = dequantize_critic_params(critic)
        return jax.grad(
            lambda params: loss_fn(critic.variables(params=params))
        )(physical_params)
    return jax.grad(loss_fn)(critic.variables())['params']

'''
from jaxrl.utils import Batch

key = agent.rng
actor = agent.actor
target_critic = agent.target_critic
critic = agent.critic
temp = agent.temp
batch = Batch(
    observations=batches.observations[0],
    actions=batches.actions[0],
    rewards=batches.rewards[0],
    masks=batches.masks[0],
    next_observations=batches.next_observations[0],
    task_ids=batches.task_ids[0])
discount = agent.discount
num_bins = agent.num_bins
v_max = agent.v_max
multitask = agent.multitask
'''
