import functools
import jax.numpy as jnp
import jax
from flax import traverse_util
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

from jaxrl.networks import dequantize_e4m3, quantize_e4m3_per_tensor
from jaxrl.utils import Batch, Model, PRNGKey, tree_norm


def _is_resident_kernel(path):
    return (
        path[-1] == 'kernel'
        and path[-2].startswith('Dense_')
        and any(part.startswith('BronetBlock_') for part in path)
    )


def _scale_path(kernel_path):
    return kernel_path[:-1] + ('kernel_scale',)


def _lag_scale_path(kernel_path):
    return kernel_path[:-1] + ('lag_scale',)


def _quantize_ensemble(kernels):
    return jax.vmap(quantize_e4m3_per_tensor)(kernels)


def _dequantize_ensemble(codes, scales):
    return jax.vmap(dequantize_e4m3)(codes, scales)


def _target_precision(target_critic: Model):
    return target_critic.apply_fn.critic_precision


def initialize_target_critic(critic: Model, target_critic: Model):
    """Initialize the target state from the online parameter tree."""
    precision = _target_precision(target_critic)
    if precision not in ('fp8_resident', 'fp8_lag'):
        return target_critic.replace(params=critic.params)

    params = traverse_util.flatten_dict(critic.params)
    metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
    target_params = dict(params)
    for path, value in params.items():
        if _is_resident_kernel(path):
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
        if _is_resident_kernel(path):
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

    online = traverse_util.flatten_dict(critic.params)
    target = traverse_util.flatten_dict(target_critic.params)
    metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
    reconstructed = dict(target)
    for path, value in target.items():
        if _is_resident_kernel(path):
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

    online = traverse_util.flatten_dict(critic.params)
    target = traverse_util.flatten_dict(target_critic.params)
    metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
    old_online = (
        traverse_util.flatten_dict(old_critic.params)
        if old_critic is not None
        else online
    )
    diagnostics = {}
    for path, online_value in online.items():
        if not _is_resident_kernel(path):
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
    if critic.fp8_meta is None:
        new_actor, info = actor.apply_gradient(actor_loss_fn)
        new_critic = critic
    else:
        grad_fn = jax.grad(actor_loss_fn, argnums=(0, 1), has_aux=True)
        (actor_grads, new_fp8_meta), info = grad_fn(
            actor.variables(), critic.fp8_meta
        )
        new_actor, info = actor.apply_variable_gradients(actor_grads, info)
        new_critic = critic.replace(fp8_meta=new_fp8_meta)
    info['actor_gnorm'] = info.pop('grad_norm')
    return new_actor, new_critic, info

def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float, multitask: bool):
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
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, target_critic, info

def update_target_critic(
    critic: Model,
    target_critic: Model,
    tau: float,
    old_critic: Model = None,
):
    precision = _target_precision(target_critic)
    if precision in ('fp8_resident', 'fp8_lag'):
        online = traverse_util.flatten_dict(critic.params)
        target = traverse_util.flatten_dict(target_critic.params)
        metadata = traverse_util.flatten_dict(target_critic.fp8_meta)
        old_online = (
            traverse_util.flatten_dict(old_critic.params)
            if old_critic is not None
            else online
        )
        new_params = {}
        new_metadata = dict(metadata)
        for path, online_value in online.items():
            target_value = target[path]
            if _is_resident_kernel(path):
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
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
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
        return (log_probs * temp().mean() - q_values).mean()

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
