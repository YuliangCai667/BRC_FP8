import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util

from jaxrl.agent.update import (
    build_actor_input,
    dequantize_critic_params,
    get_actor_gradients,
    get_critic_gradients,
    initialize_critic_optimizer,
    initialize_target_critic,
    reconstruct_carry_logical_critic_params,
    reconstruct_target_params,
    target_ema_diagnostics,
    update_actor,
    update_critic,
    update_target_critic,
    update_temperature,
)

from jaxrl.networks import NormalTanhPolicy, Critic, Temperature
from jaxrl.optimizers import MOMENT_MODES, stored_adamw, moment_diagnostics
from jaxrl.utils import Model, PRNGKey, Batch


@functools.partial(jax.jit, static_argnames=('discount', 'target_entropy', 'num_bins', 'v_max', 'multitask', 'num_tasks'),)
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None, None, None, None, None, None))
def _get_infos(
    rng: PRNGKey, 
    actor: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    batch: Batch, 
    discount: float, 
    target_entropy: float, 
    num_bins: int, 
    v_max: float,
    multitask: bool,
    num_tasks: int,
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    _, _, critic_info, _ = update_critic(
        critic_key, actor, critic, target_critic, temp, batch,
        discount, num_bins, v_max, multitask,
    )
    _, _, actor_info = update_actor(
        actor_key, actor, critic, temp, batch, num_bins, v_max, multitask, num_tasks
    )
    _, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    return {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@jax.jit
def _get_temperature(temp):
    temp_val = temp()
    return temp_val
    
@jax.jit
def _sample_actions(
    rng: PRNGKey,
    actor: Model,
    inputs: np.ndarray,
    temperature: float = 1.0,
):
    dist = actor(inputs, temperature)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    return rng, actions


@functools.partial(
    jax.jit, static_argnames=('num_bins', 'v_max', 'multitask')
)
def _estimate_bootstrap_values(
    rng,
    actor,
    critic,
    target_critic,
    observations,
    task_ids,
    num_bins,
    v_max,
    multitask,
):
    """Estimate V(s') = E_a[Q_target(s', a)] at time-limit boundaries."""
    inputs = build_actor_input(critic, observations, task_ids, multitask)
    policy = actor(inputs)
    rng, key = jax.random.split(rng)
    actions = policy.sample(seed=key)
    target_params = reconstruct_target_params(critic, target_critic)
    logits = target_critic.apply_fn.apply(
        target_critic.variables(params=target_params),
        observations,
        actions,
        task_ids,
    )
    probabilities = jax.nn.softmax(logits, axis=-1).mean(axis=0)
    support = jnp.linspace(-v_max, v_max, num_bins)
    return rng, (probabilities * support).sum(axis=-1)


@functools.partial(
    jax.jit,
    static_argnames=('discount', 'num_bins', 'v_max', 'multitask'),
)
def _get_gradient_diagnostics(
    rng,
    actor,
    critic,
    target_critic,
    temp,
    batch,
    discount,
    num_bins,
    v_max,
    multitask,
):
    _, actor_key, critic_key = jax.random.split(rng, 3)
    return {
        'actor': get_actor_gradients(
            actor_key, actor, critic, temp, batch, num_bins, v_max, multitask
        ),
        'critic': get_critic_gradients(
            critic_key,
            actor,
            critic,
            target_critic,
            temp,
            batch,
            discount,
            num_bins,
            v_max,
            multitask,
        ),
    }


@functools.partial(
    jax.jit, static_argnames=('multitask', 'num_bins', 'v_max')
)
def _get_forward_diagnostics(
    actor,
    critic,
    target_critic,
    target_reference,
    batch,
    multitask,
    num_bins,
    v_max,
):
    actor_inputs = build_actor_input(
        critic, batch.observations, batch.task_ids, multitask
    )
    policy = actor(actor_inputs)
    q_logits = critic(batch.observations, batch.actions, batch.task_ids)
    _, actor_intermediates = actor.apply_fn.apply(
        actor.variables(), actor_inputs, mutable=['intermediates']
    )
    _, critic_intermediates = critic.apply_fn.apply(
        critic.variables(),
        batch.observations,
        batch.actions,
        batch.task_ids,
        mutable=['intermediates'],
    )
    target_params = reconstruct_target_params(critic, target_critic)
    target_logits, target_intermediates = target_critic.apply_fn.apply(
        target_critic.variables(params=target_params),
        batch.observations,
        batch.actions,
        batch.task_ids,
        mutable=['intermediates'],
    )
    diagnostics = {
        'inputs': {
            'actor': actor_inputs,
            'critic_observations': batch.observations,
            'critic_actions': batch.actions,
        },
        'outputs': {
            'policy_mean': policy.distribution.loc,
            'policy_std': policy.distribution.scale_diag,
            'critic_logits': q_logits,
            'target_critic_logits': target_logits,
        },
        'activations': {
            'actor': actor_intermediates.get('intermediates', {}),
            'critic': critic_intermediates.get('intermediates', {}),
            'target_critic': _split_ensemble_intermediates(
                target_intermediates.get('intermediates', {})
            ),
        },
    }
    if target_reference is not None:
        reference_logits = target_reference(
            batch.observations, batch.actions, batch.task_ids
        )
        fp8_probabilities = jax.nn.softmax(target_logits, axis=-1)
        fp32_probabilities = jax.nn.softmax(reference_logits, axis=-1)
        support = jnp.linspace(-v_max, v_max, num_bins)
        fp8_q = (fp8_probabilities * support).sum(axis=-1)
        fp32_q = (fp32_probabilities * support).sum(axis=-1)
        member_midpoint = 0.5 * (fp8_probabilities + fp32_probabilities)
        member_js_divergence = 0.5 * jnp.sum(
            jnp.where(
                fp8_probabilities > 0,
                fp8_probabilities
                * (jnp.log(fp8_probabilities) - jnp.log(member_midpoint)),
                0.0,
            )
            + jnp.where(
                fp32_probabilities > 0,
                fp32_probabilities
                * (jnp.log(fp32_probabilities) - jnp.log(member_midpoint)),
                0.0,
            ),
            axis=-1,
        )
        member_error = fp8_q - fp32_q
        aggregate_fp8 = fp8_probabilities.mean(axis=0)
        aggregate_fp32 = fp32_probabilities.mean(axis=0)
        aggregate_midpoint = 0.5 * (aggregate_fp8 + aggregate_fp32)
        aggregate_js_divergence = 0.5 * jnp.sum(
            jnp.where(
                aggregate_fp8 > 0,
                aggregate_fp8
                * (jnp.log(aggregate_fp8) - jnp.log(aggregate_midpoint)),
                0.0,
            )
            + jnp.where(
                aggregate_fp32 > 0,
                aggregate_fp32
                * (jnp.log(aggregate_fp32) - jnp.log(aggregate_midpoint)),
                0.0,
            ),
            axis=-1,
        )
        aggregate_error = (
            (aggregate_fp8 - aggregate_fp32) * support
        ).sum(axis=-1)
        diagnostics['outputs']['target_critic_fp32_reference_logits'] = (
            reference_logits
        )
        diagnostics['target_forward_error'] = {
            'aggregate': _forward_error_summary(
                aggregate_error, aggregate_js_divergence
            ),
            **{
                f'ensemble_{member}': _forward_error_summary(
                    member_error[member], member_js_divergence[member]
                )
                for member in range(target_logits.shape[0])
            },
        }
    return diagnostics


def _forward_error_summary(error, js_divergence):
    return {
        'expected_q_mae': jnp.mean(jnp.abs(error)),
        'expected_q_signed_bias': jnp.mean(error),
        'probability_js_divergence_mean': jnp.mean(js_divergence),
    }


def _split_ensemble_intermediates(intermediates):
    split = {}
    for path, captures in traverse_util.flatten_dict(
        intermediates, sep='/'
    ).items():
        values = captures[0]
        split[path] = {
            f'ensemble_{member}': values[member]
            for member in range(values.shape[0])
        }
    return split


def _split_ensemble_arrays(flat_values):
    return {
        path: {
            f'ensemble_{member}': value[member]
            for member in range(value.shape[0])
        }
        for path, value in flat_values.items()
    }

def _update(
    rng: PRNGKey, 
    actor: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    batch: Batch, 
    discount: float, 
    tau: float, 
    target_entropy: float, 
    num_bins: int, 
    v_max: float,
    multitask: bool,
    num_tasks: int,
    collect_update_diagnostics: bool,
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    (
        new_critic,
        target_critic,
        critic_info,
        resident_diagnostics,
    ) = update_critic(
        critic_key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        discount,
        num_bins,
        v_max,
        multitask,
        collect_resident_diagnostics=collect_update_diagnostics,
    )
    target_diagnostics = (
        target_ema_diagnostics(
            new_critic, target_critic, tau, old_critic=critic
        )
        if collect_update_diagnostics
        else {}
    )
    new_target_critic = update_target_critic(
        new_critic, target_critic, tau, old_critic=critic
    )
    new_actor, new_critic, actor_info = update_actor(
        actor_key, actor, new_critic, temp, batch, num_bins, v_max, multitask, num_tasks
    )
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }, {
        'online_resident': resident_diagnostics,
        'target_ema': target_diagnostics,
    }

@functools.partial(jax.jit, static_argnames=('discount', 'tau', 'target_entropy', 'num_bins', 'v_max', 'multitask', 'num_tasks', 'num_updates', 'collect_update_diagnostics'))
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batches: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
    multitask: bool, 
    num_tasks: int,
    step: int,    
    num_updates: int,
    collect_update_diagnostics: bool,
):
    def run_update(i, state, collect_diagnostics):
        step, rng, actor, critic, target_critic, temp, info = state
        step = step + 1
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_temp,
            info,
            update_diagnostics,
        ) = _update(
            rng,
            actor,
            critic,
            target_critic,
            temp,
            jax.tree.map(lambda x: jnp.take(x, i, axis=0), batches),
            discount,
            tau,
            target_entropy,
            num_bins,
            v_max,
            multitask,
            num_tasks,
            collect_diagnostics,
        )
        return (
            step,
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_temp,
            info,
        ), update_diagnostics

    initial_state = (step, rng, actor, critic, target_critic, temp, {})
    if num_updates == 1:
        state, update_diagnostics = run_update(
            0, initial_state, collect_update_diagnostics
        )
        return (*state, update_diagnostics)

    state, _ = run_update(0, initial_state, False)

    def one_step(i, current_state):
        return run_update(i, current_state, False)[0]

    state = jax.lax.fori_loop(1, num_updates - 1, one_step, state)
    state, update_diagnostics = run_update(
        num_updates - 1, state, collect_update_diagnostics
    )
    return (*state, update_diagnostics)

class BRC(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        num_tasks: int,
        embedding_size: int = 32,
        ensemble_size: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 0.1,
        updates_per_step: int = 10,
        width_critic: int = 512,
        width_actor: int = 256,
        num_bins: int = 101,
        v_max: float = 10.0,
        task_embedding_norm: str = 'l2',
        critic_precision: str = 'fp32',
        target_critic_precision: str = 'fp32',
        fp8_amax_history_length: int = 1024,
        fp8_resident_canonicalization: bool = False,
        fp8_resident_carry: bool = False,
        fp8_all_dense_kernels: bool = False,
        fp8_input_dense_kernel: bool = False,
        fp8_output_dense_kernel: bool = False,
        critic_optimizer_state: str = 'fp32',
        critic_residual_compute_format: str = 'legacy',
    ) -> None:
        if critic_optimizer_state not in MOMENT_MODES:
            raise ValueError(f'Unknown critic_optimizer_state: {critic_optimizer_state}')
        if critic_optimizer_state != 'fp32' and (
            critic_precision != 'fp8_resident' or not fp8_resident_carry
        ):
            raise ValueError('compressed critic_optimizer_state requires resident CARRY weights')
        if fp8_resident_carry and critic_precision != 'fp8_resident':
            raise ValueError(
                'fp8_resident_carry requires critic_precision=fp8_resident'
            )
        if fp8_resident_carry and fp8_resident_canonicalization:
            raise ValueError(
                'fp8_resident_carry requires fixed-anchor canonicalization off'
            )
        
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.seed = seed
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.num_bins = num_bins
        self.v_max = v_max
        self.task_embedding_norm = task_embedding_norm
        self.critic_precision = critic_precision
        self.target_critic_precision = target_critic_precision
        self.fp8_amax_history_length = fp8_amax_history_length
        self.fp8_resident_canonicalization = fp8_resident_canonicalization
        if critic_residual_compute_format != 'legacy':
            from jaxrl.low_precision.block_formats import FORMATS
            if critic_residual_compute_format not in FORMATS:
                raise ValueError('Unsupported native residual format')
            if critic_precision != 'fp8_resident' or not fp8_resident_carry:
                raise ValueError('Native two-term compute requires resident CARRY')
        self.critic_residual_compute_format = critic_residual_compute_format
        self.fp8_resident_carry = fp8_resident_carry
        self.fp8_all_dense_kernels = fp8_all_dense_kernels
        self.fp8_input_dense_kernel = fp8_input_dense_kernel
        self.fp8_output_dense_kernel = fp8_output_dense_kernel
        self.critic_optimizer_state = critic_optimizer_state
        
        self.num_tasks = num_tasks
        self.embedding_size = embedding_size
        self.task_ids = jnp.arange(num_tasks, dtype=jnp.int32)
        
        task_embedding_init = jnp.zeros((1, embedding_size))
        task_ids_init = self.task_ids[:1]
        self.multitask = True if num_tasks > 1 else False
        
        actor_init = jnp.concatenate((observations, task_embedding_init), axis=-1) if self.multitask else observations
        
        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
            actor_def = NormalTanhPolicy(action_dim=action_dim, hidden_dims=width_actor)
            critic_def = Critic(
                num_tasks=num_tasks,
                embedding_size=embedding_size,
                ensemble_size=ensemble_size,
                hidden_dims=width_critic,
                depth=2,
                output_nodes=num_bins,
                multitask=self.multitask,
                task_embedding_norm=task_embedding_norm,
                critic_precision=critic_precision,
                fp8_amax_history_length=fp8_amax_history_length,
                fp8_resident_canonicalization=(
                    fp8_resident_canonicalization
                ),
                fp8_resident_carry=fp8_resident_carry,
                critic_residual_compute_format=critic_residual_compute_format,
                fp8_all_dense_kernels=fp8_all_dense_kernels,
                fp8_input_dense_kernel=fp8_input_dense_kernel,
                fp8_output_dense_kernel=fp8_output_dense_kernel,
            )
            critic_reference_def = Critic(
                num_tasks=num_tasks,
                embedding_size=embedding_size,
                ensemble_size=ensemble_size,
                hidden_dims=width_critic,
                depth=2,
                output_nodes=num_bins,
                multitask=self.multitask,
                task_embedding_norm=task_embedding_norm,
                critic_precision='fp32',
                fp8_all_dense_kernels=fp8_all_dense_kernels,
                fp8_input_dense_kernel=fp8_input_dense_kernel,
                fp8_output_dense_kernel=fp8_output_dense_kernel,
            )
            target_critic_def = Critic(
                num_tasks=num_tasks,
                embedding_size=embedding_size,
                ensemble_size=ensemble_size,
                hidden_dims=width_critic,
                depth=2,
                output_nodes=num_bins,
                multitask=self.multitask,
                task_embedding_norm=task_embedding_norm,
                critic_precision=target_critic_precision,
                fp8_amax_history_length=fp8_amax_history_length,
                fp8_all_dense_kernels=fp8_all_dense_kernels,
                fp8_input_dense_kernel=fp8_input_dense_kernel,
                fp8_output_dense_kernel=fp8_output_dense_kernel,
            )
            actor = Model.create(actor_def, inputs=[actor_key, actor_init], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(
                critic_def,
                inputs=[critic_key, observations, actions, task_ids_init],
                tx=(
                    optax.adamw(learning_rate=critic_lr)
                    if critic_optimizer_state == 'fp32'
                    else stored_adamw(critic_lr, mode=critic_optimizer_state)
                ),
                reference_apply_fn=critic_reference_def,
            )
            critic = initialize_critic_optimizer(critic)
            target_critic = Model.create(target_critic_def, inputs=[critic_key, observations, actions, task_ids_init])
            target_critic = initialize_target_critic(critic, target_critic)
            temp = Model.create(Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            return actor, critic, target_critic, temp, rng

        self.init_models = jax.jit(_init_models)
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seed)
        self.target_critic_reference_def = Critic(
            num_tasks=num_tasks,
            embedding_size=embedding_size,
            ensemble_size=ensemble_size,
            hidden_dims=width_critic,
            depth=2,
            output_nodes=num_bins,
            multitask=self.multitask,
            task_embedding_norm=task_embedding_norm,
            critic_precision='fp32',
            fp8_all_dense_kernels=fp8_all_dense_kernels,
            fp8_input_dense_kernel=fp8_input_dense_kernel,
            fp8_output_dense_kernel=fp8_output_dense_kernel,
        )
        self.normalizer_rng = jax.random.PRNGKey(self.seed + 104729)
        self.task_entropies = jnp.full((num_tasks,), self.target_entropy, dtype=jnp.float32)
        self.task_entropy_counts = jnp.zeros((num_tasks,), dtype=jnp.int32)
        self.last_target_ema_diagnostics = None
        self.last_online_resident_diagnostics = None
        self.step = 1

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0):
        inputs = build_actor_input(self.critic, observations, self.task_ids, self.multitask)
        rng, actions = _sample_actions(self.rng, self.actor, inputs, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def update(
        self,
        batch: Batch,
        num_updates: int,
        env_step: int,
        collect_update_diagnostics: bool = False,
    ):

        (
            step,
            rng,
            actor,
            critic,
            target_critic,
            temp,
            info,
            update_diagnostics,
        ) = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.num_bins,
            self.v_max,
            self.multitask,
            self.num_tasks,
            self.step,
            num_updates,
            collect_update_diagnostics,
        )
        entropy_by_task = info.pop('_entropy_by_task')
        entropy_counts = info.pop('_entropy_counts_by_task')
        self.task_entropies = jnp.where(
            entropy_counts > 0, entropy_by_task, self.task_entropies
        )
        self.task_entropy_counts = entropy_counts.astype(jnp.int32)
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.last_target_ema_diagnostics = (
            update_diagnostics['target_ema']
            if collect_update_diagnostics
            else None
        )
        self.last_online_resident_diagnostics = (
            update_diagnostics['online_resident']
            if collect_update_diagnostics
            else None
        )
        return info
    
    def get_infos(self, batch: Batch):
        infos = _get_infos(            
                    self.rng,
                    self.actor,
                    self.critic,
                    self.target_critic,
                    self.temp,
                    batch,
                    self.discount,
                    self.target_entropy,
                    self.num_bins,
                    self.v_max,
                    self.multitask,
                    self.num_tasks)
        return infos
    
    def get_temperature(self):
        return _get_temperature(self.temp)

    def get_task_entropies(self):
        return self.task_entropies

    def estimate_bootstrap_values(self, next_observations, truncates):
        """Return host values only when an episode actually hits a time limit."""
        rng, values = _estimate_bootstrap_values(
            self.normalizer_rng,
            self.actor,
            self.critic,
            self.target_critic,
            next_observations,
            self.task_ids,
            self.num_bins,
            self.v_max,
            self.multitask,
        )
        self.normalizer_rng = rng
        values = np.asarray(values)
        return np.where(np.asarray(truncates), values, 0.0)

    def get_tensor_diagnostics(self, batch: Batch):
        """Return device-resident trees used only at tensor-stat intervals."""
        target_params = reconstruct_target_params(
            self.critic, self.target_critic
        )
        target_reference = None
        if self.target_critic.fp8_meta is not None:
            target_reference = self.target_critic.replace(
                apply_fn=self.target_critic_reference_def,
                params=target_params,
                fp8_meta=None,
            )
        forward = _get_forward_diagnostics(
            self.actor,
            self.critic,
            self.target_critic,
            target_reference,
            batch,
            self.multitask,
            self.num_bins,
            self.v_max,
        )
        gradients = _get_gradient_diagnostics(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.num_bins,
            self.v_max,
            self.multitask,
        )
        diagnostics = {
            'params': {
                'actor': self.actor.params,
                'critic': dequantize_critic_params(self.critic),
                (
                    'target_critic_reconstructed'
                    if self.target_critic_precision == 'fp8_lag'
                    else 'target_critic_dequantized'
                ): target_params,
                'temperature': self.temp.params,
            },
            'optimizer': {
                'actor': self.actor.opt_state,
                'critic': self.critic.opt_state,
                'temperature': self.temp.opt_state,
            },
            'gradients': gradients,
            **forward,
        }
        if self.critic.fp8_meta is not None:
            diagnostics['fp8'] = self._fp8_diagnostics(self.critic)
        if self.critic_precision == 'fp8_resident':
            critic_meta_flat = traverse_util.flatten_dict(
                self.critic.fp8_meta, sep='/'
            )
            storage = {
                'kernel_scales': _split_ensemble_arrays({
                    path: value
                    for path, value in critic_meta_flat.items()
                    if path.endswith('/kernel_scale')
                }),
            }
            fixed_anchor_norms = {
                path: value
                for path, value in critic_meta_flat.items()
                if path.endswith('/kernel_anchor_norm')
            }
            if fixed_anchor_norms:
                storage['fixed_anchor_norms'] = _split_ensemble_arrays(
                    fixed_anchor_norms
                )
            if self.last_online_resident_diagnostics is not None:
                storage['last_applied_update'] = (
                    self.last_online_resident_diagnostics
                )
            diagnostics['fp8_online_storage'] = storage
            if self.fp8_resident_carry:
                logical_params = traverse_util.flatten_dict(
                    reconstruct_carry_logical_critic_params(self.critic),
                    sep='/',
                )
                physical_params = traverse_util.flatten_dict(
                    dequantize_critic_params(self.critic), sep='/'
                )
                stored_params = traverse_util.flatten_dict(
                    self.critic.params, sep='/'
                )
                resident_paths = {
                    path for path, value in stored_params.items()
                    if value.dtype == jnp.float8_e4m3fn
                }
                storage['carry_codes'] = _split_ensemble_arrays({
                    path: value
                    for path, value in critic_meta_flat.items()
                    if path.endswith('/kernel_carry')
                })
                storage['logical_kernels'] = _split_ensemble_arrays({
                    path: value
                    for path, value in logical_params.items()
                    if path in resident_paths
                })
                storage['physical_kernels'] = _split_ensemble_arrays({
                    path: value
                    for path, value in physical_params.items()
                    if path in resident_paths
                })
        if self.target_critic_precision in ('fp8_direct', 'fp8_lag'):
            diagnostics['fp8_target_forward'] = self._fp8_diagnostics(
                self.target_critic, include_output_grad=False
            )
        if self.target_critic_precision == 'fp8_resident':
            target_params_flat = traverse_util.flatten_dict(
                self.target_critic.params, sep='/'
            )
            target_meta_flat = traverse_util.flatten_dict(
                self.target_critic.fp8_meta, sep='/'
            )
            target_storage = {
                'codes': _split_ensemble_arrays({
                    path: value
                    for path, value in target_params_flat.items()
                    if value.dtype == jnp.float8_e4m3fn
                }),
                'kernel_scales': _split_ensemble_arrays({
                    path: value
                    for path, value in target_meta_flat.items()
                    if path.endswith('/kernel_scale')
                }),
            }
            if self.last_target_ema_diagnostics is not None:
                target_storage['last_applied_ema'] = (
                    self.last_target_ema_diagnostics
                )
            diagnostics['fp8_target_storage'] = target_storage
        if self.critic_optimizer_state != 'fp32':
            diagnostics['critic_moments'] = moment_diagnostics(self.critic.opt_state)
        if self.target_critic_precision == 'fp8_lag':
            target_params_flat = traverse_util.flatten_dict(
                self.target_critic.params, sep='/'
            )
            target_meta_flat = traverse_util.flatten_dict(
                self.target_critic.fp8_meta, sep='/'
            )
            lag_storage = {
                'codes': _split_ensemble_arrays({
                    path: value
                    for path, value in target_params_flat.items()
                    if value.dtype == jnp.float8_e4m3fn
                }),
                'lag_scales': _split_ensemble_arrays({
                    path: value
                    for path, value in target_meta_flat.items()
                    if path.endswith('/lag_scale')
                }),
            }
            if self.last_target_ema_diagnostics is not None:
                lag_storage['last_applied_ema'] = (
                    self.last_target_ema_diagnostics
                )
            diagnostics['fp8_target_lag'] = lag_storage
        return diagnostics

    def _fp8_diagnostics(self, model, include_output_grad=True):
        flat_meta = traverse_util.flatten_dict(model.fp8_meta, sep='/')
        fp8_max = {
            'input': jnp.asarray(jnp.finfo(jnp.float8_e4m3fn).max, jnp.float32),
            'kernel': jnp.asarray(jnp.finfo(jnp.float8_e4m3fn).max, jnp.float32),
            'output_grad': jnp.asarray(jnp.finfo(jnp.float8_e5m2).max, jnp.float32),
        }
        diagnostics = {}
        for path, history in flat_meta.items():
            if not path.endswith('_amax_history'):
                continue
            kind = path.rsplit('/', 1)[-1].removesuffix('_amax_history')
            if kind == 'output_grad' and not include_output_grad:
                continue
            layer_path = path.rsplit('/', 1)[0]
            scale = flat_meta[f'{layer_path}/{kind}_scale'][..., 0]
            current_amax = history[..., 0]
            diagnostics[f'{layer_path}/{kind}'] = {
                'scale': scale,
                'current_amax': current_amax,
                'history_amax': jnp.max(history, axis=-1),
                'saturation_risk_ratio': current_amax / (fp8_max[kind] * scale),
            }
        return diagnostics

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
        
    def save(self, path, include_optimizer=True):
        self.actor.save(f'{path}/actor.msgpack', include_optimizer)
        self.critic.save(f'{path}/critic.msgpack', include_optimizer)
        self.target_critic.save(f'{path}/target_critic.msgpack', include_optimizer)
        self.temp.save(f'{path}/temp.msgpack', include_optimizer)
        import pickle
        with open(f'{path}/agent_state.pkl', 'wb') as file:
            pickle.dump({
                'step': self.step,
                'rng': np.asarray(self.rng),
                'normalizer_rng': np.asarray(self.normalizer_rng),
                'task_entropies': np.asarray(self.task_entropies),
                'task_entropy_counts': np.asarray(self.task_entropy_counts),
            }, file)
        
    def load(self, path):
        import os
        import pickle
        target_critic = self.target_critic.load(
            f'{path}/target_critic.msgpack',
            require_fp8_metadata=(
                self.target_critic_precision in (
                    'fp8_direct', 'fp8_resident', 'fp8_lag'
                )
            ),
        )
        self.actor = self.actor.load(f'{path}/actor.msgpack')
        self.critic = self.critic.load(
            f'{path}/critic.msgpack',
            require_fp8_metadata=self.critic_precision in (
                'fp8_resident', 'fp8_current_master'
            ),
        )
        self.target_critic = target_critic
        self.temp = self.temp.load(f'{path}/temp.msgpack')
        state_path = f'{path}/agent_state.pkl'
        if os.path.exists(state_path):
            with open(state_path, 'rb') as file:
                state = pickle.load(file)
            self.step = int(state['step'])
            self.rng = jnp.asarray(state['rng'])
            self.normalizer_rng = jnp.asarray(
                state.get('normalizer_rng', self.normalizer_rng)
            )
            self.task_entropies = jnp.asarray(
                state.get('task_entropies', self.task_entropies)
            )
            self.task_entropy_counts = jnp.asarray(
                state.get('task_entropy_counts', self.task_entropy_counts)
            )
