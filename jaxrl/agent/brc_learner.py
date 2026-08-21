import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agent.update import (
    build_actor_input,
    get_actor_gradients,
    get_critic_gradients,
    update_actor,
    update_critic,
    update_target_critic,
    update_temperature,
)

from jaxrl.networks import NormalTanhPolicy, Critic, Temperature
from jaxrl.utils import Model, PRNGKey, Batch


@functools.partial(jax.jit, static_argnames=('discount', 'target_entropy', 'num_bins', 'v_max', 'multitask'),)
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None, None, None, None, None))
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
    multitask: bool
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    _, critic_info = update_critic(critic_key, actor, critic, target_critic, temp, batch, discount, num_bins, v_max, multitask)
    _, actor_info = update_actor(actor_key, actor, critic, temp, batch, num_bins, v_max, multitask) 
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


@functools.partial(jax.jit, static_argnames=('multitask',))
def _get_forward_diagnostics(actor, critic, batch, multitask):
    actor_inputs = build_actor_input(
        critic, batch.observations, batch.task_ids, multitask
    )
    policy = actor(actor_inputs)
    q_logits = critic(batch.observations, batch.actions, batch.task_ids)
    _, actor_intermediates = actor.apply_fn.apply(
        {'params': actor.params}, actor_inputs, mutable=['intermediates']
    )
    _, critic_intermediates = critic.apply_fn.apply(
        {'params': critic.params},
        batch.observations,
        batch.actions,
        batch.task_ids,
        mutable=['intermediates'],
    )
    return {
        'inputs': {
            'actor': actor_inputs,
            'critic_observations': batch.observations,
            'critic_actions': batch.actions,
        },
        'outputs': {
            'policy_mean': policy.distribution.loc,
            'policy_std': policy.distribution.scale_diag,
            'critic_logits': q_logits,
        },
        'activations': {
            'actor': actor_intermediates.get('intermediates', {}),
            'critic': critic_intermediates.get('intermediates', {}),
        },
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
    multitask: bool
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    new_critic, critic_info = update_critic(critic_key, actor, critic, target_critic, temp, batch, discount, num_bins, v_max, multitask)
    new_target_critic = update_target_critic(new_critic, target_critic, tau)
    new_actor, actor_info = update_actor(actor_key, actor, new_critic, temp, batch, num_bins, v_max, multitask) 
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@functools.partial(jax.jit, static_argnames=('discount', 'tau', 'target_entropy', 'num_bins', 'v_max', 'multitask', 'num_updates'))
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
    step: int,    
    num_updates: int
):
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update(
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
            multitask
        )
        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info

    step, rng, actor, critic, target_critic, temp, info = one_step(0, (step, rng, actor, critic, target_critic, temp, {}))
    return jax.lax.fori_loop(1, num_updates, one_step, (step, rng, actor, critic, target_critic, temp, info))

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
    ) -> None:
        
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.seed = seed
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.num_bins = num_bins
        self.v_max = v_max
        
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
            critic_def = Critic(num_tasks=num_tasks, embedding_size=embedding_size, ensemble_size=ensemble_size, hidden_dims=width_critic, depth=2, output_nodes=num_bins, multitask=self.multitask)
            actor = Model.create(actor_def, inputs=[actor_key, actor_init], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(critic_def, inputs=[critic_key, observations, actions, task_ids_init], tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions, task_ids_init])
            temp = Model.create(Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            return actor, critic, target_critic, temp, rng

        self.init_models = jax.jit(_init_models)
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seed)
        self.step = 1

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0):
        inputs = build_actor_input(self.critic, observations, self.task_ids, self.multitask)
        rng, actions = _sample_actions(self.rng, self.actor, inputs, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def update(self, batch: Batch, num_updates: int, env_step: int):

        step, rng, actor, critic, target_critic, temp, info = _do_multiple_updates(
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
            self.step,
            num_updates
        )
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
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
                    self.multitask)
        return infos
    
    def get_temperature(self):
        return _get_temperature(self.temp)

    def get_tensor_diagnostics(self, batch: Batch):
        """Return device-resident trees used only at tensor-stat intervals."""
        forward = _get_forward_diagnostics(
            self.actor, self.critic, batch, self.multitask
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
        return {
            'params': {
                'actor': self.actor.params,
                'critic': self.critic.params,
                'target_critic': self.target_critic.params,
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
            pickle.dump({'step': self.step, 'rng': np.asarray(self.rng)}, file)
        
    def load(self, path):
        import os
        import pickle
        self.actor = self.actor.load(f'{path}/actor.msgpack')
        self.critic = self.critic.load(f'{path}/critic.msgpack')
        self.target_critic = self.target_critic.load(f'{path}/target_critic.msgpack')
        self.temp = self.temp.load(f'{path}/temp.msgpack')
        state_path = f'{path}/agent_state.pkl'
        if os.path.exists(state_path):
            with open(state_path, 'rb') as file:
                state = pickle.load(file)
            self.step = int(state['step'])
            self.rng = jnp.asarray(state['rng'])
