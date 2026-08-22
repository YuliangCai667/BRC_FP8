import functools
import jax.numpy as jnp
import jax
from jaxrl.utils import Batch, Model, PRNGKey, tree_norm

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
    next_q_logits = target_critic(batch.next_observations, next_actions, batch.task_ids)
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
    return new_critic, info

def update_target_critic(critic: Model, target_critic: Model, tau: float):
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
    next_q_logits = target_critic(batch.next_observations, next_actions, batch.task_ids)
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
