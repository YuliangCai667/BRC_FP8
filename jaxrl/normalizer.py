import numpy as np
import jax.numpy as jnp
from jaxrl.utils import Batch

class RewardNormalizer(object):
    def __init__(self, num_seeds: int, target_entropy: float, discount: float = 0.99, v_max: float = 10.0, max_steps: int | None = None, return_bootstrap: str = 'reward_mean', entropy_correction: str = 'target_entropy'):
        if return_bootstrap not in ('reward_mean', 'critic'):
            raise ValueError("return_bootstrap must be 'reward_mean' or 'critic'")
        if entropy_correction not in ('target_entropy', 'empirical_per_task'):
            raise ValueError("entropy_correction must be 'target_entropy' or 'empirical_per_task'")
        self.returns_min_norm = np.zeros(num_seeds, dtype=np.float32) + np.inf
        self.returns_max_norm = np.zeros(num_seeds, dtype=np.float32) - np.inf           
        self.effective_horizon = 1 / (1 - discount)
        self.discount = discount
        self.v_max = v_max
        self.target_entropy = target_entropy        
        self.max_steps = max_steps
        self.return_bootstrap = return_bootstrap
        self.entropy_correction = entropy_correction
        self.step = 0
        self.rewards = np.zeros((num_seeds, max_steps), dtype=np.float32) if max_steps is not None else [[] for _ in range(num_seeds)]
        
    def _calculate_returns_variable_length_trajectory(self, rewards_traj: list, truncate: bool, bootstrap_value: float = 0.0):
        values = np.zeros_like(rewards_traj)
        if truncate:
            bootstrap = (
                float(bootstrap_value)
                if self.return_bootstrap == 'critic'
                else rewards_traj.mean() * self.effective_horizon
            )
        else:
            bootstrap = 0.0
        for i in reversed(range(rewards_traj.shape[0])):
            values[i] = rewards_traj[i] + self.discount * bootstrap
            bootstrap = values[i]
        return values.min(axis=-1), values.max(axis=-1)
    
    def _calculate_returns_fixed_length_trajectory(self, bootstrap_values=None):
        values = np.zeros_like(self.rewards, dtype=np.float32)
        if self.return_bootstrap == 'critic':
            bootstrap = (
                np.zeros(self.rewards.shape[0], dtype=np.float32)
                if bootstrap_values is None
                else np.asarray(bootstrap_values, dtype=np.float32)
            )
        else:
            bootstrap = self.rewards.mean(-1) * self.effective_horizon
        for i in reversed(range(values.shape[-1])):
            values[:, i] = self.rewards[:, i] + self.discount * bootstrap
            bootstrap = values[:, i]
        return values.min(axis=-1), values.max(axis=-1)
        
    def _update_variable_length_trajectory(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray, bootstrap_values=None):
        for i, reward in enumerate(rewards):
            self.rewards[i].append(reward)
        done = np.logical_or(terminal, truncate)
        if done.any():
            indx = done.nonzero()[0]
            for j in indx:
                rewards_traj = np.asarray(self.rewards[j])
                bootstrap_value = 0.0 if bootstrap_values is None else bootstrap_values[j]
                value_min, value_max = self._calculate_returns_variable_length_trajectory(
                    rewards_traj, truncate[j], bootstrap_value
                )
                self.returns_min_norm[j] = min(self.returns_min_norm[j], value_min) 
                self.returns_max_norm[j] = max(self.returns_max_norm[j], value_max) 
                self.rewards[j] = []
                
    def _update_fixed_length_trajectory(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray, bootstrap_values=None):
        self.rewards[:, self.step] = rewards
        dones = np.logical_or(terminal, truncate)
        if self.step == self.max_steps - 1:
            assert dones.all()
            v_min, v_max = self._calculate_returns_fixed_length_trajectory(bootstrap_values)
            self.returns_min_norm = np.where(v_min < self.returns_min_norm, v_min, self.returns_min_norm)
            self.returns_max_norm = np.where(v_max > self.returns_max_norm, v_max, self.returns_max_norm)            
            self.step = 0
        else:
            self.step += 1
        
    def update(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray, bootstrap_values=None):
        if self.return_bootstrap == 'critic' and np.any(truncate) and bootstrap_values is None:
            raise ValueError('critic bootstrap values are required when a trajectory truncates')
        if self.max_steps is not None:
            self._update_fixed_length_trajectory(rewards, terminal, truncate, bootstrap_values)
        else:
            self._update_variable_length_trajectory(rewards, terminal, truncate, bootstrap_values)
            
    def normalize(self, batches: Batch, temperature: np.ndarray, task_entropies=None):
        denominator = self._denominator_by_task(temperature, task_entropies)
        denominator = denominator[batches.task_ids]
        rewards = batches.rewards / denominator
        return Batch(observations=batches.observations, actions=batches.actions, rewards=rewards, masks=batches.masks, next_observations=batches.next_observations, task_ids=batches.task_ids)

    def _denominator_by_task(self, temperature, task_entropies=None):
        base = np.where(
            self.returns_max_norm > np.abs(self.returns_min_norm),
            self.returns_max_norm,
            np.abs(self.returns_min_norm),
        )
        if self.entropy_correction == 'empirical_per_task':
            if task_entropies is None:
                raise ValueError('per-task empirical entropies are required')
            # Paper: lambda_i = alpha * H_i / (1 - gamma), then
            # divide rewards by (Gbar_i + lambda_i) / v_max.
            denominator = (
                jnp.asarray(base)
                + temperature * jnp.asarray(task_entropies) * self.effective_horizon
            ) / self.v_max
        else:
            denominator = (
                base
                - temperature * self.effective_horizon * self.target_entropy / 2
            ) / self.v_max
        return denominator

    def denormalize_values(self, values, temperature, task_entropies=None):
        """Map critic values back to raw-return units for MC bootstrapping."""
        scale = np.asarray(
            self._denominator_by_task(temperature, task_entropies),
            dtype=np.float32,
        )
        # Before the first completed trajectory Gbar is infinite. The randomly
        # initialized critic is near zero, so a unit scale is the only finite,
        # neutral fallback until an empirical return scale exists.
        scale = np.where(np.isfinite(scale), scale, 1.0)
        return np.asarray(values, dtype=np.float32) * scale

    def diagnostics(self, temperature, task_entropies=None):
        """Return low-frequency host summaries of the active scaling rule."""
        base = np.where(
            self.returns_max_norm > np.abs(self.returns_min_norm),
            self.returns_max_norm,
            np.abs(self.returns_min_norm),
        )
        alpha = float(np.asarray(temperature))
        result = {'return_range_by_task': base.copy()}
        if self.entropy_correction == 'empirical_per_task':
            entropies = np.asarray(task_entropies, dtype=np.float32)
            correction = alpha * entropies * self.effective_horizon
        else:
            entropies = np.full_like(base, self.target_entropy)
            correction = -alpha * self.target_entropy * self.effective_horizon / 2
            correction = np.full_like(base, correction)
        result.update(
            task_entropy_by_task=entropies,
            entropy_correction_by_task=correction,
            reward_denominator_by_task=np.asarray(
                self._denominator_by_task(temperature, task_entropies)
            ),
        )
        return result

    def state_dict(self):
        """Persist completed-trajectory statistics only."""
        return {
            'returns_min_norm': self.returns_min_norm.copy(),
            'returns_max_norm': self.returns_max_norm.copy(),
        }

    def load_state_dict(self, state):
        self.returns_min_norm[...] = np.asarray(state['returns_min_norm'], dtype=np.float32)
        self.returns_max_norm[...] = np.asarray(state['returns_max_norm'], dtype=np.float32)
        self.step = 0
        if self.max_steps is None:
            self.rewards = [[] for _ in range(len(self.returns_min_norm))]
        else:
            self.rewards.fill(0)
