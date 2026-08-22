import unittest
from unittest import mock

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl.agent.brc_learner import BRC
from jaxrl.checkpoint import checkpoint_config_value
from jaxrl.envs import ParallelEnv
from jaxrl.networks import TaskEmbedding
from jaxrl.normalizer import RewardNormalizer
from jaxrl.paper_alignment import resolve_paper_alignment
from jaxrl.utils import Batch


class AlignmentPresetTest(unittest.TestCase):
    def test_historical_eval_seed_protocol_is_explicit(self):
        self.assertEqual(checkpoint_config_value({}, 'eval_seed_offset'), 42)
        self.assertEqual(
            checkpoint_config_value({'eval_seed_offset': 0}, 'eval_seed_offset'),
            0,
        )

    def test_presets_and_independent_override(self):
        self.assertEqual(resolve_paper_alignment(False), {
            'task_embedding_norm': 'l2',
            'return_bootstrap': 'reward_mean',
            'entropy_correction': 'target_entropy',
        })
        self.assertEqual(resolve_paper_alignment(True), {
            'task_embedding_norm': 'l1',
            'return_bootstrap': 'critic',
            'entropy_correction': 'empirical_per_task',
        })
        result = resolve_paper_alignment(True, return_bootstrap='reward_mean')
        self.assertEqual(result['task_embedding_norm'], 'l1')
        self.assertEqual(result['return_bootstrap'], 'reward_mean')


class TaskEmbeddingTest(unittest.TestCase):
    def test_l1_and_l2_constraints(self):
        task_ids = jnp.arange(4, dtype=jnp.int32)
        for norm_name, ord_value in [('l1', 1), ('l2', 2)]:
            module = TaskEmbedding(4, 8, norm=norm_name)
            variables = module.init(jax.random.PRNGKey(0), task_ids)
            embeddings = np.asarray(module.apply(variables, task_ids))
            np.testing.assert_allclose(
                np.linalg.norm(embeddings, ord=ord_value, axis=-1),
                np.ones(4),
                rtol=1e-5,
                atol=1e-6,
            )


class RewardNormalizerAlignmentTest(unittest.TestCase):
    def test_critic_bootstrap_is_used_only_for_truncation(self):
        normalizer = RewardNormalizer(
            1, target_entropy=-2.0, discount=0.5,
            return_bootstrap='critic',
        )
        minimum, maximum = normalizer._calculate_returns_variable_length_trajectory(
            np.array([1.0, 2.0], np.float32), True, bootstrap_value=10.0
        )
        self.assertAlmostEqual(minimum, 4.5)
        self.assertAlmostEqual(maximum, 7.0)
        minimum, maximum = normalizer._calculate_returns_variable_length_trajectory(
            np.array([1.0, 2.0], np.float32), False, bootstrap_value=10.0
        )
        self.assertAlmostEqual(minimum, 2.0)
        self.assertAlmostEqual(maximum, 2.0)

    def test_fixed_length_true_terminal_uses_zero_bootstrap(self):
        normalizer = RewardNormalizer(
            1, target_entropy=-2.0, discount=0.5, max_steps=2,
            return_bootstrap='critic',
        )
        normalizer.update(
            np.array([1.0]), np.array([False]), np.array([False])
        )
        normalizer.update(
            np.array([2.0]), np.array([True]), np.array([False])
        )
        np.testing.assert_allclose(normalizer.returns_min_norm, [2.0])
        np.testing.assert_allclose(normalizer.returns_max_norm, [2.0])

    def test_empirical_entropy_equation_and_value_units(self):
        normalizer = RewardNormalizer(
            2, target_entropy=-2.0, discount=0.9, v_max=10.0,
            entropy_correction='empirical_per_task',
        )
        normalizer.returns_min_norm[:] = [-5.0, -6.0]
        normalizer.returns_max_norm[:] = [20.0, 30.0]
        entropies = np.array([2.0, 4.0], np.float32)
        scale = np.asarray(normalizer._denominator_by_task(0.1, entropies))
        np.testing.assert_allclose(scale, [2.2, 3.4], rtol=1e-6)
        np.testing.assert_allclose(
            normalizer.denormalize_values([1.0, -2.0], 0.1, entropies),
            [2.2, -6.8],
            rtol=1e-6,
        )

    def test_legacy_formula_is_unchanged(self):
        normalizer = RewardNormalizer(1, target_entropy=-2.0, discount=0.9, v_max=10.0)
        normalizer.returns_min_norm[:] = [-5.0]
        normalizer.returns_max_norm[:] = [20.0]
        scale = np.asarray(normalizer._denominator_by_task(0.1))
        np.testing.assert_allclose(scale, [2.1], rtol=1e-6)


class FakeMetaWorldEnv:
    def __init__(self, seed):
        self.seed = seed
        self.observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.closed = False
        self._freeze_rand_vec = True
        self.reset_seeds = []

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return np.full(3, self.seed, dtype=np.float32), {}

    def close(self):
        self.closed = True


class MetaWorldResetProtocolTest(unittest.TestCase):
    def test_recreate_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ParallelEnv([], seed=7, metaworld_reset_mode='recreate')

    def test_resample_mode_reuses_instance_and_unfreezes_rand_vec(self):
        made = []

        def factory(_name, seed):
            env = FakeMetaWorldEnv(seed)
            made.append(env)
            return env

        with mock.patch('jaxrl.envs.make_env', side_effect=factory):
            env = ParallelEnv(
                ['fake-goal-observable'], seed=7,
                metaworld_reset_mode='resample',
            )
            original = env.envs[0]
            env.reset()
            first_seed = original.reset_seeds[-1]
            env.reset()
            self.assertIs(env.envs[0], original)
            self.assertFalse(original._freeze_rand_vec)
            self.assertFalse(original.closed)
            self.assertEqual(len(made), 1)
            self.assertNotEqual(original.reset_seeds[-1], first_seed)

    def test_frozen_mode_reuses_the_instance(self):
        made = []

        def factory(_name, seed):
            env = FakeMetaWorldEnv(seed)
            made.append(env)
            return env

        with mock.patch('jaxrl.envs.make_env', side_effect=factory):
            env = ParallelEnv(
                ['fake-goal-observable'], seed=7,
                metaworld_reset_mode='frozen',
            )
            original = env.envs[0]
            env.reset()
            self.assertIs(env.envs[0], original)
            self.assertEqual(len(made), 1)


class EmpiricalEntropyIntegrationTest(unittest.TestCase):
    def test_update_retains_per_task_entropy_without_exposing_batch_arrays(self):
        agent = BRC(
            0,
            np.zeros((1, 4), np.float32),
            np.zeros((1, 2), np.float32),
            num_tasks=2,
            width_critic=16,
            width_actor=16,
            updates_per_step=1,
            task_embedding_norm='l1',
        )
        batch = Batch(
            observations=np.zeros((1, 4, 4), np.float32),
            actions=np.zeros((1, 4, 2), np.float32),
            rewards=np.zeros((1, 4), np.float32),
            masks=np.ones((1, 4), np.float32),
            next_observations=np.zeros((1, 4, 4), np.float32),
            task_ids=np.array([[0, 0, 1, 1]], np.int32),
        )
        info = agent.update(batch, 1, 1)
        jax.tree_util.tree_map(
            lambda value: value.block_until_ready() if hasattr(value, 'block_until_ready') else value,
            info,
        )
        self.assertNotIn('_entropy_by_task', info)
        self.assertEqual(np.asarray(agent.get_task_entropies()).shape, (2,))
        np.testing.assert_array_equal(np.asarray(agent.task_entropy_counts), [2, 2])


if __name__ == '__main__':
    unittest.main()
