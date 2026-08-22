import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

from jaxrl.checkpoint import checkpoint_config_value
from jaxrl.envs import ParallelEnv


class FakeMetaWorldEnv:
    def __init__(self, seed):
        self.seed = seed
        self.observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._freeze_rand_vec = True
        self.reset_seeds = []

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return np.full(3, seed if seed is not None else -1, dtype=np.float32), {}


class MetaWorldResetProtocolTest(unittest.TestCase):
    @staticmethod
    def _factory(made):
        def factory(_name, seed):
            env = FakeMetaWorldEnv(seed)
            made.append(env)
            return env
        return factory

    def test_default_resample_reuses_env_and_advances_reproducible_seed_stream(self):
        made = []
        with mock.patch('jaxrl.envs.make_env', side_effect=self._factory(made)):
            env = ParallelEnv(['fake-goal-observable'], seed=7)
            original = env.envs[0]
            first = env.reset()
            second = env.reset()
            self.assertIs(env.envs[0], original)
            self.assertFalse(original._freeze_rand_vec)
            self.assertEqual(len(made), 1)
            self.assertFalse(np.array_equal(first, second))

        made_again = []
        with mock.patch('jaxrl.envs.make_env', side_effect=self._factory(made_again)):
            repeated = ParallelEnv(['fake-goal-observable'], seed=7)
            np.testing.assert_array_equal(repeated.reset(), first)
            np.testing.assert_array_equal(repeated.reset(), second)

    def test_frozen_mode_preserves_constructor_configuration(self):
        made = []
        with mock.patch('jaxrl.envs.make_env', side_effect=self._factory(made)):
            env = ParallelEnv(
                ['fake-goal-observable'], seed=7,
                metaworld_reset_mode='frozen',
            )
            original = env.envs[0]
            env.reset()
            self.assertIs(env.envs[0], original)
            self.assertTrue(original._freeze_rand_vec)
            self.assertEqual(len(made), 1)

    def test_historical_checkpoint_without_mode_is_frozen(self):
        self.assertEqual(checkpoint_config_value({}, 'metaworld_reset_mode'), 'frozen')
        self.assertEqual(
            checkpoint_config_value({'metaworld_reset_mode': 'resample'}, 'metaworld_reset_mode'),
            'resample',
        )


if __name__ == '__main__':
    unittest.main()
