import tempfile
import unittest

import jax
import numpy as np
import flax
from flax import traverse_util

from jaxrl.agent.brc_learner import BRC
from jaxrl.checkpoint import validate_checkpoint_config
from jaxrl.utils import Batch, LegacySaveState


def make_agent(precision):
    return BRC(
        0,
        np.zeros((1, 4), np.float32),
        np.zeros((1, 2), np.float32),
        num_tasks=2,
        width_critic=16,
        width_actor=16,
        updates_per_step=1,
        critic_precision=precision,
        fp8_amax_history_length=8,
    )


def make_batch():
    shape = (1, 8)
    return Batch(
        observations=np.ones(shape + (4,), np.float32),
        actions=np.zeros(shape + (2,), np.float32),
        rewards=np.ones(shape, np.float32),
        masks=np.ones(shape, np.float32),
        next_observations=np.full(shape + (4,), 1.1, np.float32),
        task_ids=np.zeros(shape, np.int32),
    )


def assert_trees_equal(test, left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    test.assertEqual(len(left_leaves), len(right_leaves))
    for got, expected in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


class Fp8CriticTest(unittest.TestCase):
    def test_only_online_residual_dense_layers_use_fp8(self):
        fp32 = make_agent('fp32')
        fp8 = make_agent('fp8_direct')
        self.assertIsNone(fp32.critic.fp8_meta)
        self.assertIsNotNone(fp8.critic.fp8_meta)
        self.assertIsNone(fp8.target_critic.fp8_meta)

        fp8_flat = traverse_util.flatten_dict(fp8.critic.fp8_meta, sep='/')
        for kind in ['input', 'kernel', 'output_grad']:
            histories = [
                value for path, value in fp8_flat.items()
                if path.endswith(f'{kind}_amax_history')
            ]
            self.assertEqual(len(histories), 4)
            self.assertTrue(all(value.shape == (2, 8) for value in histories))

        self.assertEqual(
            jax.tree_util.tree_structure(fp32.critic.params),
            jax.tree_util.tree_structure(fp8.critic.params),
        )
        for leaf in jax.tree_util.tree_leaves(fp8.critic.params):
            self.assertEqual(leaf.dtype, np.dtype(np.float32))
        for leaf in jax.tree_util.tree_leaves(fp8.critic.opt_state):
            if np.issubdtype(leaf.dtype, np.floating):
                self.assertEqual(leaf.dtype, np.dtype(np.float32))

    def test_critic_and_actor_paths_both_advance_fp8_history(self):
        agent = make_agent('fp8_direct')
        info = agent.update(make_batch(), 1, env_step=1)
        jax.block_until_ready(info)
        flat = traverse_util.flatten_dict(agent.critic.fp8_meta, sep='/')
        for path, history in flat.items():
            if path.endswith('_amax_history'):
                self.assertGreaterEqual(np.count_nonzero(np.asarray(history)), 4, path)

        probe = Batch(*[value[0] for value in make_batch()])
        diagnostics = agent.get_tensor_diagnostics(probe)
        jax.block_until_ready(diagnostics)
        self.assertTrue(jax.tree_util.tree_leaves(diagnostics['activations']['critic']))
        self.assertEqual(len(diagnostics['fp8']), 12)
        self.assertTrue(all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree_util.tree_leaves(diagnostics['fp8'])
        ))

    def test_fp8_checkpoint_round_trip_and_next_update(self):
        source = make_agent('fp8_direct')
        batch = make_batch()
        source.update(batch, 1, env_step=1)
        with tempfile.TemporaryDirectory() as temp:
            source.save(temp, include_optimizer=True)
            restored = make_agent('fp8_direct')
            restored.load(temp)
            assert_trees_equal(self, source.critic.params, restored.critic.params)
            assert_trees_equal(self, source.critic.fp8_meta, restored.critic.fp8_meta)
            assert_trees_equal(self, source.critic.opt_state, restored.critic.opt_state)
            source_info = source.update(batch, 1, env_step=2)
            restored_info = restored.update(batch, 1, env_step=2)
            jax.block_until_ready((source_info, restored_info))
            assert_trees_equal(self, source_info, restored_info)
            assert_trees_equal(self, source.critic.params, restored.critic.params)
            assert_trees_equal(self, source.critic.fp8_meta, restored.critic.fp8_meta)

    def test_fp32_checkpoint_initializes_fresh_fp8_metadata(self):
        source = make_agent('fp32')
        source.update(make_batch(), 1, env_step=1)
        with tempfile.TemporaryDirectory() as temp:
            source.save(temp, include_optimizer=True)
            with open(f'{temp}/critic.msgpack', 'wb') as file:
                file.write(flax.serialization.to_bytes(LegacySaveState(
                    step=source.critic.step,
                    params=source.critic.params,
                    opt_state=source.critic.opt_state,
                )))
            restored = make_agent('fp8_direct')
            initial_meta = jax.tree.map(lambda value: np.asarray(value).copy(), restored.critic.fp8_meta)
            restored.load(temp)
            assert_trees_equal(self, source.critic.params, restored.critic.params)
            assert_trees_equal(self, source.critic.opt_state, restored.critic.opt_state)
            assert_trees_equal(self, initial_meta, restored.critic.fp8_meta)

    def test_checkpoint_precision_transition_policy(self):
        self.assertTrue(validate_checkpoint_config(
            {'critic_precision': 'fp32'},
            {'critic_precision': 'fp8_direct', 'fp8_amax_history_length': 8},
        ))
        with self.assertRaises(ValueError):
            validate_checkpoint_config(
                {'critic_precision': 'fp8_direct'},
                {'critic_precision': 'fp32'},
            )


if __name__ == '__main__':
    unittest.main()
