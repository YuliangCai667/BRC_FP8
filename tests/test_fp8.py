import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np
import flax
from flax import traverse_util

from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import (
    dequantize_target_params,
    target_ema_diagnostics,
    update_target_critic,
)
from jaxrl.checkpoint import CheckpointManager, validate_checkpoint_config
from jaxrl.networks import dequantize_e4m3, quantize_e4m3_per_tensor
from jaxrl.utils import Batch, LegacySaveState


def make_agent(precision, target_precision='fp32'):
    return BRC(
        0,
        np.zeros((1, 4), np.float32),
        np.zeros((1, 2), np.float32),
        num_tasks=2,
        width_critic=16,
        width_actor=16,
        updates_per_step=1,
        critic_precision=precision,
        target_critic_precision=target_precision,
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

    def test_resident_target_scope_and_native_dot(self):
        c_mode = make_agent('fp32', 'fp8_resident')
        d_mode = make_agent('fp8_direct', 'fp8_resident')
        c_params = traverse_util.flatten_dict(c_mode.target_critic.params, sep='/')
        d_params = traverse_util.flatten_dict(d_mode.target_critic.params, sep='/')
        resident_paths = [
            path for path, value in c_params.items()
            if value.dtype == jax.numpy.float8_e4m3fn
        ]
        self.assertEqual(len(resident_paths), 4)
        self.assertTrue(all('/BronetBlock_' in path for path in resident_paths))
        self.assertTrue(all(path.endswith('/kernel') for path in resident_paths))
        for path, value in c_params.items():
            if path not in resident_paths:
                self.assertEqual(value.dtype, np.dtype(np.float32))
        for path in resident_paths:
            np.testing.assert_array_equal(
                np.asarray(c_params[path]), np.asarray(d_params[path])
            )

        metadata = traverse_util.flatten_dict(
            c_mode.target_critic.fp8_meta, sep='/'
        )
        scales = {
            path: value for path, value in metadata.items()
            if path.endswith('/kernel_scale')
        }
        self.assertEqual(len(scales), 4)
        self.assertTrue(all(value.shape == (2,) for value in scales.values()))
        self.assertTrue(all(value.dtype == np.float32 for value in scales.values()))
        for model in (c_mode.critic, d_mode.critic):
            for leaf in jax.tree_util.tree_leaves(model.params):
                self.assertEqual(leaf.dtype, np.dtype(np.float32))
            for leaf in jax.tree_util.tree_leaves(model.opt_state):
                if np.issubdtype(leaf.dtype, np.floating):
                    self.assertEqual(leaf.dtype, np.dtype(np.float32))

        target = c_mode.target_critic
        observations = np.ones((3, 4), np.float32)
        actions = np.zeros((3, 2), np.float32)
        task_ids = np.zeros((3,), np.int32)
        jaxpr = str(jax.make_jaxpr(
            lambda params, meta: target.apply_fn.apply(
                {'params': params, '_overwrite_with_gradient': meta},
                observations,
                actions,
                task_ids,
            )
        )(target.params, target.fp8_meta))
        self.assertIn('f8_e4m3fn', jaxpr)
        self.assertIn('preferred_element_type=float32', jaxpr)

    def test_resident_target_ema_and_diagnostics(self):
        agent = make_agent('fp32', 'fp8_resident')
        target_params = traverse_util.flatten_dict(agent.target_critic.params)
        target_meta = traverse_util.flatten_dict(agent.target_critic.fp8_meta)
        online_params = traverse_util.flatten_dict(
            dequantize_target_params(agent.target_critic)
        )
        for path, codes in target_params.items():
            if codes.dtype != jax.numpy.float8_e4m3fn:
                continue
            scale_path = path[:-1] + ('kernel_scale',)
            old = jax.vmap(dequantize_e4m3)(codes, target_meta[scale_path])
            amax = np.max(np.abs(np.asarray(old)), axis=(-2, -1), keepdims=True)
            mask = np.abs(np.asarray(old)) < 0.5 * amax
            delta = np.asarray(target_meta[scale_path])[:, None, None] * 0.1
            online_params[path] = old + np.where(mask, delta, 0.0)
        fp32_path = next(
            path for path, value in target_params.items()
            if value.dtype == np.float32 and path[-1] == 'bias'
        )
        online_params[fp32_path] = online_params[fp32_path] + 0.25
        online = agent.critic.replace(
            params=traverse_util.unflatten_dict(online_params)
        )

        diagnostics = target_ema_diagnostics(
            online, agent.target_critic, 0.005
        )
        jax.block_until_ready(diagnostics)
        self.assertEqual(len(diagnostics), 8)
        self.assertTrue(any(
            float(value['swallowed_update_fraction']) > 0
            for value in diagnostics.values()
        ))
        self.assertTrue(all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree_util.tree_leaves(diagnostics)
        ))

        updated = update_target_critic(online, agent.target_critic, 0.005)
        updated_params = traverse_util.flatten_dict(updated.params)
        updated_meta = traverse_util.flatten_dict(updated.fp8_meta)
        for path, codes in target_params.items():
            if codes.dtype != jax.numpy.float8_e4m3fn:
                continue
            scale_path = path[:-1] + ('kernel_scale',)
            old = jax.vmap(dequantize_e4m3)(codes, target_meta[scale_path])
            candidate = old + 0.005 * (online_params[path] - old)
            expected_codes, expected_scale = jax.vmap(
                quantize_e4m3_per_tensor
            )(candidate)
            np.testing.assert_array_equal(
                np.asarray(updated_params[path]), np.asarray(expected_codes)
            )
            np.testing.assert_array_equal(
                np.asarray(updated_meta[scale_path]), np.asarray(expected_scale)
            )
        np.testing.assert_array_equal(
            np.asarray(updated_params[fp32_path]),
            np.asarray(
                online_params[fp32_path] * 0.005
                + target_params[fp32_path] * 0.995
            ),
        )

        batch = make_batch()
        agent.update(
            batch,
            1,
            env_step=1,
            collect_target_ema_diagnostics=True,
        )
        probe = Batch(*[value[0] for value in batch])
        full_diagnostics = agent.get_tensor_diagnostics(probe)
        jax.block_until_ready(full_diagnostics)
        self.assertTrue(jax.tree_util.tree_leaves(
            full_diagnostics['activations']['target_critic']
        ))
        activation_paths = traverse_util.flatten_dict(
            full_diagnostics['activations']['target_critic'], sep='/'
        )
        self.assertTrue(any('/ensemble_0' in path for path in activation_paths))
        self.assertTrue(any('/ensemble_1' in path for path in activation_paths))
        storage = full_diagnostics['fp8_target_storage']
        self.assertEqual(len(traverse_util.flatten_dict(storage['codes'])), 8)
        self.assertEqual(len(traverse_util.flatten_dict(storage['kernel_scales'])), 8)
        self.assertEqual(len(storage['last_applied_ema']), 8)
        self.assertEqual(
            set(full_diagnostics['target_forward_error']),
            {'aggregate', 'ensemble_0', 'ensemble_1'},
        )
        self.assertTrue(all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree_util.tree_leaves(
                full_diagnostics['target_forward_error']
            )
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

    def test_resident_target_checkpoint_round_trip_and_next_update(self):
        source = make_agent('fp8_direct', 'fp8_resident')
        batch = make_batch()
        source.update(batch, 1, env_step=1)
        with tempfile.TemporaryDirectory() as temp:
            manager = CheckpointManager(
                Path(temp) / 'checkpoints',
                'resident-test',
                Path(temp) / 'run',
                ['task'],
                {
                    'critic_precision': 'fp8_direct',
                    'target_critic_precision': 'fp8_resident',
                },
            )
            checkpoint = manager.save_analysis(source, env_step=1)
            manifest = CheckpointManager.read_manifest(checkpoint)
            self.assertIn('float8_e4m3fn', manifest['parameter_dtypes'])
            self.assertIn('float32', manifest['fp8_metadata_dtypes'])
            source.save(temp, include_optimizer=True)
            restored = make_agent('fp8_direct', 'fp8_resident')
            restored.load(temp)
            assert_trees_equal(
                self, source.target_critic.params, restored.target_critic.params
            )
            assert_trees_equal(
                self, source.target_critic.fp8_meta, restored.target_critic.fp8_meta
            )
            source_info = source.update(batch, 1, env_step=2)
            restored_info = restored.update(batch, 1, env_step=2)
            jax.block_until_ready((source_info, restored_info))
            assert_trees_equal(self, source_info, restored_info)
            assert_trees_equal(
                self, source.target_critic.params, restored.target_critic.params
            )
            assert_trees_equal(
                self, source.target_critic.fp8_meta, restored.target_critic.fp8_meta
            )
            assert_trees_equal(self, source.critic.opt_state, restored.critic.opt_state)

    def test_resident_target_rejects_direct_fp32_load(self):
        source = make_agent('fp32', 'fp32')
        with tempfile.TemporaryDirectory() as temp:
            source.save(temp, include_optimizer=True)
            restored = make_agent('fp32', 'fp8_resident')
            with self.assertRaisesRegex(ValueError, 'required FP8 model metadata'):
                restored.load(temp)

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
        with self.assertRaises(ValueError):
            validate_checkpoint_config(
                {'target_critic_precision': 'fp32'},
                {'target_critic_precision': 'fp8_resident'},
            )
        with self.assertRaises(ValueError):
            validate_checkpoint_config(
                {'target_critic_precision': 'fp8_resident'},
                {'target_critic_precision': 'fp32'},
            )


if __name__ == '__main__':
    unittest.main()
