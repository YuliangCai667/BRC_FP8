import unittest

import flax
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import (
    _is_resident_kernel,
    _scale_path,
    dequantize_target_params,
    update_target_critic,
)
from jaxrl.target_simulation import (
    candidate_alphas,
    dequantize_e4m3_blocks,
    initialize_shadow_states,
    lag_recurrence_exact,
    quantize_e4m3_blocks,
    reconstruct_shadow_params,
    teacher_update_once,
    teacher_shadow_update,
    update_interleaved_block,
    update_lag_coded,
    update_naive_block_scale,
    update_naive_per_tensor,
)
from jaxrl.utils import Batch


def make_agent(online='fp8_direct', target='fp8_direct'):
    return BRC(
        3,
        np.zeros((1, 4), np.float32),
        np.zeros((1, 2), np.float32),
        num_tasks=2,
        width_critic=16,
        width_actor=16,
        updates_per_step=1,
        critic_precision=online,
        target_critic_precision=target,
        fp8_amax_history_length=8,
    )


def make_batch():
    return Batch(
        observations=np.ones((8, 4), np.float32),
        actions=np.zeros((8, 2), np.float32),
        rewards=np.ones((8,), np.float32),
        masks=np.ones((8,), np.float32),
        next_observations=np.full((8, 4), 1.1, np.float32),
        task_ids=np.arange(8, dtype=np.int32) % 2,
    )


def assert_trees_equal(test, left, right):
    test.assertEqual(jax.tree.structure(left), jax.tree.structure(right))
    for got, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


class TargetSimulationTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.shadows = initialize_shadow_states(
            self.agent.critic.params, self.agent.target_critic.params, 8
        )

    def test_unquantized_lag_recurrence_is_exact_ema(self):
        rng = np.random.RandomState(0)
        online_old = rng.randn(2, 5, 5).astype(np.float32)
        online_new = online_old + rng.randn(2, 5, 5).astype(np.float32) * 0.01
        target_old = rng.randn(2, 5, 5).astype(np.float32)
        tau = 0.005
        lag_new = lag_recurrence_exact(
            target_old - online_old, online_old, online_new, tau
        )
        reconstructed = online_new + lag_new
        expected = target_old + tau * (online_new - target_old)
        np.testing.assert_allclose(reconstructed, expected, rtol=2e-6, atol=2e-6)

    def test_lag_state_scope_dtype_and_ensemble_scaling(self):
        flat_codes = traverse_util.flatten_dict(self.shadows.lag_coded.codes)
        flat_scales = traverse_util.flatten_dict(self.shadows.lag_coded.scales)
        self.assertEqual(len(flat_codes), 4)
        self.assertTrue(all(_is_resident_kernel(path) for path in flat_codes))
        for path, codes in flat_codes.items():
            self.assertEqual(codes.dtype, jnp.float8_e4m3fn)
            self.assertEqual(codes.shape, (2, 16, 16))
            self.assertEqual(flat_scales[path].shape, (2,))
            self.assertEqual(flat_scales[path].dtype, jnp.float32)

    def test_block_quantizer_formula_and_shapes(self):
        values = jnp.arange(2 * 16 * 16, dtype=jnp.float32).reshape(2, 16, 16)
        codes, scales = quantize_e4m3_blocks(values, 8)
        restored = dequantize_e4m3_blocks(codes, scales, 8)
        self.assertEqual(codes.dtype, jnp.float8_e4m3fn)
        self.assertEqual(scales.shape, (2, 2, 2))
        expected_amax = np.asarray(values).reshape(2, 2, 8, 2, 8).transpose(
            0, 1, 3, 2, 4
        ).max(axis=(-2, -1))
        np.testing.assert_allclose(np.asarray(scales), expected_amax / 448.0)
        self.assertLess(
            float(jnp.max(jnp.abs(restored - values))),
            float(jnp.max(scales)) * 16,
        )

    def test_naive_per_tensor_matches_resident_ema(self):
        resident = make_agent('fp8_direct', 'fp8_resident')
        flat_params = traverse_util.flatten_dict(resident.critic.params)
        moved = {
            path: value + (0.01 if _is_resident_kernel(path) else 0.0)
            for path, value in flat_params.items()
        }
        online = resident.critic.replace(
            params=flax.core.freeze(traverse_util.unflatten_dict(moved))
        )
        expected = update_target_critic(online, resident.target_critic, 0.005)
        state = initialize_shadow_states(
            resident.critic.params,
            dequantize_target_params(resident.target_critic),
            8,
        ).naive_per_tensor
        state = update_naive_per_tensor(state, online.params, 0.005)
        got_codes = traverse_util.flatten_dict(state.codes)
        got_scales = traverse_util.flatten_dict(state.scales)
        expected_codes = traverse_util.flatten_dict(expected.params)
        expected_meta = traverse_util.flatten_dict(expected.fp8_meta)
        for path in got_codes:
            np.testing.assert_array_equal(got_codes[path], expected_codes[path])
            np.testing.assert_array_equal(
                got_scales[path], expected_meta[_scale_path(path)]
            )

    def test_untriggered_interleaved_blocks_are_bitwise_unchanged(self):
        state = self.shadows.interleaved_block.replace(
            phases=jax.tree.map(jnp.zeros_like, self.shadows.interleaved_block.phases)
        )
        updated = update_interleaved_block(
            state, self.agent.critic.params, 0.005, 8
        )
        assert_trees_equal(self, state.codes, updated.codes)
        assert_trees_equal(self, state.scales, updated.scales)

    def test_interleaved_phase_and_shared_block_quantizer(self):
        moved = jax.tree.map(lambda value: value + 0.05, self.agent.critic.params)
        block = update_naive_block_scale(
            self.shadows.naive_block_scale, moved, 0.005, 8
        )
        phases = jax.tree.map(
            lambda value: jnp.full_like(value, 0.9999),
            self.shadows.interleaved_block.phases,
        )
        interleaved = update_interleaved_block(
            self.shadows.interleaved_block.replace(phases=phases),
            moved, 0.005, 8,
        )
        self.assertEqual(candidate_alphas(0.005)[-1], 1.0)
        self.assertTrue(all(
            np.isfinite(np.asarray(value)).all()
            for value in jax.tree.leaves(interleaved.scales)
        ))
        self.assertEqual(
            jax.tree.structure(block.codes), jax.tree.structure(interleaved.codes)
        )
        self.assertTrue(any(
            np.count_nonzero(np.asarray(value)) > 0
            for value in jax.tree.leaves(interleaved.events)
        ))

    def test_shadow_path_does_not_change_teacher_update(self):
        batch = make_batch()
        kwargs = (
            self.agent.discount, self.agent.tau, self.agent.target_entropy,
            self.agent.num_bins, self.agent.v_max, self.agent.multitask,
            self.agent.num_tasks,
        )
        baseline = teacher_update_once(
            self.agent.rng, self.agent.actor, self.agent.critic,
            self.agent.target_critic, self.agent.temp, batch, *kwargs,
        )
        simulated = teacher_shadow_update(
            self.agent.rng, self.agent.actor, self.agent.critic,
            self.agent.target_critic, self.agent.temp, batch, self.shadows,
            *kwargs, 8,
        )
        jax.block_until_ready(simulated)
        for baseline_value, simulated_value in zip(baseline[:5], simulated[:5]):
            assert_trees_equal(self, baseline_value, simulated_value)
        assert_trees_equal(self, baseline[5], simulated[6])

    def test_lag_update_reconstructs_finite_params(self):
        moved = jax.tree.map(lambda value: value + 0.001, self.agent.critic.params)
        state = update_lag_coded(
            self.shadows.lag_coded, self.agent.critic.params, moved, 0.005
        )
        params = reconstruct_shadow_params(
            'lag_coded', state, moved, self.agent.target_critic.params, 8
        )
        self.assertTrue(all(
            np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(params)
        ))


if __name__ == '__main__':
    unittest.main()
