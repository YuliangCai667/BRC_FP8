import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax
from flax import traverse_util

from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import (
    _fixed_anchor_resident_affine,
    _optimizer_update_components,
    _probability_js,
    _resident_parameter_write,
    _scale_code_update_components,
    _update_geometry,
    dequantize_critic_params,
    dequantize_target_params,
    reconstruct_carry_logical_critic_params,
    reconstruct_target_params,
    target_ema_diagnostics,
    update_target_critic,
)
from jaxrl.checkpoint import (
    CheckpointManager,
    checkpoint_config_value,
    fixed_anchor_norm_report,
    validate_checkpoint_config,
)
from jaxrl.networks import (
    _quantize_carry_resident_kernel,
    _reconstruct_carry_logical_kernel,
    dequantize_e4m3,
    quantize_e4m3_per_tensor,
)
from jaxrl.utils import Batch, LegacySaveState


def make_agent(
    precision,
    target_precision='fp32',
    resident_canonicalization=False,
    resident_carry=False,
):
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
        fp8_resident_canonicalization=resident_canonicalization,
        fp8_resident_carry=resident_carry,
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


def assert_trees_allclose(test, left, right, *, rtol, atol):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    test.assertEqual(len(left_leaves), len(right_leaves))
    for got, expected in zip(left_leaves, right_leaves):
        got = np.asarray(got)
        expected = np.asarray(expected)
        if np.issubdtype(got.dtype, np.inexact):
            np.testing.assert_allclose(
                got, expected, rtol=rtol, atol=atol
            )
        else:
            np.testing.assert_array_equal(got, expected)


class Fp8CriticTest(unittest.TestCase):
    def test_probability_js_is_finite_when_midpoint_underflows(self):
        smallest = jnp.nextafter(jnp.float32(0), jnp.float32(1))
        left = jnp.asarray([[smallest, 1.0]], dtype=jnp.float32)
        right = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)

        value = jax.jit(_probability_js)(left, right)

        self.assertTrue(np.isfinite(np.asarray(value)).all())
        self.assertGreaterEqual(float(value[0]), 0.0)

    def test_jit_observes_materialized_e4m3_codes_before_dequantizing(self):
        candidate = jnp.asarray(
            [
                [0.03239834, -0.04254591, 0.00341173, -0.00137762],
                [0.02169952, -0.01309414, 0.00625866, -0.00643933],
                [-0.01987287, 0.00538195, 0.00177187, -0.03008418],
                [0.04048536, -0.04449455, -0.00386531, 0.01773307],
            ],
            dtype=jnp.float32,
        )
        candidates = jnp.stack((candidate, candidate * jnp.float32(0.75)))
        compiled = jax.jit(jax.vmap(_quantize_carry_resident_kernel))
        result = compiled(candidates)
        jax.block_until_ready(result)

        # This second dispatch consumes already-materialized code buffers.  It
        # must match the dequantization observed inside the first compiled graph.
        materialized_main = jax.vmap(dequantize_e4m3)(
            result['main_codes'], result['kernel_scale']
        )
        materialized_logical = jax.vmap(_reconstruct_carry_logical_kernel)(
            result['main_codes'],
            result['kernel_scale'],
            result['carry_codes'],
        )
        np.testing.assert_array_equal(
            np.asarray(result['main_physical']),
            np.asarray(materialized_main),
        )
        np.testing.assert_array_equal(
            np.asarray(result['logical_reconstruction']),
            np.asarray(materialized_logical),
        )
        main_error = np.linalg.norm(
            np.asarray(candidates - materialized_main), axis=(1, 2)
        )
        carry_error = np.linalg.norm(
            np.asarray(candidates - result['logical_reconstruction']),
            axis=(1, 2),
        )
        self.assertTrue(np.all(main_error > 0))
        self.assertTrue(np.all(carry_error < main_error))
        self.assertTrue(np.all(
            np.count_nonzero(np.asarray(result['carry_codes']), axis=(1, 2))
            > 0
        ))
        np.testing.assert_array_equal(
            np.asarray(result['carry_saturation_fraction']),
            np.zeros((2,), dtype=np.float32),
        )

    def test_carry_initialization_preserves_fp32_quantization_residual(self):
        fp32 = make_agent('fp32', 'fp8_direct')
        carry = make_agent(
            'fp8_resident', 'fp8_direct', resident_carry=True
        )
        fp32_params = traverse_util.flatten_dict(fp32.critic.params)
        stored = traverse_util.flatten_dict(carry.critic.params)
        metadata = traverse_util.flatten_dict(carry.critic.fp8_meta)
        logical = traverse_util.flatten_dict(
            reconstruct_carry_logical_critic_params(carry.critic)
        )
        physical = traverse_util.flatten_dict(
            dequantize_critic_params(carry.critic)
        )
        resident_paths = [
            path for path, value in stored.items()
            if value.dtype == jnp.float8_e4m3fn
        ]
        self.assertEqual(len(resident_paths), 4)
        for path in resident_paths:
            carry_path = path[:-1] + ('kernel_carry',)
            scale_path = path[:-1] + ('kernel_scale',)
            self.assertEqual(stored[path].dtype, jnp.float8_e4m3fn)
            self.assertEqual(metadata[carry_path].dtype, jnp.float8_e4m3fn)
            self.assertEqual(metadata[scale_path].dtype, jnp.float32)
            self.assertEqual(metadata[carry_path].shape, stored[path].shape)
            main_error = np.linalg.norm(
                np.asarray(fp32_params[path] - physical[path])
            )
            carry_error = np.linalg.norm(
                np.asarray(fp32_params[path] - logical[path])
            )
            self.assertLess(carry_error, main_error)
            for member in range(stored[path].shape[0]):
                quantized = _quantize_carry_resident_kernel(
                    fp32_params[path][member]
                )
                self.assertEqual(
                    float(quantized['carry_saturation_fraction']), 0.0
                )
        self.assertFalse(any(
            value.dtype == jnp.float32
            for path, value in metadata.items()
            if path[-1] == 'kernel_carry'
        ))
        for got, expected in zip(
            jax.tree_util.tree_leaves(
                reconstruct_carry_logical_critic_params(carry.critic)
            ),
            jax.tree_util.tree_leaves(carry.target_critic.params),
        ):
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(expected), rtol=2e-7, atol=1e-7
            )

    def test_carry_main_write_matches_current_amax_and_reduces_error(self):
        random = jax.random.normal(
            jax.random.PRNGKey(7), (37, 29), dtype=jnp.float32
        )
        mixed = random.at[0, 0].set(jnp.float32(64.0))
        mixed = mixed.at[1:8, 1:8].multiply(jnp.float32(1e-3))
        for candidate in (random, mixed, jnp.zeros((5, 3), jnp.float32)):
            expected_codes, expected_scale = quantize_e4m3_per_tensor(
                candidate
            )
            result = _quantize_carry_resident_kernel(candidate)
            np.testing.assert_array_equal(result['main_codes'], expected_codes)
            np.testing.assert_array_equal(result['kernel_scale'], expected_scale)
            np.testing.assert_array_equal(
                result['main_physical'],
                dequantize_e4m3(expected_codes, expected_scale),
            )
            self.assertEqual(result['main_codes'].dtype, jnp.float8_e4m3fn)
            self.assertEqual(result['carry_codes'].dtype, jnp.float8_e4m3fn)
            self.assertEqual(float(result['carry_saturation_fraction']), 0.0)
            main_error = float(jnp.linalg.norm(result['main_error']))
            carry_error = float(jnp.linalg.norm(result['carry_error']))
            if np.any(np.asarray(candidate)):
                self.assertLess(carry_error, main_error)
                candidate_norm = float(jnp.linalg.norm(candidate))
                self.assertLess(
                    carry_error / candidate_norm,
                    main_error / candidate_norm,
                )
            else:
                self.assertEqual(float(result['kernel_scale']), 1.0)
                self.assertEqual(main_error, 0.0)
                self.assertEqual(carry_error, 0.0)
                self.assertFalse(np.asarray(result['main_codes']).any())
                self.assertFalse(np.asarray(result['carry_codes']).any())

    def test_carry_accumulates_subgrid_updates_across_steps(self):
        reference = jnp.asarray([1.0, 0.125], dtype=jnp.float32)
        initial_codes, initial_scale = quantize_e4m3_per_tensor(reference)
        naive = dequantize_e4m3(initial_codes, initial_scale)
        carry_state = _quantize_carry_resident_kernel(reference)
        logical = carry_state['logical_reconstruction']
        small_update = jnp.asarray([0.0, 4e-4], dtype=jnp.float32)
        main_crossed_grid = False

        for _ in range(32):
            reference = reference + small_update
            naive_codes, naive_scale = quantize_e4m3_per_tensor(
                naive + small_update
            )
            naive = dequantize_e4m3(naive_codes, naive_scale)
            carry_state = _quantize_carry_resident_kernel(
                logical + small_update
            )
            logical = carry_state['logical_reconstruction']
            main_crossed_grid = main_crossed_grid or bool(
                carry_state['main_codes'][1] != initial_codes[1]
            )
            self.assertEqual(
                float(carry_state['carry_saturation_fraction']), 0.0
            )

        np.testing.assert_array_equal(naive_codes[1], initial_codes[1])
        self.assertTrue(main_crossed_grid)
        self.assertGreater(float(logical[1]), 0.125)
        self.assertLess(
            float(jnp.linalg.norm(logical - reference)),
            0.25 * float(jnp.linalg.norm(naive - reference)),
        )

    def test_carry_adamw_uses_logical_parameter_tree(self):
        agent = make_agent(
            'fp8_resident', 'fp8_direct', resident_carry=True
        )
        logical = reconstruct_carry_logical_critic_params(agent.critic)
        physical = dequantize_critic_params(agent.critic)
        grads = jax.tree.map(jnp.zeros_like, logical)
        expected_updates, expected_opt_state = agent.critic.tx.update(
            grads, agent.critic.opt_state, logical
        )
        expected_candidate = optax.apply_updates(logical, expected_updates)
        new_critic, _, candidate, _ = _resident_parameter_write(
            agent.critic, grads, {}, False
        )
        assert_trees_equal(self, candidate, expected_candidate)
        assert_trees_equal(self, new_critic.opt_state, expected_opt_state)

        physical_updates, _ = agent.critic.tx.update(
            grads, agent.critic.opt_state, physical
        )
        physical_candidate = optax.apply_updates(
            physical, physical_updates
        )
        logical_flat = traverse_util.flatten_dict(candidate)
        physical_flat = traverse_util.flatten_dict(physical_candidate)
        resident_paths = [
            path for path, value in traverse_util.flatten_dict(
                agent.critic.params
            ).items()
            if value.dtype == jnp.float8_e4m3fn
        ]
        self.assertTrue(any(
            np.any(
                np.asarray(logical_flat[path])
                != np.asarray(physical_flat[path])
            )
            for path in resident_paths
        ))
        self.assertFalse(any(
            leaf.dtype == jnp.float8_e4m3fn
            for leaf in jax.tree_util.tree_leaves(new_critic.opt_state)
            if hasattr(leaf, 'dtype')
        ))

    def test_target_ema_uses_carry_logical_online_kernel(self):
        agent = make_agent(
            'fp8_resident', 'fp8_direct', resident_carry=True
        )
        metadata = traverse_util.flatten_dict(agent.critic.fp8_meta)
        changed_metadata = dict(metadata)
        for path, value in metadata.items():
            if path[-1] == 'kernel_carry':
                changed_metadata[path] = jnp.full_like(value, 8.0)
        online = agent.critic.replace(
            fp8_meta=traverse_util.unflatten_dict(changed_metadata)
        )
        probe = Batch(*[value[0] for value in make_batch()])
        np.testing.assert_array_equal(
            np.asarray(agent.critic(
                probe.observations, probe.actions, probe.task_ids
            )),
            np.asarray(online(
                probe.observations, probe.actions, probe.task_ids
            )),
        )
        logical = traverse_util.flatten_dict(
            reconstruct_carry_logical_critic_params(online)
        )
        physical = traverse_util.flatten_dict(dequantize_critic_params(online))
        old_target = traverse_util.flatten_dict(agent.target_critic.params)
        tau = 0.25
        updated = traverse_util.flatten_dict(
            update_target_critic(online, agent.target_critic, tau).params
        )
        resident_paths = [
            path for path, value in traverse_util.flatten_dict(
                online.params
            ).items()
            if value.dtype == jnp.float8_e4m3fn
        ]
        for path in resident_paths:
            expected = logical[path] * tau + old_target[path] * (1 - tau)
            physical_only = (
                physical[path] * tau + old_target[path] * (1 - tau)
            )
            np.testing.assert_array_equal(updated[path], expected)
            self.assertTrue(np.any(
                np.asarray(updated[path]) != np.asarray(physical_only)
            ))

    def test_carry_update_reports_reconstruction_and_storage_metrics(self):
        agent = make_agent(
            'fp8_resident', 'fp8_direct', resident_carry=True
        )
        batch = make_batch()
        for env_step in range(1, 4):
            info = agent.update(
                batch,
                1,
                env_step=env_step,
                collect_update_diagnostics=env_step == 3,
            )
            jax.block_until_ready(info)
        rows = agent.last_online_resident_diagnostics['parameter_write']
        expected = {
            'carry_code_l2',
            'carry_zero_fraction',
            'carry_absmax',
            'carry_prequant_absmax',
            'carry_saturation_fraction',
            'main_write_relative_l2',
            'logical_reconstruction_relative_l2',
            'carry_error_reduction_ratio',
            'physical_kernel_norm',
            'logical_kernel_norm',
            'logical_to_physical_relative_l2',
        }
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(expected.issubset(row) for row in rows.values()))
        self.assertTrue(all(
            float(row['carry_saturation_fraction']) == 0.0
            for row in rows.values()
        ))
        self.assertTrue(all(
            float(row['logical_reconstruction_relative_l2'])
            < float(row['main_write_relative_l2'])
            for row in rows.values()
        ))
        self.assertTrue(all(
            float(row['carry_error_reduction_ratio']) > 1.0
            for row in rows.values()
        ))

        probe = Batch(*[value[0] for value in batch])
        diagnostics = agent.get_tensor_diagnostics(probe)
        jax.block_until_ready(diagnostics)
        storage = diagnostics['fp8_online_storage']
        self.assertTrue({
            'carry_codes', 'logical_kernels', 'physical_kernels'
        }.issubset(storage))
        carry_codes = jax.tree_util.tree_leaves(storage['carry_codes'])
        self.assertEqual(len(carry_codes), 8)
        self.assertTrue(all(
            value.dtype == jnp.float8_e4m3fn for value in carry_codes
        ))
        critic_grads = traverse_util.flatten_dict(
            diagnostics['gradients']['critic'], sep='/'
        )
        resident_grads = [
            value for path, value in critic_grads.items()
            if '/BronetBlock_' in path and path.endswith('/kernel')
        ]
        self.assertEqual(len(resident_grads), 4)
        self.assertTrue(all(
            np.linalg.norm(np.asarray(value)) > 0 for value in resident_grads
        ))

    def test_resident_fixed_anchor_preserves_codes_and_anchor_norms(self):
        candidate_kernel = jnp.asarray(
            [
                [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
                [[8.0, -1.0, 2.0], [0.25, -0.5, 1.0]],
            ],
            dtype=jnp.float32,
        )
        candidate_bias = jnp.asarray(
            [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
            dtype=jnp.float32,
        )
        next_codes, raw_next_scale = jax.vmap(
            quantize_e4m3_per_tensor
        )(candidate_kernel)
        original_codes = np.asarray(next_codes).copy()
        anchor_norm = jnp.asarray([5.0, 7.0], dtype=jnp.float32)

        canonical = _fixed_anchor_resident_affine(
            candidate_bias,
            next_codes,
            raw_next_scale,
            anchor_norm,
        )
        jax.block_until_ready(canonical)

        np.testing.assert_array_equal(np.asarray(next_codes), original_codes)
        canonical_norm = np.linalg.norm(
            np.asarray(canonical['canonical_kernel']), axis=(1, 2)
        )
        np.testing.assert_allclose(
            canonical_norm, np.asarray(anchor_norm), rtol=2e-6, atol=2e-6
        )
        expected_code_norm = np.linalg.norm(
            np.asarray(next_codes).astype(np.float32), axis=(1, 2)
        )
        expected_scale = np.asarray(anchor_norm) / expected_code_norm
        expected_alpha = expected_scale / np.asarray(raw_next_scale)
        np.testing.assert_allclose(canonical['alpha'], expected_alpha)
        np.testing.assert_allclose(
            canonical['canonical_scale'], expected_scale,
        )
        np.testing.assert_allclose(
            canonical['canonical_bias'],
            np.asarray(candidate_bias) * expected_alpha[:, None],
        )
        self.assertNotAlmostEqual(
            float(expected_alpha[0]), float(expected_alpha[1])
        )
        self.assertTrue(all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree_util.tree_leaves(canonical)
        ))

    def test_resident_fixed_anchor_is_stable_for_10000_writes(self):
        codes = jnp.asarray(
            [[[448.0, -256.0, 96.0], [40.0, -12.0, 1.5]]],
            dtype=jnp.float8_e4m3fn,
        )
        initial_scale = jnp.asarray([0.125], dtype=jnp.float32)
        anchor_norm = jnp.linalg.norm(
            jax.vmap(dequantize_e4m3)(codes, initial_scale), axis=(1, 2)
        )
        bias = jnp.zeros((1, 3), dtype=jnp.float32)

        def write_once(stored_scale, _):
            candidate = jax.vmap(dequantize_e4m3)(codes, stored_scale)
            next_codes, raw_scale = jax.vmap(
                quantize_e4m3_per_tensor
            )(candidate)
            result = _fixed_anchor_resident_affine(
                bias, next_codes, raw_scale, anchor_norm
            )
            physical_norm = jnp.linalg.norm(
                result['canonical_kernel'], axis=(1, 2)
            )
            return result['canonical_scale'], (
                result['canonical_scale'],
                physical_norm,
                jnp.all(next_codes == codes),
            )

        _, history = jax.jit(
            lambda: jax.lax.scan(write_once, initial_scale, None, length=10000)
        )()
        scales, physical_norms, codes_unchanged = jax.device_get(history)
        self.assertTrue(np.asarray(codes_unchanged).all())
        self.assertEqual(float(np.ptp(scales[:, 0])), 0.0)
        self.assertEqual(float(np.ptp(physical_norms[:, 0])), 0.0)
        np.testing.assert_allclose(
            physical_norms[:, 0] / float(anchor_norm[0]),
            1.0,
            rtol=2e-6,
            atol=2e-6,
        )

    def test_update_geometry_matches_projection_and_norm_identity(self):
        weight = jnp.asarray([[3.0, 4.0], [-2.0, 1.0]], dtype=jnp.float32)
        update = jnp.asarray([[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float32)
        geometry = _update_geometry(weight, update)

        weight_np = np.asarray(weight)
        update_np = np.asarray(update)
        weight_norm_sq = np.vdot(weight_np, weight_np)
        radial = (
            np.vdot(weight_np, update_np) / weight_norm_sq
        ) * weight_np
        tangential = update_np - radial
        expected_change = (
            np.vdot(weight_np + update_np, weight_np + update_np)
            - weight_norm_sq
        )

        np.testing.assert_allclose(
            geometry['tangential_update_l2'],
            np.linalg.norm(tangential),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            geometry['effective_angular_step'],
            np.linalg.norm(tangential) / np.linalg.norm(weight_np),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            geometry['predicted_norm_sq_change'],
            expected_change,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            geometry['actual_norm_sq_change'],
            expected_change,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_optimizer_weight_decay_split_matches_adamw_order(self):
        params = {
            'w': jnp.asarray([[1.5, -2.0], [0.5, 3.0]], dtype=jnp.float32)
        }
        grads = {
            'w': jnp.asarray([[0.4, -0.2], [0.1, 0.3]], dtype=jnp.float32)
        }
        learning_rate = 3e-4
        weight_decay = 1e-4
        tx = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        opt_state = tx.init(params)
        total, _, adaptive, decay = _optimizer_update_components(
            tx, grads, opt_state, params, True
        )

        np.testing.assert_allclose(
            np.asarray(total['w']),
            np.asarray(adaptive['w'] + decay['w']),
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(decay['w']),
            -learning_rate * weight_decay * np.asarray(params['w']),
            rtol=2e-3,
            atol=2e-10,
        )

    def test_scale_code_decomposition_is_exact(self):
        old_codes = jnp.asarray(
            [[1.0, -2.0], [3.0, 0.5]], dtype=jnp.float8_e4m3fn
        )
        next_codes = jnp.asarray(
            [[1.25, -1.75], [2.5, 0.625]], dtype=jnp.float8_e4m3fn
        )
        old_scale = jnp.float32(0.2)
        next_scale = jnp.float32(0.35)
        scale_only, code_only = _scale_code_update_components(
            old_codes, old_scale, next_codes, next_scale
        )
        actual = (
            dequantize_e4m3(next_codes, next_scale)
            - dequantize_e4m3(old_codes, old_scale)
        )
        np.testing.assert_allclose(
            np.asarray(scale_only + code_only),
            np.asarray(actual),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_lag_target_scope_initial_reconstruction_and_recurrence(self):
        agent = make_agent('fp8_direct', 'fp8_lag')
        raw_params = traverse_util.flatten_dict(
            agent.target_critic.params
        )
        metadata = traverse_util.flatten_dict(agent.target_critic.fp8_meta)
        lag_paths = [
            path for path, value in raw_params.items()
            if value.dtype == jax.numpy.float8_e4m3fn
        ]
        self.assertEqual(len(lag_paths), 4)
        self.assertTrue(all(
            '/BronetBlock_' in '/'.join(path) and path[-1] == 'kernel'
            for path in lag_paths
        ))
        self.assertEqual(
            sum(path[-1] == 'lag_scale' for path in metadata), 4
        )
        self.assertTrue(all(
            value.dtype == np.dtype(np.float32)
            for path, value in raw_params.items()
            if path not in lag_paths
        ))
        assert_trees_equal(
            self,
            reconstruct_target_params(agent.critic, agent.target_critic),
            agent.critic.params,
        )

        old_critic = agent.critic
        old_online = traverse_util.flatten_dict(old_critic.params)
        next_online = dict(old_online)
        for index, path in enumerate(lag_paths):
            delta = np.full(
                old_online[path].shape, (index + 1) * 1e-3, np.float32
            )
            next_online[path] = old_online[path] + delta
        new_critic = old_critic.replace(
            params=traverse_util.unflatten_dict(next_online)
        )
        updated = update_target_critic(
            new_critic, agent.target_critic, 0.005,
            old_critic=old_critic,
        )
        updated_params = traverse_util.flatten_dict(updated.params)
        updated_meta = traverse_util.flatten_dict(updated.fp8_meta)
        for path in lag_paths:
            candidate = -0.995 * (
                next_online[path] - old_online[path]
            )
            expected_codes, expected_scale = jax.vmap(
                quantize_e4m3_per_tensor
            )(candidate)
            np.testing.assert_array_equal(
                np.asarray(updated_params[path]), np.asarray(expected_codes)
            )
            np.testing.assert_array_equal(
                np.asarray(updated_meta[path[:-1] + ('lag_scale',)]),
                np.asarray(expected_scale),
            )
            reconstructed = traverse_util.flatten_dict(
                reconstruct_target_params(new_critic, updated)
            )[path]
            intended = (
                old_online[path]
                + 0.005 * (next_online[path] - old_online[path])
            )
            applied_lag = jax.vmap(dequantize_e4m3)(
                expected_codes, expected_scale
            )
            np.testing.assert_allclose(
                np.asarray(reconstructed),
                np.asarray(next_online[path] + applied_lag),
                rtol=0,
                atol=2e-9,
            )
            self.assertLess(
                float(np.linalg.norm(np.asarray(reconstructed - intended))),
                2e-5,
            )

        online_after_first = traverse_util.flatten_dict(new_critic.params)
        online_after_second = dict(online_after_first)
        for index, path in enumerate(lag_paths):
            online_after_second[path] = (
                online_after_first[path] - (index + 1) * 7e-4
            )
        second_critic = new_critic.replace(
            params=traverse_util.unflatten_dict(online_after_second)
        )
        updated_twice = update_target_critic(
            second_critic, updated, 0.005, old_critic=new_critic
        )
        first_params = traverse_util.flatten_dict(updated.params)
        first_meta = traverse_util.flatten_dict(updated.fp8_meta)
        second_params = traverse_util.flatten_dict(updated_twice.params)
        second_meta = traverse_util.flatten_dict(updated_twice.fp8_meta)
        for path in lag_paths:
            old_lag = jax.vmap(dequantize_e4m3)(
                first_params[path],
                first_meta[path[:-1] + ('lag_scale',)],
            )
            candidate = 0.995 * (
                old_lag
                - (online_after_second[path] - online_after_first[path])
            )
            expected_codes, expected_scale = jax.vmap(
                quantize_e4m3_per_tensor
            )(candidate)
            np.testing.assert_array_equal(
                np.asarray(second_params[path]), np.asarray(expected_codes)
            )
            np.testing.assert_array_equal(
                np.asarray(second_meta[path[:-1] + ('lag_scale',)]),
                np.asarray(expected_scale),
            )

    def test_lag_target_updates_forward_metadata_and_reports_underflow(self):
        agent = make_agent('fp8_direct', 'fp8_lag')
        initial_meta = traverse_util.flatten_dict(
            jax.tree.map(
                lambda value: np.asarray(value).copy(),
                agent.target_critic.fp8_meta,
            )
        )
        info = agent.update(
            make_batch(), 1, env_step=1,
            collect_update_diagnostics=True,
        )
        jax.block_until_ready(info)
        updated_meta = traverse_util.flatten_dict(agent.target_critic.fp8_meta)
        for path, before in initial_meta.items():
            after = updated_meta[path]
            if 'output_grad' in path[-1]:
                np.testing.assert_array_equal(after, before)
            elif path[-1].endswith('_amax_history'):
                self.assertGreater(np.count_nonzero(np.asarray(after)), 0)

        meta_before_diagnostics = jax.tree.map(
            lambda value: np.asarray(value).copy(),
            agent.target_critic.fp8_meta,
        )
        probe = Batch(*[value[0] for value in make_batch()])
        diagnostics = agent.get_tensor_diagnostics(probe)
        jax.block_until_ready(diagnostics)
        self.assertIn('fp8_target_forward', diagnostics)
        self.assertIn('fp8_target_lag', diagnostics)
        self.assertNotIn('fp8_target_storage', diagnostics)
        self.assertEqual(len(diagnostics['fp8_target_forward']), 8)
        self.assertEqual(
            len(diagnostics['fp8_target_lag']['last_applied_ema']), 8
        )
        assert_trees_equal(
            self, meta_before_diagnostics, agent.target_critic.fp8_meta
        )

        underflow_agent = make_agent('fp8_direct', 'fp8_lag')
        old_critic = underflow_agent.critic
        old_online = traverse_util.flatten_dict(old_critic.params)
        next_online = dict(old_online)
        lag_path = next(
            path for path, value in traverse_util.flatten_dict(
                underflow_agent.target_critic.params
            ).items()
            if value.dtype == jax.numpy.float8_e4m3fn
        )
        delta = np.full(old_online[lag_path].shape, 1.0, np.float32)
        delta[0, 0, 0] = 1e-6
        next_online[lag_path] = old_online[lag_path] + delta
        new_critic = old_critic.replace(
            params=traverse_util.unflatten_dict(next_online)
        )
        underflow = target_ema_diagnostics(
            new_critic, underflow_agent.target_critic, 0.005,
            old_critic=old_critic,
        )
        jax.block_until_ready(underflow)
        self.assertTrue(any(
            float(row['lag_candidate_underflow_fraction']) > 0
            for row in underflow.values()
        ))

    def test_lag_target_checkpoint_round_trip_and_next_update(self):
        source = make_agent('fp8_direct', 'fp8_lag')
        batch = make_batch()
        source.update(batch, 1, env_step=1)
        with tempfile.TemporaryDirectory() as temp:
            source.save(temp, include_optimizer=True)
            restored = make_agent('fp8_direct', 'fp8_lag')
            restored.load(temp)
            assert_trees_equal(
                self, source.target_critic.params,
                restored.target_critic.params,
            )
            assert_trees_equal(
                self, source.target_critic.fp8_meta,
                restored.target_critic.fp8_meta,
            )
            source_info = source.update(batch, 1, env_step=2)
            restored_info = restored.update(batch, 1, env_step=2)
            jax.block_until_ready((source_info, restored_info))
            assert_trees_equal(self, source_info, restored_info)
            assert_trees_equal(
                self, source.target_critic.params,
                restored.target_critic.params,
            )
            assert_trees_equal(
                self, source.target_critic.fp8_meta,
                restored.target_critic.fp8_meta,
            )

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

    def test_online_current_amax_fp32_master_is_matched_control(self):
        current = make_agent('fp8_current_master', 'fp8_direct')
        resident = make_agent('fp8_resident', 'fp8_direct')

        self.assertIsNotNone(current.critic.fp8_meta)
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(current.critic.params)
        ))
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(current.critic.opt_state)
            if np.issubdtype(leaf.dtype, np.floating)
        ))
        metadata = traverse_util.flatten_dict(
            current.critic.fp8_meta, sep='/'
        )
        self.assertEqual(
            sum(path.endswith('/output_grad_amax_history') for path in metadata),
            4,
        )
        self.assertEqual(
            sum(path.endswith('/output_grad_scale') for path in metadata), 4
        )
        self.assertFalse(any(
            path.endswith('/input_amax_history')
            or path.endswith('/kernel_amax_history')
            or path.endswith('/kernel_scale')
            for path in metadata
        ))

        probe = Batch(*[value[0] for value in make_batch()])
        current_logits = current.critic(
            probe.observations, probe.actions, probe.task_ids
        )
        resident_logits = resident.critic(
            probe.observations, probe.actions, probe.task_ids
        )
        np.testing.assert_allclose(
            np.asarray(current_logits),
            np.asarray(resident_logits),
            rtol=2e-5,
            atol=1e-6,
        )

        old_params = jax.tree.map(
            lambda value: np.asarray(value).copy(), current.critic.params
        )
        batch = make_batch()
        for env_step in range(1, 4):
            info = current.update(batch, 1, env_step=env_step)
            jax.block_until_ready(info)
        self.assertTrue(any(
            np.any(np.asarray(before) != np.asarray(after))
            for before, after in zip(
                jax.tree_util.tree_leaves(old_params),
                jax.tree_util.tree_leaves(current.critic.params),
            )
        ))
        updated_metadata = traverse_util.flatten_dict(
            current.critic.fp8_meta, sep='/'
        )
        output_histories = [
            value for path, value in updated_metadata.items()
            if path.endswith('/output_grad_amax_history')
        ]
        self.assertEqual(len(output_histories), 4)
        self.assertTrue(all(
            np.count_nonzero(np.asarray(value)) >= 4
            for value in output_histories
        ))

    def test_online_resident_scope_optimizer_and_target_initialization(self):
        agent = make_agent('fp8_resident', 'fp8_direct')
        params = traverse_util.flatten_dict(agent.critic.params, sep='/')
        resident_paths = [
            path for path, value in params.items()
            if value.dtype == jax.numpy.float8_e4m3fn
        ]
        self.assertEqual(len(resident_paths), 4)
        self.assertTrue(all(
            '/BronetBlock_' in path and path.endswith('/kernel')
            for path in resident_paths
        ))
        self.assertTrue(all(
            value.dtype == np.dtype(np.float32)
            for path, value in params.items()
            if path not in resident_paths
        ))
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(agent.critic.opt_state)
            if np.issubdtype(leaf.dtype, np.floating)
        ))
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(agent.target_critic.params)
        ))
        assert_trees_equal(
            self,
            dequantize_critic_params(agent.critic),
            agent.target_critic.params,
        )
        metadata = traverse_util.flatten_dict(agent.critic.fp8_meta, sep='/')
        self.assertEqual(
            sum(path.endswith('/kernel_scale') for path in metadata), 4
        )
        output_histories = {
            path: value for path, value in metadata.items()
            if path.endswith('/output_grad_amax_history')
        }
        output_scales = {
            path: value for path, value in metadata.items()
            if path.endswith('/output_grad_scale')
        }
        self.assertEqual(len(output_histories), 4)
        self.assertEqual(len(output_scales), 4)
        self.assertTrue(all(
            value.shape == (2, 8) for value in output_histories.values()
        ))
        self.assertTrue(all(
            value.shape == (2, 1) for value in output_scales.values()
        ))
        self.assertFalse(any(
            path.endswith('/input_amax_history')
            or path.endswith('/kernel_amax_history')
            for path in metadata
        ))

    def test_online_resident_scaled_backward_is_nonzero_and_persistent(self):
        agent = make_agent('fp8_resident', 'fp8_direct')
        batch = make_batch()
        for env_step in range(1, 4):
            info = agent.update(batch, 1, env_step=env_step)
            jax.block_until_ready(info)

        metadata = traverse_util.flatten_dict(agent.critic.fp8_meta, sep='/')
        output_histories = [
            value for path, value in metadata.items()
            if path.endswith('/output_grad_amax_history')
        ]
        output_scales = [
            value for path, value in metadata.items()
            if path.endswith('/output_grad_scale')
        ]
        self.assertEqual(len(output_histories), 4)
        self.assertTrue(all(
            np.count_nonzero(np.asarray(value)) >= 4
            for value in output_histories
        ))
        self.assertTrue(all(
            np.all(np.asarray(value) > 0) for value in output_scales
        ))

        probe = Batch(*[value[0] for value in batch])
        diagnostics = agent.get_tensor_diagnostics(probe)
        jax.block_until_ready(diagnostics)
        critic_grads = traverse_util.flatten_dict(
            diagnostics['gradients']['critic'], sep='/'
        )
        resident_kernel_grads = [
            value for path, value in critic_grads.items()
            if '/BronetBlock_' in path and path.endswith('/kernel')
        ]
        self.assertEqual(len(resident_kernel_grads), 4)
        self.assertTrue(all(
            np.linalg.norm(np.asarray(value)) > 0
            for value in resident_kernel_grads
        ))
        self.assertTrue(all(
            np.count_nonzero(np.asarray(value)) > 0
            for value in resident_kernel_grads
        ))

        observations = jnp.asarray(probe.observations)
        actions = jnp.asarray(probe.actions)
        task_ids = jnp.asarray(probe.task_ids)

        def q_sum(candidate_actions):
            logits = agent.critic(
                observations, candidate_actions, task_ids
            )
            probabilities = jax.nn.softmax(logits, axis=-1)
            support = jnp.linspace(-10.0, 10.0, logits.shape[-1])
            return jnp.sum(probabilities * support)

        action_gradient = jax.grad(q_sum)(actions)
        self.assertGreater(
            float(np.linalg.norm(np.asarray(action_gradient))), 0.0
        )

    def test_online_resident_update_and_nonredundant_diagnostics(self):
        agent = make_agent('fp8_resident', 'fp8_direct')
        batch = make_batch()
        old_target = jax.tree.map(
            lambda value: np.asarray(value).copy(), agent.target_critic.params
        )
        info = agent.update(
            batch, 1, env_step=1, collect_update_diagnostics=True
        )
        jax.block_until_ready(info)

        resident_paths = [
            path for path, value in traverse_util.flatten_dict(
                agent.critic.params, sep='/'
            ).items()
            if value.dtype == jax.numpy.float8_e4m3fn
        ]
        self.assertEqual(len(resident_paths), 4)
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(agent.critic.opt_state)
            if np.issubdtype(leaf.dtype, np.floating)
        ))

        physical = dequantize_critic_params(agent.critic)
        for got, online, old in zip(
            jax.tree_util.tree_leaves(agent.target_critic.params),
            jax.tree_util.tree_leaves(physical),
            jax.tree_util.tree_leaves(old_target),
        ):
            np.testing.assert_allclose(
                np.asarray(got),
                np.asarray(online) * 0.005 + old * 0.995,
                rtol=1e-6,
                atol=1e-6,
            )

        diagnostics = agent.last_online_resident_diagnostics
        self.assertEqual(
            set(diagnostics), {'parameter_write', 'function_write'}
        )
        self.assertEqual(len(diagnostics['parameter_write']), 8)
        expected_parameter_metrics = {
            'intended_update_l2',
            'intended_update_radial_cosine',
            'intended_tangential_update_l2',
            'intended_effective_angular_step',
            'actual_tangential_update_l2',
            'actual_effective_angular_step',
            'angular_retention',
            'intended_first_order_radial',
            'intended_second_order_update_sq',
            'intended_predicted_norm_sq_change',
            'intended_actual_norm_sq_change',
            'actual_first_order_radial',
            'actual_second_order_update_sq',
            'actual_predicted_norm_sq_change',
            'actual_norm_sq_change',
            'wd_intended_l2',
            'wd_intended_radial',
            'wd_applied_radial',
            'wd_radial_retention',
            'applied_to_intended_l2_ratio',
            'update_cosine',
            'swallowed_update_fraction',
            'code_unchanged_fraction',
            'code_l2',
            'code_cosine',
            'weight_relative_error',
            'quantization_error_radial_cosine',
            'scale_log2_ratio',
            'relative_scale_change',
            'scale_only_update_l2',
            'code_only_update_l2',
            'scale_update_fraction_of_actual_l2',
            'dynamic_vs_fixed_relative_difference',
            'canonicalization_factor',
            'fixed_anchor_kernel_norm',
            'code_norm',
            'pre_canonical_kernel_norm',
            'post_canonical_kernel_norm',
            'post_to_fixed_anchor_kernel_norm_ratio',
            'canonicalization_kernel_delta_l2',
        }
        self.assertTrue(all(
            set(row) == expected_parameter_metrics
            for row in diagnostics['parameter_write'].values()
        ))
        self.assertEqual(
            set(diagnostics['function_write']),
            {'aggregate', 'ensemble_0', 'ensemble_1'},
        )
        self.assertTrue(all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree_util.tree_leaves(diagnostics)
        ))
        for row in diagnostics['parameter_write'].values():
            np.testing.assert_allclose(
                np.asarray(row['intended_predicted_norm_sq_change']),
                np.asarray(row['intended_actual_norm_sq_change']),
                rtol=5e-3,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                np.asarray(row['actual_predicted_norm_sq_change']),
                np.asarray(row['actual_norm_sq_change']),
                rtol=5e-3,
                atol=2e-5,
            )
            np.testing.assert_array_equal(
                np.asarray(row['canonicalization_factor']), 1.0
            )
            np.testing.assert_array_equal(
                np.asarray(row['pre_canonical_kernel_norm']),
                np.asarray(row['post_canonical_kernel_norm']),
            )
            np.testing.assert_array_equal(
                np.asarray(row['canonicalization_kernel_delta_l2']), 0.0
            )
            np.testing.assert_array_equal(
                np.asarray(row['fixed_anchor_kernel_norm']), 0.0
            )
            np.testing.assert_array_equal(
                np.asarray(row['post_to_fixed_anchor_kernel_norm_ratio']), 0.0
            )

        probe = Batch(*[value[0] for value in batch])
        tensor_diagnostics = agent.get_tensor_diagnostics(probe)
        jax.block_until_ready(tensor_diagnostics)
        storage = tensor_diagnostics['fp8_online_storage']
        self.assertEqual(
            set(storage), {'kernel_scales', 'last_applied_update'}
        )
        self.assertNotIn('codes', storage)
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(
                tensor_diagnostics['params']['critic']
            )
        ))
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(
                tensor_diagnostics['gradients']['critic']
            )
        ))

    def test_online_resident_fixed_anchor_update_preserves_anchor_norms(self):
        agent = make_agent(
            'fp8_resident',
            'fp8_direct',
            resident_canonicalization=True,
        )
        initial_metadata = traverse_util.flatten_dict(agent.critic.fp8_meta)
        info = agent.update(
            make_batch(),
            1,
            env_step=1,
            collect_update_diagnostics=True,
        )
        jax.block_until_ready(info)
        new_physical = traverse_util.flatten_dict(
            dequantize_critic_params(agent.critic)
        )
        stored = traverse_util.flatten_dict(agent.critic.params)
        metadata = traverse_util.flatten_dict(agent.critic.fp8_meta)
        resident_paths = [
            path for path, value in stored.items()
            if value.dtype == jnp.float8_e4m3fn
        ]
        self.assertEqual(len(resident_paths), 4)
        for path in resident_paths:
            np.testing.assert_allclose(
                np.linalg.norm(np.asarray(new_physical[path]), axis=(1, 2)),
                np.asarray(
                    initial_metadata[
                        path[:-1] + ('kernel_anchor_norm',)
                    ]
                ),
                rtol=2e-6,
                atol=2e-6,
            )
            self.assertEqual(
                stored[path[:-1] + ('bias',)].dtype,
                np.dtype(np.float32),
            )
            self.assertEqual(
                metadata[path[:-1] + ('kernel_scale',)].dtype,
                np.dtype(np.float32),
            )
            self.assertEqual(
                metadata[path[:-1] + ('kernel_anchor_norm',)].dtype,
                np.dtype(np.float32),
            )
        anchor_leaves = [
            value for path, value in metadata.items()
            if path[-1] == 'kernel_anchor_norm'
        ]
        self.assertEqual(len(anchor_leaves), 4)
        self.assertEqual(sum(value.size for value in anchor_leaves), 8)
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(agent.critic.opt_state)
            if np.issubdtype(leaf.dtype, np.floating)
        ))
        diagnostics = agent.last_online_resident_diagnostics
        self.assertTrue(all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree_util.tree_leaves(diagnostics)
        ))
        for row in diagnostics['parameter_write'].values():
            self.assertGreater(float(row['canonicalization_factor']), 0.0)
            np.testing.assert_allclose(
                np.asarray(row['post_to_fixed_anchor_kernel_norm_ratio']),
                1.0,
                rtol=2e-6,
                atol=2e-6,
            )
        probe = Batch(*[value[0] for value in make_batch()])
        storage = agent.get_tensor_diagnostics(probe)['fp8_online_storage']
        self.assertIn('fixed_anchor_norms', storage)
        self.assertEqual(
            len(traverse_util.flatten_dict(storage['fixed_anchor_norms'])), 8
        )

    def test_explicitly_disabled_resident_options_match_default(self):
        default_agent = make_agent('fp8_resident', 'fp8_direct')
        explicit_off_agent = make_agent(
            'fp8_resident',
            'fp8_direct',
            resident_canonicalization=False,
            resident_carry=False,
        )
        batch = make_batch()
        for env_step in range(1, 4):
            default_info = default_agent.update(
                batch, 1, env_step=env_step
            )
            explicit_off_info = explicit_off_agent.update(
                batch, 1, env_step=env_step
            )
            jax.block_until_ready((default_info, explicit_off_info))
        for name in (
            'actor',
            'critic',
            'target_critic',
            'temp',
            'rng',
            'normalizer_rng',
            'task_entropies',
            'task_entropy_counts',
            'step',
        ):
            assert_trees_equal(
                self,
                getattr(default_agent, name),
                getattr(explicit_off_agent, name),
            )
        assert_trees_equal(self, default_info, explicit_off_info)

    def test_resident_diagnostics_do_not_change_update_trajectory(self):
        without_diagnostics = make_agent('fp8_resident', 'fp8_direct')
        with_diagnostics = make_agent('fp8_resident', 'fp8_direct')
        batch = make_batch()

        for env_step in range(1, 4):
            info_without = without_diagnostics.update(
                batch,
                1,
                env_step=env_step,
                collect_update_diagnostics=False,
            )
            info_with = with_diagnostics.update(
                batch,
                1,
                env_step=env_step,
                collect_update_diagnostics=True,
            )
            jax.block_until_ready((info_without, info_with))

        for name in (
            'actor',
            'critic',
            'target_critic',
            'temp',
            'rng',
            'normalizer_rng',
            'task_entropies',
            'task_entropy_counts',
            'step',
        ):
            assert_trees_equal(
                self,
                getattr(without_diagnostics, name),
                getattr(with_diagnostics, name),
            )
        assert_trees_equal(self, info_without, info_with)

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
            collect_update_diagnostics=True,
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

    def test_direct_target_uses_fp32_storage_and_advances_forward_metadata(self):
        agent = make_agent('fp8_direct', 'fp8_direct')
        initial_params = jax.tree.map(
            lambda value: np.asarray(value).copy(), agent.target_critic.params
        )
        initial_meta = traverse_util.flatten_dict(agent.target_critic.fp8_meta)

        self.assertIsNone(agent.target_critic.opt_state)
        self.assertTrue(all(
            leaf.dtype == np.dtype(np.float32)
            for leaf in jax.tree_util.tree_leaves(agent.target_critic.params)
        ))
        assert_trees_equal(self, agent.critic.params, agent.target_critic.params)
        self.assertTrue(all(
            value.dtype == np.dtype(np.float32)
            for value in initial_meta.values()
        ))

        info = agent.update(make_batch(), 1, env_step=1)
        jax.block_until_ready(info)
        updated_meta = traverse_util.flatten_dict(agent.target_critic.fp8_meta)
        for path, before in initial_meta.items():
            after = updated_meta[path]
            if 'output_grad' in path[-1]:
                np.testing.assert_array_equal(after, before)
            elif path[-1].endswith('_amax_history'):
                self.assertGreater(np.count_nonzero(np.asarray(after)), 0)

        initial_flat = traverse_util.flatten_dict(initial_params)
        online_flat = traverse_util.flatten_dict(agent.critic.params)
        target_flat = traverse_util.flatten_dict(agent.target_critic.params)
        for path, initial in initial_flat.items():
            np.testing.assert_allclose(
                np.asarray(target_flat[path]),
                np.asarray(online_flat[path] * 0.005 + initial * 0.995),
                rtol=1e-7,
                atol=1e-7,
            )

        meta_before_diagnostics = jax.tree.map(
            lambda value: np.asarray(value).copy(), agent.target_critic.fp8_meta
        )
        probe = Batch(*[value[0] for value in make_batch()])
        diagnostics = agent.get_tensor_diagnostics(probe)
        jax.block_until_ready(diagnostics)
        self.assertIn('fp8_target_forward', diagnostics)
        self.assertNotIn('fp8_target_storage', diagnostics)
        self.assertEqual(
            set(diagnostics['target_forward_error']),
            {'aggregate', 'ensemble_0', 'ensemble_1'},
        )
        assert_trees_equal(
            self, meta_before_diagnostics, agent.target_critic.fp8_meta
        )

    def test_direct_target_checkpoint_round_trip_and_next_update(self):
        source = make_agent('fp8_direct', 'fp8_direct')
        batch = make_batch()
        source.update(batch, 1, env_step=1)
        with tempfile.TemporaryDirectory() as temp:
            source.save(temp, include_optimizer=True)
            restored = make_agent('fp8_direct', 'fp8_direct')
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

    def test_online_resident_carry_checkpoint_round_trip_and_state_bytes(self):
        source = make_agent(
            'fp8_resident', 'fp8_direct', resident_carry=True
        )
        batch = make_batch()
        source.update(batch, 1, env_step=1)
        with tempfile.TemporaryDirectory() as temp:
            carry_path = Path(temp) / 'carry'
            base_path = Path(temp) / 'base'
            source.save(str(carry_path), include_optimizer=True)
            base = make_agent('fp8_resident', 'fp8_direct')
            base.save(str(base_path), include_optimizer=True)
            self.assertGreater(
                (carry_path / 'critic.msgpack').stat().st_size,
                (base_path / 'critic.msgpack').stat().st_size,
            )

            restored = make_agent(
                'fp8_resident', 'fp8_direct', resident_carry=True
            )
            restored.load(str(carry_path))
            for name in ('critic', 'target_critic'):
                left = getattr(source, name)
                right = getattr(restored, name)
                assert_trees_equal(self, left.params, right.params)
                assert_trees_equal(self, left.fp8_meta, right.fp8_meta)
                assert_trees_equal(self, left.opt_state, right.opt_state)

            manager = CheckpointManager(
                Path(temp) / 'checkpoints',
                'online-resident-carry-test',
                Path(temp) / 'run',
                ['task'],
                {
                    'critic_precision': 'fp8_resident',
                    'target_critic_precision': 'fp8_direct',
                    'fp8_resident_carry': True,
                    'carry_gain': 16.0,
                    'carry_dtype': 'float8_e4m3fn',
                },
            )
            checkpoint = manager.save_analysis(source, env_step=1)
            state_bytes = CheckpointManager.read_manifest(checkpoint)[
                'state_bytes'
            ]
            metadata = traverse_util.flatten_dict(source.critic.fp8_meta)
            expected_carry_bytes = sum(
                value.size * value.dtype.itemsize
                for path, value in metadata.items()
                if path[-1] == 'kernel_carry'
            )
            self.assertEqual(
                state_bytes['resident_carry_code'], expected_carry_bytes
            )
            self.assertEqual(state_bytes['resident_carry_fp32'], 0)
            self.assertEqual(
                state_bytes['resident_main_code'], expected_carry_bytes
            )
            self.assertGreater(state_bytes['resident_weight_scale'], 0)

            source_info = source.update(batch, 1, env_step=2)
            restored_info = restored.update(batch, 1, env_step=2)
            jax.block_until_ready((source_info, restored_info))
            assert_trees_equal(self, source_info, restored_info)
            assert_trees_equal(
                self, source.critic.params, restored.critic.params
            )
            assert_trees_equal(
                self, source.critic.fp8_meta, restored.critic.fp8_meta
            )
            assert_trees_allclose(
                self,
                source.critic.opt_state,
                restored.critic.opt_state,
                rtol=2e-7,
                atol=1e-8,
            )
            assert_trees_equal(
                self,
                source.target_critic.params,
                restored.target_critic.params,
            )

    def test_online_resident_canonical_checkpoint_round_trip_and_state_bytes(self):
        source = make_agent(
            'fp8_resident',
            'fp8_direct',
            resident_canonicalization=True,
        )
        batch = make_batch()
        source.update(batch, 1, env_step=1)
        with tempfile.TemporaryDirectory() as temp:
            manager = CheckpointManager(
                Path(temp) / 'checkpoints',
                'online-resident-test',
                Path(temp) / 'run',
                ['task'],
                {
                    'critic_precision': 'fp8_resident',
                    'target_critic_precision': 'fp8_direct',
                    'fp8_resident_canonicalization': True,
                },
            )
            checkpoint = manager.save_recovery(
                source,
                replay_buffer=type('Buffer', (), {
                    'estimate_size_bytes': lambda self: 0,
                    'save': lambda self, path: None,
                })(),
                reward_normalizer=type('Normalizer', (), {
                    'state_dict': lambda self: {},
                })(),
                episode_recorder=type('Episodes', (), {
                    'state_dict': lambda self: {},
                })(),
                env_step=1,
                wandb_id=None,
                save_replay_buffer=False,
            )
            manifest = CheckpointManager.read_manifest(checkpoint)
            self.assertIn('float8_e4m3fn', manifest['parameter_dtypes'])
            self.assertEqual(
                set(manifest['optimizer_dtypes']), {'float32', 'int32'}
            )
            state_bytes = manifest['state_bytes']
            self.assertGreater(state_bytes['fp8_payload'], 0)
            self.assertGreater(state_bytes['fp8_metadata'], 0)
            self.assertAlmostEqual(
                state_bytes['fp8_metadata_fraction'],
                state_bytes['fp8_metadata'] / state_bytes['persistent_total'],
            )
            report = manifest['fixed_anchor_norms']
            self.assertEqual(report['member_count'], 8)
            self.assertEqual(len(report['members']), 8)
            self.assertTrue(
                (checkpoint / 'fixed_anchor_norms.json').is_file()
            )
            self.assertEqual(
                fixed_anchor_norm_report(checkpoint), report
            )

            source.save(temp, include_optimizer=True)
            restored = make_agent(
                'fp8_resident',
                'fp8_direct',
                resident_canonicalization=True,
            )
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

    def test_direct_target_rejects_fp32_target_load(self):
        source = make_agent('fp8_direct', 'fp32')
        with tempfile.TemporaryDirectory() as temp:
            source.save(temp, include_optimizer=True)
            restored = make_agent('fp8_direct', 'fp8_direct')
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
        self.assertFalse(checkpoint_config_value(
            {}, 'fp8_resident_canonicalization'
        ))
        self.assertFalse(checkpoint_config_value({}, 'fp8_resident_carry'))
        self.assertEqual(checkpoint_config_value({}, 'carry_gain'), 16.0)
        self.assertEqual(
            checkpoint_config_value({}, 'carry_dtype'), 'float8_e4m3fn'
        )
        self.assertEqual(
            checkpoint_config_value({}, 'resolved_fp8_code_materialization'),
            'legacy_compiler_elidable_cast',
        )
        self.assertFalse(validate_checkpoint_config(
            {'critic_precision': 'fp8_resident'},
            {
                'critic_precision': 'fp8_resident',
                'fp8_resident_canonicalization': False,
            },
        ))
        self.assertEqual(
            checkpoint_config_value(
                {
                    'fp8_resident_canonicalization': True,
                    'resolved_online_fp8_canonicalization': (
                        'fixed_initial_kernel_norm_with_paired_bias'
                    ),
                },
                'resolved_online_fp8_canonicalization',
            ),
            'fixed_initial_kernel_norm_with_paired_bias',
        )
        with self.assertRaisesRegex(
            ValueError, 'resolved_online_fp8_canonicalization'
        ):
            validate_checkpoint_config(
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_canonicalization': True,
                    'resolved_online_fp8_canonicalization': (
                        'preserve_previous_kernel_norm_with_paired_bias'
                    ),
                },
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_canonicalization': True,
                    'resolved_online_fp8_canonicalization': (
                        'fixed_initial_kernel_norm_with_paired_bias'
                    ),
                },
            )
        with self.assertRaisesRegex(
            ValueError, 'fp8_resident_canonicalization'
        ):
            validate_checkpoint_config(
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_canonicalization': False,
                },
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_canonicalization': True,
                },
            )
        with self.assertRaisesRegex(ValueError, 'fp8_resident_carry'):
            validate_checkpoint_config(
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_carry': False,
                },
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_carry': True,
                },
            )
        with self.assertRaisesRegex(
            ValueError, 'resolved_fp8_code_materialization'
        ):
            validate_checkpoint_config(
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_carry': True,
                },
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_carry': True,
                    'resolved_fp8_code_materialization': (
                        'optimization_barrier_after_e4m3_cast_v1'
                    ),
                },
            )
        self.assertFalse(validate_checkpoint_config(
            {'critic_precision': 'fp8_direct'},
            {
                'critic_precision': 'fp8_direct',
                'resolved_fp8_code_materialization': (
                    'optimization_barrier_after_e4m3_cast_v1'
                ),
            },
        ))
        with self.assertRaisesRegex(ValueError, 'fp8_resident_carry'):
            make_agent('fp32', resident_carry=True)
        with self.assertRaisesRegex(ValueError, 'canonicalization'):
            make_agent(
                'fp8_resident',
                resident_canonicalization=True,
                resident_carry=True,
            )
        with self.assertRaisesRegex(
            ValueError, 'fp8_resident_canonicalization'
        ):
            validate_checkpoint_config(
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_canonicalization': True,
                },
                {
                    'critic_precision': 'fp8_resident',
                    'fp8_resident_canonicalization': False,
                },
            )
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
                {'critic_precision': 'fp8_direct'},
                {'critic_precision': 'fp8_resident'},
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
        with self.assertRaises(ValueError):
            validate_checkpoint_config(
                {
                    'critic_precision': 'fp32',
                    'target_critic_precision': 'fp8_direct',
                    'fp8_amax_history_length': 8,
                },
                {
                    'critic_precision': 'fp32',
                    'target_critic_precision': 'fp8_direct',
                    'fp8_amax_history_length': 16,
                },
            )
        with self.assertRaises(ValueError):
            validate_checkpoint_config(
                {
                    'critic_precision': 'fp8_direct',
                    'target_critic_precision': 'fp8_lag',
                    'fp8_amax_history_length': 8,
                },
                {
                    'critic_precision': 'fp8_direct',
                    'target_critic_precision': 'fp8_lag',
                    'fp8_amax_history_length': 16,
                },
            )


if __name__ == '__main__':
    unittest.main()
