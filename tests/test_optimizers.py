import tempfile
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.optimizers import (
    BlockMoment, MOMENT_MODES, decode_moment, decode_moment_tree,
    encode_moment, is_residual_kernel, optimizer_state_inventory,
    stored_adamw,
)
from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import reconstruct_carry_logical_critic_params
from jaxrl.checkpoint import checkpoint_config_value, validate_checkpoint_config
from jaxrl.utils import Batch


def tree(value):
    return {'critic': {'BronetBlock_0': {'Dense_0': {'kernel': value}},
                       'Dense_0': {'kernel': jnp.ones((2, 3, 4))}}}


def kernel(params):
    return params['critic']['BronetBlock_0']['Dense_0']['kernel']


def close_trees(left, right, rtol=1e-6, atol=1e-8):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(np.asarray(a, dtype=np.float32),
                                   np.asarray(b, dtype=np.float32), rtol=rtol, atol=atol)


class MomentCodecTests(unittest.TestCase):
    def test_subnormal_state_cannot_overflow_the_codes(self):
        # Late healthy checkpoints contain v~7e-43. RTN of amax/448 to the
        # smallest subnormal otherwise makes normalized amax=500 -> FP8 NaN.
        values = jnp.asarray([0., np.nextafter(np.float32(0), np.float32(1)),
                              1e-44, 7e-43, 1e-42, 1e-39, 1e-35], jnp.float32)
        x = jnp.broadcast_to(values[:, None], (7, 128))
        retains_subnormals = float(jax.jit(lambda a: a + a)(values[2])) > 0
        for mode in ('fp8', 'fp8_carry'):
            stored, rebuilt = jax.jit(lambda a: (encode_moment(a, mode),
                decode_moment(encode_moment(a, mode))))(x)
            self.assertTrue(np.all(np.isfinite(np.asarray(rebuilt))))
            # XLA CPU flushes subnormal arithmetic, including ordinary Adam.
            # Blackwell retains it and must preserve these nonzero states.
            if retains_subnormals:
                self.assertTrue(np.all(np.asarray(rebuilt)[1:] > 0))
            else:
                self.assertTrue(np.all(np.asarray(rebuilt)[-1] > 0))
            self.assertTrue(np.all(np.isfinite(np.asarray(stored.code).astype(np.float32))))
            np.testing.assert_array_equal(rebuilt, jax.jit(decode_moment)(stored))
            if mode == 'fp8_carry' and retains_subnormals:
                np.testing.assert_allclose(np.asarray(rebuilt).astype(np.float64),
                    np.asarray(x).astype(np.float64), rtol=0.002, atol=2e-45)

    def test_zero_tiny_blocks_and_ensemble_boundaries(self):
        for mode in MOMENT_MODES:
            zero = encode_moment(jnp.zeros((2, 5, 31)), mode)
            np.testing.assert_array_equal(decode_moment(zero), 0)
        x = jnp.linspace(-1, 1, 256).reshape(2, 128)
        x = x * jnp.array([1e-27, 1e3])[:, None]
        saved = jax.jit(encode_moment)(x)
        self.assertEqual(saved.scale.shape, (2, 1, 1))
        self.assertLess(float(saved.scale[0, 0, 0]), 1e-28)
        y = np.asarray(decode_moment(saved))
        for a, b in zip(y, np.asarray(x)):
            self.assertLess(np.linalg.norm((a - b).astype(np.float64)) /
                            np.linalg.norm(b.astype(np.float64)), 0.002)
        # Per-block scales, including a padded final block, are independent.
        x = jnp.concatenate([jnp.ones((2, 128)), jnp.full((2, 27), 1e-15)], axis=1)
        saved = encode_moment(x.reshape(2, 5, 31))
        np.testing.assert_allclose(decode_moment(saved), x.reshape(2, 5, 31), rtol=1e-6)
        self.assertEqual(saved.code.shape, (2, 5, 31))

    def test_signed_v_carry_and_real_jit_materialization(self):
        x = jax.random.uniform(jax.random.PRNGKey(17), (2, 128, 128)) * 1e-12

        @jax.jit
        def same_graph(value):
            saved = encode_moment(value)
            return saved, decode_moment(saved)

        saved, in_graph = same_graph(x)
        persisted = jax.jit(decode_moment)(saved)
        np.testing.assert_array_equal(in_graph, persisted)
        self.assertGreater(np.count_nonzero(np.asarray(saved.carry)), x.size // 2)
        self.assertTrue(np.any(np.asarray(saved.carry).astype(np.float32) < 0))
        self.assertGreaterEqual(float(jnp.min(persisted)), 0)
        self.assertLess(float(jnp.max(jnp.abs(saved.carry.astype(jnp.float32)))), 448)
        plain = decode_moment(encode_moment(x, 'fp8'))
        self.assertLess(float(jnp.linalg.norm((persisted - x) / 1e-12)),
                        float(jnp.linalg.norm((plain - x) / 1e-12)) / 20)


class StoredAdamTests(unittest.TestCase):
    def test_lossless_matches_optax_schedule_decay_and_eps(self):
        params = tree(jnp.linspace(-2, 2, 512).reshape(2, 16, 16))
        kwargs = dict(learning_rate=lambda c: 0.003 / (c + 1), b1=0.8,
                      b2=0.97, eps=2e-7, eps_root=3e-9, weight_decay=0.13,
                      mask={'critic': {'BronetBlock_0': {'Dense_0': {'kernel': True}},
                                       'Dense_0': {'kernel': False}}})
        reference = optax.adamw(**kwargs)
        candidate = stored_adamw(mode='fp32', **kwargs)
        a, b = reference.init(params), candidate.init(params)
        pa, pb = params, params
        for t in range(24):
            grads = jax.tree.map(lambda p: jnp.sin(p * 4 + t) * (t % 3 != 0), params)
            ua, a = jax.jit(reference.update)(grads, a, pa)
            ub, b = jax.jit(candidate.update)(grads, b, pb)
            pa, pb = optax.apply_updates(pa, ua), optax.apply_updates(pb, ub)
            close_trees(ua, ub)
            close_trees(a, b)
            close_trees(pa, pb)

    def test_current_update_uses_unencoded_new_moments(self):
        params = tree(jnp.ones((2, 16, 16)))
        reference = optax.adamw(3e-4, weight_decay=0.07)
        for mode in MOMENT_MODES[1:]:
            tx = stored_adamw(3e-4, mode=mode, weight_decay=0.07)
            state = tx.init(params)
            for t in range(8):
                # Includes zero-gradient historical decay, sign changes and tiny gradients.
                scale = [1.0, 0.0, -1.0, 1e-12][t % 4]
                grads = jax.tree.map(lambda p: jnp.sin(jnp.arange(p.size).reshape(p.shape) + t) * scale, params)
                old = optax.ScaleByAdamState(state[0].count,
                    decode_moment_tree(state[0].mu), decode_moment_tree(state[0].nu))
                oracle_state = (old, *state[1:])
                expected, expected_state = jax.jit(reference.update)(grads, oracle_state, params)
                actual, state = jax.jit(tx.update)(grads, state, params)
                close_trees(expected, actual)
                self.assertEqual(int(state[0].count), t + 1)
                self.assertGreaterEqual(float(jnp.min(decode_moment(kernel(state[0].nu)))), 0)
                close_trees(expected_state[0].mu['critic']['Dense_0'], state[0].mu['critic']['Dense_0'])

    def test_no_shadow_and_serialization_next_step(self):
        params = tree(jnp.ones((2, 16, 16)))
        tx = stored_adamw(3e-4)
        state = tx.init(params)
        inv = optimizer_state_inventory(state)
        codes = kernel(state[0].mu)
        self.assertIsInstance(codes, BlockMoment)
        self.assertEqual(codes.code.dtype, jnp.float8_e4m3fn)
        self.assertEqual(codes.carry.dtype, jnp.float8_e4m3fn)
        self.assertEqual(codes.scale.dtype, jnp.float32)
        selected_bytes = sum(v.size * v.dtype.itemsize for v in jax.tree.leaves((
            kernel(state[0].mu), kernel(state[0].nu))))
        self.assertEqual(selected_bytes, 512 * 4 + (512 // 128) * 8)
        self.assertEqual(inv['bytes_by_dtype']['float32'], (2 * 3 * 4 * 2 + 8) * 4)
        grads = jax.tree.map(lambda p: p * 0.0123, params)
        _, state = jax.jit(tx.update)(grads, state, params)
        restored = flax.serialization.from_bytes(tx.init(params), flax.serialization.to_bytes(state))
        close_trees(jax.jit(tx.update)(grads, state, params),
                    jax.jit(tx.update)(grads, restored, params), rtol=0, atol=0)


class LearnerMomentTests(unittest.TestCase):
    def make_agent(self, mode='fp8_carry'):
        return BRC(42, np.zeros((1, 4), np.float32), np.zeros((1, 2), np.float32),
                   num_tasks=2, width_critic=16, width_actor=16, updates_per_step=2,
                   critic_precision='fp8_resident', fp8_resident_carry=True,
                   target_critic_precision='fp8_lag', fp8_amax_history_length=8,
                   critic_optimizer_state=mode)

    def test_logical_weight_decay_and_exact_scope(self):
        agent = self.make_agent()
        logical = reconstruct_carry_logical_critic_params(agent.critic)
        grads = jax.tree.map(jnp.zeros_like, logical)
        reference = optax.adamw(3e-4)
        expected, _ = reference.update(grads, reference.init(logical), logical)
        actual, _ = agent.critic.tx.update(grads, agent.critic.opt_state, logical)
        close_trees(expected, actual, rtol=0, atol=0)
        for name in ('mu', 'nu'):
            moment = getattr(agent.critic.opt_state[0], name)
            leaves = jax.tree_util.tree_flatten_with_path(moment, is_leaf=lambda v: isinstance(v, BlockMoment))[0]
            selected = [(p, v) for p, v in leaves if isinstance(v, BlockMoment)]
            self.assertEqual(len(selected), 4)
            self.assertTrue(all(is_residual_kernel(p) for p, _ in selected))

    def test_actual_multi_updates_checkpoint_and_next_step(self):
        source = self.make_agent()
        shape = (2, 8)
        batch = Batch(np.ones(shape + (4,), np.float32), np.zeros(shape + (2,), np.float32),
                      np.ones(shape, np.float32), np.ones(shape, np.float32),
                      np.full(shape + (4,), 1.1, np.float32), np.zeros(shape, np.int32))
        before = int(source.step)
        info = source.update(batch, num_updates=2, env_step=5000)
        self.assertEqual(int(source.step) - before, 2)
        self.assertTrue(all(np.all(np.isfinite(v)) for v in jax.tree.leaves(info)))
        with tempfile.TemporaryDirectory() as tmp:
            source.save(tmp)
            restored = self.make_agent()
            restored.load(tmp)
            close_trees(source.critic.opt_state, restored.critic.opt_state, rtol=0, atol=0)
            self.assertEqual(optimizer_state_inventory(source.critic.opt_state),
                             optimizer_state_inventory(restored.critic.opt_state))
            close_trees(source.update(batch, 2, 5001), restored.update(batch, 2, 5001), rtol=0, atol=0)
            for name in ('critic', 'actor', 'target_critic', 'temp'):
                a, b = getattr(source, name), getattr(restored, name)
                close_trees(a.params, b.params, rtol=0, atol=0)
                # As in the existing resident checkpoint test, separate GPU
                # compilations can differ by an FP32 ULP in uncompressed moments.
                close_trees(a.opt_state, b.opt_state, rtol=2e-7, atol=1e-12)
                for x, y in zip(jax.tree.leaves(a.opt_state), jax.tree.leaves(b.opt_state)):
                    if str(x.dtype).startswith('float8_'):
                        np.testing.assert_array_equal(x, y)
                close_trees(a.fp8_meta, b.fp8_meta, rtol=0, atol=0)

    def test_resume_config_keeps_old_fp32_default_and_rejects_storage_change(self):
        self.assertEqual(checkpoint_config_value({}, 'critic_optimizer_state'), 'fp32')
        with self.assertRaisesRegex(ValueError, 'critic_optimizer_state'):
            validate_checkpoint_config({}, {'critic_optimizer_state': 'fp8_carry'})


if __name__ == '__main__':
    unittest.main()
