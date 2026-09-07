"""AdamW with compressed residual-kernel moments and FP32 update arithmetic.

Only storage changes: the installed Optax Adam transform computes the current
update before its new moments are encoded. No full-size FP32 moment shadow is
retained. XLA controls transient allocation; storage savings are not peak savings.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


MOMENT_MODES = ('fp32', 'bf16', 'fp8', 'fp8_carry')
MOMENT_BLOCK_SIZE = 128
MOMENT_CARRY_GAIN = 16.0


class BlockMoment(NamedTuple):
    code: jax.Array
    carry: object  # None for the single-code control.
    scale: jax.Array


def is_block_moment(value):
    return isinstance(value, BlockMoment)


def is_residual_kernel(path):
    """Match the four Bronet Dense kernels, excluding input/head and biases."""
    parts = tuple(str(getattr(key, 'key', key)) for key in path)
    return (
        len(parts) >= 2
        and parts[-1] == 'kernel'
        and parts[-2].startswith('Dense_')
        and any(part.startswith('BronetBlock_') for part in parts)
    )


def _blocks(value):
    # Flatten only within each ensemble member. Padding belongs to the final
    # block of that member and is removed from the persistent code arrays.
    members = value.shape[0]
    per_member = math.prod(value.shape[1:])
    flat = value.reshape(members, per_member)
    padding = (-per_member) % MOMENT_BLOCK_SIZE
    if padding:
        flat = jnp.pad(flat, ((0, 0), (0, padding)))
    return flat.reshape(members, -1, MOMENT_BLOCK_SIZE)


def encode_moment(value, mode='fp8_carry'):
    value = jnp.asarray(value, jnp.float32)
    if mode == 'fp32':
        return value
    if mode == 'bf16':
        return jax.lax.optimization_barrier(value.astype(jnp.bfloat16))
    if mode not in ('fp8', 'fp8_carry'):
        raise ValueError(f'Unknown moment storage: {mode}')
    blocks = _blocks(value)
    amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    scale = amax / jnp.float32(448.0)
    # A subnormal FP32 scale can round *down* so far that amax/scale exceeds
    # E4M3's finite range (e.g. amax=7e-43). Round only subnormal scales upward
    # to the next representable FP32 value. This is a representability boundary,
    # not a weight-scale floor. Backends retaining subnormals (e.g. Blackwell)
    # can preserve even the smallest nonzero FP32 state.
    tiny = jnp.float32(jnp.finfo(jnp.float32).tiny)
    scale = jnp.where(
        (amax > 0) & (scale < tiny),
        jnp.nextafter(scale, jnp.float32(jnp.inf)), scale,
    )
    scale = jnp.where(amax == 0, 1.0, scale)
    # Lift a subnormal denominator into the normal range before division.
    # Scaling both operands by a power of two preserves their ratio and avoids
    # reciprocal overflow without imposing an absolute minimum moment scale.
    lift = jnp.where(scale < tiny, jnp.float32(2**64), jnp.float32(1.0))
    normalized = (blocks * lift) / (scale * lift)
    code = jax.lax.optimization_barrier(
        normalized.astype(jnp.float8_e4m3fn)
    )
    carry = None
    if mode == 'fp8_carry':
        carry = jax.lax.optimization_barrier(
            ((normalized - code.astype(jnp.float32)) * MOMENT_CARRY_GAIN)
            .astype(jnp.float8_e4m3fn)
        )

    def unpad(array):
        return array.reshape(value.shape[0], -1)[
            :, :math.prod(value.shape[1:])
        ].reshape(value.shape)

    return BlockMoment(unpad(code), None if carry is None else unpad(carry), scale)


def decode_moment(value):
    if not is_block_moment(value):
        return jnp.asarray(value, jnp.float32)
    normalized = _blocks(value.code.astype(jnp.float32))
    if value.carry is not None:
        # Second-moment carry is signed too. Do not clip its negative values.
        normalized = normalized + _blocks(value.carry.astype(jnp.float32)) / 16.0
    decoded = (normalized * value.scale).reshape(value.code.shape[0], -1)
    return decoded[:, :math.prod(value.code.shape[1:])].reshape(value.code.shape)


def decode_moment_tree(tree):
    return jax.tree.map(decode_moment, tree, is_leaf=is_block_moment)


def encode_moment_tree(tree, mode):
    return jax.tree_util.tree_map_with_path(
        lambda path, value: encode_moment(value, mode)
        if is_residual_kernel(path) else value,
        tree,
    )


def scale_by_stored_adam(mode='fp8_carry', b1=0.9, b2=0.999,
                         eps=1e-8, eps_root=0.0):
    """Optax Adam arithmetic, with independent m/v block scales on writeback."""
    if mode not in MOMENT_MODES:
        raise ValueError(f'Unknown moment storage: {mode}')
    adam = optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root)

    def store(state):
        return optax.ScaleByAdamState(
            state.count,
            encode_moment_tree(state.mu, mode),
            encode_moment_tree(state.nu, mode),
        )

    def init_fn(params):
        # Model.create may first supply physical FP8 params. All Adam arithmetic
        # and uncompressed leaves still use the baseline FP32 dtype.
        return store(adam.init(jax.tree.map(lambda p: p.astype(jnp.float32), params)))

    def update_fn(grads, state, params=None):
        restored = optax.ScaleByAdamState(
            state.count, decode_moment_tree(state.mu), decode_moment_tree(state.nu)
        )
        updates, next_state = adam.update(grads, restored, params)
        return updates, store(next_state)

    return optax.GradientTransformation(init_fn, update_fn)


def stored_adamw(learning_rate, mode='fp8_carry', b1=0.9, b2=0.999,
                 eps=1e-8, eps_root=0.0, weight_decay=1e-4, mask=None):
    """Keep Optax's bias correction, decay mask and learning-rate schedule."""
    return optax.chain(
        scale_by_stored_adam(mode, b1, b2, eps, eps_root),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale_by_learning_rate(learning_rate),
    )


def optimizer_state_inventory(state):
    """Host metadata only: record actual resident buffers without copying them."""
    leaves = []
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        if hasattr(value, 'dtype'):
            leaves.append({
                'path': jax.tree_util.keystr(path),
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'bytes': int(value.size * value.dtype.itemsize),
            })
    by_dtype = {}
    for leaf in leaves:
        by_dtype[leaf['dtype']] = by_dtype.get(leaf['dtype'], 0) + leaf['bytes']
    return {'bytes': sum(row['bytes'] for row in leaves),
            'bytes_by_dtype': by_dtype, 'leaves': leaves}


@jax.jit
def moment_diagnostics(state):
    """Sparse diagnostics; called only at the existing tensor-stat cadence."""
    adam = state[0]
    result = {}
    for name, tree in (('mu', adam.mu), ('nu', adam.nu)):
        for path, value in jax.tree_util.tree_flatten_with_path(
            tree, is_leaf=is_block_moment
        )[0]:
            if not is_residual_kernel(path):
                continue
            restored = decode_moment(value)
            row = {
                'minimum': jnp.min(restored),
                'negative_fraction': jnp.mean(restored < 0),
                'nonfinite_count': jnp.sum(~jnp.isfinite(restored)),
            }
            if is_block_moment(value):
                row.update(scale_min=jnp.min(value.scale), scale_max=jnp.max(value.scale))
                if value.carry is not None:
                    carry = value.carry.astype(jnp.float32)
                    row.update(
                        carry_zero_fraction=jnp.mean(carry == 0),
                        carry_absmax=jnp.max(jnp.abs(carry)),
                        carry_saturation_fraction=jnp.mean(jnp.abs(carry) >= 448),
                    )
            result[name + '/' + jax.tree_util.keystr(path)] = row
    return result
