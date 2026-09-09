"""Per-tensor E4M3 operands and delayed E5M2 gradients for CARRY Dense.

State is s*(C+R/16). Forward and input gradients read both stored codes;
the optimizer receives one physical FP32 weight gradient. Gradient metadata
is overwritten once per backward, using the same rule as legacy Flax FP8.
"""
import jax
from jax import lax
import jax.numpy as jnp
from flax.linen import fp8_ops


def _dot(a, b, precision=lax.Precision.DEFAULT):
    # cuBLAS FP8 needs aligned dimensions, including batch=1 policy inference
    # and the batch reduction in Wgrad. Padding adds exact zero products.
    m, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError('Hybrid Dense reduction dimensions differ')
    a = jnp.pad(a, ((0, (-m) % 16), (0, (-k) % 16)))
    b = jnp.pad(b, ((0, (-k) % 16), (0, (-n) % 16)))
    return lax.dot_general(a, b, (((1,), (0,)), ((), ())),
                           precision=precision, preferred_element_type=jnp.float32)[:m, :n]


def _forward(x, c, r, scale, bias):
    amax = jnp.max(jnp.abs(x))
    xs = jnp.where(amax > 0, amax / jnp.float32(448), jnp.float32(1))
    qx = lax.optimization_barrier((x / xs).astype(jnp.float8_e4m3fn))
    c = lax.optimization_barrier(c.astype(jnp.float8_e4m3fn))
    r = lax.optimization_barrier(r.astype(jnp.float8_e4m3fn))
    y = (_dot(qx, c) + _dot(qx, r) * jnp.float32(1/16)) * (xs * scale) + bias
    return y, (qx, xs, c, r, scale)


@jax.custom_vjp
def hybrid_two_term_dense(x, logical_kernel, main, carry, scale, bias,
                          output_grad_scale, output_grad_amax_history):
    return _forward(x, main, carry, scale, bias)[0]


def _fwd(x, theta, c, r, s, b, grad_scale, grad_history):
    y, residual = _forward(x, c, r, s, b)
    return y, (*residual, grad_scale, grad_history)


def _bwd(residual, g):
    qx, xs, c, r, s, grad_scale, grad_history = residual
    new_scale, new_history = fp8_ops.update_fp8_meta(
        g, jnp.float8_e5m2, grad_scale, grad_history)
    scale = fp8_ops._fm32_to_float32(new_scale)
    qg = lax.optimization_barrier(fp8_ops.quantize(
        g, jnp.float8_e5m2, scale, jnp.float32))
    dx = (_dot(qg, c.T, lax.Precision.HIGHEST)
          + _dot(qg, r.T, lax.Precision.HIGHEST) * jnp.float32(1/16)) * (scale * s)
    dw = _dot(qx.T, qg, lax.Precision.HIGHEST) * (xs * scale)
    return dx, dw, None, None, None, g.sum(axis=0), new_scale, new_history


hybrid_two_term_dense.defvjp(_fwd, _bwd)
