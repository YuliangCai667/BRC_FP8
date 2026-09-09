"""The shared actor QAT / deployment codec (FP32 scales, not E8M0 MXFP8).

Matrices use [K, N], blocks of 32 along K, independently for each column.
Packed codes are actual E4M3FN values; serialize their bits, not numeric uint8s.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

CODEC = 'e4m3fn_block32_fp32scale_rtn_v1'
BLOCK_SIZE = 32


def _invalid_codec_input(_):
    raise FloatingPointError('actor export codec: nonfinite kernel or illegal FP32 scale')


def _e4m3_rtn_bits(value):
    """Round FP32 directly, avoiding backend FP32 -> BF16 -> E4M3 rounding.

    Some XLA CPU conversions double-round values near E4M3 midpoints (e.g.
    -151.99255). Explicit ties-to-even on the E4M3 grid has identical semantics
    on CPU and GPU; the final bitcast cannot be fused into a different cast.
    Input is finite, already clipped to [-448,448].
    """
    bits = lax.bitcast_convert_type(value, jnp.uint32)
    exponent = ((bits >> 23) & jnp.uint32(255)).astype(jnp.int32) - 127
    exponent = jnp.maximum(exponent, -6)
    step_bits = (exponent - 3 + 127).astype(jnp.uint32) << 23
    step = lax.bitcast_convert_type(step_bits, jnp.float32)
    mantissa = jnp.rint(jnp.abs(value) / step).astype(jnp.int32)
    magnitude = jnp.minimum((exponent + 6) * 8 + mantissa, 126).astype(jnp.uint8)
    sign = ((bits >> 24) & jnp.uint32(128)).astype(jnp.uint8)
    return lax.bitcast_convert_type(magnitude | sign, jnp.float8_e4m3fn)


def pack_export_kernel(kernel):
    kernel = jnp.asarray(kernel, jnp.float32)
    if kernel.ndim != 2 or min(kernel.shape) <= 0:
        raise ValueError('export kernel must have nonempty [K, N] shape')
    k, n = kernel.shape
    blocks = jnp.pad(kernel, ((0, (-k) % BLOCK_SIZE), (0, 0))).reshape(-1, BLOCK_SIZE, n)
    amax = jnp.max(jnp.abs(blocks), axis=1)
    scales = jnp.where(amax == 0, jnp.float32(1), amax / lax.optimization_barrier(jnp.float32(448)))
    valid = jnp.all(jnp.isfinite(kernel)) & jnp.all(jnp.isfinite(scales) & (scales > 0))
    # No callback on the normal path. Invalid traced inputs fail as well as
    # eager inputs; silently clipping infinities would hide corrupted weights.
    if isinstance(valid, jax.core.Tracer):
        def fail(_):
            jax.debug.callback(_invalid_codec_input, jnp.int32(0))
            return jnp.int32(0)
        lax.cond(valid, lambda _: jnp.int32(0), fail, operand=None)
    elif not bool(np.asarray(valid)):
        _invalid_codec_input(None)
    scales = lax.stop_gradient(scales)
    normalized = lax.optimization_barrier(jnp.clip(blocks / scales[:, None, :], -448, 448))
    codes = _e4m3_rtn_bits(normalized)
    codes = lax.stop_gradient(lax.optimization_barrier(codes))
    return codes, scales


def unpack_export_kernel(codes, scales, shape):
    k, n = shape
    if codes.shape != ((k + 31) // 32, 32, n) or scales.shape != ((k + 31) // 32, n):
        raise ValueError('code/scale shapes do not match [K, N] and block32 padding')
    return (codes.astype(jnp.float32) * scales[:, None, :]).reshape(-1, n)[:k]


def effective_export_kernel(kernel):
    codes, scales = pack_export_kernel(kernel)
    return lax.optimization_barrier(unpack_export_kernel(codes, scales, kernel.shape).astype(jnp.bfloat16))


def decoded_dense(x, weight, bias):
    """W8A16 contract: BF16 inputs/decoded weights, FP32 accumulation and bias."""
    return jnp.matmul(x.astype(jnp.bfloat16), weight.astype(jnp.bfloat16),
                      preferred_element_type=jnp.float32) + bias.astype(jnp.float32)


@jax.custom_vjp
def qat_dense(x, logical_kernel, bias):
    return decoded_dense(x, effective_export_kernel(logical_kernel), bias)


def _qat_fwd(x, kernel, bias):
    weight = effective_export_kernel(kernel)
    return decoded_dense(x, weight, bias), (x, weight)


def _qat_bwd(res, gradient):
    x, weight = res
    g = gradient.astype(jnp.float32)
    x2 = x.astype(jnp.bfloat16).astype(jnp.float32).reshape(-1, x.shape[-1])
    g2 = g.reshape(-1, g.shape[-1])
    dx = jnp.matmul(g, weight.astype(jnp.float32).T, precision=lax.Precision.HIGHEST)
    dw = jnp.matmul(x2.T, g2, precision=lax.Precision.HIGHEST)
    return dx.astype(x.dtype), dw, g2.sum(axis=0)


qat_dense.defvjp(_qat_fwd, _qat_bwd)
