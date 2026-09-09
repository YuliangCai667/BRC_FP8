"""RTN MXFP8 E4M3/E8M0-32 operands, each direction packed from its source."""
import jax
import jax.numpy as jnp
import numpy as np
from .backends.native import register

FORMATS = ('mxfp8',)
MXFP8_SCALE_POLICY = 'ceil_amax_over_448_no_overflow'

def _power_of_two(exponent):
    # Construct E8M0 powers exactly; exp2/log2 approximations under JIT can
    # otherwise move values across a quantization boundary.
    biased = (exponent.astype(jnp.int32) + 127).astype(jnp.uint32)
    bits = jnp.where(biased == 0, jnp.uint32(1 << 22), biased << 23)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)

def pack(a, fmt='mxfp8'):
    if fmt not in FORMATS:
        raise ValueError(f'Unsupported native format: {fmt}')
    rows, reduction = a.shape
    rp = (rows+127)//128*128
    kp = (reduction+127)//128*128
    blocks = jnp.pad(a.astype(jnp.float32), ((0,rp-rows),(0,kp-reduction))).reshape(rp,kp//32,32)
    amax = jnp.max(jnp.abs(blocks),axis=-1)
    amax_bits = jax.lax.bitcast_convert_type(amax, jnp.uint32)
    exponent = jnp.where(amax > 0, (amax_bits >> 23).astype(jnp.int32)-127-8, 0)
    exponent = jnp.clip(exponent,-126,127)
    # Select ceil(log2(amax / 448)): the initial exponent can leave the
    # block's maximum outside E4M3's range. Compare the actual capacity so
    # values just above a boundary cannot be rounded down by log2.
    exponent = jnp.where(amax > 448 * _power_of_two(exponent), exponent + 1, exponent)
    exponent = jnp.clip(exponent,-126,127)
    scale = _power_of_two(exponent)
    codes = jnp.clip(blocks/scale[...,None],-448,448).astype(jnp.float8_e4m3fn)
    codes = jax.lax.bitcast_convert_type(codes,jnp.uint8).reshape(rp,kp)
    scales = (exponent+127).astype(jnp.uint8)
    # cuBLAS block scale tiles: [row/128,k/4,row%32,row/32,k%4].
    scales = scales.reshape(rp//128,4,32,kp//128,4).transpose(0,3,2,1,4).reshape(-1)
    return codes, scales, (rows,reduction)

def unpack(packed):
    codes, scales, (rows,reduction) = packed
    rp,kp=codes.shape
    scales=scales.reshape(rp//128,kp//128,32,4,4).transpose(0,3,2,1,4).reshape(rp,kp//32)
    scale=_power_of_two(scales.astype(jnp.int32)-127)
    values=jax.lax.bitcast_convert_type(codes,jnp.float8_e4m3fn).astype(jnp.float32).reshape(rp,kp//32,32)*scale[...,None]
    return values.reshape(rp,kp)[:rows,:reduction]

def packed_gemm(a,b,fmt='mxfp8'):
    # b is packed as rows of B^T; reduction is contiguous for both operands.
    register()
    ac,sa,(m,k)=a; bc,sb,(n,kb)=b
    if k != kb: raise ValueError('Reduction dimensions differ')
    mp,kp=ac.shape; npad=bc.shape[0]
    y,_=jax.ffi.ffi_call('brc_mxfp8', (jax.ShapeDtypeStruct((mp,npad),jnp.float32),jax.ShapeDtypeStruct((32*1024*1024,),jnp.uint8)),vmap_method='sequential')(ac,bc,sa,sb,m=np.int64(mp),n=np.int64(npad),k=np.int64(kp))
    return y[:m,:n]

def block_gemm(a,b,fmt='mxfp8'):
    return packed_gemm(pack(a,fmt),pack(b.T,fmt),fmt)
