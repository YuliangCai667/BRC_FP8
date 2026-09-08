"""Native RTN block-scaled operands, separate from E4M3 persistent state."""
import jax
import jax.numpy as jnp
import numpy as np
from .backends.native import register

FORMATS = ('mxfp8','nvfp4','mxfp4')
FP4_VALUES = (0.,0.5,1.,1.5,2.,3.,4.,6.)

def fp4_encode(x):
    # Midpoint ties go to the even E2M1 code, including 0.25 -> 0.
    ax=jnp.abs(x)
    thresholds=jnp.asarray((0.25,0.75,1.25,1.75,2.5,3.5,5.),jnp.float32)
    above=ax[...,None]>thresholds
    tied=(ax[...,None]==thresholds)&(jnp.arange(7)%2==1)
    code=jnp.sum(above|tied,axis=-1).astype(jnp.uint8)
    return code | (jnp.signbit(x).astype(jnp.uint8)<<3)

def pack(a, fmt='mxfp8'):
    if fmt not in FORMATS: raise ValueError(f'Unsupported native format: {fmt}')
    rows,reduction=a.shape
    rp=(rows+127)//128*128;kp=(reduction+127)//128*128
    block=16 if fmt=='nvfp4' else 32
    values=jnp.pad(a.astype(jnp.float32),((0,rp-rows),(0,kp-reduction)))
    blocks=values.reshape(rp,kp//block,block)
    amax=jnp.max(jnp.abs(blocks),axis=-1)
    tensor_scale=jnp.float32(1.)
    if fmt=='nvfp4':
        global_amax=jnp.max(amax)
        tensor_scale=jnp.where(global_amax>0,global_amax/(6*448),1.)
        local=jnp.clip(amax/(6*tensor_scale),0,448).astype(jnp.float8_e4m3fn)
        # The operand must use the rounded stored scale, including under JIT.
        local=jax.lax.optimization_barrier(local)
        scale=local.astype(jnp.float32)*tensor_scale
        normalized=jnp.where(scale[...,None]>0,blocks/jnp.where(scale[...,None]>0,scale[...,None],1.),0.)
        scales=jax.lax.bitcast_convert_type(local,jnp.uint8)
    else:
        max_exponent=8 if fmt=='mxfp8' else 2
        exponent=jnp.where(amax>0,jnp.floor(jnp.log2(amax))-max_exponent,0)
        exponent=jnp.clip(exponent,-126,127)
        normalized=blocks/jnp.exp2(exponent)[...,None]
        scales=(exponent+127).astype(jnp.uint8)
        if fmt=='mxfp4':
            # E2M1 has no NaN encoding. Preserve invalid blocks through E8M0 NaN.
            scales=jnp.where(jnp.isnan(amax),jnp.uint8(255),scales)
    if fmt=='mxfp8':
        codes=jnp.clip(normalized,-448,448).astype(jnp.float8_e4m3fn)
        codes=jax.lax.bitcast_convert_type(codes,jnp.uint8).reshape(rp,kp)
    else:
        nibble=fp4_encode(normalized).reshape(rp,kp)
        codes=nibble[:,0::2]|(nibble[:,1::2]<<4)
    scales=scales.reshape(rp//128,4,32,kp//block//4,4).transpose(0,3,2,1,4).reshape(-1)
    return codes,scales,(rows,reduction),tensor_scale

def unpack(packed,fmt='mxfp8'):
    codes,scales,(rows,reduction),tensor_scale=packed
    rp=codes.shape[0];kp=codes.shape[1]*(1 if fmt=='mxfp8' else 2)
    block=16 if fmt=='nvfp4' else 32
    scales=scales.reshape(rp//128,kp//block//4,32,4,4).transpose(0,3,2,1,4).reshape(rp,kp//block)
    if fmt=='nvfp4': scale=jax.lax.bitcast_convert_type(scales,jnp.float8_e4m3fn).astype(jnp.float32)*tensor_scale
    else: scale=jnp.where(scales==255,jnp.nan,jnp.exp2(scales.astype(jnp.float32)-127))
    if fmt=='mxfp8': values=jax.lax.bitcast_convert_type(codes,jnp.float8_e4m3fn).astype(jnp.float32)
    else:
        nibble=jnp.stack((codes&15,codes>>4),axis=-1).reshape(rp,kp)
        values=jnp.asarray(FP4_VALUES,jnp.float32)[nibble&7]*jnp.where(nibble&8,-1.,1.)
    values=values.reshape(rp,kp//block,block)*scale[...,None]
    return values.reshape(rp,kp)[:rows,:reduction]

def packed_gemm(a,b,fmt='mxfp8'):
    register(fmt)
    ac,sa,(m,k),ta=a;bc,sb,(n,kb),tb=b
    if k!=kb: raise ValueError('Reduction dimensions differ')
    mp=ac.shape[0];kp=ac.shape[1]*(1 if fmt=='mxfp8' else 2);npad=bc.shape[0]
    y,_=jax.ffi.ffi_call('brc_'+fmt,(jax.ShapeDtypeStruct((mp,npad),jnp.float32),jax.ShapeDtypeStruct((32*1024*1024,),jnp.uint8)),vmap_method='sequential')(ac,bc,sa,sb,m=np.int64(mp),n=np.int64(npad),k=np.int64(kp))
    return (y[:m,:n]*ta)*tb

def block_gemm(a,b,fmt='mxfp8'):
    return packed_gemm(pack(a,fmt),pack(b.T,fmt),fmt)
