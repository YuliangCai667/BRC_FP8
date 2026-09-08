"""Two forward terms, two input-gradient terms, one physical weight gradient."""
from functools import partial
import jax
from .block_formats import pack, packed_gemm, block_gemm

def _forward(x,c,r,s,fmt):
    px=pack(x,fmt)
    return s*(packed_gemm(px,pack(c.T,fmt),fmt)+packed_gemm(px,pack(r.T,fmt),fmt)/16)

@partial(jax.custom_vjp,nondiff_argnums=(6,))
def two_term_dense(x,logical_kernel_for_optimizer,c,r,s,b,fmt='mxfp8'):
    return _forward(x,c,r,s,fmt)+b

def _fwd(x,theta,c,r,s,b,fmt):
    return _forward(x,c,r,s,fmt)+b,(x,c,r,s)

def _bwd(fmt,res,g):
    x,c,r,s=res
    pg=pack(g,fmt)
    dx=s*(packed_gemm(pg,pack(c,fmt),fmt)+packed_gemm(pg,pack(r,fmt),fmt)/16)
    dtheta=block_gemm(x.T,g,fmt)
    return dx,dtheta,None,None,None,g.sum(axis=0)

two_term_dense.defvjp(_fwd,_bwd)
