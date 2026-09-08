#!/usr/bin/env python3
"""Single real-width VJP for an external CUDA kernel trace."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
from jaxrl.low_precision.two_term_dense import two_term_dense
fmt=sys.argv[1] if len(sys.argv)>1 else 'mxfp8'
x=jnp.ones((1024,4096));c=jnp.ones((4096,4096));r=c*2;s=jnp.float32(0.001);b=jnp.zeros(4096)
@jax.jit
def probe(x,c,r,s,b):
    y,pull=jax.vjp(lambda x,t,b:two_term_dense(x,t,c,r,s,b,fmt),x,c,b)
    return y,pull(jnp.ones_like(y)*1e-6)
probe(x,c,r,s,b)[0].block_until_ready()
print('native Fprop/Dgrad/Wgrad trace complete',flush=True)
