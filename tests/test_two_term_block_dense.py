"""Algebraic VJP identity and native code/layout/gradient oracles."""
import importlib
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from jaxrl.low_precision import block_formats as bf
mod=importlib.import_module('jaxrl.low_precision.two_term_dense')

class Algebra(unittest.TestCase):
    def test_identity_quantizer(self):
        x=jnp.arange(15,dtype=jnp.float32).reshape(3,5)/11
        c=jnp.arange(20,dtype=jnp.float32).reshape(5,4)/7
        r=jnp.cos(c)*3;s=jnp.float32(0.003);b=jnp.arange(4,dtype=jnp.float32)
        theta=s*(c+r/16);g=jnp.cos(x[:,:4])
        with patch.object(mod,'pack',lambda a,f:a),patch.object(mod,'packed_gemm',lambda a,b,f:a@b.T),patch.object(mod,'block_gemm',lambda a,b,f:a@b):
            y,pull=jax.vjp(lambda x,t,b:mod.two_term_dense(x,t,c,r,s,b),x,theta,b)
            dx,dw,db=pull(g)
            np.testing.assert_allclose(y,x@theta+b,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(dx,g@theta.T,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(dw,x.T@g,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(db,g.sum(0),rtol=1e-6)

class Native(unittest.TestCase):
    def test_native_directions(self):
        if jax.default_backend() != 'gpu': self.skipTest('native GPU required')
        for m,k,n in [(1,128,128),(128,256,128),(1024,4096,4096),(4096,1024,4096)]:
            a=jax.random.normal(jax.random.key(m),(m,k))
            b=jax.random.normal(jax.random.key(k),(k,n))
            y=jax.jit(bf.block_gemm)(a,b)
            ref=bf.unpack(bf.pack(a))@bf.unpack(bf.pack(b.T)).T
            relative=float(jnp.linalg.norm(y-ref)/jnp.linalg.norm(ref))
            print({'shape':(m,k,n),'oracle_relative_l2':relative},flush=True)
            self.assertLess(relative,2e-5)
            self.assertEqual(y.dtype,jnp.float32)

    def test_carry_small_gradient(self):
        if jax.default_backend() != 'gpu': self.skipTest('native GPU required')
        x=jnp.ones((2,128));c=jnp.ones((128,128));r=jnp.ones_like(c)*8
        s=jnp.float32(1e-5);b=jnp.zeros(128);theta=s*(c+r/16)
        def run(r):
            y,pull=jax.vjp(lambda x,t,b:mod.two_term_dense(x,t,c,r,s,b),x,theta,b)
            return y,pull(jnp.full_like(y,1e-7))
        y,(dx,dw,db)=jax.jit(run)(r)
        y0,(dx0,_,_)=jax.jit(run)(jnp.zeros_like(r))
        np.testing.assert_allclose(y/y0,1.5,rtol=1e-5)
        np.testing.assert_allclose(dx/dx0,1.5,rtol=1e-5)
        self.assertTrue(np.all(np.asarray(dx)!=0))
        np.testing.assert_allclose(dw,2e-7,rtol=0.07)

if __name__=='__main__': unittest.main()
