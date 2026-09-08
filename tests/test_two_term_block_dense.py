"""Algebraic VJP identity and native code/layout/gradient oracles."""
import os
import functools
import importlib
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from jaxrl.low_precision import block_formats as bf
mod=importlib.import_module('jaxrl.low_precision.two_term_dense')

class Algebra(unittest.TestCase):
    fmt='mxfp8'
    def test_identity_quantizer(self):
        x=jnp.arange(15,dtype=jnp.float32).reshape(3,5)/11
        c=jnp.arange(20,dtype=jnp.float32).reshape(5,4)/7
        r=jnp.cos(c)*3;s=jnp.float32(0.003);b=jnp.arange(4,dtype=jnp.float32)
        theta=s*(c+r/16);g=jnp.cos(x[:,:4])
        with patch.object(mod,'pack',lambda a,f:a),patch.object(mod,'packed_gemm',lambda a,b,f:a@b.T),patch.object(mod,'block_gemm',lambda a,b,f:a@b):
            y,pull=jax.vjp(lambda x,t,b:mod.two_term_dense(x,t,c,r,s,b,self.fmt),x,theta,b)
            dx,dw,db=pull(g)
            np.testing.assert_allclose(y,x@theta+b,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(dx,g@theta.T,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(dw,x.T@g,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(db,g.sum(0),rtol=1e-6)

class Native(unittest.TestCase):
    fmt=os.environ.get('BRC_TEST_FORMAT','mxfp8')
    def test_native_nan_propagates(self):
        if jax.default_backend() != 'gpu': self.skipTest('native GPU required')
        a=jnp.ones((1,128)).at[0,0].set(jnp.nan)
        b=jnp.ones((128,128))
        y=jax.jit(functools.partial(bf.block_gemm,fmt=self.fmt))(a,b)
        self.assertTrue(bool(jnp.any(~jnp.isfinite(y))))

    def test_native_directions(self):
        if jax.default_backend() != 'gpu': self.skipTest('native GPU required')
        for m,k,n in [(1,128,128),(128,256,128),(1024,4096,4096),(4096,1024,4096)]:
            a=jax.random.normal(jax.random.key(m),(m,k))
            b=jax.random.normal(jax.random.key(k),(k,n))
            def execute_and_operands(a,b):
                pa=bf.pack(a,self.fmt);pb=bf.pack(b.T,self.fmt)
                return bf.packed_gemm(pa,pb,self.fmt),pa[0],pa[1],pa[3],pb[0],pb[1],pb[3]
            y,ac,sa,ta,bc,sb,tb=jax.jit(execute_and_operands)(a,b)
            # Oracle uses the exact device operands consumed by this GEMM.
            # Separately evaluating FP32 normalization can differ at RTN ties.
            ref=jnp.matmul(bf.unpack((ac,sa,(m,k),ta),self.fmt),
                           bf.unpack((bc,sb,(n,k),tb),self.fmt).T,
                           precision=jax.lax.Precision.HIGHEST)
            relative=float(jnp.linalg.norm(y-ref)/jnp.linalg.norm(ref))
            print({'shape':(m,k,n),'oracle_relative_l2':relative},flush=True)
            self.assertLess(relative,2e-5)
            self.assertEqual(y.dtype,jnp.float32)

    def test_carry_small_gradient(self):
        if jax.default_backend() != 'gpu': self.skipTest('native GPU required')
        x=jnp.ones((2,128));c=jnp.ones((128,128));r=jnp.ones_like(c)*8
        s=jnp.float32(1e-5);b=jnp.zeros(128);theta=s*(c+r/16)
        def run(r):
            y,pull=jax.vjp(lambda x,t,b:mod.two_term_dense(x,t,c,r,s,b,self.fmt),x,theta,b)
            return y,pull(jnp.full_like(y,1e-7))
        y,(dx,dw,db)=jax.jit(run)(r)
        y0,(dx0,_,_)=jax.jit(run)(jnp.zeros_like(r))
        np.testing.assert_allclose(y/y0,1.5,rtol=1e-5)
        np.testing.assert_allclose(dx/dx0,1.5,rtol=1e-5)
        self.assertTrue(np.all(np.asarray(dx)!=0))
        expected=np.asarray(bf.unpack(bf.pack(x.T,self.fmt),self.fmt)) @ np.asarray(bf.unpack(bf.pack(jnp.full((128,2),1e-7),self.fmt),self.fmt)).T
        np.testing.assert_allclose(dw,expected,rtol=2e-5,atol=1e-12)

class Encoding(unittest.TestCase):
    def test_nvfp4_jit_uses_the_stored_e4m3_scale(self):
        x=jax.random.normal(jax.random.key(1),(128,128))
        eager=bf.pack(x,'nvfp4')
        compiled=jax.jit(lambda x:bf.pack(x,'nvfp4'))(x)
        np.testing.assert_array_equal(eager[0],compiled[0])
        np.testing.assert_array_equal(eager[1],compiled[1])

    def test_nan_operand_cannot_be_silently_zeroed(self):
        for fmt in ('nvfp4','mxfp4'):
            x=jnp.ones((1,128)).at[0,0].set(jnp.nan)
            restored=bf.unpack(bf.pack(x,fmt),fmt)
            self.assertTrue(bool(jnp.isnan(restored[0,0])))

    def test_fp4_codes_and_midpoint_ties(self):
        values=jnp.asarray([0,.25,.5,.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,7,-.5,-6.])
        expected=[0,0,1,2,2,2,3,4,4,4,5,6,6,6,7,7,9,15]
        np.testing.assert_array_equal(bf.fp4_encode(values),expected)
    def test_pack_round_trip_storage(self):
        for fmt in ('nvfp4','mxfp4'):
            x=jnp.tile(jnp.asarray(bf.FP4_VALUES),(128,16))
            codes,scales,shape,tensor=bf.pack(x,fmt)
            self.assertEqual(codes.shape,(128,64))
            self.assertEqual(scales.shape,(1024 if fmt=='nvfp4' else 512,))
            restored=bf.unpack((codes,scales,shape,tensor),fmt)
            np.testing.assert_allclose(restored,x,rtol=1e-6,atol=1e-6)

if __name__=='__main__': unittest.main()
