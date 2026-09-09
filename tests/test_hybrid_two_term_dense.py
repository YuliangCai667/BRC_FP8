"""Hybrid CARRY algebra, legacy compatibility, and scale metadata contracts."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import fp8_ops
from jaxrl.low_precision.hybrid_two_term_dense import hybrid_two_term_dense


class Hybrid(unittest.TestCase):
    def operands(self, m=8, k=128, n=128):
        rng = np.random.default_rng(19)
        x = jnp.asarray(rng.normal(size=(m, k)), jnp.float32)
        c = jnp.asarray(rng.normal(size=(k, n))*80).astype(jnp.float8_e4m3fn)
        r = jnp.asarray(rng.normal(size=(k, n))*50).astype(jnp.float8_e4m3fn)
        s = jnp.float32(.0007)
        b = jnp.asarray(rng.normal(size=n), jnp.float32)
        g = jnp.asarray(rng.normal(size=(m, n))*1e-5, jnp.float32)
        gs = jnp.ones(1, jnp.float32)
        history = jnp.full(16, jnp.max(jnp.abs(g))*2)
        return x, c, r, s, b, g, gs, history

    def run_vjp(self, x, c, r, s, b, g, gs, history):
        theta = s*(c.astype(jnp.float32)+r.astype(jnp.float32)/16)
        def f(x, theta, b, gs, history):
            return hybrid_two_term_dense(x, theta, c, r, s, b, gs, history)
        y, pull = jax.vjp(f, x, theta, b, gs, history)
        return y, pull(g)

    def test_quantized_operand_oracle(self):
        x, c, r, s, b, g, gs, h = self.operands()
        y, (dx, dw, db, ns, nh) = jax.jit(self.run_vjp)(x,c,r,s,b,g,gs,h)
        import ml_dtypes
        xn, gn = np.asarray(x), np.asarray(g)
        xs = np.max(np.abs(xn))/np.float32(448)
        scale = np.max(np.asarray(h))/np.float32(57344)
        qx = (xn/xs).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
        qg = np.clip(gn/scale,-57344,57344).astype(ml_dtypes.float8_e5m2).astype(np.float32)
        if jax.default_backend() == 'cpu':
            # This JAX CPU version lowers FP32->FP8 through BF16. Audit the
            # algebra against materialized backend operands; GPU additionally
            # checks direct NumPy RTN above.
            qx=np.asarray(jax.jit(lambda a:jax.lax.optimization_barrier(
                (a/(jnp.max(jnp.abs(a))/448)).astype(jnp.float8_e4m3fn)))(x)).astype(np.float32)
            qg=np.asarray(jax.jit(lambda a,scale:jax.lax.optimization_barrier(
                fp8_ops.quantize(a,jnp.float8_e5m2,scale,jnp.float32)))(g,jnp.asarray(scale))).astype(np.float32)
        cn,rn = np.asarray(c).astype(np.float32),np.asarray(r).astype(np.float32)
        refs = [(qx@cn+(qx@rn)/16)*(xs*float(s))+np.asarray(b),
                (qg@cn.T+(qg@rn.T)/16)*(scale*float(s)), qx.T@qg*(xs*scale)]
        for actual, expected in zip((y,dx,dw),refs):
            err = np.linalg.norm(np.asarray(actual)-expected)/np.linalg.norm(expected)
            self.assertLess(err, 2e-6)
            self.assertEqual(actual.dtype,jnp.float32)
        np.testing.assert_allclose(db,gn.sum(0),rtol=1e-6,atol=1e-10)
        expected_s,expected_h=fp8_ops.update_fp8_meta(g,jnp.float8_e5m2,gs,h)
        np.testing.assert_array_equal(ns,expected_s)
        np.testing.assert_array_equal(nh,expected_h)

    def test_zero_carry_matches_legacy_forward_backward_and_metadata(self):
        x,c,r,s,b,g,gs,h=self.operands()
        r=jnp.zeros_like(r)
        actual=jax.jit(self.run_vjp)(x,c,r,s,b,g,gs,h)
        def legacy(x,w,b,gs,h,c,s):
            xs=jnp.max(jnp.abs(x))/448
            qx=(x/xs).astype(jnp.float8_e4m3fn)
            y=fp8_ops.quantized_dot(x,qx,xs,w,c,s,gs,h,jnp.float32,
                                   (((1,),(0,)),((),())),preferred_element_type=jnp.float32)
            return fp8_ops.out_dq(jnp.float32,xs,s,y)+b
        def run(x,c,s,b,g,gs,h):
            y,pull=jax.vjp(lambda x,w,b,gs,h:legacy(x,w,b,gs,h,c,s),x,c.astype(jnp.float32)*s,b,gs,h)
            return y,pull(g)
        expected=jax.jit(run)(x,c,s,b,g,gs,h)
        for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
            err=np.linalg.norm(np.asarray(a)-np.asarray(e))/max(np.linalg.norm(np.asarray(e)),1e-30)
            self.assertLess(err,2e-6)

    def test_carry_read_changes_forward_and_input_gradient_only(self):
        x=jnp.ones((2,128));c=jnp.ones((128,128),jnp.float8_e4m3fn)
        r=c* jnp.asarray(8,jnp.float8_e4m3fn)
        s=jnp.float32(1e-5);b=jnp.zeros(128);g=jnp.full((2,128),1e-7)
        gs=jnp.ones(1);h=jnp.full(16,1e-7)
        y,(dx,dw,db,ns,nh)=jax.jit(self.run_vjp)(x,c,r,s,b,g,gs,h)
        y0,(dx0,dw0,db0,ns0,nh0)=jax.jit(self.run_vjp)(x,c,jnp.zeros_like(r),s,b,g,gs,h)
        np.testing.assert_allclose(y/y0,1.5,rtol=1e-6)
        np.testing.assert_allclose(dx/dx0,1.5,rtol=1e-6)
        for a,e in [(dw,dw0),(db,db0),(ns,ns0),(nh,nh0)]:
            np.testing.assert_array_equal(a,e)
        np.testing.assert_allclose(dw,2e-7,rtol=1e-6)

    def test_zero_input_and_zero_gradient_finite(self):
        x,c,r,s,b,g,gs,h=self.operands()
        y,grads=jax.jit(self.run_vjp)(jnp.zeros_like(x),c,r,s,b,jnp.zeros_like(g),gs,jnp.zeros_like(h))
        np.testing.assert_array_equal(y,jnp.broadcast_to(b,y.shape))
        for a in grads[:3]: np.testing.assert_array_equal(a,jnp.zeros_like(a))
        for a in jax.tree.leaves((y,grads)): self.assertTrue(np.isfinite(np.asarray(a)).all())

    def test_transposed_and_actual_shapes(self):
        if jax.default_backend() != 'gpu': self.skipTest('full shapes use GPU')
        for m,k,n in [(1,256,256),(1024,4096,4096)]:
            values=self.operands(m,k,n)
            y,grads=jax.jit(self.run_vjp)(*values)
            for a in jax.tree.leaves((y,grads)): self.assertTrue(np.isfinite(np.asarray(a)).all())
            self.assertGreater(float(jnp.linalg.norm(grads[0])),0)


if __name__=='__main__': unittest.main()
