import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from jaxrl.low_precision.actor_export_codec import pack_export_kernel, unpack_export_kernel, qat_dense, effective_export_kernel
from jaxrl.low_precision import actor_qat_dense as aq
from jaxrl.networks import NormalTanhPolicy
from jaxrl.utils import Model


def actor():
    return aq.initialize_actor(Model.create(NormalTanhPolicy(3, hidden_dims=32,
        actor_training_recipe=aq.RECIPE, actor_export_aligned=True),
        [jax.random.PRNGKey(9), jnp.ones((4, 35))], tx=optax.adamw(3e-4)))


class Codec(unittest.TestCase):
    def test_direct_rtn_midpoints_and_signed_zero(self):
        import ml_dtypes
        from jaxrl.low_precision.actor_export_codec import _e4m3_rtn_bits
        positive = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        midpoints = (positive[:-1] + positive[1:]) * np.float32(.5)
        values = np.concatenate([positive, midpoints, np.nextafter(midpoints, -np.inf),
                                 np.nextafter(midpoints, np.inf), np.array([151.99255],np.float32)])
        values = np.concatenate([values,-values])
        codes = jax.jit(_e4m3_rtn_bits)(jnp.asarray(values))
        np.testing.assert_array_equal(np.asarray(codes).view(np.uint8), values.astype(ml_dtypes.float8_e4m3fn).view(np.uint8))

    def test_numpy_oracle_padding_zero_tiny_and_jit(self):
        import ml_dtypes
        w = np.random.default_rng(42).normal(size=(35, 7)).astype(np.float32) * 1e-12
        w[:, 0] = 0
        c, s = jax.jit(pack_export_kernel)(jnp.asarray(w))
        b = np.pad(w, ((0,29),(0,0))).reshape(2,32,7)
        mx = np.max(np.abs(b), axis=1)
        scale = np.where(mx == 0, 1, mx / np.float32(448))
        ref = np.clip(b / scale[:,None,:], -448,448).astype(ml_dtypes.float8_e4m3fn)
        np.testing.assert_array_equal(np.asarray(c).view(np.uint8), ref.view(np.uint8))
        # GPU FP32 division can differ from the CPU scalar division by one ULP.
        np.testing.assert_array_max_ulp(np.asarray(s), scale, maxulp=1)
        np.testing.assert_allclose(unpack_export_kernel(c,s,w.shape), (ref.astype(np.float32)*np.asarray(s)[:,None,:]).reshape(64,7)[:35], rtol=0,atol=0)
        self.assertEqual(str(c.dtype),'float8_e4m3fn')

    def test_quantization_survives_jit_and_roundtrip(self):
        w = jax.random.normal(jax.random.PRNGKey(2),(64,8))
        c, s = pack_export_kernel(w)
        saved = jax.lax.bitcast_convert_type(c,jnp.uint8)
        decoded = jax.lax.bitcast_convert_type(saved,jnp.float8_e4m3fn)
        expected = unpack_export_kernel(decoded,s,w.shape).astype(jnp.bfloat16)
        np.testing.assert_array_equal(jax.jit(effective_export_kernel)(w), expected)
        self.assertGreater(float(jnp.max(jnp.abs(expected.astype(jnp.float32)-w))),0)

    def test_invalid_inputs_fail_eager_and_jit(self):
        for value in (np.nan, np.inf):
            with self.assertRaises(FloatingPointError):
                pack_export_kernel(jnp.full((32,2),value))
        with self.assertRaises(Exception):
            jax.block_until_ready(jax.jit(pack_export_kernel)(jnp.full((32,2),np.inf)))

    def test_ste_has_one_physical_fp32_weight_gradient(self):
        x = jax.random.normal(jax.random.PRNGKey(4),(4,35))
        w = jax.random.normal(jax.random.PRNGKey(5),(35,3))
        b = jnp.array([.1,.2,.3])
        g = jnp.arange(12,dtype=jnp.float32).reshape(4,3)/10
        y, pull = jax.vjp(qat_dense,x,w,b)
        dx,dw,db = pull(g)
        q = effective_export_kernel(w).astype(jnp.float32)
        # NumPy FP32 oracle avoids the GPU default matmul's TF32 reduction.
        xn,qn,gn=np.asarray(x.astype(jnp.bfloat16).astype(jnp.float32)),np.asarray(q),np.asarray(g)
        np.testing.assert_allclose(y,xn@qn+np.asarray(b),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(dx,gn@qn.T,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(dw,xn.T@gn,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(db,g.sum(0))
        self.assertEqual(str(dw.dtype),'float32')


class Storage(unittest.TestCase):
    def test_storage_optimizer_and_nonzero_initial_carry(self):
        a = actor()
        kernels = {p:v for p,v in traverse_util.flatten_dict(a.params).items() if p[-1]=='kernel'}
        self.assertEqual(len(kernels),5)
        for p,v in kernels.items():
            self.assertEqual(str(v.dtype),'float8_e4m3fn' if aq.is_body(p) else 'bfloat16')
        for p,v in traverse_util.flatten_dict(a.fp8_meta).items():
            if p[-1]=='kernel_carry':
                self.assertGreater(float(jnp.mean(v.astype(jnp.float32)!=0)),.9)
        moments = a.opt_state[0]
        for x in jax.tree.leaves((moments.mu,moments.nu)):
            self.assertEqual(str(x.dtype),'float32')
        self.assertEqual(jax.tree.structure(aq.logical_params(a)),jax.tree.structure(moments.mu))

    def test_adam_uses_logical_weights_then_bf16_write_and_checkpoint(self):
        a = actor(); params = aq.logical_params(a)
        grads = jax.tree.map(lambda x:jnp.ones_like(x)*.002,params)
        updates,state = a.tx.update(grads,a.opt_state,params)
        expected = optax.apply_updates(params,updates)
        actual,_ = jax.jit(aq.apply_gradients)(a,params,grads,{})
        for p,w in traverse_util.flatten_dict(actual.params).items():
            if p[-1]=='kernel' and not aq.is_body(p):
                np.testing.assert_array_equal(w,traverse_util.flatten_dict(expected)[p].astype(jnp.bfloat16))
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'actor.msgpack');actual.save(path)
            restored=a.load(path,require_fp8_metadata=True)
            x=jnp.arange(140,dtype=jnp.float32).reshape(4,35)/100
            f=jax.jit(lambda model:model(x,return_stats=True))
            for u,v in zip(f(actual),f(restored)):
                np.testing.assert_array_equal(u,v)

    def test_phase_restore_and_pinned_recovery(self):
        from jaxrl.agent.brc_learner import BRC
        from jaxrl.checkpoint import CheckpointManager, validate_checkpoint_config
        from jaxrl.normalizer import RewardNormalizer
        from jaxrl.logger import EpisodeRecorder
        from tests.test_checkpoint import make_buffer, fill_buffer
        a = BRC(0, np.zeros((1,4),np.float32),np.zeros((1,2),np.float32),
                num_tasks=2,width_actor=16,width_critic=16,actor_training_recipe=aq.RECIPE)
        a.set_env_step(449999)
        old = jax.tree.map(np.asarray,(a.actor.params,a.actor.fp8_meta,a.actor.opt_state))
        self.assertTrue(a.set_env_step(450000))
        self.assertEqual(a.actor_phase,'export_align')
        for x,y in zip(jax.tree.leaves(old),jax.tree.leaves((a.actor.params,a.actor.fp8_meta,a.actor.opt_state))):
            np.testing.assert_array_equal(x,y)
        config=dict(actor_training_recipe=aq.RECIPE,actor_export_align_start=450000)
        with self.assertRaises(ValueError):
            validate_checkpoint_config(config,{**config,'actor_export_align_start':450001})
        with tempfile.TemporaryDirectory() as d:
            buffer=make_buffer();fill_buffer(buffer,3)
            norm=RewardNormalizer(2,target_entropy=a.target_entropy);episodes=EpisodeRecorder(2,['a','b'])
            manager=CheckpointManager(Path(d)/'checkpoints','test',d,['a','b'],config)
            pinned=manager.save_recovery(a,buffer,norm,episodes,450000,None,is_phase_transition=True)
            a.set_env_step(449999)
            manager.load_recovery(pinned,a,buffer,norm,episodes)
            self.assertEqual(a.actor_phase,'export_align')
            self.assertEqual(a.actor_env_step,450000)
            a.set_env_step(500000)
            manager.save_recovery(a,buffer,norm,episodes,500000,None,is_final=True)
            self.assertTrue((pinned/'COMPLETE').exists())

    def test_export_failure_never_marks_validation_or_eval_complete(self):
        import json
        from jaxrl.deployment_export import finish_actor_export
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ck=root/'checkpoint';ck.mkdir()
            (ck/'COMPLETE').write_text('ok')
            (ck/'manifest.json').write_text(json.dumps(dict(is_final=True,includes_replay_buffer=True,env_step=500000)))
            with patch('jaxrl.deployment_export.export_checkpoint',side_effect=ValueError('invalid scale')):
                with self.assertRaisesRegex(ValueError,'invalid scale'):
                    finish_actor_export(ck,root,None,dict(max_steps=500000))
            status=json.loads((root/'actor_export_status.json').read_text())
            self.assertTrue(status['training_complete'])
            self.assertFalse(status['export_validated'])
            self.assertFalse(status['deployment_eval_complete'])

if __name__=='__main__':
    unittest.main()
