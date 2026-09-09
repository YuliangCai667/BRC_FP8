#!/usr/bin/env python3
"""16 real full-width updates, per-task evaluation and deterministic save/load."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from jaxrl.agent.brc_learner import BRC
from jaxrl.envs import ParallelEnv
from jaxrl.env_names import get_environment_list
from jaxrl.optimizers import optimizer_state_inventory
from jaxrl.utils import Batch
from jaxrl.low_precision.backends.native import register


def host_metrics(tree):
    return jax.tree.map(lambda x:np.asarray(x).tolist(),tree)

def check_finite(tree):
    for x in jax.tree.leaves(tree):
        if hasattr(x,'dtype') and jnp.issubdtype(x.dtype,jnp.inexact):
            assert bool(jnp.all(jnp.isfinite(x))), (x.shape,x.dtype)

def main():
    p=argparse.ArgumentParser();p.add_argument('--format',choices=('legacy','hybrid','mxfp8'),default='hybrid');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    report={'format':a.format,'status':'STARTED','backend':'XLA per-tensor FP8 hybrid'}
    if a.format == 'mxfp8':
        report.update(backend='CUTLASS SM120 native block-scaled',kernel_build_hash=register()[1])
    def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2))
    save()
    env=ParallelEnv(get_environment_list('DMC_DOGS'),seed=42)
    env.action_space.seed(42)
    obs=env.reset();act=np.zeros(env.action_space.shape,np.float32)
    agent=BRC(42,obs[:1],act[:1],num_tasks=4,updates_per_step=2,width_critic=4096,task_embedding_norm='l1',critic_precision='fp8_resident',fp8_resident_carry=True,target_critic_precision='fp8_lag',critic_optimizer_state='fp8_carry',critic_residual_compute_format=a.format)
    print('initialized full learner',obs.shape,act.shape,flush=True)
    # Legal environment transitions, sampled repeatedly only for engineering smoke.
    records=[]
    for i in range(256):
        action=env.action_space.sample()
        nxt,reward,term,trunc,_=env.step(action)
        records.append((obs.copy(),action,reward,env.generate_masks(term,trunc),nxt.copy()))
        obs,_,_=env.reset_where_done(nxt,term,trunc)
    arrays=[np.stack([r[i] for r in records]) for i in range(5)]
    np.savez(a.output/'transitions.npz',**{str(i):v for i,v in enumerate(arrays)})
    rng=np.random.RandomState(20260908)
    def batch():
        ix=rng.randint(256,size=1024);tasks=rng.randint(4,size=1024)
        vals=[jnp.asarray(v[ix,tasks],jnp.float32) for v in arrays]
        # Normalized reward scale provides a legal bounded distributional target.
        vals[2]=vals[2]*0.01
        return Batch(*[v[None] for v in vals],jnp.asarray(tasks,dtype=jnp.int32)[None])
    times=[]
    for i in range(16):
        t=time.perf_counter();info=agent.update(batch(),1,i,collect_update_diagnostics=(i==15));jax.block_until_ready(info)
        times.append(time.perf_counter()-t);check_finite(info)
        print(json.dumps({'update':i+1,'step':int(agent.step),'seconds':times[-1],'critic_loss':float(info['critic_loss'])}),flush=True)
    report.update(updates=16,step=int(agent.step),update_seconds=times,optimizer_inventory=optimizer_state_inventory(agent.critic.opt_state))
    backward_meta={ '/'.join(k):np.asarray(v).tolist() for k,v in traverse_util.flatten_dict(agent.critic.fp8_meta).items()
                   if k[-1] in ('output_grad_scale','output_grad_amax_history')}
    report['backward_metadata']=backward_meta
    if a.format in ('legacy','hybrid'):
        assert all(np.max(v)>0 for k,v in backward_meta.items() if k.endswith('output_grad_amax_history'))
    check_finite((agent.actor,agent.critic,agent.target_critic,agent.temp))
    diagnostics=agent.last_online_resident_diagnostics
    report['resident_diagnostics']=host_metrics(diagnostics)
    for name,model in [('critic',agent.critic),('target',agent.target_critic),('actor',agent.actor)]:
        report[name+'_state']=[{'path':'/'.join(k),'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.size*v.dtype.itemsize} for k,v in traverse_util.flatten_dict(model.params).items()]
    assert agent.target_critic.apply_fn.critic_residual_compute_format=='legacy'
    assert not agent.critic.apply_fn.fp8_all_dense_kernels
    # Critic small-batch Fprop and dQ/da use native residuals too.
    qfun=jax.jit(lambda ac:agent.critic(obs[:1],ac,jnp.zeros(1,jnp.int32)).sum())
    dq=jax.jit(jax.grad(qfun))(jnp.zeros_like(jnp.asarray(act[:1])));check_finite(dq)
    report['dq_da_norm']=float(jnp.linalg.norm(dq))
    saved_rng=agent.rng
    ev=env.evaluate(agent,1,temperature=0.0,render=False);agent.rng=saved_rng
    report['eval']=host_metrics(ev)
    ck=a.output/'checkpoint';ck.mkdir();agent.save(str(ck))
    b=batch();info=agent.update(b,1,17);jax.block_until_ready(info)
    expected=tuple(jax.device_get(getattr(agent,n)) for n in ('actor','critic','target_critic','temp'))
    expected_rng=np.asarray(agent.rng)
    agent.load(str(ck))
    for name in ('actor','critic','target_critic','temp'): setattr(agent,name,jax.device_put(getattr(agent,name)))
    info=agent.update(b,1,17);jax.block_until_ready(info)
    actual=tuple(getattr(agent,n) for n in ('actor','critic','target_critic','temp'))
    max_error=0.
    for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        xx=np.asarray(x).astype(np.float64);yy=np.asarray(y).astype(np.float64)
        np.testing.assert_allclose(xx,yy,rtol=1e-6,atol=1e-7)
        max_error=max(max_error,float(np.max(np.abs(xx-yy),initial=0)))
    np.testing.assert_array_equal(expected_rng,agent.rng)
    report.update(status='PASSED',save_restore_max_error=max_error,memory=jax.devices()[0].memory_stats())
    save();print(json.dumps({'status':'PASSED','report':str(a.output/'report.json')}),flush=True)

if __name__=='__main__': main()
