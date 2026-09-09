#!/usr/bin/env python3
"""16 real updates per phase, full critic, recovery and package-only validation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import get_actor_gradients
from jaxrl.envs import ParallelEnv
from jaxrl.env_names import get_environment_list
from jaxrl.utils import Batch
from jaxrl.checkpoint import CheckpointManager
from jaxrl.low_precision.actor_qat_dense import RECIPE, is_body, logical_params
from jaxrl.low_precision.backends.native import register
from jaxrl.deployment_export import preprocessing_manifest, export_checkpoint, write_validation_fixture, independent_validate


def finite(tree):
    for x in jax.tree.leaves(tree):
        if hasattr(x,'dtype') and jnp.issubdtype(x.dtype,jnp.inexact):
            assert np.isfinite(np.asarray(x).astype(np.float32)).all(), (x.shape,str(x.dtype))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--format',choices=('hybrid','mxfp8'),default='hybrid');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='STARTED',backend=a.format,phases={})
    def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2))
    save()
    if a.format == 'mxfp8': register()
    env=ParallelEnv(get_environment_list('DMC_DOGS'),seed=42)
    env.action_space.seed(42)
    obs=env.reset();act=np.zeros(env.action_space.shape,np.float32)
    agent=BRC(42,obs[:1],act[:1],num_tasks=4,updates_per_step=2,width_critic=4096,
        task_embedding_norm='l1',critic_precision='fp8_resident',fp8_resident_carry=True,
        target_critic_precision='fp8_lag',critic_optimizer_state='fp8_carry',
        critic_residual_compute_format=a.format,actor_training_recipe=RECIPE,
        actor_body_compute=a.format+'_main_plus_carry')
    jax.block_until_ready(agent.actor.params)
    report['initial_carry_nonzero_fraction']={ '/'.join(p):float(np.mean(np.asarray(v).astype(np.float32)!=0))
        for p,v in traverse_util.flatten_dict(agent.actor.fp8_meta).items() if p[-1]=='kernel_carry'}
    assert all(v>.9 for v in report['initial_carry_nonzero_fraction'].values())
    print('initialized actual actor 256/depth1 + critic4096',flush=True)
    records=[]
    for _ in range(256):
        action=env.action_space.sample();nxt,reward,term,trunc,_=env.step(action)
        records.append((obs.copy(),action,reward,env.generate_masks(term,trunc),nxt.copy()))
        obs,_,_=env.reset_where_done(nxt,term,trunc)
    arrays=[np.stack([r[i] for r in records]) for i in range(5)]
    rng=np.random.RandomState(20260909)
    def batch():
        ix=rng.randint(256,size=1024);tasks=rng.randint(4,size=1024)
        vals=[jnp.asarray(v[ix,tasks],jnp.float32) for v in arrays];vals[2]=vals[2]*.01
        return Batch(*[v[None] for v in vals],jnp.asarray(tasks,jnp.int32)[None])
    for phase,start in [('carry_main',449984),('export_align',450000)]:
        before=jax.tree.map(np.asarray,(agent.actor.params,agent.actor.fp8_meta,agent.actor.opt_state))
        agent.set_env_step(start)
        for x,y in zip(jax.tree.leaves(before),jax.tree.leaves((agent.actor.params,agent.actor.fp8_meta,agent.actor.opt_state))):
            np.testing.assert_array_equal(x,y)
        assert agent.actor_phase==phase
        times=[]
        for i in range(16):
            b=batch()
            if i==15:
                ck=a.output/(phase+'_checkpoint');ck.mkdir();agent.save(str(ck))
            t=time.perf_counter();info=agent.update(b,1,start+i);jax.block_until_ready(info);finite(info)
            times.append(time.perf_counter()-t)
            print(json.dumps(dict(phase=phase,update=i+1,seconds=times[-1],actor_loss=float(info['actor_loss']),critic_loss=float(info['critic_loss']))),flush=True)
        expected=jax.tree.map(np.asarray,(agent.actor,agent.critic,agent.target_critic,agent.temp))
        expected_rng=np.asarray(agent.rng)
        agent.load(str(ck)); info=agent.update(b,1,start+15);jax.block_until_ready(info)
        actual=(agent.actor,agent.critic,agent.target_critic,agent.temp)
        maxerr=0.
        for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
            xx=np.asarray(x).astype(np.float64); yy=np.asarray(y).astype(np.float64)
            # FP8 payloads must restore exactly. FP32 scatter/reduction paths
            # may differ by one ULP after re-specializing deserialized inputs;
            # use the existing V1 recovery tolerance for physical values.
            if str(x.dtype).startswith('float8_') or str(x.dtype) == 'bfloat16':
                np.testing.assert_array_equal(xx,yy)
            else:
                np.testing.assert_allclose(xx,yy,rtol=1e-6,atol=1e-7)
            maxerr=max(maxerr,float(np.max(np.abs(xx-yy),initial=0)))
        np.testing.assert_array_equal(expected_rng,agent.rng)
        finite(actual)
        small=jax.tree.map(lambda x:x[0,:32],b)
        grad=jax.jit(get_actor_gradients,static_argnames=('num_bins','v_max','multitask'))(
            jax.random.PRNGKey(77),agent.actor,agent.critic,agent.temp,small,agent.num_bins,agent.v_max,True)
        norms={'/'.join(p):float(jnp.linalg.norm(v)) for p,v in traverse_util.flatten_dict(grad).items() if p[-1]=='kernel'}
        assert all(np.isfinite(v) and v>0 for v in norms.values()),norms
        qgrad=jax.jit(get_actor_gradients,static_argnames=('num_bins','v_max','multitask','q_only'))(
            jax.random.PRNGKey(77),agent.actor,agent.critic,agent.temp,small,agent.num_bins,agent.v_max,True,q_only=True)
        qnorms={'/'.join(p):float(jnp.linalg.norm(v)) for p,v in traverse_util.flatten_dict(qgrad).items() if p[-1]=='kernel'}
        assert all(np.isfinite(v) for v in qnorms.values()) and max(qnorms.values())>0,qnorms
        def qfun(critic, observations, actions, task_ids):
            return (jax.nn.softmax(critic(observations,actions,task_ids),axis=-1)*jnp.linspace(-10,10,101)).sum()
        dq=jax.jit(jax.grad(qfun,argnums=2))(agent.critic,small.observations,small.actions,small.task_ids);finite(dq)
        assert float(jnp.linalg.norm(dq))>0
        actions=agent.sample_actions(obs,temperature=0.0);finite(actions)
        bootstrap=agent.estimate_bootstrap_values(obs,np.ones(4,bool));finite(bootstrap)
        for p,v in traverse_util.flatten_dict(agent.actor.params).items():
            if p[-1]=='kernel': assert str(v.dtype)==('float8_e4m3fn' if is_body(p) else 'bfloat16')
        assert all(str(v.dtype)=='float32' for v in jax.tree.leaves((agent.actor.opt_state[0].mu,agent.actor.opt_state[0].nu)))
        if a.format == 'hybrid':
            histories={ '/'.join(p):float(np.max(np.asarray(v))) for p,v in traverse_util.flatten_dict(agent.actor.fp8_meta).items()
                        if p[-1]=='output_grad_amax_history'}
            assert len(histories)==2 and all(v>0 for v in histories.values()),histories
            report.setdefault('actor_gradient_history_max',{})[phase]=histories
        report['phases'][phase]=dict(updates=16,update_seconds=times,restore_max_error=maxerr,restore_rtol=1e-6,restore_atol=1e-7,
            actor_kernel_gradient_norms=norms,actor_q_only_kernel_gradient_norms=qnorms,
            dq_da_norm=float(jnp.linalg.norm(dq)),actor_phase=agent.actor_phase,step=int(agent.step))
        save()
    assert agent.target_critic.apply_fn.critic_residual_compute_format=='legacy'
    config=dict(actor_training_recipe=RECIPE,actor_body_compute=a.format+'_main_plus_carry',
                critic_residual_compute_format=a.format,resolved_task_embedding_norm='l1',actor_preprocessing=preprocessing_manifest(env))
    manager=CheckpointManager(a.output/'checkpoints','actor_smoke',a.output,get_environment_list('DMC_DOGS'),config)
    final=manager.save_analysis(agent,450015)
    package=a.output/'actor_smoke_package'
    report['export_storage']=export_checkpoint(final,package,expected_step=450015)['storage']
    fixture=a.output/'fixture.npz'
    write_validation_fixture(final,arrays[0][:16].reshape(-1,obs.shape[-1]),fixture)
    report['validation']=independent_validate(package,fixture)
    report.update(status='PASSED',memory=jax.devices()[0].memory_stats())
    save();print(json.dumps(dict(status='PASSED',report=str(a.output/'report.json'))),flush=True)

if __name__=='__main__': main()
