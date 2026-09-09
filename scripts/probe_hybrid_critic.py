#!/usr/bin/env python3
"""Fixed-state Q/action-gradient comparison; never updates learner state."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_default_matmul_precision', 'highest')
from flax import serialization
from jaxrl.networks import Critic
from jaxrl.utils import Model
from jaxrl.agent.update import dequantize_critic_params, reconstruct_carry_logical_critic_params


def metrics(value, reference):
    a=np.asarray(value,dtype=np.float64).ravel()
    b=np.asarray(reference,dtype=np.float64).ravel()
    an,bn=np.linalg.norm(a),np.linalg.norm(b)
    return dict(relative_l2=float(np.linalg.norm(a-b)/max(bn,1e-30)),
                cosine=float(np.dot(a,b)/max(an*bn,1e-30)),
                norm_ratio=float(an/max(bn,1e-30)),norm=float(an),
                zero_fraction=float(np.mean(a==0)),finite=bool(np.isfinite(a).all()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--transitions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data=np.load(a.transitions,allow_pickle=False)
    obs=data['0'][::16].reshape(-1,data['0'].shape[-1])
    actions=data['1'][::16].reshape(-1,data['1'].shape[-1])
    tasks=np.tile(np.arange(4,dtype=np.int32),len(obs)//4)
    raw=serialization.msgpack_restore((a.checkpoint/'critic.msgpack').read_bytes())
    definition=Critic(num_tasks=4,embedding_size=32,ensemble_size=2,hidden_dims=4096,
                      depth=2,output_nodes=101,multitask=True,task_embedding_norm='l1',
                      critic_precision='fp8_resident',fp8_resident_carry=True)
    critic=Model(step=raw['step'],apply_fn=definition,params=jax.tree.map(jnp.asarray,raw['params']),
                 tx=None,fp8_meta=jax.tree.map(jnp.asarray,raw['fp8_meta']))
    report={'checkpoint':str(a.checkpoint),'transitions':str(a.transitions),
            'batch_size':len(obs),'updates':0,'paths':{},
            'gradient_objective':'sum(Q_subset)/1024; preserve training batch loss scale',
            'gradient_history':'unchanged checkpoint delayed-amax metadata'}

    def q(model,obs,actions,tasks):
        logits=model(obs,actions,tasks)
        return (jax.nn.softmax(logits,axis=-1)*jnp.linspace(-10,10,101)).sum(-1).mean(0)
    evaluate=jax.jit(lambda model,obs,actions,tasks:(q(model,obs,actions,tasks),
                 jax.grad(lambda ac:q(model,obs,ac,tasks).sum()/1024)(actions)))
    for fmt,params in [('legacy',dequantize_critic_params(critic)),
                       ('hybrid',reconstruct_carry_logical_critic_params(critic))]:
        model=critic.replace(apply_fn=critic.apply_fn.clone(critic_residual_compute_format=fmt))
        ref=critic.replace(params=params,fp8_meta=None,
                           apply_fn=critic.apply_fn.clone(critic_precision='fp32',
                            critic_residual_compute_format='legacy',fp8_resident_carry=False))
        actual=evaluate(model,obs,actions,tasks)
        reference=evaluate(ref,obs,actions,tasks)
        jax.block_until_ready((actual,reference))
        report['paths'][fmt]={'q':metrics(actual[0],reference[0]),
                             'dq_da':metrics(actual[1],reference[1]),
                             'by_task':{str(t):{'q':metrics(actual[0][tasks==t],reference[0][tasks==t]),
                                              'dq_da':metrics(actual[1][tasks==t],reference[1][tasks==t])}
                                        for t in range(4)}}
        assert all(report['paths'][fmt][k]['finite'] for k in ('q','dq_da'))
        assert report['paths'][fmt]['dq_da']['norm']>0
        print(fmt,json.dumps(report['paths'][fmt]),flush=True)
    report['status']='MEASURED'
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
