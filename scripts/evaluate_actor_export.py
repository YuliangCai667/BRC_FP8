#!/usr/bin/env python3
"""Paired fresh environments, actual training policy vs independent packed loader."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from jaxrl.deployment_export import checkpoint_policy, training_statistics
from jaxrl.envs import ParallelEnv
from deployment.actor_fp8 import load_actor_export


def evaluate(checkpoint, package, episodes=10, seed=42):
    actor, embedding, source = checkpoint_policy(checkpoint)
    policy = load_actor_export(package)
    if Path(policy.manifest['source_checkpoint']).resolve() != Path(checkpoint).resolve():
        raise ValueError('paired evaluation checkpoint mismatch')
    ids = jnp.arange(len(source['task_names']))
    stats = jax.jit(training_statistics, static_argnames='norm')
    class TrainingAdapter:
        def sample_actions(self, observations, temperature=0.0):
            if temperature != 0:
                raise ValueError('this evaluation protocol is deterministic')
            return np.asarray(stats(actor, embedding, source['config']['resolved_task_embedding_norm'],
                                    observations, ids, jnp.zeros((len(ids), actor.apply_fn.action_dim)))['deterministic_actions'])
    class DeploymentAdapter:
        def sample_actions(self, observations, temperature=0.0):
            return np.asarray(policy.actions(observations, ids, deterministic=True))
    evaluations = {}
    for name, adapter in [('training', TrainingAdapter()), ('deployment', DeploymentAdapter())]:
        # ParallelEnv creation/reset consumes global NumPy RNG. Re-create with
        # exactly the same seed, rather than reusing the trainer's eval env.
        env = ParallelEnv(source['task_names'], seed=seed)
        started = time.perf_counter()
        result = env.evaluate(adapter, num_episodes=episodes, temperature=0.0, render=False)
        evaluations[name] = dict(return_by_task=np.asarray(result['return']).tolist(),
                                 success_by_task=np.asarray(result['goal']).tolist(),
                                 evaluation_seconds=time.perf_counter()-started)
        for e in env.envs:
            e.close()
    a, b = np.array(evaluations['training']['return_by_task']), np.array(evaluations['deployment']['return_by_task'])
    report = dict(status='deployment_eval_complete', source_checkpoint=str(Path(checkpoint).resolve()),
                  env_step=source['env_step'], task_names=source['task_names'], episodes_per_task=episodes,
                  seed=seed, fresh_paired_environments=True, deterministic=True,
                  training_complete=bool(source.get('is_final')), export_validated=True,
                  evaluations=evaluations, return_difference_by_task=(b-a).tolist(),
                  return_mean_training=float(a.mean()), return_mean_deployment=float(b.mean()))
    (Path(package) / 'deployment_eval.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    evaluate(a.checkpoint, a.package, a.episodes, a.seed)
