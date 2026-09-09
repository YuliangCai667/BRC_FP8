"""Checkpoint export and paired validation helpers, outside the standalone loader."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization, traverse_util
from jaxrl.low_precision.actor_export_codec import CODEC, pack_export_kernel
from jaxrl.low_precision.actor_qat_dense import logical_params, RECIPE
from deployment.actor_fp8 import policy_statistics

ROOT = Path(__file__).resolve().parents[1]


def preprocessing_manifest(env):
    tasks = []
    for name, e, od, ad in zip(env.env_names, env.envs, env.obs_dims, env.act_dims):
        # Current DMC wrapper is RescaleAction(FlattenObservation(Shimmy)).
        wrapped = e.env
        entries = None
        if hasattr(wrapped, 'env') and hasattr(wrapped.env.observation_space, 'spaces'):
            entries = [{'key': key, 'shape': list(space.shape)}
                       for key, space in wrapped.env.observation_space.spaces.items()]
        tasks.append(dict(name=name, observation_dim=int(od), action_dim=int(ad),
                          flatten_fields=entries,
                          physical_action_low=np.asarray(wrapped.action_space.low).tolist(),
                          physical_action_high=np.asarray(wrapped.action_space.high).tolist()))
    return dict(padded_observation_dim=int(env.observation_space.shape[-1]),
                input_concatenation='observation then normalized task embedding',
                observation_dtype='float32_at_actor_input',
                observation_normalizer=dict(type='identity', epsilon=None, clipping=None, running_statistics=None),
                padding=dict(side='right', value=0), tasks=tasks,
                actions=dict(output='tanh(mean + std * Gaussian)', deterministic='tanh(mean)',
                             clip=[-1, 1], mask='take first task.action_dim entries',
                             scaling='RescaleAction: low + (a + 1) * (high - low) / 2'),
                reward_normalizer='training only; no effect on policy observation preprocessing')


def export_checkpoint(checkpoint, output, *, expected_step=None, source_root=None):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    source_root = Path(source_root or ROOT)
    if not (checkpoint / 'COMPLETE').exists():
        raise ValueError('export requires a complete checkpoint')
    source = json.loads((checkpoint / 'manifest.json').read_text())
    config = source['config']
    if config.get('actor_training_recipe') != RECIPE or source.get('actor_phase') != 'export_align':
        raise ValueError('export requires the aligned actor recipe checkpoint')
    if expected_step is not None and source['env_step'] != expected_step:
        raise ValueError('source checkpoint is not the requested final step')
    actor = serialization.msgpack_restore((checkpoint / 'actor.msgpack').read_bytes())
    actor = jax.tree.map(jnp.asarray, actor)
    params = logical_params(SimpleNamespace(params=actor['params'], fp8_meta=actor['fp8_meta']))
    flat = traverse_util.flatten_dict(params, sep='/')
    raw_actor = traverse_util.flatten_dict(actor['params'], sep='/')
    critic = serialization.msgpack_restore((checkpoint / 'critic.msgpack').read_bytes())
    critic_flat = traverse_util.flatten_dict(critic['params'], sep='/')
    embedding = critic_flat.get('task_embedding/embeddings/embedding')
    del critic, critic_flat
    output.mkdir(parents=True, exist_ok=False)
    arrays, descriptors, kernels = {}, {}, {}
    def add(key, value):
        value = jnp.asarray(value)
        dtype = str(value.dtype)
        if dtype == 'float8_e4m3fn':
            disk = np.asarray(jax.lax.bitcast_convert_type(value, jnp.uint8))
        elif dtype == 'bfloat16':
            disk = np.asarray(jax.lax.bitcast_convert_type(value, jnp.uint16))
        else:
            disk = np.asarray(value)
        if not np.isfinite(np.asarray(value).astype(np.float32)).all():
            raise ValueError(f'nonfinite actor state: {key}')
        arrays[key] = disk
        descriptors[key] = dict(dtype=dtype, disk_dtype=disk.dtype.name, shape=list(disk.shape))
    kernel_elements = code_bytes = scale_bytes = padding_bytes = 0
    for path, value in flat.items():
        if path.endswith('/kernel'):
            p = path.removesuffix('/kernel')
            codes, scales = pack_export_kernel(value)
            add(p + '/codes', codes)
            add(p + '/scales', scales)
            kernels[p] = dict(shape=list(value.shape), padded_k=int(codes.shape[0] * 32),
                              block_axis=0, block_size=32, layout='[ceil(K/32),32,N]',
                              codes=p + '/codes', scales=p + '/scales')
            kernel_elements += value.size
            code_bytes += codes.size
            scale_bytes += scales.size * 4
            padding_bytes += codes.size - value.size
        else:
            add(path, raw_actor[path])
    if embedding is not None:
        add('task_embedding', embedding)
    architecture = dict(width=int(flat['BroNet_0/Dense_0/kernel'].shape[1]),
                        depth=len({p.split('/')[1] for p in flat if '/BronetBlock_' in p}),
                        action_dim=int(flat['Dense_0/kernel'].shape[1]),
                        log_std_min=-10.0, log_std_max=2.0,
                        activation='relu', layer_norm_epsilon=1e-6,
                        layer_norm_use_fast_variance=True)
    manifest = dict(schema_version=1, codec=CODEC, kernels=kernels, arrays=descriptors,
                    architecture=architecture, multitask=embedding is not None,
                    embedding_norm=config['resolved_task_embedding_norm'],
                    task_names=source['task_names'], task_ids={name:i for i,name in enumerate(source['task_names'])},
                    preprocessing=config['actor_preprocessing'],
                    source_checkpoint=str(checkpoint), source_env_step=source['env_step'],
                    source_actor_phase=source['actor_phase'],
                    base_commit=config.get('actor_base_commit'), source_commit=config.get('actor_source_commit'),
                    config=config, compute='W8A16, BF16 decoded weights/inputs, FP32 accumulation/output',
                    storage=dict(kernel_elements=int(kernel_elements), fp32_kernel_bytes=int(kernel_elements*4),
                                 code_bytes=int(code_bytes), scale_bytes=int(scale_bytes),
                                 padding_code_bytes=int(padding_bytes),
                                 auxiliary_bytes=int(sum(a.nbytes for a in arrays.values())-code_bytes-scale_bytes),
                                 parameter_bytes=int(sum(a.nbytes for a in arrays.values())),
                                 kernel_compression_ratio=float(kernel_elements*4/(code_bytes+scale_bytes))))
    np.savez(output / 'weights.npz', **arrays)
    manifest['storage']['weights_npz_bytes'] = (output / 'weights.npz').stat().st_size
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for name in ('__init__.py', 'actor_fp8.py'):
        shutil.copy2(source_root / 'deployment' / name, output / name)
    shutil.copy2(source_root / 'jaxrl/low_precision/actor_export_codec.py', output / 'actor_export_codec.py')
    (output / 'README.md').write_text(
        '# Actor FP8 deployment package\n\n'
        'All Dense kernels: one E4M3FN code plus FP32 block32 scale. '
        'Auxiliary parameters keep original precision. No CARRY, optimizer, or critic network.\n\n'
        'Reference inference uses W8A16; it does not imply native FP8 GEMM speedup. '
        'Requires Python, JAX, Flax and Distrax. Copy this directory anywhere, then:\n\n'
        '```python\nfrom PACKAGE_NAME import load_actor_export\n'
        "policy = load_actor_export('/path/to/PACKAGE_NAME')\n"
        'actions = policy.actions(flat_padded_obs, task_ids, deterministic=True)\n'
        'actions = policy.actions(flat_padded_obs, task_ids, deterministic=False, rng=key)\n```\n\n'
        'PACKAGE_NAME is this directory name. See manifest for task order, observation '
        'flattening, zero padding, identity observation normalizer and action scaling. '
        'Outputs are normalized actions; take the first action_dim entries for each task, '
        'then use the environment action rescaling recorded in the manifest.\n')
    return manifest


def checkpoint_policy(checkpoint):
    """Recreate the actual training actor apply path, without allocating a critic."""
    from jaxrl.networks import NormalTanhPolicy
    from jaxrl.utils import Model
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    raw = jax.tree.map(jnp.asarray, serialization.msgpack_restore((checkpoint / 'actor.msgpack').read_bytes()))
    flat = traverse_util.flatten_dict(raw['params'], sep='/')
    definition = NormalTanhPolicy(action_dim=flat['Dense_0/kernel'].shape[1],
        hidden_dims=flat['BroNet_0/Dense_0/kernel'].shape[1],
        depth=len({p.split('/')[1] for p in flat if '/BronetBlock_' in p}),
        actor_training_recipe=RECIPE, actor_export_aligned=manifest['actor_phase'] == 'export_align')
    actor = Model(step=raw['step'], apply_fn=definition, params=raw['params'], tx=None, fp8_meta=raw['fp8_meta'])
    critic_raw = serialization.msgpack_restore((checkpoint / 'critic.msgpack').read_bytes())
    emb = traverse_util.flatten_dict(critic_raw['params'], sep='/').get('task_embedding/embeddings/embedding')
    return actor, (None if emb is None else jnp.asarray(emb)), manifest


def training_statistics(actor, embedding, norm, observations, task_ids, noise):
    observations = jnp.asarray(observations, jnp.float32)
    if embedding is not None:
        emb = embedding[task_ids]
        emb = emb / jnp.linalg.norm(emb, ord=1 if norm == 'l1' else 2, axis=-1, keepdims=True)
        observations = jnp.concatenate((observations, emb), axis=-1)
    mean, raw, log_std, _ = actor(observations, return_stats=True)
    stats = policy_statistics(mean, log_std, noise)
    stats['raw_log_std'] = raw
    return stats


def write_validation_fixture(checkpoint, observations, output):
    actor, embedding, manifest = checkpoint_policy(checkpoint)
    observations = jnp.asarray(observations, jnp.float32)
    task_ids = jnp.tile(jnp.arange(len(manifest['task_names'])), observations.shape[0] // len(manifest['task_names']))
    if observations.shape[0] % len(manifest['task_names']):
        raise ValueError('validation observations must cover every task equally')
    noise = jax.random.normal(jax.random.PRNGKey(773), (len(task_ids), actor.apply_fn.action_dim))
    stats = jax.jit(training_statistics, static_argnames='norm')(
        actor, embedding, manifest['config']['resolved_task_embedding_norm'], observations, task_ids, noise)
    batch1 = [jax.jit(training_statistics, static_argnames='norm')(
        actor, embedding, manifest['config']['resolved_task_embedding_norm'],
        observations[i:i+1], task_ids[i:i+1], noise[i:i+1]) for i in range(len(manifest['task_names']))]
    np.savez(output, observations=np.asarray(observations), task_ids=np.asarray(task_ids), noise=np.asarray(noise),
             **{k: np.asarray(v) for k, v in stats.items()},
             **{'batch1_'+k:np.concatenate([np.asarray(row[k]) for row in batch1]) for k in stats})


def independent_validate(package, fixture, source_root=None):
    source_root = Path(source_root or ROOT)
    subprocess.run([sys.executable, str(source_root / 'scripts/validate_actor_export.py'),
                    '--package', str(package), '--fixture', str(fixture)], check=True, cwd='/tmp',
                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    return json.loads((Path(package) / 'export_validation.json').read_text())


def finish_actor_export(checkpoint, run_dir, env, config, recorder=None, wandb_run=None):
    """Completion is gated on separate checkpoint, validation and eval stages."""
    run_dir = Path(run_dir)
    status_path = run_dir / 'actor_export_status.json'
    status = dict(training_complete=False, export_validated=False, deployment_eval_complete=False)
    step = config['max_steps']
    source_root = Path(config.get('actor_export_source_root', ROOT))
    def mark(stage, **details):
        status[stage] = True
        status.update(details)
        status_path.write_text(json.dumps(status, indent=2))
        if recorder is not None:
            recorder.record_event(stage, step, int(source.get('update_step', 0)), path=str(checkpoint), **details)
            recorder.flush(int(source.get('update_step', 0)))
        if wandb_run is not None:
            wandb_run.summary['actor_export/' + stage] = True
    source = json.loads((Path(checkpoint) / 'manifest.json').read_text())
    if not source.get('is_final') or not source.get('includes_replay_buffer') or not (Path(checkpoint) / 'COMPLETE').exists():
        raise ValueError('automatic export requires final complete recovery checkpoint with replay')
    if source['env_step'] != step:
        raise ValueError('final checkpoint step mismatch')
    mark('training_complete')
    try:
        export_root = run_dir / 'export'
        export_root.mkdir(exist_ok=True)
        package = export_root / f'actor_fp8_{step//1000}k'
        export_checkpoint(checkpoint, package, expected_step=step, source_root=source_root)
        # Real observations: collect all four tasks at 16 environment states.
        obs = env.reset()
        fixture_obs = []
        for _ in range(16):
            fixture_obs.append(obs.copy())
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            obs, _, _ = env.reset_where_done(obs, term, trunc)
        fixture = run_dir / 'artifacts/actor_export_validation_fixture.npz'
        fixture.parent.mkdir(exist_ok=True)
        write_validation_fixture(checkpoint, np.concatenate(fixture_obs), fixture)
        validation = independent_validate(package, fixture, source_root=source_root)
        if not validation.get('batch1_validated'):
            raise ValueError('final deployment validation must include batch-one outputs for every task')
        mark('export_validated', package=str(package))
        subprocess.run([sys.executable, str(source_root / 'scripts/evaluate_actor_export.py'),
                        '--checkpoint', str(checkpoint), '--package', str(package),
                        '--episodes', str(config['eval_episodes']),
                        '--seed', str(config['seed'] + config['eval_seed_offset'])], check=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
        evaluation = json.loads((package / 'deployment_eval.json').read_text())
        curve_path = run_dir / 'eval_metrics.jsonl'
        if curve_path.exists():
            rows = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
            fields = ('env_step','return_by_task','return_mean','actor_phase')
            learning = dict(at_450k=[{k:r[k] for k in fields if k in r} for r in rows if r['env_step']==450000],
                            at_500k=[{k:r[k] for k in fields if k in r} for r in rows if r['env_step']==500000],
                            last_three_steps=[r['env_step'] for r in rows[-3:]],
                            last_three_return_by_task=np.mean([r['return_by_task'] for r in rows[-3:]],axis=0).tolist() if rows else [],
                            paired_deployment_evaluation=evaluation)
            (package / 'learning_summary.json').write_text(json.dumps(learning,indent=2))
        inventory = {str(p.relative_to(package)):p.stat().st_size for p in package.rglob('*') if p.is_file()}
        inventory_path = package / 'package_inventory.json'
        inventory_report = dict(files=inventory, package_total_bytes=0)
        # Include this inventory's own bytes, without a content hash or relying
        # on an NPZ size estimate. The integer's digit count converges quickly.
        for _ in range(4):
            payload = json.dumps(inventory_report,indent=2)
            inventory_report['package_total_bytes'] = sum(inventory.values())+len(payload.encode())
        inventory_path.write_text(json.dumps(inventory_report,indent=2))
        mark('deployment_eval_complete')
        if wandb_run is not None:
            for key in ('return_mean_training', 'return_mean_deployment'):
                wandb_run.summary['actor_export/' + key] = evaluation[key]
            for task, value in zip(evaluation['task_names'], evaluation['evaluations']['deployment']['return_by_task']):
                wandb_run.summary['actor_export/return/' + task] = value
        return status
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error))
        status_path.write_text(json.dumps(status, indent=2))
        if recorder is not None:
            recorder.record_event('actor_export_failed', step, error=str(error), completion=status)
        raise
