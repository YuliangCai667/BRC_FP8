"""Package-only JAX actor. Packed kernels remain the runtime's persistent state."""
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import distrax
try:
    from .actor_export_codec import CODEC, unpack_export_kernel, decoded_dense
except ImportError:
    from jaxrl.low_precision.actor_export_codec import CODEC, unpack_export_kernel, decoded_dense


def policy_statistics(mean, log_std, noise):
    std = jnp.exp(log_std)
    pre_tanh = mean + std * noise
    action = jnp.tanh(pre_tanh)
    base = distrax.MultivariateNormalDiag(mean, std)
    log_prob = base.log_prob(pre_tanh) - distrax.Block(distrax.Tanh(), 1).forward_log_det_jacobian(pre_tanh)
    return dict(mean=mean, log_std=log_std, std=std, pre_tanh=pre_tanh,
                deterministic_actions=jnp.tanh(mean), actions=jnp.clip(action, -1, 1), log_prob=log_prob)


class ExportedActor:
    def __init__(self, manifest, arrays):
        self.manifest = manifest
        self.arrays = arrays
        self._statistics = jax.jit(self._forward_statistics)

    def _dense(self, x, path, arrays):
        item = self.manifest['kernels'][path]
        w = unpack_export_kernel(arrays[item['codes']], arrays[item['scales']], tuple(item['shape']))
        w = lax.optimization_barrier(w.astype(jnp.bfloat16))
        return decoded_dense(x, w, arrays[path + '/bias'])

    @staticmethod
    def _norm(x, path, arrays):
        return nn.LayerNorm().apply({'params': {'scale': arrays[path + '/scale'], 'bias': arrays[path + '/bias']}}, x)

    def _forward_statistics(self, arrays, obs, task_ids, noise):
        # ParallelEnv supplies flattened, zero-padded observations. There is no
        # observation running normalizer in this training baseline.
        obs = obs.astype(jnp.float32)
        if self.manifest['multitask']:
            emb = arrays['task_embedding'][task_ids]
            ord_value = 1 if self.manifest['embedding_norm'] == 'l1' else 2
            emb = emb / jnp.linalg.norm(emb, ord=ord_value, axis=-1, keepdims=True)
            obs = jnp.concatenate((obs, emb), axis=-1)
        x = self._dense(obs, 'BroNet_0/Dense_0', arrays)
        x = nn.relu(self._norm(x, 'BroNet_0/LayerNorm_0', arrays))
        for i in range(self.manifest['architecture']['depth']):
            p = f'BroNet_0/BronetBlock_{i}'
            res = self._dense(x, p + '/Dense_0', arrays)
            res = nn.relu(self._norm(res, p + '/LayerNorm_0', arrays))
            res = self._dense(res, p + '/Dense_1', arrays)
            x = self._norm(res, p + '/LayerNorm_1', arrays) + x
        mean = self._dense(x, 'Dense_0', arrays)
        raw = self._dense(x, 'Dense_1', arrays)
        cfg = self.manifest['architecture']
        log_std = cfg['log_std_min'] + (cfg['log_std_max'] - cfg['log_std_min']) * 0.5 * (1 + nn.tanh(raw))
        stats = policy_statistics(mean, log_std, noise)
        stats['raw_log_std'] = raw
        return stats

    def statistics(self, obs, task_ids, noise):
        obs = jnp.asarray(obs, jnp.float32)
        task_ids = jnp.asarray(task_ids, jnp.int32)
        noise = jnp.asarray(noise, jnp.float32)
        if obs.shape[-1] != self.manifest['preprocessing']['padded_observation_dim']:
            raise ValueError('expected flattened, zero-padded observations of manifest dimension')
        if task_ids.shape != obs.shape[:-1] or noise.shape != obs.shape[:-1] + (self.manifest['architecture']['action_dim'],):
            raise ValueError('task IDs / noise batch shape mismatch')
        if not isinstance(task_ids, jax.core.Tracer):
            ids = np.asarray(task_ids)
            if np.any(ids < 0) or np.any(ids >= len(self.manifest['task_names'])):
                raise ValueError('task ID outside exported task table')
        return self._statistics(self.arrays, obs, task_ids, noise)

    def actions(self, obs, task_ids, deterministic=True, rng=None, noise=None):
        shape = np.shape(obs)[:-1] + (self.manifest['architecture']['action_dim'],)
        if noise is None:
            if deterministic:
                noise = jnp.zeros(shape, jnp.float32)
            elif rng is not None:
                noise = jax.random.normal(rng, shape)
            else:
                raise ValueError('stochastic actions require rng or explicit Gaussian noise')
        stats = self.statistics(obs, task_ids, noise)
        return stats['deterministic_actions' if deterministic else 'actions']

    @property
    def persistent_bytes(self):
        return sum(int(a.size * a.dtype.itemsize) for a in self.arrays.values())


def load_actor_export(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema_version'] != 1 or manifest['codec'] != CODEC:
        raise ValueError('unsupported actor export schema or codec')
    if manifest['preprocessing']['observation_normalizer']['type'] != 'identity':
        raise ValueError('this runtime only supports the exported identity observation preprocessing')
    arrays = {}
    with np.load(path / 'weights.npz', allow_pickle=False) as data:
        if set(data.files) != set(manifest['arrays']):
            raise ValueError('archive arrays do not match manifest')
        for key, desc in manifest['arrays'].items():
            a = data[key]
            if a.dtype.name != desc['disk_dtype'] or list(a.shape) != desc['shape']:
                raise ValueError(f'archive dtype/shape mismatch: {key}')
            value = jnp.asarray(a)
            if desc['dtype'] == 'float8_e4m3fn':
                value = lax.bitcast_convert_type(value, jnp.float8_e4m3fn)
            elif desc['dtype'] == 'bfloat16':
                value = lax.bitcast_convert_type(value, jnp.bfloat16)
            if not np.isfinite(np.asarray(value).astype(np.float32)).all():
                raise ValueError(f'nonfinite exported array: {key}')
            arrays[key] = value
    for item in manifest['kernels'].values():
        scales = arrays[item['scales']]
        if not np.all(np.asarray(scales) > 0):
            raise ValueError('illegal export scale')
        k, n = item['shape']
        if arrays[item['codes']].shape != ((k+31)//32, 32, n) or scales.shape != ((k+31)//32, n):
            raise ValueError('invalid block32 layout')
    return ExportedActor(manifest, arrays)
