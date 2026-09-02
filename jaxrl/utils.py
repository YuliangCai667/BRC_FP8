import os
import collections
from typing import Any, Optional, Sequence

import flax
import jax
import jax.numpy as jnp
import optax
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'task_ids'])

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

@flax.struct.dataclass
class SaveState:
    step: int
    params: Params
    opt_state: Optional[optax.OptState] = None
    fp8_meta: Optional[Params] = None


@flax.struct.dataclass
class LegacySaveState:
    step: int
    params: Params
    opt_state: Optional[optax.OptState] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: flax.linen.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    reference_apply_fn: Optional[flax.linen.Module] = flax.struct.field(
        pytree_node=False, default=None
    )
    opt_state: Optional[optax.OptState] = None
    fp8_meta: Optional[Params] = None

    @classmethod
    def create(cls,
               model_def: flax.linen.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None,
               reference_apply_fn: Optional[flax.linen.Module] = None):
        variables = model_def.init(*inputs)

        params = variables['params']
        fp8_meta = variables.get(OVERWRITE_WITH_GRADIENT)

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   reference_apply_fn=reference_apply_fn,
                   params=params,
                   tx=tx,
                   opt_state=opt_state,
                   fp8_meta=fp8_meta)

    def variables(self, params=None, fp8_meta=None):
        variables = {'params': self.params if params is None else params}
        meta = self.fp8_meta if fp8_meta is None else fp8_meta
        if meta is not None:
            variables[OVERWRITE_WITH_GRADIENT] = meta
        return variables

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply(self.variables(), *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.variables())
        return self.apply_variable_gradients(grads, info)

    def apply_variable_gradients(self, grads, info):
        param_grads = grads['params']
        grad_norm = tree_norm(param_grads)
        info['grad_norm'] = grad_norm

        updates, new_opt_state = self.tx.update(param_grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_fp8_meta = grads.get(OVERWRITE_WITH_GRADIENT, self.fp8_meta)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state,
                            fp8_meta=new_fp8_meta), info
    
    def get_gradient(self, loss_fn):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.variables())
        return grads['params']

    def save(self, save_path: str, include_optimizer: bool = True):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(SaveState(
                step=self.step,
                params=self.params,
                opt_state=self.opt_state if include_optimizer else None,
                fp8_meta=self.fp8_meta,
            )))

    def load(self, load_path: str, require_fp8_metadata: bool = False):
        with open(load_path, 'rb') as f:
            contents = f.read()
        raw_state = flax.serialization.msgpack_restore(contents)
        source_has_fp8 = raw_state.get('fp8_meta') is not None
        if require_fp8_metadata and not source_has_fp8:
            raise ValueError('checkpoint is missing required FP8 model metadata')
        if source_has_fp8 and self.fp8_meta is None:
            raise ValueError('cannot load an FP8 model state into an FP32 model')
        target_opt_state = self.opt_state if raw_state.get('opt_state') is not None else None
        if 'fp8_meta' in raw_state:
            target_fp8_meta = self.fp8_meta if source_has_fp8 else None
            saved_state = flax.serialization.from_state_dict(
                SaveState(
                    step=self.step,
                    params=self.params,
                    opt_state=target_opt_state,
                    fp8_meta=target_fp8_meta,
                ),
                raw_state,
            )
        else:
            saved_state = flax.serialization.from_state_dict(
                LegacySaveState(
                    step=self.step,
                    params=self.params,
                    opt_state=target_opt_state,
                ),
                raw_state,
            )
        loaded_fp8_meta = getattr(saved_state, 'fp8_meta', None)
        if self.fp8_meta is not None and loaded_fp8_meta is None:
            loaded_fp8_meta = self.fp8_meta
        return self.replace(
            step=int(saved_state.step),
            params=saved_state.params,
            opt_state=self.opt_state if saved_state.opt_state is None else saved_state.opt_state,
            fp8_meta=loaded_fp8_meta,
        )
