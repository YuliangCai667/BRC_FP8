"""Actor-only storage and logical Adam path; no critic ensemble axis."""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import traverse_util
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
import optax

from .actor_export_codec import qat_dense

RECIPE = 'carry_body_bf16_edge_qat'


def enabled(actor):
    return getattr(actor.apply_fn, 'actor_training_recipe', 'fp32') == RECIPE


def is_body(path):
    return path[-1] == 'kernel' and any(p.startswith('BronetBlock_') for p in path)


class ActorQatDense(nn.Module):
    features: int
    body: bool = False
    export_aligned: bool = False
    body_compute: str = 'mxfp8_main_plus_carry'
    fp8_amax_history_length: int = 1024
    kernel_init: object = nn.initializers.orthogonal(jnp.sqrt(2))

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features), jnp.float32)
        bias = self.param('bias', nn.initializers.zeros_init(), (self.features,), jnp.float32)
        if self.body:
            if self.body_compute == 'hybrid_main_plus_carry':
                grad_scale = self.variable(OVERWRITE_WITH_GRADIENT, 'output_grad_scale',
                                           lambda: jnp.ones(1, jnp.float32)).value
                grad_history = self.variable(OVERWRITE_WITH_GRADIENT, 'output_grad_amax_history',
                    lambda: jnp.zeros(self.fp8_amax_history_length, jnp.float32)).value
            scale = self.variable(OVERWRITE_WITH_GRADIENT, 'kernel_scale', lambda: jnp.float32(1)).value
            carry = self.variable(OVERWRITE_WITH_GRADIENT, 'kernel_carry',
                                  lambda: jnp.zeros(kernel.shape, jnp.float8_e4m3fn)).value
            if kernel.dtype == jnp.float8_e4m3fn:
                from jaxrl.networks import _reconstruct_carry_logical_kernel
                main = kernel
                logical = _reconstruct_carry_logical_kernel(main, scale, carry)
            else:
                logical = kernel.astype(jnp.float32)
                main = self.get_variable('actor_payload', 'kernel')
            if not self.export_aligned and not self.is_initializing():
                if main is None:
                    raise ValueError('logical actor body requires its stored main payload')
                args = (x.reshape(-1, x.shape[-1]), logical,
                        jax.lax.stop_gradient(main), jax.lax.stop_gradient(carry),
                        jax.lax.stop_gradient(scale), bias)
                if self.body_compute == 'hybrid_main_plus_carry':
                    from .hybrid_two_term_dense import hybrid_two_term_dense
                    out = hybrid_two_term_dense(*args, grad_scale, grad_history)
                elif self.body_compute == 'mxfp8_main_plus_carry':
                    from .two_term_dense import two_term_dense
                    out = two_term_dense(*args, 'mxfp8')
                else:
                    raise ValueError('Unsupported actor body compute')
                return out.reshape(x.shape[:-1] + (self.features,))
        else:
            logical = kernel.astype(jnp.float32)
        return qat_dense(x, logical, bias)


def logical_params(actor):
    from jaxrl.networks import _reconstruct_carry_logical_kernel
    flat = traverse_util.flatten_dict(actor.params)
    meta = traverse_util.flatten_dict(actor.fp8_meta or {})
    return traverse_util.unflatten_dict({
        p: (_reconstruct_carry_logical_kernel(v, meta[p[:-1] + ('kernel_scale',)],
                                            meta[p[:-1] + ('kernel_carry',)])
            if is_body(p) else v.astype(jnp.float32))
        for p, v in flat.items()
    })


def logical_variables(actor, params, fp8_meta=None):
    variables = actor.variables(params=params, fp8_meta=fp8_meta)
    variables['actor_payload'] = jax.tree.map(jax.lax.stop_gradient, actor.params)
    return variables


def write_candidate(actor, candidate):
    from jaxrl.networks import _quantize_carry_resident_kernel
    flat = traverse_util.flatten_dict(candidate)
    meta = traverse_util.flatten_dict(actor.fp8_meta or {})
    for path, weight in flat.items():
        if is_body(path):
            state = _quantize_carry_resident_kernel(weight)
            flat[path] = state['main_codes']
            meta[path[:-1] + ('kernel_scale',)] = state['kernel_scale']
            meta[path[:-1] + ('kernel_carry',)] = state['carry_codes']
        elif path[-1] == 'kernel':
            flat[path] = jax.lax.optimization_barrier(weight.astype(jnp.bfloat16))
    return actor.replace(params=traverse_util.unflatten_dict(flat),
                         fp8_meta=traverse_util.unflatten_dict(meta))


def initialize_actor(actor):
    if not enabled(actor):
        return actor
    actor = write_candidate(actor, actor.params)
    return actor.replace(opt_state=actor.tx.init(logical_params(actor)))


def apply_gradients(actor, params, grads, info, backward_metadata=None):
    from jaxrl.utils import tree_norm
    if backward_metadata is not None and not actor.apply_fn.actor_export_aligned:
        from jaxrl.agent.update import _merge_resident_backward_metadata
        actor = actor.replace(fp8_meta=_merge_resident_backward_metadata(actor.fp8_meta, backward_metadata))
    updates, state = actor.tx.update(grads, actor.opt_state, params)
    candidate = optax.apply_updates(params, updates)
    info['grad_norm'] = tree_norm(grads)
    return write_candidate(actor.replace(step=actor.step + 1, opt_state=state), candidate), info
