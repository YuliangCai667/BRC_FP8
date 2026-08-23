import functools
from typing import Callable

from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT, update_fp8_meta
import distrax


E4M3_MAX = jnp.float32(448.0)


def _e4m3_amax_and_scale(value: jnp.ndarray):
    value = jnp.asarray(value, dtype=jnp.float32)
    amax = jnp.max(jnp.abs(value))
    scale = jnp.where(amax > 0, amax / E4M3_MAX, jnp.float32(1.0))
    return amax, scale


def quantize_e4m3_per_tensor(value: jnp.ndarray):
    value = jnp.asarray(value, dtype=jnp.float32)
    _, scale = _e4m3_amax_and_scale(value)
    return (value / scale).astype(jnp.float8_e4m3fn), scale


def dequantize_e4m3(codes: jnp.ndarray, scale: jnp.ndarray):
    scale = jnp.asarray(scale, dtype=jnp.float32)
    scale = jnp.reshape(scale, scale.shape + (1,) * (codes.ndim - scale.ndim))
    return codes.astype(jnp.float32) * scale


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class Fp8DirectDotGeneralOp(nn.Fp8DirectDotGeneralOp):
    """Flax FP8 direct dot with optional forward-only metadata updates."""

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        if (
            self.is_mutable_collection(OVERWRITE_WITH_GRADIENT)
            and not self.is_initializing()
        ):
            inputs, kernel = args[:2]
            inputs = jnp.asarray(inputs, dtype=kernel.dtype)
            input_scale, input_history = update_fp8_meta(
                inputs,
                self.e4m3_dtype,
                self.input_scale.value,
                self.input_amax_history.value,
            )
            kernel_scale, kernel_history = update_fp8_meta(
                kernel,
                self.e4m3_dtype,
                self.kernel_scale.value,
                self.kernel_amax_history.value,
            )
            self.input_scale.value = input_scale
            self.input_amax_history.value = input_history
            self.kernel_scale.value = kernel_scale
            self.kernel_amax_history.value = kernel_history
        return outputs


class ResidentFp8Dense(nn.Module):
    features: int
    kernel_init: Callable = default_init()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        kernel_shape = (inputs.shape[-1], self.features)
        if self.has_variable('params', 'kernel'):
            initial_kernel = jnp.zeros(kernel_shape, dtype=jnp.float8_e4m3fn)
            initial_scale = jnp.float32(1.0)
        else:
            fp32_kernel = self.kernel_init(
                self.make_rng('params'), kernel_shape, jnp.float32
            )
            initial_kernel, initial_scale = quantize_e4m3_per_tensor(fp32_kernel)

        kernel = self.param(
            'kernel',
            lambda _key, _shape, _dtype: initial_kernel,
            kernel_shape,
            jnp.float8_e4m3fn,
        )
        kernel_scale = self.variable(
            OVERWRITE_WITH_GRADIENT,
            'kernel_scale',
            lambda: initial_scale,
        ).value
        bias = self.param(
            'bias', self.bias_init, (self.features,), jnp.float32
        )

        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        activation_amax, activation_scale = _e4m3_amax_and_scale(inputs)
        activation_codes = (inputs / activation_scale).astype(jnp.float8_e4m3fn)
        outputs = lax.dot_general(
            activation_codes,
            kernel,
            (((activation_codes.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        outputs *= activation_scale * jnp.asarray(kernel_scale, dtype=jnp.float32)
        outputs += jnp.reshape(bias, (1,) * (outputs.ndim - 1) + (-1,))

        self.sow('intermediates', 'activation', inputs)
        self.sow('intermediates', 'activation_amax', activation_amax)
        self.sow('intermediates', 'activation_scale', activation_scale)
        return outputs


class BronetBlock(nn.Module):
    hidden_dims: int
    activations: Callable[[jnp.ndarray], jnp.ndarray]
    fp8_direct: bool = False
    fp8_resident: bool = False
    fp8_amax_history_length: int = 1024

    def _dense(self, name: str):
        if self.fp8_resident:
            return ResidentFp8Dense(
                self.hidden_dims,
                kernel_init=default_init(),
                name=name,
            )
        if not self.fp8_direct:
            return nn.Dense(self.hidden_dims, kernel_init=default_init())
        return nn.Dense(
            self.hidden_dims,
            kernel_init=default_init(),
            dot_general_cls=functools.partial(
                Fp8DirectDotGeneralOp,
                amax_history_length=self.fp8_amax_history_length,
            ),
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        res = self._dense('Dense_0')(x)
        self.sow('intermediates', 'dense_0_output', res)
        res = nn.LayerNorm()(res)
        res = self.activations(res)
        res = self._dense('Dense_1')(res)
        self.sow('intermediates', 'dense_1_output', res)
        res = nn.LayerNorm()(res)
        return res + x

class BroNet(nn.Module):
    hidden_dims: int
    depth: int
    add_final_layer: bool = False
    output_nodes: int = 101
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    fp8_residual_blocks: bool = False
    resident_fp8_residual_blocks: bool = False
    fp8_amax_history_length: int = 1024

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.hidden_dims, kernel_init=default_init())(x)
        self.sow('intermediates', 'input_dense_output', x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        for i in range(self.depth):
            x = BronetBlock(
                self.hidden_dims,
                self.activations,
                fp8_direct=self.fp8_residual_blocks,
                fp8_resident=self.resident_fp8_residual_blocks,
                fp8_amax_history_length=self.fp8_amax_history_length,
            )(x)
        if self.add_final_layer:
            x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            self.sow('intermediates', 'final_dense_output', x)
        return x

class TaskEmbedding(nn.Module): 
    num_tasks: int
    embedding_size: int
    norm: str = 'l2'
    
    def setup(self):
        self.embeddings = nn.Embed(self.num_tasks, self.embedding_size)
        
    def __call__(self, x: jnp.ndarray):
        emb = self.embeddings(x)
        ord_value = 1 if self.norm == 'l1' else 2
        norm = jnp.linalg.norm(emb, ord=ord_value, axis=-1, keepdims=True)
        emb = emb/norm
        return emb

class QValue(nn.Module):
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    critic_precision: str = 'fp32'
    fp8_amax_history_length: int = 1024
    
    def setup(self):
        self.critic = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            activations=self.activations,
            add_final_layer=True,
            output_nodes=self.output_nodes,
            fp8_residual_blocks=self.critic_precision == 'fp8_direct',
            resident_fp8_residual_blocks=self.critic_precision == 'fp8_resident',
            fp8_amax_history_length=self.fp8_amax_history_length,
        )

    def __call__(self, inputs: jnp.ndarray):
        q_value = self.critic(inputs)
        return q_value    
    
class QValueEnsemble(nn.Module):
    ensemble_size: int = 2
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    critic_precision: str = 'fp32'
    fp8_amax_history_length: int = 1024
    
    def setup(self):
        variable_axes = {'params': 0, 'intermediates': 0}
        if self.critic_precision in ('fp8_direct', 'fp8_resident'):
            variable_axes[OVERWRITE_WITH_GRADIENT] = 0
        VmapCritic = nn.vmap(QValue,
                             variable_axes=variable_axes,
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.ensemble_size)
        self.q_value_ensemble = VmapCritic(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            activations=self.activations,
            output_nodes=self.output_nodes,
            critic_precision=self.critic_precision,
            fp8_amax_history_length=self.fp8_amax_history_length,
        )

    def __call__(self, inputs: jnp.ndarray):
        q_values = self.q_value_ensemble(inputs)
        return q_values
    
class Critic(nn.Module):
    num_tasks: int
    embedding_size: int
    ensemble_size: int = 2
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    multitask: bool = False
    task_embedding_norm: str = 'l2'
    critic_precision: str = 'fp32'
    fp8_amax_history_length: int = 1024
    
    def setup(self):
        if self.multitask:
            self.task_embedding = TaskEmbedding(
                self.num_tasks, self.embedding_size, norm=self.task_embedding_norm
            )
        self.q_value_ensemble = QValueEnsemble(
            ensemble_size=self.ensemble_size,
            hidden_dims=self.hidden_dims, 
            depth=self.depth,
            activations=self.activations,
            output_nodes=self.output_nodes,
            critic_precision=self.critic_precision,
            fp8_amax_history_length=self.fp8_amax_history_length,
        )

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, task_ids: jnp.ndarray, return_embeddings: bool = False):
        if self.multitask is False:
            inputs = jnp.concatenate((observations, actions), axis=-1)
        else:
            task_embedding = self.task_embedding(task_ids)
            if return_embeddings:
                return task_embedding
            inputs = jnp.concatenate((observations, actions, task_embedding), axis=-1)            
        q_values = self.q_value_ensemble(inputs)
        return q_values

class NormalTanhPolicy(nn.Module):
    action_dim: int
    hidden_dims: int = 256
    depth: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    log_std_scale: float = 1.0
    log_std_min: float =  -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0):
        outputs = BroNet(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=None)(observations)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(self.log_std_scale))(outputs)
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + nn.tanh(log_stds))
        stds = jnp.exp(log_stds)
        stds = stds * temperature
        base_dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)
        tanh_dist = distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), 1))
        return tanh_dist
    
class Temperature(nn.Module):
    initial_temperature: float = 1.0
    
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)
