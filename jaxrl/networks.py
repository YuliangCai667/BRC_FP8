import functools
from typing import Callable

from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import fp8_ops
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT, update_fp8_meta
import distrax


E4M3_MAX = jnp.float32(448.0)
CARRY_GAIN = jnp.float32(16.0)
CARRY_INV_GAIN = jnp.float32(0.0625)
CARRY_DTYPE = jnp.float8_e4m3fn


def _e4m3_amax_and_scale(value: jnp.ndarray):
    value = jnp.asarray(value, dtype=jnp.float32)
    amax = jnp.max(jnp.abs(value))
    scale = jnp.where(amax > 0, amax / E4M3_MAX, jnp.float32(1.0))
    return amax, scale


def quantize_e4m3_per_tensor(value: jnp.ndarray):
    value = jnp.asarray(value, dtype=jnp.float32)
    _, scale = _e4m3_amax_and_scale(value)
    codes = (value / scale).astype(jnp.float8_e4m3fn)
    # The stored FP8 value is an observable numerical boundary.  Without this
    # barrier the GPU compiler can forward the pre-cast FP32 value into an
    # immediate widening consumer, erasing the quantization residual while the
    # returned code buffer itself is still genuinely E4M3.
    codes = lax.optimization_barrier(codes)
    return codes, scale


def dequantize_e4m3(codes: jnp.ndarray, scale: jnp.ndarray):
    scale = jnp.asarray(scale, dtype=jnp.float32)
    scale = jnp.reshape(scale, scale.shape + (1,) * (codes.ndim - scale.ndim))
    return codes.astype(jnp.float32) * scale


def _reconstruct_carry_logical_kernel(
    main_codes: jnp.ndarray,
    kernel_scale: jnp.ndarray,
    carry_codes: jnp.ndarray,
):
    """Reconstruct s * (C + R / 16) without persistent FP32 state."""
    scale = jnp.asarray(kernel_scale, dtype=jnp.float32)
    scale = jnp.reshape(
        scale, scale.shape + (1,) * (main_codes.ndim - scale.ndim)
    )
    normalized_logical = (
        main_codes.astype(jnp.float32)
        + carry_codes.astype(jnp.float32) * CARRY_INV_GAIN
    )
    return scale * normalized_logical


def _quantize_carry_resident_kernel(candidate_kernel: jnp.ndarray):
    """Quantize one resident kernel and its shared-scale E4M3 carry."""
    candidate_kernel = jnp.asarray(candidate_kernel, dtype=jnp.float32)
    main_codes, kernel_scale = quantize_e4m3_per_tensor(candidate_kernel)
    main_physical = dequantize_e4m3(main_codes, kernel_scale)
    normalized_residual = (
        candidate_kernel / jnp.asarray(kernel_scale, dtype=jnp.float32)
        - main_codes.astype(jnp.float32)
    )
    carry_prequant = normalized_residual * CARRY_GAIN
    saturation = (~jnp.isfinite(carry_prequant)) | (
        jnp.abs(carry_prequant) > E4M3_MAX
    )
    clipped = jnp.clip(carry_prequant, -E4M3_MAX, E4M3_MAX)
    carry_codes = clipped.astype(CARRY_DTYPE)
    # Carry is a second persistent FP8 boundary.  It needs its own barrier;
    # otherwise a same-graph reconstruction can consume ``clipped`` directly
    # even though the returned/stored carry buffer is genuinely E4M3.
    carry_codes = lax.optimization_barrier(carry_codes)
    logical_reconstruction = _reconstruct_carry_logical_kernel(
        main_codes, kernel_scale, carry_codes
    )
    return {
        'main_codes': main_codes,
        'kernel_scale': kernel_scale,
        'carry_codes': carry_codes,
        'main_physical': main_physical,
        'logical_reconstruction': logical_reconstruction,
        'main_error': candidate_kernel - main_physical,
        'carry_error': candidate_kernel - logical_reconstruction,
        'carry_prequant_absmax': jnp.max(
            jnp.abs(carry_prequant).astype(jnp.float32)
        ),
        'carry_saturation_fraction': jnp.mean(
            saturation.astype(jnp.float32)
        ),
    }


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
    fixed_anchor: bool = False
    carry: bool = False
    critic_residual_compute_format: str = 'legacy'
    fp8_amax_history_length: int = 1024

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        kernel_shape = (inputs.shape[-1], self.features)
        if self.has_variable('params', 'kernel'):
            initial_kernel = jnp.zeros(kernel_shape, dtype=jnp.float8_e4m3fn)
            initial_scale = jnp.float32(1.0)
            initial_carry = jnp.zeros(kernel_shape, dtype=CARRY_DTYPE)
        else:
            fp32_kernel = self.kernel_init(
                self.make_rng('params'), kernel_shape, jnp.float32
            )
            if self.carry:
                initial = _quantize_carry_resident_kernel(fp32_kernel)
                initial_kernel = initial['main_codes']
                initial_scale = initial['kernel_scale']
                initial_carry = initial['carry_codes']
            else:
                initial_kernel, initial_scale = quantize_e4m3_per_tensor(
                    fp32_kernel
                )
                initial_carry = jnp.zeros(kernel_shape, dtype=CARRY_DTYPE)

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
        if self.carry:
            self.variable(
                OVERWRITE_WITH_GRADIENT,
                'kernel_carry',
                lambda: initial_carry,
            )
        output_grad_scale = self.variable(
            OVERWRITE_WITH_GRADIENT,
            'output_grad_scale',
            lambda: jnp.ones((1,), dtype=jnp.float32),
        ).value
        output_grad_amax_history = self.variable(
            OVERWRITE_WITH_GRADIENT,
            'output_grad_amax_history',
            lambda: jnp.zeros(
                (self.fp8_amax_history_length,), dtype=jnp.float32
            ),
        ).value
        if self.fixed_anchor:
            self.variable(
                OVERWRITE_WITH_GRADIENT,
                'kernel_anchor_norm',
                lambda: jnp.asarray(
                    jnp.linalg.norm(
                        dequantize_e4m3(initial_kernel, initial_scale)
                    ),
                    dtype=jnp.float32,
                ),
            )
        bias = self.param(
            'bias', self.bias_init, (self.features,), jnp.float32
        )

        kernel_scale = jnp.asarray(kernel_scale, dtype=jnp.float32)
        if kernel.dtype == jnp.float8_e4m3fn:
            kernel_codes = kernel
            physical_kernel = dequantize_e4m3(kernel_codes, kernel_scale)
        else:
            physical_kernel = jnp.asarray(kernel, dtype=jnp.float32)
            kernel_codes = (physical_kernel / kernel_scale).astype(
                jnp.float8_e4m3fn
            )

        if self.critic_residual_compute_format != 'legacy':
            from jaxrl.low_precision.two_term_dense import two_term_dense
            carry_codes = self.get_variable(OVERWRITE_WITH_GRADIENT, 'kernel_carry')
            if not self.carry or carry_codes is None:
                raise ValueError('Two-term compute requires persistent CARRY weights')
            codes = lax.stop_gradient(lax.optimization_barrier(kernel_codes))
            carry_codes = lax.stop_gradient(carry_codes)
            scale = lax.stop_gradient(kernel_scale)
            x = jnp.asarray(inputs, dtype=jnp.float32)
            out = two_term_dense(x.reshape(-1, x.shape[-1]), physical_kernel,
                                 codes.astype(jnp.float32), carry_codes.astype(jnp.float32),
                                 scale, bias, self.critic_residual_compute_format)
            self.sow('intermediates', 'activation', x)
            return out.reshape(x.shape[:-1] + (self.features,))

        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        activation_amax, activation_scale = _e4m3_amax_and_scale(inputs)
        activation_codes = (inputs / activation_scale).astype(jnp.float8_e4m3fn)
        dimension_numbers = (
            ((activation_codes.ndim - 1,), (0,)),
            ((), ()),
        )
        outputs = fp8_ops.quantized_dot(
            inputs,
            activation_codes,
            activation_scale,
            physical_kernel,
            kernel_codes,
            kernel_scale,
            output_grad_scale,
            output_grad_amax_history,
            jnp.float32,
            dimension_numbers,
            preferred_element_type=jnp.float32,
        )
        outputs = fp8_ops.out_dq(
            jnp.float32,
            activation_scale,
            kernel_scale,
            outputs,
        )
        outputs += jnp.reshape(bias, (1,) * (outputs.ndim - 1) + (-1,))

        self.sow('intermediates', 'activation', inputs)
        self.sow('intermediates', 'activation_amax', activation_amax)
        self.sow('intermediates', 'activation_scale', activation_scale)
        return outputs


class CurrentAmaxFp8Dense(nn.Module):
    """FP32-master Dense with resident-equivalent current-amax FP8 compute."""

    features: int
    kernel_init: Callable = default_init()
    bias_init: Callable = nn.initializers.zeros_init()
    fp8_amax_history_length: int = 1024

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (inputs.shape[-1], self.features),
            jnp.float32,
        )
        bias = self.param(
            'bias', self.bias_init, (self.features,), jnp.float32
        )
        output_grad_scale = self.variable(
            OVERWRITE_WITH_GRADIENT,
            'output_grad_scale',
            lambda: jnp.ones((1,), dtype=jnp.float32),
        ).value
        output_grad_amax_history = self.variable(
            OVERWRITE_WITH_GRADIENT,
            'output_grad_amax_history',
            lambda: jnp.zeros(
                (self.fp8_amax_history_length,), dtype=jnp.float32
            ),
        ).value

        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        kernel = jnp.asarray(kernel, dtype=jnp.float32)
        activation_amax, activation_scale = _e4m3_amax_and_scale(inputs)
        kernel_amax, kernel_scale = _e4m3_amax_and_scale(kernel)
        activation_codes = (inputs / activation_scale).astype(jnp.float8_e4m3fn)
        kernel_codes = (kernel / kernel_scale).astype(jnp.float8_e4m3fn)
        dimension_numbers = (
            ((activation_codes.ndim - 1,), (0,)),
            ((), ()),
        )
        outputs = fp8_ops.quantized_dot(
            inputs,
            activation_codes,
            activation_scale,
            kernel,
            kernel_codes,
            kernel_scale,
            output_grad_scale,
            output_grad_amax_history,
            jnp.float32,
            dimension_numbers,
            preferred_element_type=jnp.float32,
        )
        outputs = fp8_ops.out_dq(
            jnp.float32,
            activation_scale,
            kernel_scale,
            outputs,
        )
        outputs += jnp.reshape(bias, (1,) * (outputs.ndim - 1) + (-1,))

        self.sow('intermediates', 'activation', inputs)
        self.sow('intermediates', 'activation_amax', activation_amax)
        self.sow('intermediates', 'activation_scale', activation_scale)
        self.sow('intermediates', 'kernel_amax', kernel_amax)
        self.sow('intermediates', 'kernel_scale', kernel_scale)
        return outputs


def _precision_dense(
    features: int,
    name: str,
    *,
    fp8_direct: bool = False,
    fp8_resident: bool = False,
    fp8_current_master: bool = False,
    fp8_resident_fixed_anchor: bool = False,
    fp8_resident_carry: bool = False,
    critic_residual_compute_format: str = 'legacy',
    fp8_amax_history_length: int = 1024,
):
    if fp8_resident:
        return ResidentFp8Dense(
            features,
            kernel_init=default_init(),
            fixed_anchor=fp8_resident_fixed_anchor,
            carry=fp8_resident_carry,
            critic_residual_compute_format=critic_residual_compute_format,
            fp8_amax_history_length=fp8_amax_history_length,
            name=name,
        )
    if fp8_current_master:
        return CurrentAmaxFp8Dense(
            features,
            kernel_init=default_init(),
            fp8_amax_history_length=fp8_amax_history_length,
            name=name,
        )
    if fp8_direct:
        return nn.Dense(
            features,
            kernel_init=default_init(),
            dot_general_cls=functools.partial(
                Fp8DirectDotGeneralOp,
                amax_history_length=fp8_amax_history_length,
            ),
            name=name,
        )
    return nn.Dense(features, kernel_init=default_init(), name=name)


class BronetBlock(nn.Module):
    hidden_dims: int
    activations: Callable[[jnp.ndarray], jnp.ndarray]
    fp8_direct: bool = False
    fp8_resident: bool = False
    fp8_current_master: bool = False
    fp8_resident_fixed_anchor: bool = False
    fp8_resident_carry: bool = False
    critic_residual_compute_format: str = 'legacy'
    fp8_amax_history_length: int = 1024

    def _dense(self, name: str):
        return _precision_dense(
            self.hidden_dims,
            name,
            fp8_direct=self.fp8_direct,
            fp8_resident=self.fp8_resident,
            fp8_current_master=self.fp8_current_master,
            fp8_resident_fixed_anchor=self.fp8_resident_fixed_anchor,
            fp8_resident_carry=self.fp8_resident_carry,
            critic_residual_compute_format=self.critic_residual_compute_format,
            fp8_amax_history_length=self.fp8_amax_history_length,
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
    current_master_fp8_residual_blocks: bool = False
    resident_fp8_fixed_anchor: bool = False
    resident_fp8_carry: bool = False
    critic_residual_compute_format: str = 'legacy'
    fp8_all_dense_kernels: bool = False
    fp8_input_dense_kernel: bool = False
    fp8_output_dense_kernel: bool = False
    fp8_amax_history_length: int = 1024

    def _edge_dense(self, name: str, features: int):
        enabled = self.fp8_all_dense_kernels or (
            self.fp8_input_dense_kernel
            if name == 'Dense_0'
            else self.fp8_output_dense_kernel
        )
        return _precision_dense(
            features,
            name,
            fp8_direct=enabled and self.fp8_residual_blocks,
            fp8_resident=enabled and self.resident_fp8_residual_blocks,
            fp8_current_master=(
                enabled and self.current_master_fp8_residual_blocks
            ),
            fp8_resident_fixed_anchor=self.resident_fp8_fixed_anchor,
            fp8_resident_carry=self.resident_fp8_carry,
            fp8_amax_history_length=self.fp8_amax_history_length,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = self._edge_dense('Dense_0', self.hidden_dims)(x)
        self.sow('intermediates', 'input_dense_output', x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        for i in range(self.depth):
            x = BronetBlock(
                self.hidden_dims,
                self.activations,
                fp8_direct=self.fp8_residual_blocks,
                fp8_resident=self.resident_fp8_residual_blocks,
                fp8_current_master=self.current_master_fp8_residual_blocks,
                fp8_resident_fixed_anchor=self.resident_fp8_fixed_anchor,
                fp8_resident_carry=self.resident_fp8_carry,
                critic_residual_compute_format=self.critic_residual_compute_format,
                fp8_amax_history_length=self.fp8_amax_history_length,
            )(x)
        if self.add_final_layer:
            x = self._edge_dense('Dense_1', self.output_nodes)(x)
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
    fp8_resident_fixed_anchor: bool = False
    fp8_resident_carry: bool = False
    critic_residual_compute_format: str = 'legacy'
    fp8_all_dense_kernels: bool = False
    fp8_input_dense_kernel: bool = False
    fp8_output_dense_kernel: bool = False
    
    def setup(self):
        self.critic = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            activations=self.activations,
            add_final_layer=True,
            output_nodes=self.output_nodes,
            fp8_residual_blocks=self.critic_precision in (
                'fp8_direct', 'fp8_lag'
            ),
            resident_fp8_residual_blocks=self.critic_precision == 'fp8_resident',
            current_master_fp8_residual_blocks=(
                self.critic_precision == 'fp8_current_master'
            ),
            resident_fp8_fixed_anchor=self.fp8_resident_fixed_anchor,
            resident_fp8_carry=self.fp8_resident_carry,
            critic_residual_compute_format=self.critic_residual_compute_format,
            fp8_all_dense_kernels=self.fp8_all_dense_kernels,
            fp8_input_dense_kernel=self.fp8_input_dense_kernel,
            fp8_output_dense_kernel=self.fp8_output_dense_kernel,
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
    fp8_resident_fixed_anchor: bool = False
    fp8_resident_carry: bool = False
    critic_residual_compute_format: str = 'legacy'
    fp8_all_dense_kernels: bool = False
    fp8_input_dense_kernel: bool = False
    fp8_output_dense_kernel: bool = False
    
    def setup(self):
        variable_axes = {'params': 0, 'intermediates': 0}
        if self.critic_precision in (
            'fp8_direct', 'fp8_resident', 'fp8_current_master', 'fp8_lag'
        ):
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
            fp8_resident_fixed_anchor=self.fp8_resident_fixed_anchor,
            fp8_resident_carry=self.fp8_resident_carry,
            critic_residual_compute_format=self.critic_residual_compute_format,
            fp8_all_dense_kernels=self.fp8_all_dense_kernels,
            fp8_input_dense_kernel=self.fp8_input_dense_kernel,
            fp8_output_dense_kernel=self.fp8_output_dense_kernel,
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
    fp8_resident_canonicalization: bool = False
    fp8_resident_carry: bool = False
    critic_residual_compute_format: str = 'legacy'
    fp8_all_dense_kernels: bool = False
    fp8_input_dense_kernel: bool = False
    fp8_output_dense_kernel: bool = False
    
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
            fp8_resident_fixed_anchor=self.fp8_resident_canonicalization,
            fp8_resident_carry=self.fp8_resident_carry,
            critic_residual_compute_format=self.critic_residual_compute_format,
            fp8_all_dense_kernels=self.fp8_all_dense_kernels,
            fp8_input_dense_kernel=self.fp8_input_dense_kernel,
            fp8_output_dense_kernel=self.fp8_output_dense_kernel,
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
