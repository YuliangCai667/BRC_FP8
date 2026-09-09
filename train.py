import os

os.environ['MUJOCO_GL'] = 'egl'

import time
import json
from pathlib import Path

import jax
import numpy as np
from absl import app, flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.checkpoint import CheckpointManager, validate_checkpoint_config
from jaxrl.env_names import get_environment_list
from jaxrl.envs import ParallelEnv
from jaxrl.experiment import ExperimentRecorder, collect_jax_memory_stats, summarize_tree
from jaxrl.logger import EpisodeRecorder, get_wandb_video
from jaxrl.normalizer import RewardNormalizer
from jaxrl.optimizers import optimizer_state_inventory
from jaxrl.paper_alignment import resolve_paper_alignment
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.utils import Batch


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer(
    'eval_seed_offset', 0,
    'Offset added to --seed for the independent evaluation environment.',
)
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps per task.')
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Replay buffer capacity per task.')
flags.DEFINE_integer('start_training', 5000, 'Number of steps before training starts.')
flags.DEFINE_string('env_names', 'cheetah-run', 'Environment name or named task group.')
flags.DEFINE_boolean('log_to_wandb', True, 'Whether to mirror metrics to W&B.')
flags.DEFINE_string('wandb_name', 'auto', 'W&B display name; auto uses the seed.')
flags.DEFINE_enum('critic_residual_compute_format', 'legacy', ['legacy', 'hybrid', 'mxfp8'], 'Online residual compute: legacy main-only or CARRY two-term.')
flags.DEFINE_enum('critic_residual_compute_terms', 'main_plus_carry', ['main_only', 'main_plus_carry'], 'Residual compute terms; legacy always reads main only.')
flags.DEFINE_enum('critic_residual_compute_rounding', 'rtn', ['rtn'], 'Deterministic compute operand rounding.')
flags.DEFINE_boolean('critic_residual_compute_rht', False, 'Must be false for V1.')
flags.DEFINE_boolean('offline_evaluation', True, 'Whether to perform deterministic evaluations.')
flags.DEFINE_boolean('render', True, 'Whether to log evaluation videos.')
flags.DEFINE_integer('updates_per_step', 2, 'Number of updates per environment step.')
flags.DEFINE_integer('width_critic', 4096, 'Width of the critic network.')
flags.DEFINE_enum(
    'critic_precision', 'fp32',
    ['fp32', 'fp8_direct', 'fp8_resident', 'fp8_current_master'],
    'Precision used by the online critic residual-block Dense layers.',
)
flags.DEFINE_boolean(
    'fp8_resident_canonicalization',
    False,
    'Anchor each online resident kernel to its fixed initialization '
    'Frobenius norm and scale the paired bias by the same factor.',
)
flags.DEFINE_boolean(
    'fp8_resident_carry',
    False,
    'Persist one shared-scale E4M3 carry code per resident kernel element.',
)
flags.DEFINE_enum(
    'critic_optimizer_state', 'fp32', ['fp32', 'bf16', 'fp8', 'fp8_carry'],
    'Adam moment storage for the four residual kernels; arithmetic stays FP32.',
)
flags.DEFINE_boolean(
    'fp8_all_dense_kernels',
    False,
    'Extend selected online and target FP8 modes to the input and '
    'distributional-output Dense kernels.',
)
flags.DEFINE_boolean(
    'fp8_input_dense_kernel',
    False,
    'Extend selected online and target FP8 modes to the critic input Dense '
    'kernel.',
)
flags.DEFINE_boolean(
    'fp8_output_dense_kernel',
    False,
    'Extend selected online and target FP8 modes to the critic '
    'distributional-output Dense kernel.',
)
flags.DEFINE_enum(
    'target_critic_precision', 'fp32',
    ['fp32', 'fp8_direct', 'fp8_resident', 'fp8_lag'],
    'Target critic residual Dense mode: FP32, FP8-direct compute with FP32 '
    'parameters, resident E4M3 kernels, or resident E4M3 lag state.',
)
flags.DEFINE_integer(
    'fp8_amax_history_length', 1024,
    'FP8-direct and resident-backward per-tensor amax history length.',
)
flags.DEFINE_boolean(
    'paper_alignment', False,
    'Enable L1 and empirical-entropy alignment while keeping reward-mean return-scale estimation.',
)
flags.DEFINE_enum(
    'task_embedding_norm', 'auto', ['auto', 'l1', 'l2'],
    'Task embedding norm; auto follows --paper_alignment.',
)
flags.DEFINE_enum(
    'return_bootstrap', 'auto', ['auto', 'reward_mean', 'critic'],
    'Time-limit return bootstrap; auto follows --paper_alignment.',
)
flags.DEFINE_enum(
    'entropy_correction', 'auto',
    ['auto', 'target_entropy', 'empirical_per_task'],
    'Reward-scale entropy correction; auto follows --paper_alignment.',
)
flags.DEFINE_enum(
    'metaworld_reset_mode', 'frozen', ['frozen', 'resample'],
    'Keep one MetaWorld rand_vec or resample it on each reset.',
)

flags.DEFINE_string('run_root', 'runs', 'Root directory for local experiment data.')
flags.DEFINE_string('run_id', 'auto', 'Local run id; auto generates one.')
flags.DEFINE_integer('metrics_interval', 1000, 'Training metric sampling interval; 0 disables.')
flags.DEFINE_integer('metrics_flush_interval', 1000, 'Episode/local file flush interval; 0 disables.')
flags.DEFINE_float('system_metrics_interval_sec', 10.0, 'Host/GPU monitor interval.')
flags.DEFINE_integer('profile_interval', 25000, 'Component profiling interval; 0 disables.')
flags.DEFINE_integer('profile_window', 10, 'Number of synchronized steps in a profiling window.')
flags.DEFINE_integer('tensor_stats_interval', 25000, 'Tensor diagnostic interval; 0 disables.')
flags.DEFINE_integer('analysis_checkpoint_interval', 50000, 'Parameter checkpoint interval; 0 disables.')
flags.DEFINE_integer('recovery_checkpoint_interval', 100000, 'Recovery checkpoint interval; 0 disables.')
flags.DEFINE_integer('keep_last_analysis_checkpoints', 2, 'Recent analysis checkpoints to retain.')
flags.DEFINE_integer('keep_last_recovery_checkpoints', 1, 'Recent recovery checkpoints to retain.')
flags.DEFINE_boolean('save_replay_buffer', True, 'Include valid replay data in recovery checkpoints.')
flags.DEFINE_string('resume_from', '', 'Recovery checkpoint or run directory to resume.')
flags.DEFINE_enum('actor_training_recipe', 'fp32', ['fp32', 'carry_body_bf16_edge_qat'], 'Actor storage and update recipe.')
flags.DEFINE_enum('actor_body_compute', 'mxfp8_main_plus_carry', ['mxfp8_main_plus_carry', 'hybrid_main_plus_carry'], 'Main phase actor residual compute.')
flags.DEFINE_enum('actor_edge_storage', 'bf16', ['bf16'], 'QAT edge persistent dtype.')
flags.DEFINE_boolean('actor_weight_qat', True, 'Enable edge weight QAT in the actor recipe.')
flags.DEFINE_enum('actor_export_codec', 'e4m3fn_block32_fp32scale_rtn_v1', ['e4m3fn_block32_fp32scale_rtn_v1'], 'Shared QAT / export codec.')
flags.DEFINE_integer('actor_export_align_start', 450000, 'Environment step to switch all actor calls to W8A16.')
flags.DEFINE_boolean('actor_export_on_finish', False, 'Export and independently validate/evaluate the final checkpoint.')


def _block_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, 'block_until_ready') else value,
        tree,
    )


def _host_metrics(tree):
    tree = jax.device_get(tree)
    result = {}
    nan_count = inf_count = 0
    for key, value in tree.items():
        array = np.asarray(value)
        nan_count += int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0
        inf_count += int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0
        result[key] = array.item() if array.ndim == 0 else array
    result['update_nan_count'] = nan_count
    result['update_inf_count'] = inf_count
    return result


def _eval_aggregates(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    return {
        f'{prefix}_mean': float(values.mean()),
        f'{prefix}_median': float(np.median(values)),
        f'{prefix}_min': float(values.min()),
        f'{prefix}_p10': float(np.percentile(values, 10)),
        f'{prefix}_p90': float(np.percentile(values, 90)),
    }


def _save_probe_batch(path: Path, batch: Batch):
    np.savez(
        path,
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        masks=batch.masks,
        next_observations=batch.next_observations,
        task_ids=batch.task_ids,
    )


def _load_probe_batch(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return Batch(**{field: data[field] for field in Batch._fields})


def _profile_active(step: int):
    if FLAGS.profile_interval <= 0 or FLAGS.profile_window <= 0:
        return False
    return step % FLAGS.profile_interval < FLAGS.profile_window


def _fp8_dense_scope():
    input_dense = (
        FLAGS.fp8_all_dense_kernels or FLAGS.fp8_input_dense_kernel
    )
    output_dense = (
        FLAGS.fp8_all_dense_kernels or FLAGS.fp8_output_dense_kernel
    )
    if input_dense and output_dense:
        return 'all_dense_kernels'
    if input_dense:
        return 'residual_and_input_dense_kernels'
    if output_dense:
        return 'residual_and_output_dense_kernels'
    return 'residual_dense_kernels'


def main(_):
    program_start = time.perf_counter()
    interval_flags = {
        'metrics_interval': FLAGS.metrics_interval,
        'metrics_flush_interval': FLAGS.metrics_flush_interval,
        'profile_interval': FLAGS.profile_interval,
        'tensor_stats_interval': FLAGS.tensor_stats_interval,
        'analysis_checkpoint_interval': FLAGS.analysis_checkpoint_interval,
        'recovery_checkpoint_interval': FLAGS.recovery_checkpoint_interval,
    }
    for name, value in interval_flags.items():
        if value < 0:
            raise ValueError(f'--{name} must be >= 0')
    if FLAGS.profile_interval > 0 and not 0 < FLAGS.profile_window < FLAGS.profile_interval:
        raise ValueError('--profile_window must be > 0 and smaller than --profile_interval')
    if FLAGS.batch_size <= 0 or FLAGS.max_steps <= 0 or FLAGS.replay_buffer_size <= 0:
        raise ValueError('batch_size, max_steps, and replay_buffer_size must be positive')
    if FLAGS.fp8_amax_history_length <= 0:
        raise ValueError('--fp8_amax_history_length must be positive')
    if (
        FLAGS.fp8_resident_canonicalization
        and FLAGS.critic_precision != 'fp8_resident'
    ):
        raise ValueError(
            '--fp8_resident_canonicalization requires '
            '--critic_precision=fp8_resident'
        )
    if FLAGS.fp8_resident_carry and FLAGS.critic_precision != 'fp8_resident':
        raise ValueError(
            '--fp8_resident_carry requires --critic_precision=fp8_resident'
        )
    if (
        FLAGS.fp8_resident_carry
        and FLAGS.fp8_resident_canonicalization
    ):
        raise ValueError(
            '--fp8_resident_carry requires '
            '--fp8_resident_canonicalization=false'
        )

    resolved_alignment = resolve_paper_alignment(
        FLAGS.paper_alignment,
        FLAGS.task_embedding_norm,
        FLAGS.return_bootstrap,
        FLAGS.entropy_correction,
    )
    if FLAGS.critic_residual_compute_rht:
        raise ValueError('V1 fixes RHT off')
    if FLAGS.critic_residual_compute_format != 'legacy' and FLAGS.critic_residual_compute_terms != 'main_plus_carry':
        raise ValueError('hybrid/MXFP8 residual compute requires main_plus_carry')
    config = FLAGS.flag_values_dict()
    actor_qat = FLAGS.actor_training_recipe != 'fp32'
    if actor_qat:
        import signal
        def actor_stop(signum, frame):
            raise KeyboardInterrupt(f'training stopped by signal {signum}')
        signal.signal(signal.SIGTERM, actor_stop)
    if actor_qat and (not FLAGS.actor_weight_qat or FLAGS.actor_export_align_start < 0):
        raise ValueError('actor recipe requires weight QAT and a nonnegative alignment step')
    if FLAGS.actor_export_on_finish and (not actor_qat or FLAGS.max_steps < FLAGS.actor_export_align_start):
        raise ValueError('final export requires the aligned actor recipe')
    if actor_qat:
        import subprocess
        config.update(actor_base_commit='affa7ea',
                      actor_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      resolved_actor_width=256, resolved_actor_depth=1,
                      resolved_actor_optimizer='adamw_fp32_moments',
                      resolved_actor_adamw=dict(learning_rate=3e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4))
    if FLAGS.critic_residual_compute_format == 'mxfp8':
        from jaxrl.low_precision.backends.native import register
        import importlib.metadata
        dependencies = {name: importlib.metadata.version(name) for name in ('jax','jaxlib','flax','optax')}
        config.update(method_version=1,
                      online_residual_compute_format=FLAGS.critic_residual_compute_format,
                      online_residual_compute_terms='main_plus_carry',
                      online_residual_compute_rounding='rtn', online_residual_compute_rht=False,
                      online_residual_nvfp4_weight_scaling='1d',
                      weight_state_codec='carry_e4m3_gain16',
                      optimizer_state_codec='dual_fp8_m_v_block128_gain16', target_state_codec='lag',
                      kernel_build_hash=register()[1],
                      dependency_manifest=dependencies)
    config.update({
        'carry_gain': 16.0,
        'optimizer_moment_block_size': 128,
        'optimizer_moment_carry_gain': 16.0,
        'carry_dtype': 'float8_e4m3fn',
        'resolved_fp8_code_materialization': (
            'optimization_barrier_after_e4m3_cast_v1'
        ),
    })
    config.update({f'resolved_{key}': value for key, value in resolved_alignment.items()})
    fp8_dense_scope = _fp8_dense_scope()
    fp8_lag_scope = fp8_dense_scope.replace('_kernels', '_lag_state')
    config.update({
        'resolved_online_fp8_compute_scope': (
            fp8_dense_scope
            if FLAGS.critic_precision in (
                'fp8_direct', 'fp8_resident', 'fp8_current_master'
            )
            else 'none'
        ),
        'resolved_online_fp8_storage_scope': (
            fp8_dense_scope
            if FLAGS.critic_precision == 'fp8_resident'
            else 'none'
        ),
        'resolved_online_parameter_storage': (
            (
                (
                    f'e4m3_main_and_e4m3_carry_{fp8_dense_scope}_otherwise_fp32'
                )
                if FLAGS.fp8_resident_carry
                else (
                    f'e4m3_{fp8_dense_scope}_otherwise_fp32'
                )
            )
            if FLAGS.critic_precision == 'fp8_resident'
            else 'fp32_all_parameters'
        ),
        'resolved_fp8_resident_state': (
            (
                'carry_e4m3_shared_scale_gain16'
                if FLAGS.fp8_resident_carry
                else 'main_e4m3_shared_scale'
            )
            if FLAGS.critic_precision == 'fp8_resident'
            else 'disabled'
        ),
        'resolved_online_fp8_weight_scaling': (
            'dynamic_current_amax_per_tensor'
            if FLAGS.critic_precision in (
                'fp8_resident', 'fp8_current_master'
            )
            else (
                'delayed_amax_history_per_tensor'
                if FLAGS.critic_precision == 'fp8_direct'
                else 'none'
            )
        ),
        'resolved_online_fp32_weight_master': (
            'none'
            if FLAGS.critic_precision == 'fp8_resident'
            else 'persistent_fp32_parameters'
        ),
        'resolved_online_optimizer_state': (
            'fp32_adamw' if FLAGS.critic_optimizer_state == 'fp32'
            else f'{FLAGS.critic_optimizer_state}_residual_moments_fp32_adamw'
        ),
        'resolved_online_fp8_backward': (
            'scale_aware_e5m2_delayed_amax_custom_vjp'
            if FLAGS.critic_precision in (
                'fp8_resident', 'fp8_current_master'
            )
            else (
                'flax_e5m2_delayed_amax_custom_vjp'
                if FLAGS.critic_precision == 'fp8_direct'
                else 'none'
            )
        ),
        'resolved_online_fp8_canonicalization': (
            'fixed_initial_kernel_norm_with_paired_bias'
            if FLAGS.fp8_resident_canonicalization
            else 'disabled'
        ),
    })
    target_fp8_compute = FLAGS.target_critic_precision in (
        'fp8_direct', 'fp8_resident', 'fp8_lag'
    )
    config.update({
        'resolved_target_fp8_compute_scope': (
            fp8_dense_scope
            if target_fp8_compute else 'none'
        ),
        'resolved_target_fp8_storage_scope': (
            fp8_dense_scope
            if FLAGS.target_critic_precision == 'fp8_resident'
            else (
                fp8_lag_scope
                if FLAGS.target_critic_precision == 'fp8_lag'
                else 'none'
            )
        ),
        'resolved_target_parameter_storage': (
            f'e4m3_{fp8_dense_scope}_otherwise_fp32'
            if FLAGS.target_critic_precision == 'fp8_resident'
            else (
                f'e4m3_lag_{fp8_dense_scope}_otherwise_fp32'
                if FLAGS.target_critic_precision == 'fp8_lag'
                else 'fp32_all_parameters'
            )
        ),
        'resolved_target_fp8_weight_scaling': (
            'dynamic_current_amax_per_tensor'
            if FLAGS.target_critic_precision == 'fp8_resident'
            else (
                'delayed_amax_history_per_tensor'
                if FLAGS.target_critic_precision in ('fp8_direct', 'fp8_lag')
                else 'none'
            )
        ),
        'resolved_target_fp8_activation_scaling': (
            'current_amax_per_tensor'
            if FLAGS.target_critic_precision == 'fp8_resident'
            else (
                'delayed_amax_history_per_tensor'
                if FLAGS.target_critic_precision in ('fp8_direct', 'fp8_lag')
                else 'none'
            )
        ),
        'resolved_target_fp8_scale_state_update': (
            'bootstrap_forward_input_kernel_only'
            if FLAGS.target_critic_precision == 'fp8_direct'
            else (
                'ema_requantization_kernel_scale'
                if FLAGS.target_critic_precision == 'fp8_resident'
                else (
                    'bootstrap_forward_input_kernel_and_lag_requantization'
                    if FLAGS.target_critic_precision == 'fp8_lag'
                    else 'none'
                )
            )
        ),
        'resolved_target_reconstruction': (
            'fp32_online_plus_dequantized_lag'
            if FLAGS.target_critic_precision == 'fp8_lag'
            else 'none'
        ),
        'resolved_target_kernel_cache': 'none',
        'resolved_target_lag_scaling': (
            'dynamic_current_amax_per_tensor'
            if FLAGS.target_critic_precision == 'fp8_lag'
            else 'none'
        ),
    })
    if FLAGS.critic_residual_compute_format == 'mxfp8':
        import subprocess
        from jaxrl.low_precision import block_formats
        config.update(
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            comparison_group='BRC_V1_TWOTERM_20260908',
            resolved_online_fp8_backward='two_term_native_block_scaled_physical_vjp',
            resolved_online_compute_weight_scaling='independent_main_carry_reduction_blocks',
            resolved_online_compute_scale_policy=block_formats.MXFP8_SCALE_POLICY,
            resolved_online_fp8_weight_scaling='state_current_amax_per_tensor_compute_independent_block_scales',
            resolved_evaluation_rng='separate_env_preserve_learner_and_numpy_rng',
        )
    elif FLAGS.critic_residual_compute_format == 'hybrid':
        config.update(
            method_version=2, comparison_group='BRC_HYBRID_CARRY',
            online_residual_compute_format='hybrid', online_residual_compute_terms='main_plus_carry',
            weight_state_codec='carry_e4m3_gain16',
            optimizer_state_codec=FLAGS.critic_optimizer_state, target_state_codec='lag',
            resolved_online_fp8_backward='two_term_e5m2_delayed_amax_physical_vjp',
            resolved_online_compute_weight_scaling='shared_state_scale_main_carry',
            resolved_online_compute_scale_policy='activation_current_tensor_gradient_delayed_tensor',
            resolved_online_fp8_weight_scaling='state_current_amax_per_tensor',
            resolved_evaluation_rng='separate_env_preserve_learner_and_numpy_rng',
        )
    env_names = get_environment_list(FLAGS.env_names)
    resume_manifest = None
    resume_checkpoint = None
    existing_run_dir = None
    requested_run_id = FLAGS.run_id
    wandb_resume_id = None
    precision_transition = False
    source_precision = None

    if FLAGS.resume_from:
        resume_checkpoint = CheckpointManager.resolve_recovery_checkpoint(FLAGS.resume_from)
        resume_manifest = CheckpointManager.read_manifest(resume_checkpoint)
        existing_run_dir = resume_manifest['run_dir']
        requested_run_id = resume_manifest['run_id']
        wandb_resume_id = resume_manifest.get('wandb_id')
        if resume_manifest['task_names'] != env_names:
            raise ValueError('checkpoint task names/order do not match --env_names')
        previous_config = resume_manifest.get('config', {})
        precision_transition = validate_checkpoint_config(previous_config, config)
        source_precision = previous_config.get('critic_precision', 'fp32')

    wandb_run = None
    wandb_initialization_start = time.perf_counter()
    if FLAGS.log_to_wandb:
        import wandb
        init_kwargs = dict(
            config=config,
            entity=os.environ.get('WANDB_ENTITY', ''),
            project=os.environ.get('WANDB_PROJECT', ''),
            group=FLAGS.env_names,
            name=str(FLAGS.seed) if FLAGS.wandb_name == 'auto' else FLAGS.wandb_name,
        )
        if wandb_resume_id:
            init_kwargs.update(id=wandb_resume_id, resume='allow')
        try:
            wandb_run = wandb.init(**init_kwargs)
        except Exception as error:
            print(f'[recorder] warning: W&B initialization failed: {error}')
    wandb_initialization_sec = time.perf_counter() - wandb_initialization_start

    if requested_run_id == 'auto' and wandb_run is not None:
        requested_run_id = wandb_run.id
    recorder = ExperimentRecorder(
        run_root=FLAGS.run_root,
        env_group=FLAGS.env_names,
        config=config,
        task_names=env_names,
        seed=FLAGS.seed,
        run_id=requested_run_id,
        wandb_run=wandb_run,
        system_metrics_interval_sec=FLAGS.system_metrics_interval_sec,
        existing_run_dir=existing_run_dir,
        start_monotonic=program_start,
    )
    recorder.record_event(
        'wandb_initialization_finished',
        wandb_initialization_sec=wandb_initialization_sec,
        enabled=FLAGS.log_to_wandb,
        initialized=wandb_run is not None,
    )

    env_step = 0
    agent = None
    normal_exit = False
    initialization_start = time.perf_counter()
    try:
        env = ParallelEnv(
            env_names, seed=FLAGS.seed,
            metaworld_reset_mode=FLAGS.metaworld_reset_mode,
        )
        eval_env = ParallelEnv(
            env_names, seed=FLAGS.seed + FLAGS.eval_seed_offset,
            metaworld_reset_mode=FLAGS.metaworld_reset_mode,
        ) if FLAGS.offline_evaluation else None
        num_tasks = len(env.envs)
        if actor_qat:
            from jaxrl.deployment_export import preprocessing_manifest
            config['actor_preprocessing'] = preprocessing_manifest(env)
            config['actor_export_source_root'] = str(recorder.run_dir / 'artifacts' / 'source_snapshot')
            recorder._write_config(config)
            if wandb_run is not None:
                wandb_run.config.update({'actor_preprocessing': config['actor_preprocessing']})
        agent = BRC(
            FLAGS.seed,
            env.observation_space.sample()[:1],
            env.action_space.sample()[:1],
            num_tasks=num_tasks,
            updates_per_step=FLAGS.updates_per_step,
            width_critic=FLAGS.width_critic,
            task_embedding_norm=resolved_alignment['task_embedding_norm'],
            critic_precision=FLAGS.critic_precision,
            target_critic_precision=FLAGS.target_critic_precision,
            fp8_amax_history_length=FLAGS.fp8_amax_history_length,
            fp8_resident_canonicalization=(
                FLAGS.fp8_resident_canonicalization
            ),
            fp8_resident_carry=FLAGS.fp8_resident_carry,
            critic_residual_compute_format=FLAGS.critic_residual_compute_format,
            critic_optimizer_state=FLAGS.critic_optimizer_state,
            fp8_all_dense_kernels=FLAGS.fp8_all_dense_kernels,
            fp8_input_dense_kernel=FLAGS.fp8_input_dense_kernel,
            fp8_output_dense_kernel=FLAGS.fp8_output_dense_kernel,
            actor_training_recipe=FLAGS.actor_training_recipe,
            actor_body_compute=FLAGS.actor_body_compute,
            actor_export_align_start=FLAGS.actor_export_align_start,
        )
        resource_devices = tuple(jax.devices())
        replay_buffer = ParallelReplayBuffer(
            env.observation_space,
            env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_tasks=num_tasks,
        )
        reward_normalizer = RewardNormalizer(
            num_tasks,
            target_entropy=agent.target_entropy,
            discount=agent.discount,
            return_bootstrap=resolved_alignment['return_bootstrap'],
            entropy_correction=resolved_alignment['entropy_correction'],
        )
        episode_recorder = EpisodeRecorder(num_tasks, env_names)
        checkpoint_manager = CheckpointManager(
            recorder.run_dir / 'checkpoints',
            recorder.run_id,
            recorder.run_dir,
            env_names,
            config,
            FLAGS.keep_last_analysis_checkpoints,
            FLAGS.keep_last_recovery_checkpoints,
        )

        if resume_checkpoint is not None:
            resume_manifest = checkpoint_manager.load_recovery(
                resume_checkpoint, agent, replay_buffer, reward_normalizer, episode_recorder
            )
            env_step = int(resume_manifest['env_step'])
            episode_recorder.reset_partial()
            recorder.record_event(
                'resume_reset', env_step, agent.step,
                checkpoint=str(resume_checkpoint),
                detail='environment and unfinished trajectories reset; replay/optimizer restored',
            )
            if precision_transition:
                recorder.record_event(
                    'precision_transition', env_step, agent.step,
                    checkpoint=str(resume_checkpoint),
                    from_precision=source_precision,
                    to_precision=FLAGS.critic_precision,
                    fp8_metadata='initialized_from_defaults',
                )

        observations = env.reset()
        initialization_sec = time.perf_counter() - initialization_start
        recorder.record_event(
            'initialization_finished', env_step, agent.step,
            initialization_sec=initialization_sec,
            resumed=resume_checkpoint is not None,
            critic_optimizer_inventory=optimizer_state_inventory(agent.critic.opt_state),
        )

        first_update_sec = None
        latest_update_info = None
        latest_normalized_rewards = None
        best_eval_success = -np.inf
        profile_samples = []
        completed_profile = {}
        probe_batch = None
        probe_path = recorder.run_dir / 'artifacts' / 'probe_batch.npz'
        if probe_path.exists():
            probe_batch = _load_probe_batch(probe_path)

        window_start = time.perf_counter()
        window_start_step = env_step

        for i in range(env_step + 1, FLAGS.max_steps + 1):
            env_step = i
            phase_changed = agent.set_env_step(i)
            if phase_changed:
                recorder.record_event('phase_transition', i, agent.step, component='actor',
                                      actor_phase=agent.actor_phase, optimizer_reset=False, carry_cleared=False)
                if wandb_run is not None:
                    wandb_run.summary['actor_phase'] = agent.actor_phase
            profiling = _profile_active(i) and i >= FLAGS.start_training
            profile = {}

            start = time.perf_counter()
            actions = (
                env.action_space.sample()
                if i < FLAGS.start_training
                else agent.sample_actions(observations, temperature=1.0)
            )
            if profiling:
                profile['action_sample_sec'] = time.perf_counter() - start

            start = time.perf_counter()
            next_observations, rewards, terms, truns, goals = env.step(actions)
            if profiling:
                profile['env_step_sec'] = time.perf_counter() - start

            bootstrap_values = None
            bootstrap_values_normalized = None
            if resolved_alignment['return_bootstrap'] == 'critic' and np.any(truns):
                bootstrap_values_normalized = agent.estimate_bootstrap_values(
                    next_observations, truns
                )
                bootstrap_values = reward_normalizer.denormalize_values(
                    bootstrap_values_normalized,
                    agent.get_temperature(),
                    agent.get_task_entropies(),
                )
            reward_normalizer.update(
                rewards, terms, truns, bootstrap_values=bootstrap_values
            )
            events = episode_recorder.update(rewards, goals, terms, truns, env_step=i)
            for event in events:
                event['episode_wall_time_sec'] = recorder.wall_time_sec
                if bootstrap_values is not None:
                    task_id = event['task_id']
                    event['return_bootstrap_normalized'] = float(
                        bootstrap_values_normalized[task_id]
                    )
                    event['return_bootstrap_raw'] = float(bootstrap_values[task_id])
            recorder.queue_episode_events(events)
            episode_recorder.drain_events()
            masks = env.generate_masks(terms, truns)
            replay_buffer.insert(observations, actions, rewards, masks, next_observations)
            observations = next_observations
            observations, terms, truns = env.reset_where_done(observations, terms, truns)

            if i >= FLAGS.start_training:
                start = time.perf_counter()
                batches = replay_buffer.sample(FLAGS.batch_size, FLAGS.updates_per_step)
                batches = reward_normalizer.normalize(
                    batches,
                    agent.get_temperature(),
                    task_entropies=agent.get_task_entropies(),
                )
                latest_normalized_rewards = batches.rewards
                if profiling:
                    profile['replay_sample_sec'] = time.perf_counter() - start

                start = time.perf_counter()
                latest_update_info = agent.update(
                    batches,
                    FLAGS.updates_per_step,
                    i,
                    collect_update_diagnostics=(
                        (
                            FLAGS.critic_precision == 'fp8_resident'
                            or FLAGS.target_critic_precision in (
                                'fp8_resident', 'fp8_lag'
                            )
                        )
                        and FLAGS.tensor_stats_interval > 0
                        and i % FLAGS.tensor_stats_interval == 0
                    ),
                )
                if FLAGS.critic_residual_compute_format != 'legacy':
                    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(latest_update_info)):
                        raise FloatingPointError('Native V1 learner produced NaN/Inf')
                if phase_changed:
                    _block_tree(latest_update_info)
                    recorder.record_event('actor_phase_first_update_finished', i, agent.step,
                        actor_phase=agent.actor_phase, update_including_compile_seconds=time.perf_counter()-start,
                        **collect_jax_memory_stats(resource_devices))
                if first_update_sec is None or profiling:
                    _block_tree(latest_update_info)
                if first_update_sec is None:
                    first_update_sec = time.perf_counter() - start
                    recorder.record_event(
                        'first_update_finished', i, agent.step,
                        jit_and_first_update_sec=first_update_sec,
                    )
                    window_start = time.perf_counter()
                    window_start_step = i
                if profiling:
                    profile['update_sec'] = time.perf_counter() - start
                    profile_samples.append(profile)
                    if i % FLAGS.profile_interval == FLAGS.profile_window - 1:
                        completed_profile = {
                            f'profile_{key[:-4]}_ms': 1000.0 * np.mean([row[key] for row in profile_samples])
                            for key in profile_samples[0]
                        }
                        completed_profile['profile_samples'] = len(profile_samples)
                        profile_samples = []

            metrics_due = FLAGS.metrics_interval > 0 and i % FLAGS.metrics_interval == 0
            if metrics_due:
                if latest_update_info is not None:
                    _block_tree(latest_update_info)
                now = time.perf_counter()
                active_sec = max(now - window_start, 1e-9)
                active_steps = i - window_start_step
                episode_summary = episode_recorder.interval_summary(reset=True)
                train_metrics = {
                    'actor_export_aligned': int(agent.actor_phase == 'export_align'),
                    **(_host_metrics(latest_update_info) if latest_update_info is not None else {}),
                    **episode_summary,
                    **completed_profile,
                    'replay_buffer_size': replay_buffer.size,
                    'reward_raw_mean': float(np.mean(rewards)),
                    'reward_raw_std': float(np.std(rewards)),
                    'reward_raw_by_task': np.asarray(rewards),
                    'action_mean': float(np.mean(actions)),
                    'action_std': float(np.std(actions)),
                    'action_saturation_fraction': float(np.mean(np.abs(actions) >= 0.99)),
                    'fp8_direct_enabled': float(FLAGS.critic_precision == 'fp8_direct'),
                    'fp8_resident_enabled': float(
                        FLAGS.critic_precision == 'fp8_resident'
                    ),
                    'fp8_resident_carry_enabled': float(
                        FLAGS.fp8_resident_carry
                    ),
                    'fp8_current_master_enabled': float(
                        FLAGS.critic_precision == 'fp8_current_master'
                    ),
                    'target_fp8_direct_enabled': float(
                        FLAGS.target_critic_precision == 'fp8_direct'
                    ),
                    'target_fp8_resident_enabled': float(
                        FLAGS.target_critic_precision == 'fp8_resident'
                    ),
                    'target_fp8_lag_enabled': float(
                        FLAGS.target_critic_precision == 'fp8_lag'
                    ),
                    'window_train_sec': active_sec,
                    'env_steps_per_sec': active_steps / active_sec,
                    'transitions_per_sec': active_steps * num_tasks / active_sec,
                    'updates_per_sec': active_steps * FLAGS.updates_per_step / active_sec,
                    'jit_and_first_update_sec': first_update_sec if first_update_sec is not None else np.nan,
                    **collect_jax_memory_stats(resource_devices),
                }
                train_metrics.update(reward_normalizer.diagnostics(
                    agent.get_temperature(), agent.get_task_entropies()
                ))
                train_metrics['task_entropy_batch_count_by_task'] = np.asarray(
                    agent.task_entropy_counts
                )
                if latest_normalized_rewards is not None:
                    train_metrics.update(
                        reward_normalized_mean=float(np.mean(latest_normalized_rewards)),
                        reward_normalized_std=float(np.std(latest_normalized_rewards)),
                    )
                recorder.record_train(i, agent.step, train_metrics)
                recorder.flush(agent.step)
                completed_profile = {}
                print(
                    f"step={i} transitions={i * num_tasks} "
                    f"throughput={train_metrics['transitions_per_sec']:.1f}/s "
                    f"return={train_metrics['episode_return_mean']:.3f} "
                    f"success={train_metrics['episode_success_mean']:.3f} "
                    f"wall={recorder.wall_time_sec:.1f}s"
                )
                window_start = time.perf_counter()
                window_start_step = i
            elif FLAGS.metrics_flush_interval > 0 and i % FLAGS.metrics_flush_interval == 0:
                recorder.flush(agent.step)

            maintenance_performed = False
            eval_success = None
            if (
                FLAGS.offline_evaluation
                and FLAGS.eval_interval > 0
                and i >= FLAGS.start_training
                and i % FLAGS.eval_interval == 0
            ):
                maintenance_performed = True
                if latest_update_info is not None:
                    _block_tree(latest_update_info)
                eval_start = time.perf_counter()
                training_rng = agent.rng
                numpy_rng = np.random.get_state()
                try:
                    eval_stats = eval_env.evaluate(
                        agent, num_episodes=FLAGS.eval_episodes, temperature=0.0, render=FLAGS.render
                    )
                finally:
                    agent.rng = training_rng
                    np.random.set_state(numpy_rng)
                eval_sec = time.perf_counter() - eval_start
                renders = eval_stats.pop('renders', None)
                eval_metrics = {
                    'actor_phase': agent.actor_phase,
                    'actor_export_aligned': int(agent.actor_phase == 'export_align'),
                    'return_by_task': np.asarray(eval_stats['return']),
                    'success_by_task': np.asarray(eval_stats['goal']),
                    **_eval_aggregates(eval_stats['return'], 'return'),
                    **_eval_aggregates(eval_stats['goal'], 'success'),
                    'evaluation_sec': eval_sec,
                }
                recorder.record_eval(i, agent.step, eval_metrics)
                eval_success = eval_metrics['success_mean']
                print(
                    f"eval step={i} return={eval_metrics['return_mean']:.3f} "
                    f"success={eval_success:.3f} time={eval_sec:.1f}s"
                )
                if renders is not None and wandb_run is not None:
                    try:
                        wandb_run.log({'env_step': i, 'eval/renders': get_wandb_video(renders)})
                    except Exception as error:
                        recorder.record_event('video_log_failed', i, agent.step, error=str(error))

            if (
                FLAGS.tensor_stats_interval > 0
                and i >= FLAGS.start_training
                and i % FLAGS.tensor_stats_interval == 0
            ):
                maintenance_performed = True
                tensor_start = time.perf_counter()
                try:
                    if probe_batch is None:
                        probe_batch = replay_buffer.make_probe_batch(
                            batch_size=min(256, FLAGS.batch_size), seed=FLAGS.seed + 9173
                        )
                        _save_probe_batch(probe_path, probe_batch)
                    normalized_probe = reward_normalizer.normalize(
                        probe_batch,
                        agent.get_temperature(),
                        task_entropies=agent.get_task_entropies(),
                    )
                    diagnostic_trees = agent.get_tensor_diagnostics(normalized_probe)
                    tensor_stats = {}
                    for category, tree in diagnostic_trees.items():
                        tensor_stats.update(summarize_tree(tree, prefix=category))
                    tensor_stats.update(recorder.collect_registered_tensor_stats(
                        agent=agent, batch=normalized_probe, env_step=i
                    ))
                    recorder.record_tensor_stats(i, agent.step, tensor_stats)
                    recorder.record_event(
                        'tensor_stats_finished', i, agent.step,
                        tensor_count=len(tensor_stats),
                        tensor_stats_sec=time.perf_counter() - tensor_start,
                    )
                except Exception as error:
                    recorder.record_event('tensor_stats_failed', i, agent.step, error=str(error))
                    print(f'[recorder] warning: tensor statistics failed: {error}')

            new_best = eval_success is not None and eval_success > best_eval_success
            if new_best:
                best_eval_success = eval_success
            analysis_due = (
                FLAGS.analysis_checkpoint_interval > 0
                and i >= FLAGS.start_training
                and i % FLAGS.analysis_checkpoint_interval == 0
            )
            if analysis_due or new_best:
                maintenance_performed = True
                recorder.flush(agent.step)
                checkpoint_start = time.perf_counter()
                try:
                    path = checkpoint_manager.save_analysis(
                        agent, i, is_best=new_best, is_final=i == FLAGS.max_steps
                    )
                    recorder.record_event(
                        'analysis_checkpoint_saved', i, agent.step,
                        path=str(path), checkpoint_sec=time.perf_counter() - checkpoint_start,
                        is_best=new_best,
                    )
                except Exception as error:
                    recorder.record_event('analysis_checkpoint_failed', i, agent.step, error=str(error))
                    print(f'[checkpoint] warning: analysis checkpoint failed: {error}')

            recovery_due = (
                FLAGS.recovery_checkpoint_interval > 0
                and i >= FLAGS.start_training
                and i % FLAGS.recovery_checkpoint_interval == 0
            ) or (actor_qat and i == FLAGS.actor_export_align_start)
            if recovery_due:
                maintenance_performed = True
                recorder.flush(agent.step)
                checkpoint_start = time.perf_counter()
                try:
                    path = checkpoint_manager.save_recovery(
                        agent, replay_buffer, reward_normalizer, episode_recorder,
                        i, wandb_run.id if wandb_run is not None else None,
                        save_replay_buffer=FLAGS.save_replay_buffer,
                        is_final=i == FLAGS.max_steps,
                        is_phase_transition=actor_qat and i == FLAGS.actor_export_align_start,
                    )
                    manifest = CheckpointManager.read_manifest(path)
                    if actor_qat and i == FLAGS.actor_export_align_start and not manifest['includes_replay_buffer']:
                        raise IOError('phase transition requires a complete recovery checkpoint including replay')
                    recorder.record_event(
                        'recovery_checkpoint_saved', i, agent.step,
                        path=str(path), checkpoint_sec=time.perf_counter() - checkpoint_start,
                        includes_replay_buffer=manifest['includes_replay_buffer'],
                        degraded_reason=manifest.get('degraded_reason'),
                    )
                except Exception as error:
                    recorder.record_event('recovery_checkpoint_failed', i, agent.step, error=str(error))
                    if actor_qat and i == FLAGS.actor_export_align_start:
                        raise
                    print(f'[checkpoint] warning: recovery checkpoint failed: {error}')

            if maintenance_performed:
                recorder.flush(agent.step)
                window_start = time.perf_counter()
                window_start_step = i

        recorder.flush(agent.step)
        final_start = time.perf_counter()
        try:
            if FLAGS.analysis_checkpoint_interval > 0:
                checkpoint_manager.save_analysis(agent, env_step, is_final=True)
            if FLAGS.recovery_checkpoint_interval > 0 or FLAGS.actor_export_on_finish:
                final_checkpoint = checkpoint_manager.save_recovery(
                    agent, replay_buffer, reward_normalizer, episode_recorder,
                    env_step, wandb_run.id if wandb_run is not None else None,
                    save_replay_buffer=FLAGS.save_replay_buffer, is_final=True,
                )
            recorder.record_event(
                'final_checkpoints_finished', env_step, agent.step,
                checkpoint_sec=time.perf_counter() - final_start,
                **collect_jax_memory_stats(resource_devices),
            )
        except Exception as error:
            recorder.record_event('final_checkpoint_failed', env_step, agent.step, error=str(error))
            if FLAGS.actor_export_on_finish:
                raise
            print(f'[checkpoint] warning: final checkpoint failed: {error}')
        if FLAGS.actor_export_on_finish:
            from jaxrl.deployment_export import finish_actor_export
            finish_actor_export(final_checkpoint, recorder.run_dir, env, config, recorder, wandb_run)
        normal_exit = True
    except BaseException as error:
        recorder.record_event(
            'run_interrupted', env_step, agent.step if agent is not None else 0,
            error_type=type(error).__name__, error=str(error),
        )
        if actor_qat and agent is not None and 'checkpoint_manager' in locals():
            try:
                state = (agent.actor, agent.critic, agent.target_critic, agent.temp)
                finite_state = all(np.isfinite(np.asarray(x).astype(np.float32)).all()
                                   for x in jax.tree.leaves(state))
                if finite_state:
                    path = checkpoint_manager.save_recovery(
                        agent, replay_buffer, reward_normalizer, episode_recorder, env_step,
                        wandb_run.id if wandb_run is not None else None, save_replay_buffer=True)
                    recorder.record_event('interruption_recovery_saved', env_step, agent.step,
                                          path=str(path), partial_environment_step_possible=True)
                else:
                    path = recorder.run_dir / 'artifacts' / f'nonfinite_state_step_{env_step}'
                    path.mkdir(exist_ok=True)
                    agent.save(str(path))
                    (path / 'NOT_RESUMABLE').write_text(str(error))
                    recorder.record_event('nonfinite_state_saved', env_step, agent.step,
                                          path=str(path), resume_from='last complete periodic recovery')
            except BaseException as save_error:
                recorder.record_event('interruption_checkpoint_failed', env_step, agent.step, error=str(save_error))
        raise
    finally:
        recorder.close(env_step, agent.step if agent is not None else 0)
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=0 if normal_exit else 1)
            except Exception:
                pass


if __name__ == '__main__':
    app.run(main)
