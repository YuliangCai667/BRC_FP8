import os

os.environ['MUJOCO_GL'] = 'egl'

import time

import jax
import numpy as np
from absl import app, flags

from jaxrl.agent.training_learner import TrainingBRC as BRC
from jaxrl.checkpoint import CheckpointManager, validate_checkpoint_config
from jaxrl.env_names import get_environment_list
from jaxrl.envs import ParallelEnv
from jaxrl.experiment import ExperimentRecorder, collect_jax_memory_stats
from jaxrl.logger import EpisodeRecorder, get_wandb_video
from jaxrl.normalizer import RewardNormalizer
from jaxrl.optimizers import optimizer_state_inventory
from jaxrl.paper_alignment import resolve_paper_alignment
from jaxrl.replay_buffer import ParallelReplayBuffer


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
flags.DEFINE_integer('analysis_checkpoint_interval', 50000, 'Parameter checkpoint interval; 0 disables.')
flags.DEFINE_integer('recovery_checkpoint_interval', 100000, 'Recovery checkpoint interval; 0 disables.')
flags.DEFINE_integer('keep_last_analysis_checkpoints', 2, 'Recent analysis checkpoints to retain.')
flags.DEFINE_integer('keep_last_recovery_checkpoints', 1, 'Recent recovery checkpoints to retain.')
flags.DEFINE_boolean('save_replay_buffer', True, 'Include valid replay data in recovery checkpoints.')
flags.DEFINE_string('resume_from', '', 'Recovery checkpoint or run directory to resume.')


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


def main(_):
    program_start = time.perf_counter()
    interval_flags = {
        'metrics_interval': FLAGS.metrics_interval,
        'metrics_flush_interval': FLAGS.metrics_flush_interval,
        'analysis_checkpoint_interval': FLAGS.analysis_checkpoint_interval,
        'recovery_checkpoint_interval': FLAGS.recovery_checkpoint_interval,
    }
    for name, value in interval_flags.items():
        if value < 0:
            raise ValueError(f'--{name} must be >= 0')
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
    config = FLAGS.flag_values_dict()
    config.update({
        'training_runtime': 'pure_training_v1',
        'carry_gain': 16.0,
        'optimizer_moment_block_size': 128,
        'optimizer_moment_carry_gain': 16.0,
        'carry_dtype': 'float8_e4m3fn',
        'resolved_fp8_code_materialization': (
            'optimization_barrier_after_e4m3_cast_v1'
        ),
    })
    config.update({f'resolved_{key}': value for key, value in resolved_alignment.items()})
    config.update({
        'resolved_online_fp8_compute_scope': (
            'residual_dense_kernels'
            if FLAGS.critic_precision in (
                'fp8_direct', 'fp8_resident', 'fp8_current_master'
            )
            else 'none'
        ),
        'resolved_online_fp8_storage_scope': (
            'residual_dense_kernels'
            if FLAGS.critic_precision == 'fp8_resident'
            else 'none'
        ),
        'resolved_online_parameter_storage': (
            (
                'e4m3_main_and_e4m3_carry_residual_dense_kernels_otherwise_fp32'
                if FLAGS.fp8_resident_carry
                else 'e4m3_residual_dense_kernels_otherwise_fp32'
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
            'residual_dense_kernels' if target_fp8_compute else 'none'
        ),
        'resolved_target_fp8_storage_scope': (
            'residual_dense_kernels'
            if FLAGS.target_critic_precision == 'fp8_resident'
            else (
                'residual_dense_lag_state'
                if FLAGS.target_critic_precision == 'fp8_lag'
                else 'none'
            )
        ),
        'resolved_target_parameter_storage': (
            'e4m3_residual_dense_kernels_otherwise_fp32'
            if FLAGS.target_critic_precision == 'fp8_resident'
            else (
                'e4m3_lag_residual_dense_kernels_otherwise_fp32'
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
            entity='',
            project='',
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
        system_metrics_interval_sec=0,
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
            critic_optimizer_state=FLAGS.critic_optimizer_state,
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
        best_eval_success = -np.inf
        window_start = time.perf_counter()
        window_start_step = env_step

        for i in range(env_step + 1, FLAGS.max_steps + 1):
            env_step = i

            actions = (
                env.action_space.sample()
                if i < FLAGS.start_training
                else agent.sample_actions(observations, temperature=1.0)
            )

            next_observations, rewards, terms, truns, goals = env.step(actions)

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
                batches = replay_buffer.sample(FLAGS.batch_size, FLAGS.updates_per_step)
                batches = reward_normalizer.normalize(
                    batches,
                    agent.get_temperature(),
                    task_entropies=agent.get_task_entropies(),
                )

                start = time.perf_counter()
                latest_update_info = agent.update(
                    batches,
                    FLAGS.updates_per_step,
                    i,
                )
                if first_update_sec is None:
                    _block_tree(latest_update_info)
                    first_update_sec = time.perf_counter() - start
                    recorder.record_event(
                        'first_update_finished', i, agent.step,
                        jit_and_first_update_sec=first_update_sec,
                    )
                    window_start = time.perf_counter()
                    window_start_step = i

            metrics_due = FLAGS.metrics_interval > 0 and i % FLAGS.metrics_interval == 0
            if metrics_due:
                if latest_update_info is not None:
                    _block_tree(latest_update_info)
                now = time.perf_counter()
                active_sec = max(now - window_start, 1e-9)
                active_steps = i - window_start_step
                episode_summary = episode_recorder.interval_summary(reset=True)
                train_metrics = {
                    **(_host_metrics(latest_update_info) if latest_update_info is not None else {}),
                    **episode_summary,
                    'replay_buffer_size': replay_buffer.size,
                    'window_train_sec': active_sec,
                    'env_steps_per_sec': active_steps / active_sec,
                    'transitions_per_sec': active_steps * num_tasks / active_sec,
                    'updates_per_sec': active_steps * FLAGS.updates_per_step / active_sec,
                    'jit_and_first_update_sec': first_update_sec if first_update_sec is not None else np.nan,
                    **collect_jax_memory_stats(resource_devices),
                }
                recorder.record_train(i, agent.step, train_metrics)
                recorder.flush(agent.step)
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
                eval_stats = eval_env.evaluate(
                    agent, num_episodes=FLAGS.eval_episodes, temperature=0.0, render=FLAGS.render
                )
                eval_sec = time.perf_counter() - eval_start
                renders = eval_stats.pop('renders', None)
                eval_metrics = {
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
            )
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
                    )
                    manifest = CheckpointManager.read_manifest(path)
                    recorder.record_event(
                        'recovery_checkpoint_saved', i, agent.step,
                        path=str(path), checkpoint_sec=time.perf_counter() - checkpoint_start,
                        includes_replay_buffer=manifest['includes_replay_buffer'],
                        degraded_reason=manifest.get('degraded_reason'),
                    )
                except Exception as error:
                    recorder.record_event('recovery_checkpoint_failed', i, agent.step, error=str(error))
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
            if FLAGS.recovery_checkpoint_interval > 0:
                checkpoint_manager.save_recovery(
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
            print(f'[checkpoint] warning: final checkpoint failed: {error}')
        normal_exit = True
    except BaseException as error:
        recorder.record_event(
            'run_interrupted', env_step, agent.step if agent is not None else 0,
            error_type=type(error).__name__, error=str(error),
        )
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
