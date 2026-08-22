import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import jax
import numpy as np
from absl import app, flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.checkpoint import CheckpointManager
from jaxrl.env_names import get_environment_list
from jaxrl.envs import ParallelEnv
from jaxrl.experiment import ExperimentRecorder, collect_jax_memory_stats, summarize_tree
from jaxrl.logger import EpisodeRecorder, get_wandb_video
from jaxrl.normalizer import RewardNormalizer
from jaxrl.paper_alignment import resolve_paper_alignment
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.utils import Batch


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
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
flags.DEFINE_boolean(
    'paper_alignment', False,
    'Enable the paper-aligned L1/bootstrap/empirical-entropy preset.',
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
    'metaworld_reset_mode', 'frozen', ['frozen', 'recreate'],
    'Keep one MetaWorld rand_vec or reconstruct the task after each episode.',
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

    resolved_alignment = resolve_paper_alignment(
        FLAGS.paper_alignment,
        FLAGS.task_embedding_norm,
        FLAGS.return_bootstrap,
        FLAGS.entropy_correction,
    )
    config = FLAGS.flag_values_dict()
    config.update({f'resolved_{key}': value for key, value in resolved_alignment.items()})
    env_names = get_environment_list(FLAGS.env_names)
    resume_manifest = None
    resume_checkpoint = None
    existing_run_dir = None
    requested_run_id = FLAGS.run_id
    wandb_resume_id = None

    if FLAGS.resume_from:
        resume_checkpoint = CheckpointManager.resolve_recovery_checkpoint(FLAGS.resume_from)
        resume_manifest = CheckpointManager.read_manifest(resume_checkpoint)
        existing_run_dir = resume_manifest['run_dir']
        requested_run_id = resume_manifest['run_id']
        wandb_resume_id = resume_manifest.get('wandb_id')
        if resume_manifest['task_names'] != env_names:
            raise ValueError('checkpoint task names/order do not match --env_names')
        for key in ['env_names', 'seed', 'width_critic', 'updates_per_step',
                    'batch_size', 'replay_buffer_size', 'metaworld_reset_mode',
                    'resolved_task_embedding_norm', 'resolved_return_bootstrap',
                    'resolved_entropy_correction']:
            previous = resume_manifest.get('config', {}).get(key)
            current = config.get(key)
            if previous is not None and current is not None and previous != current:
                raise ValueError(f'checkpoint configuration mismatch for {key}: {previous} != {current}')

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
            env_names, seed=FLAGS.seed + 42,
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

        observations = env.reset()
        initialization_sec = time.perf_counter() - initialization_start
        recorder.record_event(
            'initialization_finished', env_step, agent.step,
            initialization_sec=initialization_sec,
            resumed=resume_checkpoint is not None,
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
                latest_update_info = agent.update(batches, FLAGS.updates_per_step, i)
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
