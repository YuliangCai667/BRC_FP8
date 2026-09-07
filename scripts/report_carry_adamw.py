#!/usr/bin/env python3
"""Summarize observed CARRY-AdamW results without declaring a live run complete.

Reads only local recorder artifacts. The report remains explicitly incomplete
until terminal training, all scheduled evaluations, and final checkpoints exist.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/DMC_DOGS'
BASELINE = RUNS / 'brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_jitbarrier_v1_target_lag_fresh2_s42'
CANDIDATE = RUNS / 'brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_target_lag_adam_fp8_carry_v1_s42'


def rows(path):
    if not path.exists():
        return []
    text = path.read_text()
    # A writer may be appending the final line right now. Never swallow errors
    # in complete records, and do not equate a temporarily absent line to failure.
    return [json.loads(line) for line in text.splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def summarize_run(run):
    events = rows(run / 'events.jsonl')
    train = rows(run / 'train_metrics.jsonl')
    evaluations = rows(run / 'eval_metrics.jsonl')
    tensors = rows(run / 'tensor_stats.jsonl')
    # Use the recorder's config JSON inside a manifest if one exists; config.yaml
    # is the authoritative startup config before the first checkpoint.
    import yaml
    config = yaml.safe_load((run / 'config.yaml').read_text())
    horizon = int(config['max_steps'])
    latest_step = max([r.get('env_step', 0) for r in events + train + evaluations], default=0)
    finished = [r for r in events if r.get('event') == 'run_finished']
    final_event = finished[-1] if finished else None
    terminal_horizon = bool(final_event and final_event['env_step'] == horizon)
    evaluation_steps = {r['env_step'] for r in evaluations}
    expected_evaluations = set(range(config['eval_interval'], horizon + 1, config['eval_interval']))
    complete_evaluations = expected_evaluations <= evaluation_steps
    finite_evaluations = all(math.isfinite(r['return_mean']) and
        all(math.isfinite(v) for v in r['return_by_task']) for r in evaluations)
    failures = [r for r in events if 'failed' in r.get('event', '') or r.get('event') == 'run_interrupted']
    checkpoint_data = {}
    for kind in ('analysis', 'recovery'):
        path = run / 'checkpoints' / f'{kind}_step_{horizon:012d}'
        if (path / 'COMPLETE').exists():
            manifest = json.loads((path / 'manifest.json').read_text())
            checkpoint_data[kind] = {
                'path': str(path), 'env_step': manifest['env_step'],
                'update_step': manifest['update_step'],
                'bytes': directory_bytes(path),
                'model_msgpack_bytes': sum(p.stat().st_size for p in path.glob('*.msgpack')),
                'critic_msgpack_bytes': (path / 'critic.msgpack').stat().st_size,
                'replay_bytes': directory_bytes(path / 'replay_buffer'),
                'includes_optimizer': manifest.get('includes_optimizer', False),
                'includes_replay_buffer': manifest.get('includes_replay_buffer', False),
                'state_bytes': manifest['state_bytes'],
                'critic_optimizer_inventory': manifest.get('critic_optimizer_inventory'),
            }
    checkpoint_complete = (
        len(checkpoint_data) == 2
        and checkpoint_data['recovery']['includes_optimizer']
        and checkpoint_data['recovery']['includes_replay_buffer']
    )
    finite_update_records = [r for r in train if 'update_nan_count' in r]
    numerical_failures = [r['env_step'] for r in finite_update_records
        if r.get('update_nan_count', 0) or r.get('update_inf_count', 0)]
    tensor_nonfinite = sum(r['nan_count'] + r['inf_count'] for r in tensors)
    moment_health = {}
    for label, prefix, suffix in (
        ('v_negative_fraction_max', 'critic_moments/nu/', 'negative_fraction'),
        ('moment_nonfinite_count_max', 'critic_moments/', 'nonfinite_count'),
        ('carry_saturation_fraction_max', 'critic_moments/', 'carry_saturation_fraction'),
        ('carry_zero_fraction_max', 'critic_moments/', 'carry_zero_fraction'),
    ):
        values = [r['mean'] for r in tensors if r['tensor'].startswith(prefix)
                  and r['tensor'].endswith(suffix)]
        moment_health[label] = max(values) if values else None
    profile = [r for r in train if 'profile_update_ms' in r]
    per_update = [r['profile_update_ms'] / config['updates_per_step'] for r in profile]
    # The first profile window compiles the diagnostics graph. Preserve its
    # cost and all-window mean as well as the subsequent-window measurement.
    warm = per_update[1:]
    peak = [r['jax_memory_peak_bytes_in_use_mb'] for r in events + train
            if 'jax_memory_peak_bytes_in_use_mb' in r]
    returns = [r['return_mean'] for r in evaluations]
    quantized_moments_healthy = True
    if config.get('critic_optimizer_state', 'fp32') == 'fp8_carry':
        quantized_moments_healthy = all(moment_health[key] == 0 for key in (
            'v_negative_fraction_max', 'moment_nonfinite_count_max',
            'carry_saturation_fraction_max'))
        quantized_moments_healthy &= (
            moment_health['carry_zero_fraction_max'] is not None
            and moment_health['carry_zero_fraction_max'] < 1
        )
    observed_complete = (terminal_horizon and complete_evaluations and checkpoint_complete
                         and finite_evaluations and not numerical_failures
                         and tensor_nonfinite == 0 and not failures
                         and quantized_moments_healthy)
    return {
        'run_dir': str(run.resolve()), 'run_id': run.name,
        'config': config, 'horizon': horizon, 'latest_env_step': latest_step,
        'latest_actual_learner_updates': max([r.get('update_step', 1) for r in events + train], default=1) - 1,
        'complete': observed_complete,
        'completion_evidence': {
            'terminal_event_at_horizon': terminal_horizon,
            'all_scheduled_evaluations_present': complete_evaluations,
            'missing_evaluation_steps': sorted(expected_evaluations - evaluation_steps),
            'finite_evaluations': finite_evaluations,
            'final_analysis_and_recovery_complete': checkpoint_complete,
            'recorded_quantized_moments_healthy': quantized_moments_healthy,
            'recorded_failures': failures,
        },
        'return': {'evaluation_count': len(evaluations),
                   'latest': returns[-1] if returns else None,
                   'latest_step': evaluations[-1]['env_step'] if evaluations else None,
                   'best': max(returns) if returns else None,
                   'tail3_mean': statistics.mean(returns[-3:]) if len(returns) >= 3 else None,
                   'latest_by_task': evaluations[-1]['return_by_task'] if evaluations else None},
        'numerics': {'logged_update_rows': len(finite_update_records),
                     'nonfinite_update_steps': numerical_failures,
                     'tensor_records': len(tensors), 'tensor_nonfinite_count': tensor_nonfinite,
                     'moments': moment_health},
        'performance': {
            'jax_allocator_peak_mib_observed': max(peak) if peak else None,
            'profile_windows': len(profile),
            'profile_ms_per_actual_learner_update_mean_all': statistics.mean(per_update) if per_update else None,
            'first_profile_window_ms_per_update': per_update[0] if per_update else None,
            'subsequent_profile_ms_per_update_mean': statistics.mean(warm) if warm else None,
            'subsequent_profile_ms_per_update_median': statistics.median(warm) if warm else None,
            'total_wall_time_sec': final_event.get('total_wall_time_sec') if final_event else None,
            'timing_notes': 'Synchronized recorder windows include the existing diagnostic update at each cadence. The first window includes compilation. The CARRY-AdamW run shares GPU2; this comparison is not a contention-controlled speedup benchmark.',
        },
        'final_checkpoints': checkpoint_data,
    }


def report(baseline, candidate):
    a, b = summarize_run(baseline), summarize_run(candidate)
    keys = ('seed', 'max_steps', 'env_names', 'eval_seed_offset', 'start_training',
            'batch_size', 'updates_per_step', 'width_critic', 'critic_precision',
            'target_critic_precision', 'fp8_resident_carry', 'fp8_resident_canonicalization',
            'fp8_all_dense_kernels', 'fp8_input_dense_kernel', 'fp8_output_dense_kernel',
            'fp8_amax_history_length', 'replay_buffer_size', 'eval_interval', 'eval_episodes',
            'resolved_task_embedding_norm', 'resolved_return_bootstrap', 'resolved_entropy_correction',
            'resolved_online_fp8_backward', 'resolved_online_fp8_weight_scaling',
            'resolved_fp8_code_materialization')
    defaults = {'fp8_all_dense_kernels': False, 'fp8_input_dense_kernel': False,
                'fp8_output_dense_kernel': False}
    differences = {k: [a['config'].get(k, defaults.get(k)), b['config'].get(k, defaults.get(k))]
                   for k in keys if a['config'].get(k, defaults.get(k)) != b['config'].get(k, defaults.get(k))}
    return {'baseline': a, 'candidate': b, 'material_protocol_differences': differences,
            'comparison_complete': a['complete'] and b['complete'] and not differences,
            'notes': ['One seed; no across-seed uncertainty estimate.',
                      'State payload bytes, serialized checkpoint bytes and allocator peak are distinct.',
                      'Missing final evidence keeps this report incomplete.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/carry_adamw_validation/run_comparison.json')
    args = parser.parse_args()
    result = report(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'report': str(args.output.resolve()),
                      'comparison_complete': result['comparison_complete'],
                      'candidate_env_step': result['candidate']['latest_env_step'],
                      'candidate_return': result['candidate']['return']['latest']}))
