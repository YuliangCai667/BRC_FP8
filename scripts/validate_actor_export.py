#!/usr/bin/env python3
"""Fresh process: import only the copied package and a test-input/output fixture."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import jax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    # Remove the checkout completely; this process cannot import jaxrl/learner.
    root = Path(__file__).resolve().parents[1]
    sys.path[:] = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)]
    sys.path.insert(0, str(package.parent))
    module = importlib.import_module(package.name)
    codec = importlib.import_module(package.name + '.actor_export_codec')
    def forbidden(*args, **kwargs):
        raise AssertionError('loader/inference must not re-quantize kernels')
    codec.pack_export_kernel = forbidden
    codec.effective_export_kernel = forbidden
    policy = module.load_actor_export(package)
    with np.load(args.fixture, allow_pickle=False) as data:
        observations, ids, noise = data['observations'], data['task_ids'], data['noise']
        actual = policy.statistics(observations, ids, noise)
        report = dict(status='export_validated', fixture=str(args.fixture),
                      package_only_process=True, quantizer_calls_forbidden=True,
                      tasks_covered=np.unique(ids).tolist(), comparisons={})
        for key, value in actual.items():
            expected, value = data[key], np.asarray(value)
            atol = 1e-4 if key == 'log_prob' else 1e-6
            np.testing.assert_allclose(value, expected, rtol=1e-5, atol=atol, err_msg=key)
            report['comparisons'][key] = dict(bitwise_equal=bool(np.array_equal(value, expected)),
                max_abs_error=float(np.max(np.abs(value-expected))), rtol=1e-5, atol=atol)
        np.testing.assert_array_equal(policy.actions(observations, ids), actual['deterministic_actions'])
        np.testing.assert_array_equal(policy.actions(observations, ids, deterministic=False, noise=noise), actual['actions'])
        batch1 = [policy.statistics(observations[i:i+1],ids[i:i+1],noise[i:i+1]) for i in range(len(policy.manifest['task_names']))]
        report['batch1_comparisons'] = {}
        for key in actual:
            if 'batch1_'+key not in data:
                continue  # Older fixtures validate their recorded batch shape.
            expected = data['batch1_'+key]
            got = np.concatenate([np.asarray(row[key]) for row in batch1])
            atol = 1e-4 if key == 'log_prob' else 1e-6
            np.testing.assert_allclose(got,expected,rtol=1e-5,atol=atol,err_msg='batch1_'+key)
            report['batch1_comparisons'][key] = dict(bitwise_equal=bool(np.array_equal(got,expected)),
                                                    max_abs_error=float(np.max(np.abs(got-expected))))
        report['batch1_validated'] = len(report['batch1_comparisons']) == len(actual)
    # Batch-one benchmark includes on-the-fly weight decoding and synchronization.
    obs1, ids1 = observations[:1], ids[:1]
    jax.block_until_ready(policy.actions(obs1, ids1))
    times = []
    for _ in range(100):
        start = time.perf_counter()
        jax.block_until_ready(policy.actions(obs1, ids1))
        times.append(time.perf_counter() - start)
    report.update(batch1_latency_ms=dict(median=float(np.median(times)*1000), p95=float(np.percentile(times,95)*1000),
                                         repetitions=100, includes_python_dispatch=True),
                  runtime_persistent_parameter_bytes=policy.persistent_bytes,
                  runtime_arrays=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype),bytes=int(v.size*v.dtype.itemsize))
                                  for k,v in policy.arrays.items()],
                  runtime_memory=jax.devices()[0].memory_stats(), runtime_device=str(jax.devices()[0]),
                  full_bf16_kernel_cache=False,
                  package_bytes_before_validation_report=sum(p.stat().st_size for p in package.iterdir() if p.is_file()))
    assert not any(k.startswith('jaxrl') for k in sys.modules), 'fresh validation imported training code'
    (package / 'export_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'comparisons': report['comparisons']}), flush=True)


if __name__ == '__main__':
    main()
