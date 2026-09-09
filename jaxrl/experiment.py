"""Low-overhead, local-first experiment recording utilities."""

from __future__ import annotations

import csv
import functools
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np


SCHEMA_VERSION = 1


def collect_jax_memory_stats(devices) -> dict[str, float]:
    """Read allocator counters after an existing synchronization boundary."""
    fields = [
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_reserved",
        "peak_bytes_reserved",
        "bytes_limit",
        "largest_alloc_size",
        "largest_free_block_bytes",
        "num_allocs",
    ]
    metrics: dict[str, float] = {"jax_memory_stats_available": 0}
    totals = {field: 0 for field in fields}
    available_devices = 0
    for logical_index, device in enumerate(devices):
        if getattr(device, "platform", "") != "gpu":
            continue
        try:
            stats = device.memory_stats() or {}
        except Exception:
            continue
        if "bytes_in_use" not in stats:
            continue
        available_devices += 1
        for field in fields:
            value = stats.get(field)
            if value is None or value < 0:
                continue
            totals[field] += value
            suffix = "count" if field == "num_allocs" else "mb"
            converted = float(value) if field == "num_allocs" else float(value) / 1024 ** 2
            metrics[f"jax_memory/device_{logical_index}/{field}_{suffix}"] = converted
    if available_devices:
        metrics["jax_memory_stats_available"] = 1
        metrics["jax_memory_device_count"] = available_devices
        for field, value in totals.items():
            if value <= 0 and field not in ("bytes_in_use", "num_allocs"):
                continue
            suffix = "count" if field == "num_allocs" else "mb"
            metrics[f"jax_memory_{field}_{suffix}"] = (
                float(value) if field == "num_allocs" else float(value) / 1024 ** 2
            )
        in_use = totals["bytes_in_use"]
        reserved = totals["bytes_reserved"]
        limit = totals["bytes_limit"]
        if reserved > 0:
            metrics["jax_memory_active_to_reserved_ratio"] = in_use / reserved
        if limit > 0:
            metrics["jax_memory_active_to_limit_ratio"] = in_use / limit
    return metrics


def _path_name(path) -> str:
    parts = []
    for entry in path:
        value = getattr(entry, "key", getattr(entry, "name", getattr(entry, "idx", entry)))
        parts.append(str(value))
    return "/".join(parts)


@functools.lru_cache(maxsize=8)
def _summary_kernel(sample_size: int):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def summarize(array):
        array = jnp.asarray(array)
        flat = array.reshape(-1).astype(jnp.float32)
        stride = max(flat.size // sample_size, 1)
        sample = flat[::stride][:sample_size]
        finite = jnp.isfinite(flat)
        finite_values = jnp.where(finite, flat, 0.0)
        sample = jnp.nan_to_num(sample)
        quantiles = jnp.percentile(sample, jnp.asarray([1.0, 50.0, 99.0, 99.9]))
        count = jnp.maximum(finite.sum(), 1)
        mean = finite_values.sum() / count
        variance = jnp.square(jnp.where(finite, flat - mean, 0.0)).sum() / count
        return jnp.asarray([
            mean,
            jnp.sqrt(variance),
            jnp.min(jnp.where(finite, flat, jnp.inf)),
            jnp.max(jnp.where(finite, flat, -jnp.inf)),
            jnp.max(jnp.abs(finite_values)),
            quantiles[0], quantiles[1], quantiles[2], quantiles[3],
            jnp.abs(finite_values).sum(),
            jnp.sqrt(jnp.square(finite_values).sum()),
            (flat == 0).mean(),
            jnp.isnan(flat).sum(),
            jnp.isinf(flat).sum(),
        ])
    return summarize


def summarize_tree(tree: Any, prefix: str = "", sample_size: int = 4096) -> dict[str, dict]:
    """Reduce large device tensors to small scalar summaries before host transfer."""
    import jax

    sample_size = max(int(sample_size), 32)
    summarize = _summary_kernel(sample_size)

    result = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        try:
            if not hasattr(leaf, "shape") or np.prod(leaf.shape, dtype=np.int64) == 0:
                continue
            values = np.asarray(jax.device_get(summarize(leaf)), dtype=np.float64)
            name = "/".join(part for part in [prefix, _path_name(path)] if part)
            result[name] = {
                "dtype": str(leaf.dtype),
                "shape": list(leaf.shape),
                "mean": values[0], "std": values[1], "min": values[2], "max": values[3],
                "absmax": values[4], "p1": values[5], "p50": values[6],
                "p99": values[7], "p99_9": values[8], "l1_norm": values[9],
                "l2_norm": values[10], "zero_fraction": values[11],
                "nan_count": int(values[12]), "inf_count": int(values[13]),
                "quantiles_from_sample": min(int(np.prod(leaf.shape)), sample_size),
            }
        except (TypeError, ValueError):
            continue
    return result


def jsonable(value: Any):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    array = np.asarray(value)
    if array.ndim == 0:
        return jsonable(array.item())
    return jsonable(array.tolist())


def safe_command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class JsonlSink:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1024 * 1024)
        self._lock = threading.Lock()

    def write(self, rows):
        with self._lock:
            for row in rows:
                self._file.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

    def flush(self):
        with self._lock:
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.flush()
            self._file.close()


class SystemMonitor:
    """Samples host and NVIDIA metrics without mandatory extra dependencies."""

    FIELDS = [
        "timestamp", "wall_time_sec", "cpu_percent", "process_cpu_percent",
        "process_rss_mb", "ram_used_mb", "ram_total_mb", "gpu_index",
        "gpu_util_percent", "gpu_device_memory_used_mb", "gpu_device_memory_total_mb",
        "gpu_power_w", "gpu_temperature_c",
    ]

    def __init__(self, path: Path, interval_sec: float, start_time: float):
        self.path = path
        self.interval_sec = max(float(interval_sec), 1.0)
        self.start_time = start_time
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_cpu = None
        self._last_proc = None
        self._warned_gpu = False
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    @staticmethod
    def _read_cpu_ticks():
        with open("/proc/stat", encoding="utf-8") as file:
            values = [int(x) for x in file.readline().split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    @staticmethod
    def _read_proc_ticks():
        fields = Path(f"/proc/{os.getpid()}/stat").read_text().split()
        return int(fields[13]) + int(fields[14])

    @staticmethod
    def _read_memory():
        entries = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            entries[key] = int(value.strip().split()[0]) / 1024.0
        rss_kb = 0.0
        for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = float(line.split()[1])
                break
        return rss_kb / 1024.0, entries["MemTotal"] - entries["MemAvailable"], entries["MemTotal"]

    @staticmethod
    def _visible_gpu_indices():
        result = []
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","):
            if token.strip().isdigit():
                result.append(int(token.strip()))
        return result

    def _gpu_rows(self):
        if self._nvml is not None:
            try:
                visible = self._visible_gpu_indices()
                indices = visible or list(range(self._nvml.nvmlDeviceGetCount()))
                rows = []
                for index in indices:
                    handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
                    utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    power = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    temperature = self._nvml.nvmlDeviceGetTemperature(
                        handle, self._nvml.NVML_TEMPERATURE_GPU
                    )
                    rows.append([
                        float(index), float(utilization.gpu), memory.used / 1024 ** 2,
                        memory.total / 1024 ** 2, power, float(temperature),
                    ])
                return rows
            except Exception:
                self._nvml = None
        output = safe_command([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        if not output:
            if not self._warned_gpu:
                print("[recorder] warning: GPU system metrics are unavailable")
                self._warned_gpu = True
            return []
        visible = self._visible_gpu_indices()
        rows = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6 or not parts[0].isdigit():
                continue
            if visible and int(parts[0]) not in visible:
                continue
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue
        return rows

    def _sample(self):
        total, idle = self._read_cpu_ticks()
        proc = self._read_proc_ticks()
        cpu_percent = process_cpu_percent = 0.0
        if self._last_cpu is not None:
            total_delta = max(total - self._last_cpu[0], 1)
            idle_delta = idle - self._last_cpu[1]
            proc_delta = proc - self._last_proc
            cpu_percent = 100.0 * (1.0 - idle_delta / total_delta)
            process_cpu_percent = 100.0 * proc_delta / total_delta * (os.cpu_count() or 1)
        self._last_cpu = (total, idle)
        self._last_proc = proc
        rss, ram_used, ram_total = self._read_memory()
        common = {
            "timestamp": time.time(),
            "wall_time_sec": time.perf_counter() - self.start_time,
            "cpu_percent": cpu_percent,
            "process_cpu_percent": process_cpu_percent,
            "process_rss_mb": rss,
            "ram_used_mb": ram_used,
            "ram_total_mb": ram_total,
        }
        gpu_rows = self._gpu_rows() or [["", "", "", "", "", ""]]
        return [{
            **common,
            "gpu_index": gpu[0], "gpu_util_percent": gpu[1],
                    "gpu_device_memory_used_mb": gpu[2], "gpu_device_memory_total_mb": gpu[3],
            "gpu_power_w": gpu[4], "gpu_temperature_c": gpu[5],
        } for gpu in gpu_rows]

    def _run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8", buffering=1) as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDS)
            if new_file:
                writer.writeheader()
            while not self._stop.is_set():
                try:
                    writer.writerows(self._sample())
                except Exception as error:
                    print(f"[recorder] warning: system monitor sample failed: {error}")
                self._stop.wait(self.interval_sec)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="system-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self.interval_sec + 1.0, 5.0))
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class ExperimentRecorder:
    """Buffered local recorder with optional, failure-isolated W&B mirroring."""

    def __init__(
        self,
        run_root: str,
        env_group: str,
        config: Mapping[str, Any],
        task_names: list[str],
        seed: int,
        run_id: str = "auto",
        wandb_run=None,
        system_metrics_interval_sec: float = 10.0,
        existing_run_dir: Optional[str] = None,
        start_monotonic: Optional[float] = None,
    ):
        self.start_monotonic = time.perf_counter() if start_monotonic is None else start_monotonic
        self.task_names = list(task_names)
        self.num_tasks = len(task_names)
        self.seed = int(seed)
        self.wandb_run = wandb_run
        self.resuming = bool(existing_run_dir)
        if existing_run_dir:
            self.run_dir = Path(existing_run_dir).resolve()
            self.run_id = self.run_dir.name
        else:
            if run_id == "auto":
                run_id = time.strftime("%Y%m%d_%H%M%S") + f"_s{seed}_{uuid.uuid4().hex[:8]}"
            self.run_id = run_id
            self.run_dir = (Path(run_root) / env_group / run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        self._episodes = JsonlSink(self.run_dir / "episodes.jsonl")
        self._train = JsonlSink(self.run_dir / "train_metrics.jsonl")
        self._eval = JsonlSink(self.run_dir / "eval_metrics.jsonl")
        self._tensor = JsonlSink(self.run_dir / "tensor_stats.jsonl")
        self._events = JsonlSink(self.run_dir / "events.jsonl")
        self._tensor_providers: dict[str, Callable[..., Mapping[str, Any]]] = {}
        self._pending_episode_events: list[dict] = []

        config_dict = dict(config)
        if self.resuming:
            resume_stamp = time.strftime("%Y%m%d_%H%M%S")
            resume_path = self.run_dir / "artifacts" / f"resume_config_{resume_stamp}.json"
            resume_path.write_text(json.dumps(jsonable(config_dict), indent=2), encoding="utf-8")
        else:
            self._write_config(config_dict)
            self._write_metadata(config_dict)
        self.system_monitor = None
        if system_metrics_interval_sec > 0:
            self.system_monitor = SystemMonitor(
                self.run_dir / "system_metrics.csv", system_metrics_interval_sec, self.start_monotonic
            )
            self.system_monitor.start()
        if self.wandb_run is not None:
            try:
                self.wandb_run.define_metric("env_step")
                self.wandb_run.define_metric("train/*", step_metric="env_step")
                self.wandb_run.define_metric("train_by_task/*", step_metric="env_step")
                self.wandb_run.define_metric("eval/*", step_metric="env_step")
                self.wandb_run.define_metric("eval_by_task/*", step_metric="env_step")
                self.wandb_run.define_metric("tensor/*", step_metric="env_step")
            except Exception as error:
                print(f"[recorder] warning: W&B metric definition failed: {error}")

    @property
    def wall_time_sec(self):
        return time.perf_counter() - self.start_monotonic

    def common_fields(self, env_step: int, update_step: int):
        return {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "seed": self.seed,
            "env_step": int(env_step), "global_transition": int(env_step) * self.num_tasks,
            "update_step": int(update_step), "wall_time_sec": self.wall_time_sec,
            "timestamp": time.time(),
        }

    def _write_config(self, config):
        try:
            import yaml
            text = yaml.safe_dump(jsonable(config), sort_keys=True)
        except Exception:
            text = json.dumps(jsonable(config), indent=2, ensure_ascii=False)
        (self.run_dir / "config.yaml").write_text(text, encoding="utf-8")

    def _write_metadata(self, config):
        versions = {}
        for package in ["jax", "jaxlib", "flax", "optax", "gymnasium", "metaworld", "wandb", "numpy"]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        ptxas_path = shutil.which("ptxas")
        metadata = {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id,
            "created_at": time.time(), "command": " ".join(shlex.quote(arg) for arg in sys.argv),
            "cwd": os.getcwd(), "hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version, "versions": versions, "task_names": self.task_names,
            "seed": self.seed, "git_commit": safe_command(["git", "rev-parse", "HEAD"]),
            "git_status": safe_command(["git", "status", "--short"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "config": jsonable(config),
            "cuda_environment": {
                "CUDA_ROOT": os.environ.get("CUDA_ROOT"),
                "CUDA_HOME": os.environ.get("CUDA_HOME"),
                "ptxas_path": ptxas_path,
                "ptxas_version": safe_command([ptxas_path, "--version"]) if ptxas_path else None,
            },
            "jax_memory_environment": {
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
                "XLA_CLIENT_MEM_FRACTION": os.environ.get("XLA_CLIENT_MEM_FRACTION"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            },
        }
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        git_diff = safe_command(["git", "diff", "--binary"])
        if git_diff:
            (self.run_dir / "artifacts" / "git_diff.patch").write_text(git_diff, encoding="utf-8")
        freeze = safe_command([sys.executable, "-m", "pip", "freeze"])
        if freeze:
            (self.run_dir / "artifacts" / "pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")
        snapshot_root = self.run_dir / "artifacts" / "source_snapshot"
        sources = [Path("train.py"), *Path("jaxrl").rglob("*.py")]
        if config.get('actor_training_recipe', 'fp32') != 'fp32':
            sources.extend(Path('deployment').rglob('*.py'))
            sources.extend(Path('scripts').glob('*actor*.py'))
            sources.extend([Path('scripts/run_actor_qat_fp8_export.sh'), Path('configs/actor_qat_fp8_export_v1.yaml')])
        for source in sources:
            if source.is_file():
                target = snapshot_root / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def register_tensor_stats_provider(self, name: str, provider: Callable[..., Mapping[str, Any]]):
        self._tensor_providers[name] = provider

    def register_fp8_stats_provider(self, provider: Callable[..., Mapping[str, Any]]):
        self.register_tensor_stats_provider("fp8", provider)

    def queue_episode_events(self, events: list[dict]):
        self._pending_episode_events.extend(events)

    def flush_episode_events(self, update_step: int):
        rows = []
        for event in self._pending_episode_events:
            env_step = event.get("env_step") or 0
            rows.append({**self.common_fields(env_step, update_step), **event})
        if rows:
            self._episodes.write(rows)
            self._pending_episode_events.clear()

    @staticmethod
    def _flatten_wandb(prefix: str, metrics: Mapping[str, Any], task_names: list[str]):
        result = {}
        for key, value in metrics.items():
            array = np.asarray(value)
            if array.ndim == 0:
                scalar = array.item()
                if isinstance(scalar, (int, float, np.number)) and np.isfinite(scalar):
                    result[f"{prefix}/{key}"] = scalar
            elif array.ndim == 1 and len(array) == len(task_names):
                for task_name, item in zip(task_names, array):
                    if np.isfinite(item):
                        result[f"{prefix}_by_task/{task_name}/{key}"] = item
        return result

    def _wandb_log(self, payload: dict, env_step: int):
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log({"env_step": int(env_step), **payload})
        except Exception as error:
            print(f"[recorder] warning: W&B logging failed: {error}")

    def record_train(self, env_step: int, update_step: int, metrics: Mapping[str, Any]):
        row = {
            **self.common_fields(env_step, update_step),
            "update_metric_aggregation": "periodic_sample",
            "episode_metric_aggregation": "interval_summary",
            **metrics,
        }
        self._train.write([row])
        payload = self._flatten_wandb("train", metrics, self.task_names)
        payload.update(timestep=int(env_step), global_transition=int(env_step) * self.num_tasks,
                       wall_time_sec=self.wall_time_sec)
        self._wandb_log(payload, env_step)

    def record_eval(self, env_step: int, update_step: int, metrics: Mapping[str, Any]):
        self._eval.write([{**self.common_fields(env_step, update_step), **metrics}])
        self._wandb_log(self._flatten_wandb("eval", metrics, self.task_names), env_step)

    def record_tensor_stats(self, env_step: int, update_step: int, stats: Mapping[str, Any]):
        rows = [{**self.common_fields(env_step, update_step), "tensor": name, **value}
                for name, value in stats.items()]
        if rows:
            self._tensor.write(rows)
            absmax = [float(value.get("absmax", 0.0)) for value in stats.values()
                      if isinstance(value, Mapping)]
            nan_count = sum(int(value.get("nan_count", 0)) for value in stats.values()
                            if isinstance(value, Mapping))
            inf_count = sum(int(value.get("inf_count", 0)) for value in stats.values()
                            if isinstance(value, Mapping))
            self._wandb_log({
                "tensor/tensor_count": len(rows),
                "tensor/max_absmax": max(absmax, default=0.0),
                "tensor/nan_count": nan_count,
                "tensor/inf_count": inf_count,
            }, env_step)

    def collect_registered_tensor_stats(self, **kwargs):
        stats = {}
        for name, provider in self._tensor_providers.items():
            try:
                provided = provider(**kwargs)
                stats.update({f"{name}/{key}": value for key, value in provided.items()})
            except Exception as error:
                self.record_event("tensor_provider_failed", error=str(error), provider=name)
        return stats

    def record_event(self, event: str, env_step: int = 0, update_step: int = 0, **fields):
        self._events.write([{**self.common_fields(env_step, update_step), "event": event, **fields}])

    def flush(self, update_step: int = 0):
        self.flush_episode_events(update_step)
        for sink in [self._episodes, self._train, self._eval, self._tensor, self._events]:
            sink.flush()

    def close(self, env_step: int = 0, update_step: int = 0):
        self.record_event("run_finished", env_step, update_step, total_wall_time_sec=self.wall_time_sec)
        self.flush(update_step)
        if self.system_monitor is not None:
            self.system_monitor.stop()
        for sink in [self._episodes, self._train, self._eval, self._tensor, self._events]:
            sink.close()
