import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jaxrl.experiment import ExperimentRecorder, collect_jax_memory_stats
from jaxrl.logger import EpisodeRecorder


class EpisodeRecorderTest(unittest.TestCase):
    def test_asynchronous_episode_events_and_summary(self):
        recorder = EpisodeRecorder(2, ["left", "right"])
        recorder.update(
            np.array([1.0, 2.0]), np.array([0, 1]),
            np.array([False, False]), np.array([False, False]), env_step=1,
        )
        events = recorder.update(
            np.array([3.0, 4.0]), np.array([1, 0]),
            np.array([True, False]), np.array([False, True]), env_step=2,
        )
        self.assertEqual([event["task_name"] for event in events], ["left", "right"])
        self.assertEqual([event["episode_return"] for event in events], [4.0, 6.0])
        self.assertEqual([event["episode_success"] for event in events], [1, 1])
        summary = recorder.interval_summary()
        np.testing.assert_array_equal(summary["episodes_by_task"], [1, 1])
        np.testing.assert_allclose(summary["episode_return_by_task"], [4.0, 6.0])
        self.assertEqual(summary["episodes_completed"], 2)


class ExperimentRecorderTest(unittest.TestCase):
    def test_jax_allocator_memory_metrics_are_distinct_from_device_memory(self):
        class FakeGpu:
            platform = "gpu"

            def memory_stats(self):
                return {
                    "bytes_in_use": 100 * 1024 ** 2,
                    "peak_bytes_in_use": 150 * 1024 ** 2,
                    "bytes_reserved": 200 * 1024 ** 2,
                    "bytes_limit": 400 * 1024 ** 2,
                    "num_allocs": 12,
                }

        metrics = collect_jax_memory_stats([FakeGpu()])
        self.assertEqual(metrics["jax_memory_bytes_in_use_mb"], 100.0)
        self.assertEqual(metrics["jax_memory_peak_bytes_in_use_mb"], 150.0)
        self.assertEqual(metrics["jax_memory_bytes_reserved_mb"], 200.0)
        self.assertEqual(metrics["jax_memory_active_to_reserved_ratio"], 0.5)
        self.assertEqual(metrics["jax_memory_active_to_limit_ratio"], 0.25)

    def test_local_files_exist_without_wandb(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = ExperimentRecorder(
                temp, "test", {"x": 1}, ["a", "b"], 7,
                run_id="run", system_metrics_interval_sec=0,
            )
            recorder.queue_episode_events([{
                "event": "episode_end", "env_step": 5, "task_id": 0,
                "task_name": "a", "episode_return": 3.0,
                "episode_success": 1, "episode_length": 2,
            }])
            recorder.record_train(5, 2, {"loss": 1.5, "by_task": np.array([1.0, 2.0])})
            recorder.record_eval(5, 2, {"return_by_task": np.array([3.0, 4.0])})
            recorder.close(5, 2)
            root = Path(temp) / "test" / "run"
            for name in ["metadata.json", "config.yaml", "episodes.jsonl",
                         "train_metrics.jsonl", "eval_metrics.jsonl", "events.jsonl"]:
                self.assertTrue((root / name).exists(), name)
            episode = json.loads((root / "episodes.jsonl").read_text().splitlines()[0])
            self.assertEqual(episode["global_transition"], 10)
            self.assertEqual(episode["task_name"], "a")

    def test_wandb_failure_does_not_break_local_recording(self):
        class FailingRun:
            def define_metric(self, *args, **kwargs):
                raise RuntimeError("offline")

            def log(self, *args, **kwargs):
                raise RuntimeError("offline")

        with tempfile.TemporaryDirectory() as temp:
            recorder = ExperimentRecorder(
                temp, "test", {}, ["a"], 0, run_id="run",
                wandb_run=FailingRun(), system_metrics_interval_sec=0,
            )
            recorder.record_train(1, 1, {"loss": 2.0})
            recorder.close(1, 1)
            lines = (Path(temp) / "test" / "run" / "train_metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
