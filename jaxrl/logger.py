"""Episode accounting and legacy evaluation logging helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def log_to_wandb(step: int, infos: dict, suffix: str = ""):
    """Compatibility helper used by older entrypoints."""
    import wandb

    dict_to_log = {"timestep": step}
    for info_key, values in infos.items():
        values = np.atleast_1d(np.asarray(values))
        for task_id, value in enumerate(values):
            dict_to_log[f"task{task_id}/{info_key}{suffix}"] = value
    wandb.log(dict_to_log, step=step)


def get_wandb_video(renders: np.ndarray, fps: int = 15):
    import wandb

    return [wandb.Video(render, fps=fps, format="mp4") for render in renders]


class EpisodeRecorder:
    """Tracks every completed episode without performing I/O on environment steps."""

    def __init__(self, num_tasks: int, task_names: Optional[Iterable[str]] = None):
        self.num_tasks = int(num_tasks)
        self.task_names = list(task_names or [f"task_{i}" for i in range(num_tasks)])
        if len(self.task_names) != self.num_tasks:
            raise ValueError("task_names must have one entry per task")

        self.current_returns = np.zeros(self.num_tasks, dtype=np.float64)
        self.current_success = np.zeros(self.num_tasks, dtype=np.bool_)
        self.current_lengths = np.zeros(self.num_tasks, dtype=np.int64)
        self.interval_return_sums = np.zeros(self.num_tasks, dtype=np.float64)
        self.interval_success_sums = np.zeros(self.num_tasks, dtype=np.float64)
        self.interval_length_sums = np.zeros(self.num_tasks, dtype=np.int64)
        self.interval_counts = np.zeros(self.num_tasks, dtype=np.int64)
        self.total_counts = np.zeros(self.num_tasks, dtype=np.int64)
        self._pending_events: list[dict] = []

    def update(
        self,
        rewards: np.ndarray,
        goals: np.ndarray,
        terminals: np.ndarray,
        truncates: np.ndarray,
        env_step: Optional[int] = None,
    ) -> list[dict]:
        rewards = np.asarray(rewards)
        goals = np.asarray(goals)
        done = np.logical_or(terminals, truncates)

        self.current_returns += rewards
        self.current_success |= goals > 0
        self.current_lengths += 1

        completed = []
        for task_id in np.flatnonzero(done):
            episode_return = float(self.current_returns[task_id])
            episode_success = int(self.current_success[task_id])
            episode_length = int(self.current_lengths[task_id])
            event = {
                "event": "episode_end",
                "task_id": int(task_id),
                "task_name": self.task_names[task_id],
                "episode_return": episode_return,
                "episode_success": episode_success,
                "episode_length": episode_length,
                "env_step": None if env_step is None else int(env_step),
            }
            completed.append(event)
            self._pending_events.append(event)
            self.interval_return_sums[task_id] += episode_return
            self.interval_success_sums[task_id] += episode_success
            self.interval_length_sums[task_id] += episode_length
            self.interval_counts[task_id] += 1
            self.total_counts[task_id] += 1
            self.current_returns[task_id] = 0.0
            self.current_success[task_id] = False
            self.current_lengths[task_id] = 0
        return completed

    def drain_events(self) -> list[dict]:
        events, self._pending_events = self._pending_events, []
        return events

    def interval_summary(self, reset: bool = True) -> dict:
        counts = self.interval_counts.copy()
        denom = np.maximum(counts, 1)
        summary = {
            "episode_return_by_task": self.interval_return_sums / denom,
            "episode_success_by_task": self.interval_success_sums / denom,
            "episode_length_by_task": self.interval_length_sums / denom,
            "episodes_by_task": counts,
            "episodes_total_by_task": self.total_counts.copy(),
        }
        completed_mask = counts > 0
        if completed_mask.any():
            completed_returns = summary["episode_return_by_task"][completed_mask]
            completed_success = summary["episode_success_by_task"][completed_mask]
            completed_lengths = summary["episode_length_by_task"][completed_mask]
            summary.update(
                episode_return_mean=float(completed_returns.mean()),
                episode_return_median=float(np.median(completed_returns)),
                episode_return_p10=float(np.percentile(completed_returns, 10)),
                episode_return_p90=float(np.percentile(completed_returns, 90)),
                episode_success_mean=float(completed_success.mean()),
                episode_length_mean=float(completed_lengths.mean()),
                episodes_completed=int(counts.sum()),
            )
        else:
            summary.update(
                episode_return_mean=np.nan,
                episode_return_median=np.nan,
                episode_return_p10=np.nan,
                episode_return_p90=np.nan,
                episode_success_mean=np.nan,
                episode_length_mean=np.nan,
                episodes_completed=0,
            )
        if reset:
            self.interval_return_sums.fill(0)
            self.interval_success_sums.fill(0)
            self.interval_length_sums.fill(0)
            self.interval_counts.fill(0)
        return summary

    def reset_partial(self):
        """Discard trajectories that cannot be continued after an env reset."""
        self.current_returns.fill(0)
        self.current_success.fill(False)
        self.current_lengths.fill(0)

    def state_dict(self) -> dict:
        return {
            "total_counts": self.total_counts.copy(),
            "interval_return_sums": self.interval_return_sums.copy(),
            "interval_success_sums": self.interval_success_sums.copy(),
            "interval_length_sums": self.interval_length_sums.copy(),
            "interval_counts": self.interval_counts.copy(),
        }

    def load_state_dict(self, state: dict):
        self.total_counts[...] = np.asarray(state["total_counts"], dtype=np.int64)
        for name in ["interval_return_sums", "interval_success_sums",
                     "interval_length_sums", "interval_counts"]:
            if name in state:
                getattr(self, name)[...] = np.asarray(state[name], dtype=getattr(self, name).dtype)
        self.reset_partial()

    def _get_scores(self):
        summary = self.interval_summary(reset=True)
        result = {
            "goal_online": summary["episode_success_by_task"],
            "return_online": summary["episode_return_by_task"],
        }
        print(result)
        return result

    def log(self, flags, agent, replay_buffer, reward_normalizer, step, eval_env=None, render=False):
        """Legacy combined logger retained for callers other than train.py."""
        batches_info = reward_normalizer.normalize(
            replay_buffer.sample_task_batches(), agent.get_temperature()
        )
        infos = {**agent.get_infos(batches_info), **self._get_scores()}
        if flags.offline_evaluation:
            eval_stats = eval_env.evaluate(
                agent, num_episodes=flags.eval_episodes, temperature=0.0, render=render
            )
            if render:
                eval_stats["renders"] = get_wandb_video(eval_stats["renders"])
            infos = {**infos, **eval_stats}
        if flags.log_to_wandb:
            log_to_wandb(step, infos)
        return infos
