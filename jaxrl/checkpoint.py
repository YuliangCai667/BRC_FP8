"""Atomic analysis and recovery checkpoints for BRC experiments."""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1

RESUME_CONFIG_KEYS = [
    "env_names", "seed", "width_critic", "updates_per_step", "batch_size",
    "replay_buffer_size", "metaworld_reset_mode", "eval_seed_offset",
    "resolved_task_embedding_norm", "resolved_return_bootstrap",
    "resolved_entropy_correction", "critic_precision", "target_critic_precision",
    "fp8_amax_history_length",
]


def checkpoint_config_value(config: Mapping[str, Any], key: str):
    """Interpret protocol fields missing from historical checkpoints."""
    if key == "metaworld_reset_mode":
        return config.get(key, "frozen")
    if key == "eval_seed_offset":
        return config.get(key, 42)
    if key == "critic_precision":
        return config.get(key, "fp32")
    if key == "target_critic_precision":
        return config.get(key, "fp32")
    if key == "fp8_amax_history_length":
        return config.get(key, 1024)
    return config.get(key)


def validate_checkpoint_config(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Validate resume compatibility and report an FP32-to-FP8 transition."""
    old_precision = checkpoint_config_value(previous, "critic_precision")
    new_precision = checkpoint_config_value(current, "critic_precision")
    precision_transition = old_precision == "fp32" and new_precision == "fp8_direct"
    if old_precision != new_precision and not precision_transition:
        raise ValueError(
            f"checkpoint configuration mismatch for critic_precision: "
            f"{old_precision} != {new_precision}"
        )
    old_target_precision = checkpoint_config_value(previous, "target_critic_precision")
    new_target_precision = checkpoint_config_value(current, "target_critic_precision")
    if old_target_precision != new_target_precision:
        raise ValueError(
            f"checkpoint configuration mismatch for target_critic_precision: "
            f"{old_target_precision} != {new_target_precision}"
        )
    for key in RESUME_CONFIG_KEYS:
        if key in ("critic_precision", "target_critic_precision"):
            continue
        if (
            key == "fp8_amax_history_length"
            and old_precision != "fp8_direct"
            and old_target_precision not in ("fp8_direct", "fp8_lag")
        ):
            continue
        old = checkpoint_config_value(previous, key)
        new = checkpoint_config_value(current, key)
        if old is not None and new is not None and old != new:
            raise ValueError(f"checkpoint configuration mismatch for {key}: {old} != {new}")
    return precision_transition


def _tree_nbytes(tree) -> int:
    import jax

    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            itemsize = getattr(leaf.dtype, "itemsize", None)
            if itemsize is None:
                itemsize = np.dtype(leaf.dtype).itemsize
            total += int(np.prod(leaf.shape, dtype=np.int64)) * int(itemsize)
    return total


class CheckpointManager:
    def __init__(
        self,
        checkpoint_root: str | Path,
        run_id: str,
        run_dir: str | Path,
        task_names: list[str],
        config: Mapping[str, Any],
        keep_last_analysis: int = 2,
        keep_last_recovery: int = 1,
    ):
        self.root = Path(checkpoint_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.run_dir = str(Path(run_dir).resolve())
        self.task_names = list(task_names)
        self.config = dict(config)
        self.keep_last_analysis = max(int(keep_last_analysis), 0)
        self.keep_last_recovery = max(int(keep_last_recovery), 0)

    @staticmethod
    def read_manifest(checkpoint_path: str | Path) -> dict:
        path = Path(checkpoint_path).resolve()
        if not (path / "COMPLETE").exists():
            raise ValueError(f"checkpoint is incomplete: {path}")
        with (path / "manifest.json").open(encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema: {manifest.get('schema_version')}")
        return manifest

    @classmethod
    def resolve_recovery_checkpoint(cls, path: str | Path) -> Path:
        path = Path(path).resolve()
        if (path / "COMPLETE").exists():
            manifest = cls.read_manifest(path)
            if manifest.get("kind") != "recovery":
                raise ValueError("--resume_from must point to a recovery checkpoint")
            if not manifest.get("includes_replay_buffer"):
                raise ValueError("checkpoint has no replay buffer and cannot safely resume training")
            return path
        candidates = []
        if (path / "checkpoints").is_dir():
            for candidate in sorted((path / "checkpoints").glob("recovery_step_*")):
                if not (candidate / "COMPLETE").exists():
                    continue
                try:
                    if cls.read_manifest(candidate).get("includes_replay_buffer"):
                        candidates.append(candidate)
                except Exception:
                    continue
        if not candidates:
            raise ValueError(f"no complete recovery checkpoint found under {path}")
        return candidates[-1]

    def _agent_bytes(self, agent, include_optimizer: bool):
        models = [agent.actor, agent.critic, agent.target_critic, agent.temp]
        total = sum(_tree_nbytes(model.params) for model in models)
        total += sum(_tree_nbytes(model.fp8_meta) for model in models)
        if include_optimizer:
            total += sum(_tree_nbytes(model.opt_state) for model in models)
        return total

    def _prepare_temp(self, final_path: Path) -> Path:
        temp_path = self.root / f".tmp_{final_path.name}_{os.getpid()}"
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True)
        return temp_path

    @staticmethod
    def _validate_files(path: Path, include_replay: bool):
        required = [
            "actor.msgpack", "critic.msgpack", "target_critic.msgpack",
            "temp.msgpack", "agent_state.pkl", "manifest.json",
        ]
        if include_replay:
            required.append("replay_buffer/manifest.json")
        for name in required:
            target = path / name
            if not target.exists() or (target.is_file() and target.stat().st_size == 0):
                raise IOError(f"checkpoint validation failed: missing/empty {name}")

    def _commit(self, temp_path: Path, final_path: Path):
        (temp_path / "COMPLETE").write_text("ok\n", encoding="utf-8")
        if final_path.exists():
            if (final_path / "COMPLETE").exists():
                shutil.rmtree(temp_path)
                return final_path
            shutil.rmtree(final_path)
        os.replace(temp_path, final_path)
        return final_path

    def _base_manifest(self, kind: str, env_step: int, agent, **fields):
        models = [agent.actor, agent.critic, agent.target_critic, agent.temp]
        parameter_dtypes = sorted({str(leaf.dtype) for model in models
                                   for leaf in __import__('jax').tree_util.tree_leaves(model.params)})
        fp8_metadata_dtypes = sorted({
            str(leaf.dtype)
            for model in models
            for leaf in __import__('jax').tree_util.tree_leaves(model.fp8_meta)
        })
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": kind,
            "created_at": time.time(),
            "env_step": int(env_step),
            "update_step": int(agent.step),
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "task_names": self.task_names,
            "parameter_dtypes": parameter_dtypes,
            "fp8_metadata_dtypes": fp8_metadata_dtypes,
            "config": self.config,
            **fields,
        }

    def save_analysis(self, agent, env_step: int, is_best: bool = False, is_final: bool = False):
        final_path = self.root / f"analysis_step_{int(env_step):012d}"
        if (final_path / "COMPLETE").exists():
            if is_best:
                self._set_best(final_path, env_step)
            return final_path
        temp_path = self._prepare_temp(final_path)
        try:
            agent.save(str(temp_path), include_optimizer=False)
            manifest = self._base_manifest(
                "analysis", env_step, agent, includes_optimizer=False,
                includes_replay_buffer=False, is_final=bool(is_final),
            )
            (temp_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self._validate_files(temp_path, include_replay=False)
            result = self._commit(temp_path, final_path)
            if is_best:
                self._set_best(result, env_step)
            self._prune("analysis", self.keep_last_analysis)
            return result
        except Exception:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise

    def save_recovery(
        self,
        agent,
        replay_buffer,
        reward_normalizer,
        episode_recorder,
        env_step: int,
        wandb_id: Optional[str],
        save_replay_buffer: bool = True,
        is_final: bool = False,
    ):
        final_path = self.root / f"recovery_step_{int(env_step):012d}"
        if (final_path / "COMPLETE").exists():
            return final_path
        temp_path = self._prepare_temp(final_path)
        replay_saved = False
        degraded_reason = None
        try:
            required = int(self._agent_bytes(agent, True) * 1.15)
            if save_replay_buffer:
                required += int(replay_buffer.estimate_size_bytes() * 1.05)
            free = shutil.disk_usage(self.root).free
            if save_replay_buffer and free < required:
                degraded_reason = f"insufficient disk: required={required}, free={free}"
                save_replay_buffer = False

            agent.save(str(temp_path), include_optimizer=True)
            state = {
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "reward_normalizer": reward_normalizer.state_dict(),
                "episode_recorder": episode_recorder.state_dict(),
            }
            with (temp_path / "training_state.pkl").open("wb") as file:
                pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
            if save_replay_buffer:
                replay_buffer.save(str(temp_path / "replay_buffer"))
                replay_saved = True
            manifest = self._base_manifest(
                "recovery", env_step, agent, includes_optimizer=True,
                includes_replay_buffer=replay_saved, degraded_reason=degraded_reason,
                wandb_id=wandb_id, is_final=bool(is_final),
            )
            (temp_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self._validate_files(temp_path, include_replay=replay_saved)
            result = self._commit(temp_path, final_path)
            self._prune("recovery", self.keep_last_recovery)
            return result
        except Exception:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise

    def load_recovery(self, path, agent, replay_buffer, reward_normalizer, episode_recorder):
        checkpoint = self.resolve_recovery_checkpoint(path)
        manifest = self.read_manifest(checkpoint)
        if manifest["task_names"] != self.task_names:
            raise ValueError("checkpoint task names/order do not match the current run")
        validate_checkpoint_config(manifest.get("config", {}), self.config)
        agent.load(str(checkpoint))
        with (checkpoint / "training_state.pkl").open("rb") as file:
            state = pickle.load(file)
        reward_normalizer.load_state_dict(state["reward_normalizer"])
        episode_recorder.load_state_dict(state["episode_recorder"])
        if manifest.get("includes_replay_buffer"):
            replay_buffer.load(str(checkpoint / "replay_buffer"))
        else:
            raise ValueError("recovery checkpoint has no replay buffer; safe training resume is unavailable")
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        return manifest

    def _set_best(self, path: Path, env_step: int):
        pointer = {"path": path.name, "env_step": int(env_step)}
        temp = self.root / ".best.json.tmp"
        temp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        os.replace(temp, self.root / "best.json")

    def _prune(self, kind: str, keep_last: int):
        candidates = sorted(
            path for path in self.root.glob(f"{kind}_step_*") if (path / "COMPLETE").exists()
        )
        if kind == "recovery":
            resumable = []
            degraded = []
            for path in candidates:
                try:
                    if self.read_manifest(path).get("includes_replay_buffer"):
                        resumable.append(path)
                    else:
                        degraded.append(path)
                except Exception:
                    continue
            preserved = set(resumable[-keep_last:] if keep_last else [])
            if degraded:
                preserved.add(degraded[-1])
        else:
            preserved = set(candidates[-keep_last:] if keep_last else [])
        best_file = self.root / "best.json"
        if kind == "analysis" and best_file.exists():
            try:
                preserved.add(self.root / json.loads(best_file.read_text())["path"])
            except Exception:
                pass
        final_candidates = []
        for path in candidates:
            try:
                if self.read_manifest(path).get("is_final"):
                    final_candidates.append(path)
            except Exception:
                continue
        if final_candidates:
            preserved.add(final_candidates[-1])
        for path in candidates:
            if path not in preserved:
                shutil.rmtree(path)
