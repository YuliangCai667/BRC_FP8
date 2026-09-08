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
    "critic_residual_compute_format", "critic_residual_compute_terms",
    "critic_residual_compute_rounding", "critic_residual_compute_rht",
    "method_version", "kernel_build_hash", "dependency_manifest_hash",
    "env_names", "seed", "width_critic", "updates_per_step", "batch_size",
    "replay_buffer_size", "metaworld_reset_mode", "eval_seed_offset",
    "resolved_task_embedding_norm", "resolved_return_bootstrap",
    "resolved_entropy_correction", "critic_precision", "target_critic_precision",
    "fp8_amax_history_length", "fp8_resident_canonicalization",
    "fp8_resident_carry", "carry_gain", "carry_dtype",
    "fp8_all_dense_kernels", "fp8_input_dense_kernel",
    "fp8_output_dense_kernel",
    "critic_optimizer_state", "optimizer_moment_block_size",
    "optimizer_moment_carry_gain", "resolved_online_optimizer_state",
    "resolved_fp8_resident_state",
    "resolved_fp8_code_materialization",
    "resolved_online_fp8_canonicalization", "resolved_online_fp8_backward",
]


def checkpoint_config_value(config: Mapping[str, Any], key: str):
    """Interpret protocol fields missing from historical checkpoints."""
    if key == "critic_optimizer_state":
        return config.get(key, "fp32")
    if key == "optimizer_moment_block_size":
        return config.get(key, 128)
    if key == "optimizer_moment_carry_gain":
        return config.get(key, 16.0)
    if key == "resolved_online_optimizer_state":
        return config.get(key, "fp32_adamw")
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
    if key == "fp8_resident_canonicalization":
        return config.get(key, False)
    if key == "fp8_resident_carry":
        return config.get(key, False)
    if key in (
        "fp8_all_dense_kernels",
        "fp8_input_dense_kernel",
        "fp8_output_dense_kernel",
    ):
        return config.get(key, False)
    if key == "carry_gain":
        return config.get(key, 16.0)
    if key == "carry_dtype":
        return config.get(key, "float8_e4m3fn")
    if key == "resolved_fp8_resident_state":
        if key in config:
            return config[key]
        return (
            "carry_e4m3_shared_scale_gain16"
            if config.get("fp8_resident_carry", False)
            else "main_e4m3_shared_scale"
        )
    if key == "resolved_fp8_code_materialization":
        return config.get(key, "legacy_compiler_elidable_cast")
    if key == "resolved_online_fp8_canonicalization":
        if key in config:
            return config[key]
        return (
            "legacy_moving_anchor_or_unspecified"
            if config.get("fp8_resident_canonicalization", False)
            else "disabled"
        )
    if key == "resolved_online_fp8_backward":
        if key in config:
            return config[key]
        precision = config.get("critic_precision", "fp32")
        if precision == "fp8_resident":
            return "generic_autodiff_unscaled_fp8"
        if precision == "fp8_direct":
            return "flax_e5m2_delayed_amax_custom_vjp"
        return "none"
    return config.get(key)


def fixed_anchor_norm_report(checkpoint_path: str | Path) -> Optional[dict]:
    """Measure resident fixed-anchor invariants from saved bytes in NumPy FP64."""
    import flax
    from flax import traverse_util

    checkpoint_path = Path(checkpoint_path).resolve()
    with (checkpoint_path / "critic.msgpack").open("rb") as file:
        raw_state = flax.serialization.msgpack_restore(file.read())
    if raw_state.get("fp8_meta") is None:
        return None
    params = traverse_util.flatten_dict(raw_state["params"], sep="/")
    metadata = traverse_util.flatten_dict(raw_state["fp8_meta"], sep="/")
    anchor_paths = sorted(
        path for path in metadata if path.endswith("/kernel_anchor_norm")
    )
    if not anchor_paths:
        return None

    members = {}
    ratios = []
    for anchor_path in anchor_paths:
        layer_path = anchor_path.rsplit("/", 1)[0]
        codes = np.asarray(params[f"{layer_path}/kernel"]).astype(np.float64)
        scales = np.asarray(
            metadata[f"{layer_path}/kernel_scale"], dtype=np.float64
        )
        anchors = np.asarray(metadata[anchor_path], dtype=np.float64)
        reduce_axes = tuple(range(1, codes.ndim))
        code_norms = np.sqrt(
            np.sum(np.square(codes), axis=reduce_axes, dtype=np.float64)
        )
        layer_ratios = scales * code_norms / anchors
        for member, ratio in enumerate(layer_ratios):
            key = f"{layer_path}/ensemble_{member}"
            members[key] = {
                "anchor_norm": float(anchors[member]),
                "stored_scale": float(scales[member]),
                "code_norm_float64": float(code_norms[member]),
                "anchor_norm_ratio": float(ratio),
            }
            ratios.append(float(ratio))

    ratios = np.asarray(ratios, dtype=np.float64)
    return {
        "method": "numpy_float64_from_serialized_critic_checkpoint",
        "member_count": int(ratios.size),
        "min_anchor_norm_ratio": float(np.min(ratios)),
        "max_anchor_norm_ratio": float(np.max(ratios)),
        "max_abs_deviation_from_one": float(np.max(np.abs(ratios - 1.0))),
        "members": members,
    }


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
        if key == "resolved_online_fp8_backward" and precision_transition:
            continue
        if key == "resolved_fp8_code_materialization":
            materialized_precisions = {"fp8_resident", "fp8_current_master"}
            materialized_target_precisions = {"fp8_resident", "fp8_lag"}
            if (
                old_precision not in materialized_precisions
                and new_precision not in materialized_precisions
                and old_target_precision not in materialized_target_precisions
                and new_target_precision not in materialized_target_precisions
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


def _tree_nbytes_matching(tree, predicate) -> int:
    import jax

    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        if not predicate(leaf.dtype):
            continue
        total += int(np.prod(leaf.shape, dtype=np.int64)) * int(leaf.dtype.itemsize)
    return total


def _tree_nbytes_at_paths(tree, path_predicate, dtype_predicate=None) -> int:
    from flax import traverse_util

    if tree is None:
        return 0
    total = 0
    for path, leaf in traverse_util.flatten_dict(tree).items():
        if not path_predicate(path):
            continue
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        if dtype_predicate is not None and not dtype_predicate(leaf.dtype):
            continue
        total += int(np.prod(leaf.shape, dtype=np.int64)) * int(
            leaf.dtype.itemsize
        )
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

    @staticmethod
    def _write_fixed_anchor_report(path: Path) -> Optional[dict]:
        report = fixed_anchor_norm_report(path)
        if report is not None:
            (path / "fixed_anchor_norms.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        return report

    def _base_manifest(self, kind: str, env_step: int, agent, **fields):
        from jaxrl.optimizers import optimizer_state_inventory
        models = [agent.actor, agent.critic, agent.target_critic, agent.temp]
        parameter_dtypes = sorted({str(leaf.dtype) for model in models
                                   for leaf in __import__('jax').tree_util.tree_leaves(model.params)})
        fp8_metadata_dtypes = sorted({
            str(leaf.dtype)
            for model in models
            for leaf in __import__('jax').tree_util.tree_leaves(model.fp8_meta)
        })
        parameter_bytes = sum(_tree_nbytes(model.params) for model in models)
        fp8_payload_bytes = sum(
            _tree_nbytes_matching(
                model.params, lambda dtype: str(dtype).startswith("float8_")
            )
            for model in models
        )
        fp8_metadata_bytes = sum(
            _tree_nbytes(model.fp8_meta) for model in models
        )
        resident_main_code_bytes = _tree_nbytes_at_paths(
            agent.critic.params,
            lambda path: path[-1] == "kernel",
            lambda dtype: str(dtype) == "float8_e4m3fn",
        )
        resident_carry_code_bytes = _tree_nbytes_at_paths(
            agent.critic.fp8_meta,
            lambda path: path[-1] == "kernel_carry",
        )
        resident_carry_fp32_bytes = _tree_nbytes_at_paths(
            agent.critic.fp8_meta,
            lambda path: path[-1] == "kernel_carry",
            lambda dtype: str(dtype) == "float32",
        )
        resident_weight_scale_bytes = _tree_nbytes_at_paths(
            agent.critic.fp8_meta,
            lambda path: path[-1] == "kernel_scale",
        )
        optimizer_bytes = (
            sum(_tree_nbytes(model.opt_state) for model in models)
            if fields.get("includes_optimizer")
            else 0
        )
        persistent_state_bytes = (
            parameter_bytes + fp8_metadata_bytes + optimizer_bytes
        )
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
            "optimizer_dtypes": (
                sorted({
                    str(leaf.dtype)
                    for model in models
                    for leaf in __import__('jax').tree_util.tree_leaves(
                        model.opt_state
                    )
                    if hasattr(leaf, 'dtype')
                })
                if fields.get("includes_optimizer")
                else []
            ),
            "state_bytes": {
                "parameters": parameter_bytes,
                "fp8_payload": fp8_payload_bytes,
                "high_precision_parameters": parameter_bytes - fp8_payload_bytes,
                "fp8_metadata": fp8_metadata_bytes,
                "resident_main_code": resident_main_code_bytes,
                "resident_carry_code": resident_carry_code_bytes,
                "resident_carry_fp32": resident_carry_fp32_bytes,
                "resident_weight_scale": resident_weight_scale_bytes,
                "optimizer": optimizer_bytes,
                "persistent_total": persistent_state_bytes,
                "fp8_metadata_fraction": (
                    fp8_metadata_bytes / persistent_state_bytes
                    if persistent_state_bytes
                    else 0.0
                ),
            },
            "config": self.config,
            "critic_optimizer_inventory": (
                optimizer_state_inventory(agent.critic.opt_state)
                if fields.get("includes_optimizer") else None
            ),
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
            fixed_anchor_report = self._write_fixed_anchor_report(temp_path)
            manifest = self._base_manifest(
                "analysis", env_step, agent, includes_optimizer=False,
                includes_replay_buffer=False, is_final=bool(is_final),
                fixed_anchor_norms=fixed_anchor_report,
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
            fixed_anchor_report = self._write_fixed_anchor_report(temp_path)
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
                fixed_anchor_norms=fixed_anchor_report,
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
