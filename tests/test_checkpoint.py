import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import jax
import numpy as np

from jaxrl.agent.brc_learner import BRC
from jaxrl.checkpoint import CheckpointManager
from jaxrl.experiment import summarize_tree
from jaxrl.logger import EpisodeRecorder
from jaxrl.normalizer import RewardNormalizer
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.utils import Batch


def make_buffer(capacity=5):
    space = gym.spaces.Box(-1.0, 1.0, shape=(2, 4), dtype=np.float32)
    return ParallelReplayBuffer(space, action_dim=2, capacity=capacity, num_tasks=2)


def fill_buffer(buffer, count):
    for step in range(count):
        obs = np.full((2, 4), step, np.float32)
        action = np.full((2, 2), step + 0.5, np.float32)
        reward = np.array([step, step + 1], np.float32)
        mask = np.ones(2, np.float32)
        buffer.insert(obs, action, reward, mask, obs + 1)


class ReplayBufferCheckpointTest(unittest.TestCase):
    def _round_trip(self, insert_count):
        source = make_buffer()
        fill_buffer(source, insert_count)
        with tempfile.TemporaryDirectory() as temp:
            source.save(temp, target_chunk_bytes=128)
            restored = make_buffer()
            restored.load(temp)
            self.assertEqual(restored.size, source.size)
            self.assertEqual(restored.insert_index, source.insert_index)
            valid = source.capacity if source.size == source.capacity else source.size
            for name in ["observations", "actions", "rewards", "masks", "next_observations"]:
                np.testing.assert_array_equal(
                    getattr(restored, name)[:, :valid], getattr(source, name)[:, :valid]
                )

    def test_not_full(self):
        self._round_trip(3)

    def test_wrapped(self):
        self._round_trip(8)


class FullCheckpointTest(unittest.TestCase):
    def test_recovery_round_trip_and_tensor_diagnostics(self):
        agent = BRC(
            0, np.zeros((1, 4), np.float32), np.zeros((1, 2), np.float32),
            num_tasks=2, width_critic=16, width_actor=16, updates_per_step=1,
        )
        buffer = make_buffer()
        fill_buffer(buffer, 4)
        normalizer = RewardNormalizer(2, target_entropy=agent.target_entropy)
        normalizer.returns_min_norm[:] = [-3.0, -4.0]
        normalizer.returns_max_norm[:] = [5.0, 6.0]
        episodes = EpisodeRecorder(2, ["a", "b"])
        episodes.total_counts[:] = [2, 3]
        batch = buffer.make_probe_batch(4, seed=3)
        batch_for_update = Batch(*[np.expand_dims(value, 0) for value in batch])
        agent.update(batch_for_update, 1, 1)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), agent.actor.params)
        expected_rng = np.asarray(agent.rng).copy()
        expected_normalizer_rng = np.asarray(agent.normalizer_rng).copy()
        expected_task_entropies = np.asarray(agent.task_entropies).copy()
        expected_task_entropy_counts = np.asarray(agent.task_entropy_counts).copy()
        expected_actor = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), agent.actor.params)
        expected_opt = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), agent.actor.opt_state)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            manager = CheckpointManager(
                root, "run", temp, ["a", "b"],
                {"env_names": "test", "width_critic": 16, "updates_per_step": 1},
            )
            path = manager.save_recovery(
                agent, buffer, normalizer, episodes, env_step=10,
                wandb_id=None, save_replay_buffer=True,
            )
            agent.rng = jax.random.PRNGKey(999)
            agent.normalizer_rng = jax.random.PRNGKey(998)
            agent.task_entropies = np.zeros(2, np.float32)
            agent.task_entropy_counts = np.zeros(2, np.int32)
            buffer.size = 0
            episodes.total_counts[:] = 0
            manifest = manager.load_recovery(path, agent, buffer, normalizer, episodes)
            self.assertEqual(manifest["env_step"], 10)
            np.testing.assert_array_equal(np.asarray(agent.rng), expected_rng)
            np.testing.assert_array_equal(
                np.asarray(agent.normalizer_rng), expected_normalizer_rng
            )
            np.testing.assert_array_equal(
                np.asarray(agent.task_entropies), expected_task_entropies
            )
            np.testing.assert_array_equal(
                np.asarray(agent.task_entropy_counts), expected_task_entropy_counts
            )
            self.assertEqual(buffer.size, 4)
            np.testing.assert_array_equal(episodes.total_counts, [2, 3])
            np.testing.assert_array_equal(normalizer.returns_min_norm, [-3.0, -4.0])
            np.testing.assert_array_equal(normalizer.returns_max_norm, [5.0, 6.0])
            for got, expected in zip(jax.tree_util.tree_leaves(agent.actor.params),
                                     jax.tree_util.tree_leaves(expected_actor)):
                np.testing.assert_array_equal(np.asarray(got), expected)
            for got, expected in zip(jax.tree_util.tree_leaves(agent.actor.opt_state),
                                     jax.tree_util.tree_leaves(expected_opt)):
                np.testing.assert_array_equal(np.asarray(got), expected)
            manager.save_recovery(
                agent, buffer, normalizer, episodes, env_step=20,
                wandb_id=None, save_replay_buffer=True, is_final=True,
            )
            complete = [p for p in root.glob("recovery_step_*") if (p / "COMPLETE").exists()]
            self.assertEqual([p.name for p in complete], ["recovery_step_000000000020"])
            degraded = manager.save_recovery(
                agent, buffer, normalizer, episodes, env_step=25,
                wandb_id=None, save_replay_buffer=False,
            )
            self.assertFalse(CheckpointManager.read_manifest(degraded)["includes_replay_buffer"])
            self.assertEqual(
                CheckpointManager.resolve_recovery_checkpoint(temp).name,
                "recovery_step_000000000020",
            )
            incomplete = root / "recovery_step_000000000030"
            incomplete.mkdir()
            with self.assertRaises(ValueError):
                CheckpointManager.read_manifest(incomplete)

        diagnostics = agent.get_tensor_diagnostics(batch)
        self.assertIn("gradients", diagnostics)
        self.assertIn("activations", diagnostics)
        stats = summarize_tree(diagnostics["outputs"], "outputs")
        self.assertIn("outputs/critic_logits", stats)
        self.assertIn("p99_9", stats["outputs/critic_logits"])


if __name__ == "__main__":
    unittest.main()
