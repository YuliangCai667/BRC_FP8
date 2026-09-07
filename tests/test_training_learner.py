import tempfile
import unittest

import jax
import numpy as np

from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.training_learner import TrainingBRC, TRAIN_METRICS
from jaxrl.utils import Batch


def agent(cls, mode='fp8_carry'):
    return cls(42, np.zeros((1, 4), np.float32), np.zeros((1, 2), np.float32),
               num_tasks=2, width_critic=16, width_actor=16, updates_per_step=2,
               critic_precision='fp8_resident', fp8_resident_carry=True,
               target_critic_precision='fp8_lag', fp8_amax_history_length=8,
               critic_optimizer_state=mode)


def batch():
    rng = np.random.RandomState(12)
    return Batch(rng.randn(2, 8, 4).astype(np.float32),
                 rng.randn(2, 8, 2).astype(np.float32),
                 rng.randn(2, 8).astype(np.float32),
                 np.ones((2, 8), np.float32),
                 rng.randn(2, 8, 4).astype(np.float32),
                 np.tile(np.arange(8, dtype=np.int32) % 2, (2, 1)))


def equal_tree(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if str(getattr(x, 'dtype', '')).startswith('float8_'):
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=2e-6, atol=1e-7)


def equal_agents(a, b):
    for name in ('actor', 'critic', 'target_critic', 'temp'):
        left, right = getattr(a, name), getattr(b, name)
        equal_tree((left.params, left.opt_state, left.fp8_meta),
                   (right.params, right.opt_state, right.fp8_meta))
    equal_tree((a.rng, a.step, a.task_entropies, a.task_entropy_counts),
               (b.rng, b.step, b.task_entropies, b.task_entropy_counts))


class TrainingLearnerTests(unittest.TestCase):
    def test_training_matches_research_state_and_required_metrics(self):
        for mode in ('fp32', 'fp8_carry'):
            reference, pure = agent(BRC, mode), agent(TrainingBRC, mode)
            inputs = batch()
            for i in range(3):
                expected = reference.update(inputs, 2, i)
                actual = pure.update(inputs, 2, i)
                self.assertEqual(set(actual), set(TRAIN_METRICS))
                equal_tree(actual, {key: expected[key] for key in TRAIN_METRICS})
                equal_agents(reference, pure)
            self.assertIsNone(pure.last_target_ema_diagnostics)
            self.assertIsNone(pure.last_online_resident_diagnostics)

    def test_research_checkpoint_and_pure_resume_next_update(self):
        reference = agent(BRC)
        inputs = batch()
        reference.update(inputs, 2, 1)
        with tempfile.TemporaryDirectory() as path:
            reference.save(path)
            pure = agent(TrainingBRC)
            pure.load(path)
            equal_agents(reference, pure)
            expected = reference.update(inputs, 2, 2)
            actual = pure.update(inputs, 2, 2)
            equal_tree(actual, {key: expected[key] for key in TRAIN_METRICS})
            equal_agents(reference, pure)
            pure.save(path)
            restored = agent(TrainingBRC)
            restored.load(path)
            equal_agents(pure, restored)


if __name__ == '__main__':
    unittest.main()
