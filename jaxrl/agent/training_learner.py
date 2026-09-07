"""Training entry with no tensor diagnostics and only basic scalar logging.

Keep the research learner's compiled update boundary: changing the surrounding
JIT can alter GPU rounding at FP8 quantization boundaries. Filtering the returned
scalar dictionary in Python does not change that computation or retain tensors.
"""

from jaxrl.agent.brc_learner import BRC


TRAIN_METRICS = (
    'critic_loss', 'actor_loss', 'temp_loss', 'temperature', 'entropy', 'q_mean',
)


class TrainingBRC(BRC):
    """Original numerical update/checkpoint format, without tensor diagnostics."""

    def update(self, batch, num_updates, env_step):
        info = super().update(batch, num_updates, env_step,
                              collect_update_diagnostics=False)
        return {key: info[key] for key in TRAIN_METRICS}
