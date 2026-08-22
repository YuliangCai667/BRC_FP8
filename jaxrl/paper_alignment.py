"""Resolve independently testable paper-alignment choices."""

PAPER_PRESET = {
    "task_embedding_norm": "l1",
    # Keep the return scale independent of critic predictions. Critic
    # bootstrapping remains available as an explicit ablation override.
    "return_bootstrap": "reward_mean",
    "entropy_correction": "empirical_per_task",
}

LEGACY_PRESET = {
    "task_embedding_norm": "l2",
    "return_bootstrap": "reward_mean",
    "entropy_correction": "target_entropy",
}


def resolve_paper_alignment(
    enabled: bool,
    task_embedding_norm: str = "auto",
    return_bootstrap: str = "auto",
    entropy_correction: str = "auto",
) -> dict[str, str]:
    """Resolve the preset while allowing one-at-a-time ablations."""
    preset = PAPER_PRESET if enabled else LEGACY_PRESET
    requested = {
        "task_embedding_norm": task_embedding_norm,
        "return_bootstrap": return_bootstrap,
        "entropy_correction": entropy_correction,
    }
    return {
        name: preset[name] if value == "auto" else value
        for name, value in requested.items()
    }
