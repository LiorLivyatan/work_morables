"""Null controls for the specificity contrast: random directions and shuffled tags."""
from __future__ import annotations
import numpy as np


def random_unit_direction_at_norm(
    hidden_dim: int, target_norm: float, rng: np.random.Generator
) -> np.ndarray:
    v = rng.standard_normal(hidden_dim).astype(np.float32)
    v /= np.linalg.norm(v)
    v *= target_norm
    return v


def shuffled_tag_indices(
    n_total: int, n_positives: int, n_perms: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Return n_perms random subsets of size n_positives from range(n_total)."""
    return [rng.choice(n_total, size=n_positives, replace=False) for _ in range(n_perms)]
