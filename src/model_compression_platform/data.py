"""Synthetic tabular data: generic, separable multi-class blobs.

No real or employer data is involved. Each class is a Gaussian blob in a
low-dimensional feature space, with a fixed seed so runs are reproducible.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class SyntheticConfig:
    n_features: int = 8
    n_classes: int = 3
    n_samples: int = 600
    seed: int = 42
    class_separation: float = 2.0


def generate_dataset(cfg: SyntheticConfig | None = None) -> tuple[list[list[float]], list[int]]:
    """Return (features, labels) with one blob per class."""
    cfg = cfg or SyntheticConfig()
    rng = random.Random(cfg.seed)
    # Class centres: spread along a rotating direction so classes overlap
    # only mildly; each centre is a fresh random direction scaled up.
    centres: list[list[float]] = []
    for _ in range(cfg.n_classes):
        direction = [rng.gauss(0.0, 1.0) for _ in range(cfg.n_features)]
        norm = math.sqrt(sum(v * v for v in direction)) or 1.0
        centres.append([cfg.class_separation * v / norm for v in direction])
    xs: list[list[float]] = []
    ys: list[int] = []
    for i in range(cfg.n_samples):
        label = i % cfg.n_classes  # balanced, deterministic ordering
        centre = centres[label]
        xs.append([centre[j] + rng.gauss(0.0, 1.0) for j in range(cfg.n_features)])
        ys.append(label)
    # Shuffle deterministically so train/test splits are not class-ordered.
    order = list(range(cfg.n_samples))
    rng.shuffle(order)
    return [xs[i] for i in order], [ys[i] for i in order]


def split_dataset(
    xs: list[list[float]], ys: list[int], test_fraction: float = 0.25
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    """Deterministic tail split (data is already shuffled by generation)."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    n_test = int(len(xs) * test_fraction)
    return xs[:-n_test], ys[:-n_test], xs[-n_test:], ys[-n_test:]


def standardize(
    train: list[list[float]], test: list[list[float]]
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    """Zero-mean / unit-variance using TRAIN statistics only (no leakage)."""
    if not train:
        raise ValueError("train set must not be empty")
    n_feat = len(train[0])
    means = [sum(row[j] for row in train) / len(train) for j in range(n_feat)]
    stds = []
    for j in range(n_feat):
        var = sum((row[j] - means[j]) ** 2 for row in train) / len(train)
        stds.append(math.sqrt(var) or 1.0)

    def _apply(rows: list[list[float]]) -> list[list[float]]:
        return [[(row[j] - means[j]) / stds[j] for j in range(n_feat)] for row in rows]

    return _apply(train), _apply(test), means, stds
