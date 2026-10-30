"""Magnitude pruning: unstructured (per-weight) and structured (per-neuron).

Unstructured pruning zeroes the smallest-magnitude weights; the model
keeps its shape but gains sparsity (a sparse format is what actually
shrinks it on disk). Structured pruning removes whole hidden neurons
with the smallest outgoing-weight norm, which genuinely shrinks the
dense matrices and therefore also cuts inference FLOPs.
"""
from __future__ import annotations

import math

from model_compression_platform.model import DenseClassifier


def sparsity(model: DenseClassifier) -> float:
    """Fraction of weights (w1 + w2) that are exactly zero."""
    weights = [w for row in model.w1 for w in row] + [w for row in model.w2 for w in row]
    if not weights:
        return 0.0
    return sum(1 for w in weights if w == 0.0) / len(weights)


def prune_unstructured(model: DenseClassifier, fraction: float) -> DenseClassifier:
    """Zero the `fraction` smallest-|weight| entries across w1 and w2.

    A short fine-tune-free pass is intentional: this module isolates the
    pruning damage so the evaluator can measure it honestly; callers
    that want recovery should retrain afterwards (see pipeline).
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    pruned = model.clone()
    if fraction == 0.0:
        return pruned
    weights = [(abs(w), "w1", j, i) for j, row in enumerate(pruned.w1) for i, w in enumerate(row)]
    weights += [(abs(w), "w2", k, j) for k, row in enumerate(pruned.w2) for j, w in enumerate(row)]
    weights.sort(key=lambda t: t[0])
    n_cut = int(len(weights) * fraction)
    for _, matrix, a, b in weights[:n_cut]:
        if matrix == "w1":
            pruned.w1[a][b] = 0.0
        else:
            pruned.w2[a][b] = 0.0
    return pruned


def retrain_masked(model: DenseClassifier, xs: list[list[float]], ys: list[int],
                   epochs: int | None = None) -> DenseClassifier:
    """Fine-tune a pruned model without resurrecting zeroed weights.

    The mask is derived from the current weights (nonzero = trainable),
    so a pruned weight keeps a zero gradient and stays zero. Returns the
    same model object, mutated, for call chaining.
    """
    mask_w1 = [[w != 0.0 for w in row] for row in model.w1]
    mask_w2 = [[w != 0.0 for w in row] for row in model.w2]
    if epochs is not None:
        original_epochs = model.cfg.epochs
        model.cfg = type(model.cfg)(**{**model.cfg.__dict__, "epochs": epochs})
        try:
            model.train(xs, ys, mask=(mask_w1, mask_w2))
        finally:
            model.cfg = type(model.cfg)(**{**model.cfg.__dict__, "epochs": original_epochs})
    else:
        model.train(xs, ys, mask=(mask_w1, mask_w2))
    return model


def prune_structured(model: DenseClassifier, n_neurons: int) -> DenseClassifier:
    """Drop the `n_neurons` hidden units with smallest outgoing-norm.

    Importance of hidden unit j = L2 norm of its outgoing row w2[:, j]
    (plus its incoming row, so a unit that is dead in both directions
    scores lowest). Removed units disappear from w1, b1 and w2, so the
    resulting model is genuinely smaller and faster, not just sparser.
    """
    hidden = model.cfg.hidden_units
    if not 0 <= n_neurons < hidden:
        raise ValueError(f"n_neurons must be in [0, {hidden})")
    if n_neurons == 0:
        return model.clone()
    scores = []
    for j in range(hidden):
        out_norm = math.sqrt(sum(model.w2[k][j] ** 2 for k in range(model.cfg.n_classes)))
        in_norm = math.sqrt(sum(w ** 2 for w in model.w1[j]))
        scores.append((out_norm + in_norm, j))
    scores.sort()
    drop = {j for _, j in scores[:n_neurons]}
    keep = [j for j in range(hidden) if j not in drop]
    pruned = model.clone()
    pruned.w1 = [model.w1[j] for j in keep]
    pruned.b1 = [model.b1[j] for j in keep]
    pruned.w2 = [[row[j] for j in keep] for row in model.w2]
    pruned.cfg = type(model.cfg)(**{**model.cfg.__dict__, "hidden_units": len(keep)})
    return pruned
