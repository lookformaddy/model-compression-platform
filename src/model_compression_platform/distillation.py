"""Knowledge distillation: teacher -> smaller student, with temperature.

The student is trained on a blend of (a) the hard labels and (b) the
teacher's temperature-softened probabilities. Softening with temperature
T > 1 exposes the teacher's "dark knowledge" — which wrong classes it
finds almost-plausible — and is what lets a much smaller student recover
accuracy that hard-label training alone would not reach.
"""
from __future__ import annotations

import math

from model_compression_platform.model import DenseClassifier, ModelConfig


def _softmax_temp(logits_or_probs: list[float], temperature: float) -> list[float]:
    # Input here is a probability vector from the teacher; recover a
    # temperature-softened distribution by raising to 1/T and renormalising
    # (equivalent to softmax(logits / T) up to a constant).
    powered = [max(p, 1e-12) ** (1.0 / temperature) for p in logits_or_probs]
    total = sum(powered)
    return [p / total for p in powered]


def distill(
    teacher: DenseClassifier,
    student_cfg: ModelConfig,
    xs: list[list[float]],
    ys: list[int],
    temperature: float = 3.0,
    alpha: float = 0.7,
    epochs: int | None = None,
) -> DenseClassifier:
    """Train `student_cfg` model; alpha weights the soft (teacher) loss.

    Loss per example = alpha * CE(student, teacher_soft) +
    (1 - alpha) * CE(student, hard label). Implemented as a blended
    target distribution, which is exactly that loss's gradient.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    student = DenseClassifier(student_cfg)
    cfg = student.cfg
    if epochs is not None:
        cfg = ModelConfig(**{**cfg.__dict__, "epochs": epochs})
        student.cfg = cfg
    # Teacher soft targets, computed once (teacher is frozen).
    teacher_targets = []
    for x in xs:
        probs, _ = teacher.forward(x)
        teacher_targets.append(_softmax_temp(probs, temperature))
    for _ in range(cfg.epochs):
        gw1 = [[0.0] * cfg.n_features for _ in range(cfg.hidden_units)]
        gb1 = [0.0] * cfg.hidden_units
        gw2 = [[0.0] * cfg.hidden_units for _ in range(cfg.n_classes)]
        gb2 = [0.0] * cfg.n_classes
        for x, y, soft in zip(xs, ys, teacher_targets):
            probs, hidden = student.forward(x)
            # Standard Hinton gradient: the soft term is scaled by T^2
            # (the 1/T from differentiating softmax(z/T), times T^2 to
            # keep the soft and hard losses on a comparable scale), and
            # the hard term uses the ordinary unsoftened probabilities.
            student_soft = _softmax_temp(probs, temperature)
            delta2 = []
            for k in range(cfg.n_classes):
                hard = probs[k] - (1.0 if k == y else 0.0)
                soft_term = temperature * (student_soft[k] - soft[k])
                delta2.append(alpha * soft_term + (1.0 - alpha) * hard)
            for k in range(cfg.n_classes):
                gb2[k] += delta2[k]
                for j in range(cfg.hidden_units):
                    gw2[k][j] += delta2[k] * hidden[j]
            delta1 = []
            for j in range(cfg.hidden_units):
                back = sum(delta2[k] * student.w2[k][j] for k in range(cfg.n_classes))
                delta1.append(back * (1.0 - hidden[j] ** 2))
            for j in range(cfg.hidden_units):
                gb1[j] += delta1[j]
                for i in range(cfg.n_features):
                    gw1[j][i] += delta1[j] * x[i]
        n = max(len(xs), 1)
        lr = cfg.learning_rate / n
        for j in range(cfg.hidden_units):
            student.b1[j] -= lr * gb1[j]
            for i in range(cfg.n_features):
                student.w1[j][i] -= lr * gw1[j][i]
        for k in range(cfg.n_classes):
            student.b2[k] -= lr * gb2[k]
            for j in range(cfg.hidden_units):
                student.w2[k][j] -= lr * gw2[k][j]
    return student
