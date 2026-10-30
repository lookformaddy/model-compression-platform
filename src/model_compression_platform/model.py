"""From-scratch dense classifier: one hidden layer, trained by SGD.

Pure standard library (lists + math). A hidden layer with tanh activation
and a softmax output is expressive enough for the synthetic blobs while
staying small enough that pruning and quantization effects are visible
and every weight can be inspected.
"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _matvec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(w * v for w, v in zip(row, vec)) for row in matrix]


@dataclass
class ModelConfig:
    n_features: int = 8
    hidden_units: int = 16
    n_classes: int = 3
    learning_rate: float = 0.05
    epochs: int = 40
    seed: int = 42


class DenseClassifier:
    """Two-layer network: input -> tanh hidden -> softmax output."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        rng = random.Random(self.cfg.seed)
        scale1 = math.sqrt(1.0 / self.cfg.n_features)
        scale2 = math.sqrt(1.0 / self.cfg.hidden_units)
        self.w1 = [[rng.gauss(0.0, scale1) for _ in range(self.cfg.n_features)]
                   for _ in range(self.cfg.hidden_units)]
        self.b1 = [0.0] * self.cfg.hidden_units
        self.w2 = [[rng.gauss(0.0, scale2) for _ in range(self.cfg.hidden_units)]
                   for _ in range(self.cfg.n_classes)]
        self.b2 = [0.0] * self.cfg.n_classes

    # -- inference -----------------------------------------------------
    def forward(self, x: list[float]) -> tuple[list[float], list[float]]:
        hidden_pre = [v + b for v, b in zip(_matvec(self.w1, x), self.b1)]
        hidden = [math.tanh(v) for v in hidden_pre]
        logits = [v + b for v, b in zip(_matvec(self.w2, hidden), self.b2)]
        return _softmax(logits), hidden

    def predict(self, x: list[float]) -> int:
        probs, _ = self.forward(x)
        return max(range(len(probs)), key=lambda i: probs[i])

    def predict_batch(self, xs: list[list[float]]) -> list[int]:
        return [self.predict(x) for x in xs]

    def accuracy(self, xs: list[list[float]], ys: list[int]) -> float:
        if not ys:
            return 0.0
        correct = sum(1 for x, y in zip(xs, ys) if self.predict(x) == y)
        return correct / len(ys)

    # -- training ------------------------------------------------------
    def train(self, xs: list[list[float]], ys: list[int],
              mask: tuple[list[list[bool]], list[list[bool]]] | None = None) -> list[float]:
        """Full-batch gradient descent; returns per-epoch mean CE loss.

        `mask` is an optional (mask_w1, mask_w2) pair of booleans marking
        trainable weights. It exists for masked fine-tuning after
        pruning: gradients of pruned (zeroed) weights are forced to zero,
        so retraining cannot resurrect weights the pruning step removed.
        """
        cfg = self.cfg
        history: list[float] = []
        for _ in range(cfg.epochs):
            # Accumulate gradients over the whole batch.
            gw1 = [[0.0] * cfg.n_features for _ in range(cfg.hidden_units)]
            gb1 = [0.0] * cfg.hidden_units
            gw2 = [[0.0] * cfg.hidden_units for _ in range(cfg.n_classes)]
            gb2 = [0.0] * cfg.n_classes
            total_loss = 0.0
            for x, y in zip(xs, ys):
                probs, hidden = self.forward(x)
                total_loss += -math.log(max(probs[y], 1e-12))
                # Output-layer gradient: probs - one_hot.
                delta2 = probs[:]
                delta2[y] -= 1.0
                for k in range(cfg.n_classes):
                    gb2[k] += delta2[k]
                    for j in range(cfg.hidden_units):
                        gw2[k][j] += delta2[k] * hidden[j]
                # Hidden-layer gradient through tanh derivative (1 - h^2).
                delta1 = []
                for j in range(cfg.hidden_units):
                    back = sum(delta2[k] * self.w2[k][j] for k in range(cfg.n_classes))
                    delta1.append(back * (1.0 - hidden[j] ** 2))
                for j in range(cfg.hidden_units):
                    gb1[j] += delta1[j]
                    for i in range(cfg.n_features):
                        gw1[j][i] += delta1[j] * x[i]
            n = max(len(xs), 1)
            lr = cfg.learning_rate / n
            if mask is not None:
                mask_w1, mask_w2 = mask
                for j in range(cfg.hidden_units):
                    for i in range(cfg.n_features):
                        if not mask_w1[j][i]:
                            gw1[j][i] = 0.0
                for k in range(cfg.n_classes):
                    for j in range(cfg.hidden_units):
                        if not mask_w2[k][j]:
                            gw2[k][j] = 0.0
            for j in range(cfg.hidden_units):
                self.b1[j] -= lr * gb1[j]
                for i in range(cfg.n_features):
                    self.w1[j][i] -= lr * gw1[j][i]
            for k in range(cfg.n_classes):
                self.b2[k] -= lr * gb2[k]
                for j in range(cfg.hidden_units):
                    self.w2[k][j] -= lr * gw2[k][j]
            history.append(total_loss / n)
        return history

    # -- size / structure ----------------------------------------------
    def parameter_count(self) -> int:
        return (len(self.w1) * len(self.w1[0]) + len(self.b1)
                + len(self.w2) * len(self.w2[0]) + len(self.b2))

    def float32_bytes(self) -> int:
        """In-memory weight size at 4 bytes per parameter."""
        return self.parameter_count() * 4

    def clone(self) -> "DenseClassifier":
        other = DenseClassifier(self.cfg)
        other.w1 = copy.deepcopy(self.w1)
        other.b1 = self.b1[:]
        other.w2 = copy.deepcopy(self.w2)
        other.b2 = self.b2[:]
        return other

    # -- serialisation ---------------------------------------------------
    def to_state(self) -> dict:
        return {"config": self.cfg.__dict__, "w1": self.w1, "b1": self.b1,
                "w2": self.w2, "b2": self.b2}

    @classmethod
    def from_state(cls, state: dict) -> "DenseClassifier":
        model = cls(ModelConfig(**state["config"]))
        model.w1 = state["w1"]
        model.b1 = state["b1"]
        model.w2 = state["w2"]
        model.b2 = state["b2"]
        return model

    def to_json(self) -> str:
        return json.dumps(self.to_state())

    @classmethod
    def from_json(cls, payload: str) -> "DenseClassifier":
        return cls.from_state(json.loads(payload))
