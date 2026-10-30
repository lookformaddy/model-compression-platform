"""Accuracy / size / latency trade-off evaluation and promotion gates.

Latency is measured as mean wall-clock predict time over the test set
(plus a FLOP proxy, which is hardware-independent). A variant promotes
only if it stays within the accuracy budget AND meets the size-reduction
floor — a variant that is merely smaller, or merely accurate, is not
enough. Gates are explicit so a promotion decision is auditable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class VariantMetrics:
    name: str
    accuracy: float
    size_bytes: int
    mean_latency_ms: float
    flops_per_sample: int
    extra: dict = field(default_factory=dict)

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024.0


@dataclass
class PromotionGate:
    max_accuracy_drop: float = 0.05  # absolute points vs baseline
    min_size_reduction: float = 0.20  # fraction smaller than baseline
    max_latency_ms: float | None = None  # optional ceiling


@dataclass
class TradeOffReport:
    baseline: VariantMetrics
    variants: list[VariantMetrics]
    gate: PromotionGate

    def rows(self) -> list[dict]:
        out = []
        for v in self.variants:
            acc_drop = self.baseline.accuracy - v.accuracy
            size_red = 1.0 - v.size_bytes / self.baseline.size_bytes if self.baseline.size_bytes else 0.0
            out.append({
                "name": v.name,
                "accuracy": round(v.accuracy, 4),
                "accuracy_drop": round(acc_drop, 4),
                "size_bytes": v.size_bytes,
                "size_reduction": round(size_red, 4),
                "mean_latency_ms": round(v.mean_latency_ms, 4),
                "promoted": promotion_check(self.baseline, v, self.gate),
            })
        return out

    def summary(self) -> str:
        lines = [f"Baseline {self.baseline.name}: acc={self.baseline.accuracy:.4f} "
                 f"size={self.baseline.size_bytes}B latency={self.baseline.mean_latency_ms:.3f}ms"]
        for row in self.rows():
            verdict = "PROMOTE" if row["promoted"] else "HOLD"
            lines.append(
                f"[{verdict}] {row['name']}: acc={row['accuracy']:.4f} "
                f"(drop {row['accuracy_drop']:.4f}) size={row['size_bytes']}B "
                f"(-{row['size_reduction']:.0%}) latency={row['mean_latency_ms']:.3f}ms")
        return "\n".join(lines)


def flops_per_sample(model) -> int:
    """Multiply-adds for one forward pass (hidden layer + output layer)."""
    n_feat = getattr(model, "n_features", None) or model.cfg.n_features
    hidden = getattr(model, "hidden_units", None) or model.cfg.hidden_units
    n_classes = getattr(model, "n_classes", None) or model.cfg.n_classes
    return n_feat * hidden + hidden * n_classes


def evaluate_variant(name: str, model, xs: list[list[float]], ys: list[int],
                     size_bytes: int | None = None) -> VariantMetrics:
    start = time.perf_counter()
    preds = model.predict_batch(xs)
    elapsed = time.perf_counter() - start
    acc = sum(1 for p, y in zip(preds, ys) if p == y) / len(ys) if ys else 0.0
    if size_bytes is None:
        if hasattr(model, "stored_bytes"):
            size_bytes = model.stored_bytes()
        elif hasattr(model, "float32_bytes"):
            size_bytes = model.float32_bytes()
        else:
            size_bytes = 0
    return VariantMetrics(
        name=name, accuracy=acc, size_bytes=size_bytes,
        mean_latency_ms=(elapsed / max(len(xs), 1)) * 1000.0,
        flops_per_sample=flops_per_sample(model),
    )


def promotion_check(baseline: VariantMetrics, variant: VariantMetrics, gate: PromotionGate) -> bool:
    if baseline.accuracy - variant.accuracy > gate.max_accuracy_drop:
        return False
    if baseline.size_bytes and (1.0 - variant.size_bytes / baseline.size_bytes) < gate.min_size_reduction:
        return False
    if gate.max_latency_ms is not None and variant.mean_latency_ms > gate.max_latency_ms:
        return False
    return True
