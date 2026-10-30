"""End-to-end pipeline: train -> prune -> quantize -> distill -> gate.

Each compressed variant is evaluated against the same held-out test set
and registered as a versioned artifact, so the trade-off report and the
registry always agree about what was shipped and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from model_compression_platform.data import SyntheticConfig, generate_dataset, split_dataset, standardize
from model_compression_platform.distillation import distill
from model_compression_platform.evaluation import (
    PromotionGate, TradeOffReport, evaluate_variant,
)
from model_compression_platform.model import DenseClassifier, ModelConfig
from model_compression_platform.pruning import (
    prune_structured, prune_unstructured, retrain_masked, sparsity,
)
from model_compression_platform.quantization import quantize_per_channel, quantize_per_tensor
from model_compression_platform.registry import ArtifactMeta, ArtifactRegistry


@dataclass
class PipelineConfig:
    data: SyntheticConfig = field(default_factory=SyntheticConfig)
    teacher: ModelConfig = field(default_factory=lambda: ModelConfig(hidden_units=16, epochs=40))
    student: ModelConfig = field(default_factory=lambda: ModelConfig(hidden_units=8, epochs=40))
    prune_fraction: float = 0.5
    structured_drop: int = 4
    gate: PromotionGate = field(default_factory=PromotionGate)


def run_pipeline(cfg: PipelineConfig | None = None,
                 registry_root: str | Path | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    xs, ys = generate_dataset(cfg.data)
    x_train, y_train, x_test, y_test = split_dataset(xs, ys)
    x_train, x_test, _, _ = standardize(x_train, x_test)

    teacher = DenseClassifier(cfg.teacher)
    teacher.train(x_train, y_train)

    pruned = prune_unstructured(teacher, cfg.prune_fraction)
    retrain_masked(pruned, x_train, y_train)  # recovery fine-tune that keeps pruned weights at zero
    structured = prune_structured(teacher, cfg.structured_drop)
    quantized_tensor = quantize_per_tensor(teacher)
    quantized_channel = quantize_per_channel(teacher)
    student = distill(teacher, cfg.student, x_train, y_train)

    baseline = evaluate_variant("original", teacher, x_test, y_test)
    variants = [
        evaluate_variant("pruned-unstructured-50pct", pruned, x_test, y_test),
        evaluate_variant("pruned-structured", structured, x_test, y_test),
        evaluate_variant("quantized-int8-per-tensor", quantized_tensor, x_test, y_test,
                         size_bytes=quantized_tensor.stored_bytes()),
        evaluate_variant("quantized-int8-per-channel", quantized_channel, x_test, y_test,
                         size_bytes=quantized_channel.stored_bytes()),
        evaluate_variant("distilled-student", student, x_test, y_test),
    ]
    report = TradeOffReport(baseline=baseline, variants=variants, gate=cfg.gate)

    registry = ArtifactRegistry(registry_root) if registry_root else None
    if registry is not None:
        registry.save(ArtifactMeta("original", "1.0.0", "none", baseline.accuracy,
                                   baseline.size_bytes, True), teacher.to_state())
        registry.save(ArtifactMeta("pruned-unstructured", "1.0.0", "unstructured-pruning",
                                   variants[0].accuracy, variants[0].size_bytes,
                                   report.rows()[0]["promoted"], parent="original"), pruned.to_state())
        registry.save(ArtifactMeta("pruned-structured", "1.0.0", "structured-pruning",
                                   variants[1].accuracy, variants[1].size_bytes,
                                   report.rows()[1]["promoted"], parent="original"), structured.to_state())
        registry.save(ArtifactMeta("distilled-student", "1.0.0", "distillation",
                                   variants[4].accuracy, variants[4].size_bytes,
                                   report.rows()[4]["promoted"], parent="original"), student.to_state())

    return {
        "baseline_accuracy": baseline.accuracy,
        "baseline_size_bytes": baseline.size_bytes,
        "sparsity_unstructured": sparsity(pruned),
        "structured_hidden_units": structured.cfg.hidden_units,
        "quant_per_channel_max_error": quantized_channel.max_abs_error(teacher.w1, teacher.w2),
        "report": report.summary(),
        "rows": report.rows(),
        "promoted": [r["name"] for r in report.rows() if r["promoted"]],
    }
