"""Model compression platform: train, prune, quantize, distill, gate, ship.

Pure standard-library core. No third-party ML framework is used: the
dense classifier, pruning, quantization and distillation code below is
written from scratch so every compression step is auditable.
"""

from model_compression_platform.data import SyntheticConfig, generate_dataset, split_dataset
from model_compression_platform.model import DenseClassifier
from model_compression_platform.pruning import (
    prune_structured, prune_unstructured, retrain_masked, sparsity,
)
from model_compression_platform.quantization import QuantizedModel, quantize_per_channel, quantize_per_tensor
from model_compression_platform.distillation import distill
from model_compression_platform.evaluation import PromotionGate, TradeOffReport, evaluate_variant, promotion_check
from model_compression_platform.registry import ArtifactRegistry

__all__ = [
    "SyntheticConfig",
    "generate_dataset",
    "split_dataset",
    "DenseClassifier",
    "prune_structured",
    "prune_unstructured",
    "retrain_masked",
    "sparsity",
    "QuantizedModel",
    "quantize_per_channel",
    "quantize_per_tensor",
    "distill",
    "PromotionGate",
    "TradeOffReport",
    "evaluate_variant",
    "promotion_check",
    "ArtifactRegistry",
]

__version__ = "0.1.0"
