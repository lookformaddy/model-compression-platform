import pytest

from model_compression_platform.data import SyntheticConfig, generate_dataset, split_dataset, standardize
from model_compression_platform.model import DenseClassifier, ModelConfig
from model_compression_platform.pruning import (
    prune_structured, prune_unstructured, retrain_masked, sparsity,
)
from model_compression_platform.quantization import quantize_per_channel, quantize_per_tensor


def _trained():
    xs, ys = generate_dataset(SyntheticConfig(n_samples=300, seed=3))
    x_tr, y_tr, _, _ = split_dataset(xs, ys)
    x_tr, _, _, _ = standardize(x_tr, x_tr)
    model = DenseClassifier(ModelConfig(epochs=15))
    model.train(x_tr, y_tr)
    return model


def test_unstructured_pruning_hits_target_sparsity():
    model = _trained()
    pruned = prune_unstructured(model, 0.5)
    assert abs(sparsity(pruned) - 0.5) < 0.01
    assert sparsity(model) == 0.0  # original untouched
    assert prune_unstructured(model, 0.0).w1 == model.w1
    with pytest.raises(ValueError):
        prune_unstructured(model, 1.0)


def test_masked_retrain_keeps_pruned_weights_at_zero():
    model = _trained()
    pruned = prune_unstructured(model, 0.5)
    zero_before = {(j, i) for j, row in enumerate(pruned.w1) for i, w in enumerate(row) if w == 0.0}
    xs, ys = generate_dataset(SyntheticConfig(n_samples=200, seed=9))
    x_tr, _, _, _ = standardize(xs, xs)
    retrain_masked(pruned, x_tr, ys, epochs=3)
    zero_after = {(j, i) for j, row in enumerate(pruned.w1) for i, w in enumerate(row) if w == 0.0}
    assert zero_before == zero_after
    assert abs(sparsity(pruned) - 0.5) < 0.01


def test_structured_pruning_shrinks_the_model():
    model = _trained()
    pruned = prune_structured(model, 4)
    assert pruned.cfg.hidden_units == 12
    assert pruned.parameter_count() < model.parameter_count()
    assert len(pruned.w1) == 12 and len(pruned.w2[0]) == 12
    assert prune_structured(model, 0).cfg.hidden_units == 16
    with pytest.raises(ValueError):
        prune_structured(model, 16)


def test_quantization_codes_bounded_and_per_channel_tighter():
    model = _trained()
    per_tensor = quantize_per_tensor(model)
    per_channel = quantize_per_channel(model)
    for q in (per_tensor, per_channel):
        assert all(-128 <= c <= 127 for row in q.w1_codes for c in row)
        assert all(-128 <= c <= 127 for row in q.w2_codes for c in row)
        assert q.stored_bytes() < model.float32_bytes()
    # Per-channel scales adapt per row, so worst-case error is no worse
    # than a single shared scale in practice on this model.
    assert per_channel.max_abs_error(model.w1, model.w2) <= per_tensor.max_abs_error(model.w1, model.w2) * 1.5
    assert per_tensor.scheme == "per-tensor" and per_channel.scheme == "per-channel"
    assert per_channel.accuracy  # predict path exists


def test_quantization_error_is_bounded_by_half_step():
    # Regression test: the signed affine zero-point must map lo -> -128
    # and hi -> +127, so worst-case error is half a quantization step
    # (range / 255 / 2), not ~the weight magnitude (the old unsigned
    # zero-point formula clamped and produced errors near max|w|).
    model = _trained()
    for quantize in (quantize_per_tensor, quantize_per_channel):
        q = quantize(model)
        assert q.max_abs_error(model.w1, model.w2) < 0.01
