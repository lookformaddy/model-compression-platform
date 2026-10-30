import math

import pytest

from model_compression_platform.data import SyntheticConfig, generate_dataset, split_dataset, standardize
from model_compression_platform.model import DenseClassifier, ModelConfig


def _small_model():
    return DenseClassifier(ModelConfig(epochs=15))


def test_dataset_is_balanced_and_reproducible():
    xs1, ys1 = generate_dataset(SyntheticConfig(n_samples=90, seed=7))
    xs2, ys2 = generate_dataset(SyntheticConfig(n_samples=90, seed=7))
    assert xs1 == xs2 and ys1 == ys2
    assert len(xs1) == 90 and len(xs1[0]) == 8
    assert all(ys1.count(c) == 30 for c in range(3))


def test_split_and_standardize_use_train_stats_only():
    xs, ys = generate_dataset(SyntheticConfig(n_samples=200, seed=1))
    x_tr, y_tr, x_te, y_te = split_dataset(xs, ys, test_fraction=0.25)
    assert len(x_te) == 50 and len(x_tr) == 150
    x_tr_s, x_te_s, means, stds = standardize(x_tr, x_te)
    for j in range(8):
        assert abs(sum(r[j] for r in x_tr_s) / len(x_tr_s)) < 1e-9
    with pytest.raises(ValueError):
        split_dataset(xs, ys, test_fraction=1.5)
    with pytest.raises(ValueError):
        standardize([], x_te)


def test_training_reduces_loss_and_beats_chance():
    xs, ys = generate_dataset(SyntheticConfig(n_samples=300, seed=3))
    x_tr, y_tr, x_te, y_te = split_dataset(xs, ys)
    x_tr, x_te, _, _ = standardize(x_tr, x_te)
    model = _small_model()
    history = model.train(x_tr, y_tr)
    assert history[-1] < history[0]
    assert model.accuracy(x_te, y_te) > 0.5  # chance is 0.33


def test_model_serialisation_round_trip():
    model = _small_model()
    restored = DenseClassifier.from_json(model.to_json())
    assert restored.w1 == model.w1 and restored.w2 == model.w2
    assert restored.parameter_count() == model.parameter_count() == 16 * 8 + 16 + 16 * 3 + 3
    assert restored.float32_bytes() == restored.parameter_count() * 4
    assert restored.clone().w1 == model.w1
