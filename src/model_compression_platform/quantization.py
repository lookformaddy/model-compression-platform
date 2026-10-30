"""Post-training int8 affine quantization (no retraining required).

Each weight tensor is mapped to int8 via  scale = (max - min) / 255  and
zero_point = round(-min / scale), clamped to [-128, 127]. Per-tensor uses
one (scale, zero_point) per matrix; per-channel uses one per output row,
which handles rows with very different magnitudes far better at the
cost of a few extra floats. Dequantized weights approximate the original
and can be evaluated with the same DenseClassifier code path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QuantParams:
    scale: float
    zero_point: int


def _quantize_row(values: list[float]) -> tuple[list[int], QuantParams]:
    lo, hi = min(values), max(values)
    if hi == lo:  # constant row: code 0 dequantizes to -zero_point ~= lo
        zero_point = max(-128, min(127, -int(round(lo))))
        return [0] * len(values), QuantParams(scale=1.0, zero_point=zero_point)
    scale = (hi - lo) / 255.0
    # Signed affine mapping: lo -> -128 and hi -> +127, so
    # zero_point = -128 - round(lo / scale). (The unsigned variant,
    # zero_point = round(-lo / scale), overflows the signed range for
    # asymmetric rows and silently clamps — that was a real bug here.)
    zero_point = -128 - int(round(lo / scale))
    zero_point = max(-128, min(127, zero_point))
    codes = []
    for v in values:
        code = int(round(v / scale)) + zero_point
        codes.append(max(-128, min(127, code)))
    return codes, QuantParams(scale=scale, zero_point=zero_point)


def _dequantize_row(codes: list[int], params: QuantParams) -> list[float]:
    return [(c - params.zero_point) * params.scale for c in codes]


@dataclass
class QuantizedModel:
    """A DenseClassifier-shaped model stored as int8 codes + scales.

    `predict` dequantizes on the fly (exactly what a quantized inference
    engine does, minus the integer kernel). `stored_bytes` is the honest
    on-disk size: 1 byte per weight + 8 bytes per (scale, zero_point).
    """

    n_features: int
    hidden_units: int
    n_classes: int
    w1_codes: list[list[int]]
    w1_params: list[QuantParams]
    b1: list[float]
    w2_codes: list[list[int]]
    w2_params: list[QuantParams]
    b2: list[float]
    scheme: str  # "per-tensor" or "per-channel"

    def _dequantized_weights(self) -> tuple[list[list[float]], list[list[float]]]:
        w1 = [_dequantize_row(row, p) for row, p in zip(self.w1_codes, self.w1_params)]
        w2 = [_dequantize_row(row, p) for row, p in zip(self.w2_codes, self.w2_params)]
        return w1, w2

    def predict(self, x: list[float]) -> int:
        from model_compression_platform.model import DenseClassifier, ModelConfig

        w1, w2 = self._dequantized_weights()
        probe = DenseClassifier(ModelConfig(
            n_features=self.n_features, hidden_units=self.hidden_units,
            n_classes=self.n_classes))
        probe.w1, probe.w2, probe.b1, probe.b2 = w1, w2, self.b1[:], self.b2[:]
        return probe.predict(x)

    def predict_batch(self, xs: list[list[float]]) -> list[int]:
        return [self.predict(x) for x in xs]

    def accuracy(self, xs: list[list[float]], ys: list[int]) -> float:
        if not ys:
            return 0.0
        return sum(1 for x, y in zip(xs, ys) if self.predict(x) == y) / len(ys)

    def stored_bytes(self) -> int:
        n_weights = sum(len(row) for row in self.w1_codes) + sum(len(row) for row in self.w2_codes)
        # Per-tensor shares ONE (scale, zero_point) object across all rows
        # of a matrix; count distinct objects, not list entries, or the
        # shared scale would be charged once per row.
        distinct_params = {id(p) for p in self.w1_params} | {id(p) for p in self.w2_params}
        n_bias = len(self.b1) + len(self.b2)
        return n_weights * 1 + len(distinct_params) * 8 + n_bias * 4

    def max_abs_error(self, original_w1: list[list[float]], original_w2: list[list[float]]) -> float:
        w1, w2 = self._dequantized_weights()
        errors = [abs(a - b) for ra, rb in zip(w1, original_w1) for a, b in zip(ra, rb)]
        errors += [abs(a - b) for ra, rb in zip(w2, original_w2) for a, b in zip(ra, rb)]
        return max(errors) if errors else 0.0


def _quantize_matrix(matrix: list[list[float]], per_channel: bool) -> tuple[list[list[int]], list[QuantParams]]:
    if per_channel:
        out = [_quantize_row(row) for row in matrix]
        return [codes for codes, _ in out], [params for _, params in out]
    flat = [w for row in matrix for w in row]
    codes, params = _quantize_row(flat)
    rows, cols = len(matrix), len(matrix[0]) if matrix else 0
    return [codes[r * cols:(r + 1) * cols] for r in range(rows)], [params] * rows


def quantize_per_tensor(model) -> QuantizedModel:
    w1_codes, w1_params = _quantize_matrix(model.w1, per_channel=False)
    w2_codes, w2_params = _quantize_matrix(model.w2, per_channel=False)
    return QuantizedModel(model.cfg.n_features, model.cfg.hidden_units, model.cfg.n_classes,
                          w1_codes, w1_params, model.b1[:], w2_codes, w2_params,
                          model.b2[:], scheme="per-tensor")


def quantize_per_channel(model) -> QuantizedModel:
    w1_codes, w1_params = _quantize_matrix(model.w1, per_channel=True)
    w2_codes, w2_params = _quantize_matrix(model.w2, per_channel=True)
    return QuantizedModel(model.cfg.n_features, model.cfg.hidden_units, model.cfg.n_classes,
                          w1_codes, w1_params, model.b1[:], w2_codes, w2_params,
                          model.b2[:], scheme="per-channel")
