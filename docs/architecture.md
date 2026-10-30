# Architecture

## Modules

| Module | Responsibility |
|---|---|
| `data.py` | Synthetic tabular data: one Gaussian blob per class, seeded, balanced, deterministically shuffled. `standardize` uses train statistics only (no leakage). |
| `model.py` | `DenseClassifier`: input -> tanh hidden -> softmax. Full-batch SGD with cross-entropy, JSON serialisation, parameter counting. `train()` accepts an optional weight mask for masked fine-tuning. |
| `pruning.py` | `prune_unstructured` (per-weight magnitude), `prune_structured` (whole-neuron removal by outgoing+incoming norm), `sparsity`, and `retrain_masked` (fine-tune that keeps pruned weights at zero by zeroing their gradients). |
| `quantization.py` | Signed int8 affine quantization. `_quantize_row` maps `[lo, hi]` to `[-128, 127]` with `scale = (hi-lo)/255` and `zero_point = -128 - round(lo/scale)`. `QuantizedModel` dequantizes on the fly for inference and reports honest stored bytes (1 byte/weight + 8 bytes per *distinct* scale pair + 4 bytes/bias). |
| `distillation.py` | Teacher -> student with temperature `T` (default 3.0) and blend `alpha` (default 0.7). Gradient is the standard Hinton form: `alpha * T * (student_soft - teacher_soft) + (1-alpha) * (probs - one_hot)`. |
| `evaluation.py` | `evaluate_variant` (accuracy, wall-clock latency, FLOP proxy), `PromotionGate`, `TradeOffReport` (rows + human-readable summary). |
| `registry.py` | `ArtifactRegistry`: `<root>/<name>/v<version>/artifact.json` + a `current` pointer file. `save` refuses duplicate versions; `rollback` only moves the pointer. |
| `pipeline.py` | `run_pipeline`: the full train -> prune -> quantize -> distill -> evaluate -> register flow, returning the report rows and promotion list. |
| `service.py` | Optional FastAPI app (`/health`, `/predict` per variant). The module degrades to a `RuntimeError` on import use when FastAPI is absent; the core never imports it. |
| `cli.py` | Argument parsing for the pipeline (registry root, gate thresholds, `--json`). |

## Data flow (diagram description)

```
  [synthetic blobs] --standardize (train stats)--> [train / test split]
        |
        v
  [DenseClassifier teacher, 16 hidden] --train--> baseline metrics
        |
        +-- prune_unstructured 50% --retrain_masked--> pruned variant
        +-- prune_structured -4 neurons -----------> smaller dense variant
        +-- quantize per-tensor / per-channel -----> int8 variants
        +-- distill to 8-hidden student -----------> student variant
        |
        v
  [evaluation: same test set, accuracy/size/latency/FLOPs]
        |
        v
  [promotion gate: acc drop <= 0.05 AND size reduction >= 20%]
        |
        v
  [registry: versioned artifacts + current pointer + rollback]
```

## Known bugs found and fixed during development

1. **Unsigned zero-point (quantization).** The first implementation used
   `zero_point = round(-lo / scale)`, which assumes unsigned `[0, 255]`
   codes. With signed `[-128, 127]` codes it clamped for asymmetric
   rows and produced weight errors near `max|w|` (~0.68). Fixed to the
   signed mapping above; max error is now ~0.003 (half a step).
2. **Shared scale double-counted (size accounting).** Per-tensor
   quantization shares one scale pair across rows, but `stored_bytes`
   counted it once per row, overstating int8 size by ~50%. Fixed by
   counting distinct scale objects.
3. **Fine-tuning resurrected pruned weights.** Plain retraining after
   pruning updated zeroed weights back to nonzero, silently erasing the
   sparsity. Fixed with gradient masking (`retrain_masked`); a test
   asserts the zero set is unchanged by masked retraining.
4. **Distillation gradient shadowing.** A loop variable named `soft`
   was overwritten by the soft-gradient term, crashing on the second
   class. Renamed to `soft_term`.

## Extension points

- Sparse storage (CSR) for unstructured pruning so its disk size
  reflects its sparsity.
- Quantization-aware fine-tuning and per-group scales.
- Slice-aware gates (per-class accuracy floors) alongside the global gate.
- A real integer inference kernel behind `QuantizedModel.predict`
  instead of dequantize-then-float.
