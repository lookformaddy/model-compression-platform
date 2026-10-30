# model-compression-platform

A model-compression platform built from scratch (no sklearn, no ML
framework): train a small dense classifier on synthetic tabular data,
then shrink it three ways — magnitude pruning, post-training int8
quantization, and knowledge distillation — evaluate every variant on
accuracy / size / latency, and promote only the variants that pass
explicit gates into a versioned artifact registry with rollback.

## The enterprise problem

Production models are usually too big, too slow, or too expensive for
where they need to run: on-device inference, latency-sensitive serving,
or cost-capped fleets. Teams reach for compression, but each technique
trades something different, and "smaller" is not the same as "safe to
ship." A pruned model can look smaller while its dense file size is
unchanged; a quantized model can be 4x smaller while silently losing
accuracy on the classes that matter; a distilled student can match the
teacher on average while failing on a slice. This platform makes those
trade-offs measurable and the promotion decision auditable: same test
set, same metrics, explicit gates, versioned artifacts, reversible
rollback.

## What it does

1. **Trains a from-scratch dense classifier** — one tanh hidden layer,
   softmax output, full-batch gradient descent, all standard library.
   Data is synthetic (Gaussian blobs, generic tabular domain), so
   nothing real or proprietary is involved.
2. **Prunes it two ways:**
   - *Unstructured magnitude pruning* zeroes the smallest-|weight|
     weights (50% by default). A masked fine-tune recovers accuracy
     without resurrecting pruned weights.
   - *Structured pruning* removes whole hidden neurons with the
     smallest outgoing-weight norm, genuinely shrinking the matrices
     and cutting FLOPs.
3. **Quantizes it two ways** — post-training int8 affine quantization,
   per-tensor (one scale pair per matrix) and per-channel (one per
   output row). On the default model both are near-lossless (max weight
   error ~0.003, accuracy drop ≤ 0.007).
4. **Distills it** — a half-size student (8 hidden units vs 16) trained
   on the teacher's temperature-softened probabilities blended with hard
   labels, using the standard temperature-scaled (T²) gradient.
5. **Evaluates and gates** — every variant gets accuracy, stored size,
   mean latency, and FLOPs on the same held-out set. The default gate
   promotes only variants within 0.05 accuracy drop AND at least 20%
   smaller. On the default run, structured pruning, both quantized
   variants, and the distilled student promote; unstructured pruning
   does not (see trade-offs below).
6. **Registers versioned artifacts** — each shipped variant is saved
   under `<registry>/<name>/v<version>/` with metadata, and `current`
   is a pointer, so rollback moves the pointer and is itself reversible.

## Architecture overview

```
synthetic tabular data (generic blobs, seeded)
        │
        ▼
┌──────────────────┐
│ DenseClassifier  │  from-scratch MLP: input -> tanh -> softmax
│ (model.py)       │  trained by full-batch SGD
└──────┬───────────┘
       │ teacher
       ├────► pruning.py ────────► unstructured (sparse) + structured (smaller dense)
       ├────► quantization.py ───► int8 affine, per-tensor + per-channel
       └────► distillation.py ───► smaller student, temperature-softened teacher targets
       │
       ▼
┌──────────────────┐
│ evaluation.py    │  accuracy / size / latency / FLOPs + promotion gates
└──────┬───────────┘
       ▼
┌──────────────────┐
│ registry.py      │  versioned artifacts, current pointer, rollback
└──────────────────┘
Optional: service.py exposes a FastAPI /predict surface (skipped when
FastAPI is absent); cli.py / pipeline.py orchestrate the full run.
```

See `docs/architecture.md` for module details and the diagram narrative.

## How to run it

```bash
python3 -m pytest -q            # core tests (stdlib only; FastAPI test skips if absent)
python3 examples/run_demo.py    # train -> prune -> quantize -> distill -> trade-off report
python3 -m model_compression_platform --help   # CLI (needs src on PYTHONPATH or pip install -e .)
```

Example output (default run, 150 held-out rows):

```
Baseline original: acc=0.7600 size=780B latency=0.028ms
[HOLD] pruned-unstructured-50pct: acc=0.7600 (drop 0.0000) size=780B (-0%)
[PROMOTE] pruned-structured: acc=0.7733 size=588B (-25%)
[PROMOTE] quantized-int8-per-tensor: acc=0.7600 size=268B (-66%)
[PROMOTE] quantized-int8-per-channel: acc=0.7533 size=404B (-48%)
[PROMOTE] distilled-student: acc=0.7533 size=396B (-49%)
```

## Key design decisions and trade-offs

- **No ML framework, on purpose.** The classifier, pruning, quantization
  and distillation are small enough to audit line by line, and the
  compression effects are attributable to the technique, not to a
  library's hidden defaults.
- **Unstructured pruning does not shrink dense storage.** Zeroing 50%
  of weights leaves the file the same size unless a sparse format (and
  sparse-capable hardware) is used — which is exactly why it HOLDs under
  the gate while structured pruning promotes. Reporting that honestly
  beats claiming a 2x saving that does not exist on disk.
- **Per-channel vs per-tensor quantization.** Per-channel adapts to
  rows with different magnitudes and is slightly more accurate here,
  but on a tiny model its per-row scale pairs cost more than they save
  (404B vs 268B). At production scale the overhead is negligible and
  per-channel usually wins; at this scale per-tensor wins on size.
- **Distillation matches, not beats, hard-label training here.**
  The student reaches 0.753 vs the teacher's 0.760 at half the size.
  Soft targets carry the teacher's ranking of wrong classes, but they
  also dilute the hard signal, so on a small synthetic task the student
  converges to parity rather than superiority — the honest result, and
  the reason the gate measures the student on the same test set instead
  of trusting the technique's reputation.
- **Gates are two-sided by design.** Accuracy alone would promote a
  no-op; size alone would promote a broken model. Both conditions must
  hold, and the gate parameters are CLI flags, not buried constants.
- **Rollback moves a pointer, never deletes.** A bad promotion and a
  bad rollback are both reversible because history is append-only.

## What it demonstrates

End-to-end model compression as a shipping discipline: from-scratch
model training, three compression techniques with their real (not
assumed) trade-offs, hardware-honest size accounting, gated promotion,
and a versioned registry with rollback — the pieces a Director-level
ML platform review actually probes.

## Layout

```
src/model_compression_platform/  data, model, pruning, quantization,
                                 distillation, evaluation, registry,
                                 pipeline, service, cli
tests/                           pytest coverage incl. conftest.py (src on sys.path)
examples/run_demo.py             end-to-end demo
docs/architecture.md             design detail
docs/github-actions-ci.yml       CI definition (move to .github/workflows/ci.yml to activate)
```
