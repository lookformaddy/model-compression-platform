"""End-to-end demo: train -> prune -> quantize -> distill -> trade-off report.

Run from the project root:  python3 examples/run_demo.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_compression_platform.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    registry = Path(tempfile.mkdtemp(prefix="mcp-demo-registry-"))
    result = run_pipeline(registry_root=registry)
    print(result["report"])
    print(f"\nBaseline accuracy: {result['baseline_accuracy']:.4f}")
    print(f"Unstructured sparsity: {result['sparsity_unstructured']:.0%}")
    print(f"Structured hidden units: {result['structured_hidden_units']}")
    print(f"Per-channel quant max weight error: {result['quant_per_channel_max_error']:.5f}")
    print(f"Promoted: {', '.join(result['promoted']) or '(none)'}")
    print(f"Registry: {registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
