"""CLI: run the full compression pipeline and print the trade-off report."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from model_compression_platform.evaluation import PromotionGate
from model_compression_platform.pipeline import PipelineConfig, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-compression-platform")
    parser.add_argument("--registry", default=None, help="Registry root dir (default: temp dir)")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--min-size-reduction", type=float, default=0.20)
    parser.add_argument("--json", action="store_true", help="Print rows as JSON")
    args = parser.parse_args(argv)
    cfg = PipelineConfig(gate=PromotionGate(
        max_accuracy_drop=args.max_accuracy_drop,
        min_size_reduction=args.min_size_reduction))
    registry_root = args.registry or tempfile.mkdtemp(prefix="mcp-registry-")
    result = run_pipeline(cfg, registry_root=registry_root)
    if args.json:
        print(json.dumps(result["rows"], indent=2))
    else:
        print(result["report"])
        print(f"\nPromoted: {', '.join(result['promoted']) or '(none)'}")
        print(f"Registry: {registry_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
