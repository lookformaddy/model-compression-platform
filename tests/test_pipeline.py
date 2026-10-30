from model_compression_platform.data import SyntheticConfig
from model_compression_platform.model import ModelConfig
from model_compression_platform.pipeline import PipelineConfig, run_pipeline


def test_pipeline_end_to_end(tmp_path):
    cfg = PipelineConfig(
        data=SyntheticConfig(n_samples=300, seed=11),
        teacher=ModelConfig(epochs=30),
        student=ModelConfig(hidden_units=8, epochs=30),
    )
    result = run_pipeline(cfg, registry_root=tmp_path / "registry")
    assert result["baseline_accuracy"] > 0.5
    assert len(result["rows"]) == 5
    assert result["structured_hidden_units"] == 12
    assert result["sparsity_unstructured"] > 0.4
    assert result["quant_per_channel_max_error"] > 0
    names = [r["name"] for r in result["rows"]]
    assert "quantized-int8-per-channel" in names and "distilled-student" in names
    # Quantized variants genuinely shrink. Per-tensor is ~1/4 of float32;
    # per-channel carries one scale pair per row, so on this tiny model
    # its saving is smaller (~48%) — the overhead trade-off, stated honestly.
    for row in result["rows"]:
        if row["name"] == "quantized-int8-per-tensor":
            assert row["size_reduction"] > 0.5
        if row["name"] == "quantized-int8-per-channel":
            assert row["size_reduction"] > 0.4
