import os
import tempfile

from transformers import LlamaConfig, LlamaForCausalLM

from mindnlp.wizard.merge.config import InputModelDefinition, MergeConfiguration
from mindnlp.wizard.merge.merge import MergeOptions, run_merge


def _make_tiny_llama(path: str, *, safe_serialization: bool) -> str:
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=4,
        num_hidden_layers=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=safe_serialization)
    return path


def test_merge_mixed_weight_formats_safetensors_and_bin(tmp_path):
    model_safetensors = _make_tiny_llama(
        str(tmp_path / "mixed_safe"), safe_serialization=True
    )
    model_bin = _make_tiny_llama(str(tmp_path / "mixed_bin"), safe_serialization=False)

    cfg = MergeConfiguration(
        merge_method="linear",
        dtype="bfloat16",
        models=[
            InputModelDefinition(model=model_safetensors, parameters={"weight": 0.5}),
            InputModelDefinition(model=model_bin, parameters={"weight": 0.5}),
        ],
    )

    with tempfile.TemporaryDirectory() as out_dir:
        run_merge(cfg, out_path=out_dir, options=MergeOptions(device="CPU"))
        assert os.path.exists(os.path.join(out_dir, "config.json"))
        assert os.path.exists(os.path.join(out_dir, "model.safetensors")) or os.path.exists(
            os.path.join(out_dir, "model.safetensors.index.json")
        )
