import os
from pathlib import Path

import yaml
from click.testing import CliRunner
from transformers import LlamaConfig, LlamaForCausalLM

from mindnlp.wizard.merge.common import ModelReference
from mindnlp.wizard.merge.scripts.run_yaml import main as run_yaml_main


def _make_tiny_llama(path: str, vocab_size: int = 64) -> str:
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=4,
        num_hidden_layers=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return path


def test_run_yaml_cli_end_to_end_cpu(tmp_path):
    model_a = _make_tiny_llama(str(tmp_path / "cli_model_a"))
    model_b = _make_tiny_llama(str(tmp_path / "cli_model_b"))
    out_dir = tmp_path / "out"
    recipe_file = tmp_path / "recipe.yaml"

    recipe = {
        "merge_method": "linear",
        "dtype": "bfloat16",
        "models": [
            {"model": model_a, "parameters": {"weight": 0.6}},
            {"model": model_b, "parameters": {"weight": 0.4}},
        ],
    }
    recipe_file.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        run_yaml_main,
        [
            str(recipe_file),
            str(out_dir),
            "--device",
            "CPU",
            "--no-multi-npu",
            "--max-tensor-mem-gb",
            "0.000001",
            "--split-pieces",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists() or (
        out_dir / "model.safetensors.index.json"
    ).exists()
    # E2E CLI should persist recipe + execution report for debugging.
    assert (out_dir / "wizard_config.yml").exists()
    assert (out_dir / "wizard_execution_report.json").exists()


def test_run_yaml_cli_lora_reference_compat_cpu(tmp_path, monkeypatch):
    model_a = _make_tiny_llama(str(tmp_path / "cli_lora_model_a"))
    model_b = _make_tiny_llama(str(tmp_path / "cli_lora_model_b"))
    lora_a = tmp_path / "fake_lora_a"
    lora_b = tmp_path / "fake_lora_b"
    lora_a.mkdir()
    lora_b.mkdir()

    out_dir = tmp_path / "out_lora"
    recipe_file = tmp_path / "recipe_lora.yaml"

    recipe = {
        "merge_method": "linear",
        "dtype": "bfloat16",
        "models": [
            {"model": f"{model_a}+{lora_a}", "parameters": {"weight": 0.6}},
            {"model": f"{model_b}+{lora_b}", "parameters": {"weight": 0.4}},
        ],
    }
    recipe_file.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

    def _fake_merged(
        self,
        cache_dir=None,
        trust_remote_code=False,
        lora_merge_dtype=None,
    ):
        # Keep test offline/stable: emulate successful LoRA merge output path.
        return ModelReference(model=self.model)

    monkeypatch.setattr(ModelReference, "merged", _fake_merged, raising=False)

    runner = CliRunner()
    result = runner.invoke(
        run_yaml_main,
        [str(recipe_file), str(out_dir), "--device", "CPU", "--no-multi-npu"],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists() or (
        out_dir / "model.safetensors.index.json"
    ).exists()
