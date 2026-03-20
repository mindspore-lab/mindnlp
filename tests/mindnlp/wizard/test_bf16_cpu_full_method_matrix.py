import os
import tempfile
from typing import Any, Dict, List, Tuple

import mindspore
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from mindnlp.wizard.merge.config import InputModelDefinition, MergeConfiguration
from mindnlp.wizard.merge.merge import MergeOptions, run_merge
from mindnlp.wizard.merge.merge_methods import REGISTERED_MERGE_METHODS


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


@pytest.fixture(scope="session")
def tiny_triplet(tmp_path_factory) -> Tuple[str, str, str]:
    a = _make_tiny_llama(str(tmp_path_factory.mktemp("bf16_matrix_a")))
    b = _make_tiny_llama(str(tmp_path_factory.mktemp("bf16_matrix_b")))
    c = _make_tiny_llama(str(tmp_path_factory.mktemp("bf16_matrix_c")))
    return a, b, c


def _method_recipe(method_name: str, models: Tuple[str, str, str]) -> Dict[str, Any]:
    a, b, c = models
    base_recipe: Dict[str, Any] = {
        "merge_method": method_name,
        "dtype": "bfloat16",
    }

    # Known stable shapes from existing wizard parity tests.
    if method_name == "passthrough":
        base_recipe["models"] = [
            {"model": a, "parameters": {"scale": 1.0}},
        ]
        return base_recipe

    if method_name in {"slerp", "arcee_fusion", "nearswap"}:
        base_recipe["base_model"] = a
        base_recipe["models"] = [
            {"model": a, "parameters": {"weight": 0.5}},
            {"model": b, "parameters": {"weight": 0.5}},
        ]
        base_recipe["parameters"] = {"t": 0.5}
        return base_recipe

    if method_name == "nuslerp":
        base_recipe["base_model"] = c
        base_recipe["models"] = [
            {"model": a, "parameters": {"weight": 0.5}},
            {"model": b, "parameters": {"weight": 0.5}},
        ]
        base_recipe["parameters"] = {
            "nuslerp_row_wise": False,
            "nuslerp_flatten": False,
        }
        return base_recipe

    if method_name == "model_stock":
        base_recipe["base_model"] = c
        base_recipe["models"] = [
            {"model": a, "parameters": {"weight": 0.5}},
            {"model": b, "parameters": {"weight": 0.5}},
        ]
        return base_recipe

    if method_name in {"sce"}:
        base_recipe["base_model"] = c
        base_recipe["models"] = [
            {"model": a, "parameters": {"weight": 0.5}},
            {"model": b, "parameters": {"weight": 0.5}},
        ]
        base_recipe["parameters"] = {"select_topk": 0.5}
        return base_recipe

    if method_name in {"ramplus_tl"}:
        base_recipe["base_model"] = c
        base_recipe["models"] = [
            {"model": a, "parameters": {"weight": 0.5}},
            {"model": b, "parameters": {"weight": 0.5}},
        ]
        base_recipe["parameters"] = {"r": 0.1, "alpha": 0.2}
        return base_recipe

    if method_name in {
        "task_arithmetic",
        "ties",
        "dare_ties",
        "dare_linear",
        "breadcrumbs",
        "breadcrumbs_ties",
        "della",
        "della_linear",
        "ram",
        "multislerp",
        "karcher",
    }:
        base_recipe["base_model"] = c
        base_recipe["models"] = [
            {"model": a, "parameters": {"weight": 0.5}},
            {"model": b, "parameters": {"weight": 0.5}},
        ]
        if method_name in {"ties", "dare_ties", "dare_linear", "breadcrumbs", "breadcrumbs_ties", "della", "della_linear"}:
            base_recipe["parameters"] = {"density": 0.5}
        return base_recipe

    # Fallback for newly added methods: 2-model merge.
    base_recipe["models"] = [
        {"model": a, "parameters": {"weight": 0.5}},
        {"model": b, "parameters": {"weight": 0.5}},
    ]
    return base_recipe


@pytest.mark.parametrize("method_name", sorted(list(REGISTERED_MERGE_METHODS.keys())))
def test_registered_methods_bf16_cpu_safe(method_name: str, tiny_triplet: Tuple[str, str, str]):
    mindspore.set_context(device_target="CPU")
    recipe = _method_recipe(method_name, tiny_triplet)
    cfg = MergeConfiguration.model_validate(recipe)
    with tempfile.TemporaryDirectory() as out_dir:
        run_merge(cfg, out_path=out_dir, options=MergeOptions(device="CPU"))
        assert os.path.exists(os.path.join(out_dir, "config.json"))
        assert os.path.exists(os.path.join(out_dir, "model.safetensors")) or os.path.exists(
            os.path.join(out_dir, "model.safetensors.index.json")
        )
