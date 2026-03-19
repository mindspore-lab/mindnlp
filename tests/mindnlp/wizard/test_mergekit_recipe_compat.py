import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from transformers import LlamaConfig, LlamaForCausalLM

from mindnlp.wizard.merge.config import (
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
)
from mindnlp.wizard.merge.merge import MergeOptions, run_merge


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
def tiny_models(tmp_path_factory):
    a = _make_tiny_llama(str(tmp_path_factory.mktemp("recipe_compat_a")))
    b = _make_tiny_llama(str(tmp_path_factory.mktemp("recipe_compat_b")))
    c = _make_tiny_llama(str(tmp_path_factory.mktemp("recipe_compat_c")))
    return [a, b, c]


def _run_recipe(recipe: Dict[str, Any]) -> None:
    cfg = MergeConfiguration.model_validate(recipe)
    with tempfile.TemporaryDirectory() as out_dir:
        run_merge(cfg, out_path=out_dir, options=MergeOptions())
        assert os.path.exists(os.path.join(out_dir, "config.json"))
        assert os.path.exists(os.path.join(out_dir, "model.safetensors")) or os.path.exists(
            os.path.join(out_dir, "model.safetensors.index.json")
        )


def _remap_models(recipe: Dict[str, Any], tiny_models: List[str]) -> Dict[str, Any]:
    """Map external recipe model ids to local tiny models for offline CI."""
    out = dict(recipe)
    if "models" in out and isinstance(out["models"], list):
        remapped = []
        for idx, model_entry in enumerate(out["models"]):
            entry = dict(model_entry)
            entry["model"] = tiny_models[idx % len(tiny_models)]
            remapped.append(entry)
        out["models"] = remapped
    if out.get("base_model"):
        out["base_model"] = tiny_models[0]
    if "slices" in out and isinstance(out["slices"], list):
        remapped_slices = []
        for slice_entry in out["slices"]:
            s = dict(slice_entry)
            if "sources" in s and isinstance(s["sources"], list):
                remapped_sources = []
                for idx, src in enumerate(s["sources"]):
                    src_out = dict(src)
                    src_out["model"] = tiny_models[idx % len(tiny_models)]
                    # Tiny Llama fixtures have 2 transformer layers.
                    if "layer_range" in src_out:
                        src_out["layer_range"] = [0, 2]
                    remapped_sources.append(src_out)
                s["sources"] = remapped_sources
            remapped_slices.append(s)
        out["slices"] = remapped_slices
    return out


@pytest.mark.parametrize(
    "recipe",
    [
        {
            "merge_method": "linear",
            "dtype": "float16",
            "models": [
                {"model": "dummy_a", "parameters": {"weight": 0.6}},
                {"model": "dummy_b", "parameters": {"weight": 0.4}},
            ],
        },
        {
            "merge_method": "ties",
            "base_model": "dummy_a",
            "dtype": "float16",
            "parameters": {"normalize": True, "int8_mask": True},
            "models": [
                {"model": "dummy_a", "parameters": {"density": 0.7, "weight": 0.5}},
                {"model": "dummy_b", "parameters": {"density": 0.5, "weight": 0.5}},
            ],
        },
        {
            "merge_method": "slerp",
            "base_model": "dummy_a",
            "dtype": "float16",
            "parameters": {"t": 0.35},
            "models": [
                {"model": "dummy_a", "parameters": {"weight": 0.5}},
                {"model": "dummy_b", "parameters": {"weight": 0.5}},
            ],
        },
    ],
)
def test_local_mergekit_style_recipes(recipe: Dict[str, Any], tiny_models: List[str]):
    remapped = _remap_models(recipe, tiny_models)
    _run_recipe(remapped)


MERGEKIT_EXAMPLE_RECIPES = {
    "linear": {
        "merge_method": "linear",
        "dtype": "float16",
        "models": [
            {"model": "model_a", "parameters": {"weight": 0.6}},
            {"model": "model_b", "parameters": {"weight": 0.4}},
        ],
    },
    "ties": {
        "merge_method": "ties",
        "base_model": "model_a",
        "dtype": "float16",
        "parameters": {"normalize": True, "int8_mask": True},
        "models": [
            {"model": "model_a", "parameters": {"density": 0.7, "weight": 0.5}},
            {"model": "model_b", "parameters": {"density": 0.5, "weight": 0.5}},
        ],
    },
    "arcee_fusion": {
        "merge_method": "arcee_fusion",
        "base_model": "model_a",
        "dtype": "float16",
        "models": [
            {"model": "model_a", "parameters": {"weight": 0.5}},
            {"model": "model_b", "parameters": {"weight": 0.5}},
        ],
    },
}


@pytest.mark.parametrize("example_name", list(MERGEKIT_EXAMPLE_RECIPES.keys()))
def test_mergekit_examples_are_compatible_after_model_remap(
    example_name: str, tiny_models: List[str]
):
    """
    Skeleton compatibility gate:
    - uses inline MergeKit-style example recipes (no external file dependency)
    - remaps placeholder model ids to local tiny models
    - runs wizard end-to-end
    """
    recipe = MERGEKIT_EXAMPLE_RECIPES[example_name]
    remapped = _remap_models(recipe, tiny_models)
    _run_recipe(remapped)


def test_recipe_shape_validation_guard(tiny_models: List[str]):
    # Keep one negative case so compat test fails loudly on schema regressions.
    with pytest.raises(Exception):
        MergeConfiguration(
            merge_method="linear",
            models=[InputModelDefinition(model=tiny_models[0])],
            slices=[
                InputSliceDefinition(model=tiny_models[0], layer_range=(0, 1))
            ],
        )
