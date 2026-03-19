import os
import tempfile
from typing import Callable, Optional

import numpy as np
import pytest
from transformers import AutoConfig, CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

from mindnlp.wizard.merge.config import InputModelDefinition, MergeConfiguration
from mindnlp.wizard.merge.io.lazy_tensor_loader import LazyTensorLoader
from mindnlp.wizard.merge.merge import MergeOptions, run_merge


def _run_and_check_merge(
    config: MergeConfiguration,
    validate: Optional[Callable[[str], None]] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        run_merge(config, out_path=tmpdir, options=MergeOptions())
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        assert (
            os.path.exists(os.path.join(tmpdir, "model.safetensors.index.json"))
            or os.path.exists(os.path.join(tmpdir, "model.safetensors"))
        ), "No model produced by merge"

        loader = LazyTensorLoader.from_disk(tmpdir, lazy_loader=False)
        for tensor_name in sorted(loader.index.tensor_paths.keys()):
            tensor = loader.get_tensor(tensor_name)
            assert np.isfinite(tensor.asnumpy()).all(), f"NaN/Inf found in {tensor_name}"

        if validate:
            validate(tmpdir)


def _make_pico_llava(path: str):
    vision_config = CLIPVisionConfig(
        image_size=32,
        patch_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
    )
    text_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=4,
        num_hidden_layers=2,
    )
    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_seq_length=16,
    )
    model = LlavaForConditionalGeneration(config=llava_config)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="session")
def vlm_a(tmp_path_factory):
    return _make_pico_llava(str(tmp_path_factory.mktemp("wizard_vlm_a")))


@pytest.fixture(scope="session")
def vlm_b(tmp_path_factory):
    return _make_pico_llava(str(tmp_path_factory.mktemp("wizard_vlm_b")))


@pytest.fixture(scope="session")
def vlm_c(tmp_path_factory):
    return _make_pico_llava(str(tmp_path_factory.mktemp("wizard_vlm_c")))


class TestVlmMergeParity:
    def _two_model_config(
        self,
        model_a: str,
        model_b: str,
        *,
        merge_method: str,
        base_model: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> MergeConfiguration:
        cfg = MergeConfiguration(
            merge_method=merge_method,
            base_model=base_model,
            models=[
                InputModelDefinition(model=model_a, parameters={"weight": 0.6}),
                InputModelDefinition(model=model_b, parameters={"weight": 0.4}),
            ],
            dtype="bfloat16",
        )
        if params:
            cfg.parameters = params
        return cfg

    def _validate_llava(self, model_path: str):
        cfg = AutoConfig.from_pretrained(model_path)
        assert cfg.model_type == "llava"

    def test_linear_vlm_merge(self, vlm_a, vlm_b):
        config = self._two_model_config(vlm_a, vlm_b, merge_method="linear")
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_slerp_vlm_merge(self, vlm_a, vlm_b):
        config = self._two_model_config(
            vlm_a, vlm_b, merge_method="slerp", base_model=vlm_a, params={"t": 0.35}
        )
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_nuslerp_vlm_merge(self, vlm_a, vlm_b, vlm_c):
        config = self._two_model_config(
            vlm_a,
            vlm_b,
            merge_method="nuslerp",
            base_model=vlm_c,
            params={"nuslerp_row_wise": False, "nuslerp_flatten": False},
        )
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_task_arithmetic_vlm_merge(self, vlm_a, vlm_b, vlm_c):
        config = self._two_model_config(
            vlm_a, vlm_b, merge_method="task_arithmetic", base_model=vlm_c
        )
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_breadcrumbs_vlm_merge(self, vlm_a, vlm_b, vlm_c):
        config = self._two_model_config(
            vlm_a, vlm_b, merge_method="breadcrumbs", base_model=vlm_c
        )
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_ties_vlm_merge(self, vlm_a, vlm_b, vlm_c):
        config = self._two_model_config(
            vlm_a,
            vlm_b,
            merge_method="ties",
            base_model=vlm_c,
            params={"density": 0.3},
        )
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_dare_ties_vlm_merge(self, vlm_a, vlm_b, vlm_c):
        config = self._two_model_config(
            vlm_a,
            vlm_b,
            merge_method="dare_ties",
            base_model=vlm_c,
            params={"density": 0.66},
        )
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_model_stock_vlm_merge(self, vlm_a, vlm_b, vlm_c):
        config = self._two_model_config(
            vlm_b, vlm_c, merge_method="model_stock", base_model=vlm_a
        )
        _run_and_check_merge(config, validate=self._validate_llava)

    def test_model_stock_filterwise_vlm_merge(self, vlm_a, vlm_b, vlm_c):
        config = self._two_model_config(
            vlm_b,
            vlm_c,
            merge_method="model_stock",
            base_model=vlm_a,
            params={"filter_wise": True},
        )
        _run_and_check_merge(config, validate=self._validate_llava)
