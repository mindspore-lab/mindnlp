import os
import tempfile
from typing import Callable, Optional

import numpy as np
import pytest
from transformers import AutoConfig, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from mindnlp.wizard.merge.config import (
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mindnlp.wizard.merge.io.lazy_tensor_loader import LazyTensorLoader
from mindnlp.wizard.merge.merge import MergeOptions, run_merge


def _run_and_check_merge(
    config: MergeConfiguration,
    check_nan: bool = True,
    validate: Optional[Callable[[str], None]] = None,
    options: Optional[MergeOptions] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        run_merge(config, out_path=tmpdir, options=options or MergeOptions())
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        assert (
            os.path.exists(os.path.join(tmpdir, "model.safetensors.index.json"))
            or os.path.exists(os.path.join(tmpdir, "model.safetensors"))
        ), "No model produced by merge"

        if check_nan:
            loader = LazyTensorLoader.from_disk(tmpdir, lazy_loader=False)
            for tensor_name in sorted(loader.index.tensor_paths.keys()):
                tensor = loader.get_tensor(tensor_name)
                assert np.isfinite(tensor.asnumpy()).all(), f"NaN/Inf found in {tensor_name}"

        if validate:
            validate(tmpdir)


def _make_picollama(path: str, vocab_size: int = 64):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=4,
        num_hidden_layers=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def _make_gpt2size(path: str):
    cfg = GPT2Config(
        n_ctx=128,
        n_embd=64,
        n_head=4,
        n_layer=4,
        n_positions=128,
        vocab_size=128,
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="session")
def model_a(tmp_path_factory):
    return _make_picollama(str(tmp_path_factory.mktemp("wizard_basic_a")))


@pytest.fixture(scope="session")
def model_b(tmp_path_factory):
    return _make_picollama(str(tmp_path_factory.mktemp("wizard_basic_b")))


@pytest.fixture(scope="session")
def model_c(tmp_path_factory):
    return _make_picollama(str(tmp_path_factory.mktemp("wizard_basic_c")))


@pytest.fixture(scope="session")
def gpt2_like(tmp_path_factory):
    return _make_gpt2size(str(tmp_path_factory.mktemp("wizard_gpt2_like")))


class TestBasicMergeParity:
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
                InputModelDefinition(model=model_a, parameters={"weight": 0.5}),
                InputModelDefinition(model=model_b, parameters={"weight": 0.5}),
            ],
            dtype="bfloat16",
        )
        if params:
            cfg.parameters = params
        return cfg

    def test_gpt2_copy(self, gpt2_like):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model=gpt2_like)],
            dtype="bfloat16",
        )
        _run_and_check_merge(config)

    def test_gpt2_stack(self, gpt2_like):
        config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[InputSliceDefinition(model=gpt2_like, layer_range=(0, 4))]
                )
            ]
            * 2,
            dtype="bfloat16",
        )

        def _validate(model_path: str):
            model_config = AutoConfig.from_pretrained(model_path)
            assert model_config.n_layer == 8

        _run_and_check_merge(config, validate=_validate)

    def test_passthrough_scale(self, model_a):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[
                InputModelDefinition(
                    model=model_a,
                    parameters={
                        "scale": [
                            {"filter": "o_proj", "value": 0},
                            {"value": 1},
                        ]
                    },
                )
            ],
            dtype="bfloat16",
        )

        def _validate(model_path: str):
            loader = LazyTensorLoader.from_disk(model_path, lazy_loader=False)
            saw_any = False
            for name in loader.index.tensor_paths:
                if "o_proj" in name:
                    param = loader.get_tensor(name).asnumpy()
                    assert (param == 0).all()
                    saw_any = True
            assert saw_any, "No o_proj parameters found"

        _run_and_check_merge(config, validate=_validate)

    def test_linear_merge(self, model_a, model_b):
        config = self._two_model_config(model_a, model_b, merge_method="linear")
        _run_and_check_merge(config)

    def test_slerp_merge(self, model_a, model_b):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="slerp",
            base_model=model_a,
            params={"t": 0.35},
        )
        _run_and_check_merge(config)

    def test_slerp_merge_chunked(self, model_a, model_b):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="slerp",
            base_model=model_a,
            params={"t": 0.35},
        )
        _run_and_check_merge(
            config,
            options=MergeOptions(
                device="CPU",
                max_tensor_mem_gb=0.000001,
                split_pieces=2,
            ),
        )

    def test_nuslerp_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="nuslerp",
            base_model=model_c,
            params={"nuslerp_row_wise": False, "nuslerp_flatten": False},
        )
        _run_and_check_merge(config)

    def test_nuslerp_merge_chunked(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="nuslerp",
            base_model=model_c,
            params={"nuslerp_row_wise": False, "nuslerp_flatten": False},
        )
        _run_and_check_merge(
            config,
            options=MergeOptions(
                device="CPU",
                max_tensor_mem_gb=0.000001,
                split_pieces=2,
            ),
        )

    def test_task_arithmetic_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a, model_b, merge_method="task_arithmetic", base_model=model_c
        )
        _run_and_check_merge(config)

    def test_ties_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="ties",
            base_model=model_c,
            params={"density": 0.3},
        )
        _run_and_check_merge(config)

    def test_sce_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="sce",
            base_model=model_c,
            params={"select_topk": 0.5},
        )
        _run_and_check_merge(config)

    def test_ram_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="ram",
            base_model=model_c,
        )
        _run_and_check_merge(config)

    def test_multislerp_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="multislerp",
            base_model=model_c,
        )
        _run_and_check_merge(config)

    def test_model_stock_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="model_stock",
            base_model=model_c,
        )
        _run_and_check_merge(config)

    def test_model_stock_filterwise_chunked_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="model_stock",
            base_model=model_c,
            params={"filter_wise": True},
        )
        _run_and_check_merge(
            config,
            options=MergeOptions(
                device="CPU",
                max_tensor_mem_gb=0.000001,
                split_pieces=2,
            ),
        )

    def test_arcee_fusion_merge(self, model_a, model_b):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="arcee_fusion",
            base_model=model_a,
        )
        _run_and_check_merge(config)

    def test_arcee_fusion_merge_chunked(self, model_a, model_b):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="arcee_fusion",
            base_model=model_a,
        )
        _run_and_check_merge(
            config,
            options=MergeOptions(
                device="CPU",
                max_tensor_mem_gb=0.000001,
                split_pieces=2,
            ),
        )

    def test_karcher_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="karcher",
            base_model=model_c,
            params={"max_iter": 5, "tol": 1e-5},
        )
        _run_and_check_merge(config)

    def test_karcher_merge_chunked(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="karcher",
            base_model=model_c,
            params={"max_iter": 5, "tol": 1e-5},
        )
        _run_and_check_merge(
            config,
            options=MergeOptions(
                device="CPU",
                max_tensor_mem_gb=0.000001,
                split_pieces=2,
            ),
        )

    def test_della_merge(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="della",
            base_model=model_c,
            params={"density": 0.66, "epsilon": 0.05, "lambda": 0.5},
        )
        _run_and_check_merge(config)

    def test_della_merge_int8_mask_chunked(self, model_a, model_b, model_c):
        config = self._two_model_config(
            model_a,
            model_b,
            merge_method="della",
            base_model=model_c,
            params={
                "density": 0.66,
                "epsilon": 0.05,
                "lambda": 0.5,
                "int8_mask": True,
            },
        )
        _run_and_check_merge(
            config,
            options=MergeOptions(
                device="CPU",
                max_tensor_mem_gb=0.000001,
                split_pieces=2,
            ),
        )
