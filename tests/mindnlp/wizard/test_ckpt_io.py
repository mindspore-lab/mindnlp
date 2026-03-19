# Copyright 2026 MindSpore Wizard Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for .ckpt read/write support and bf16 precision safety."""

import json
import os
import tempfile
from typing import Dict

import mindspore
import numpy as np
import pytest

from mindnlp.wizard.merge.io.tensor_writer import TensorWriter
from mindnlp.wizard.merge.io.lazy_tensor_loader import ShardedTensorIndex, LazyTensorLoader
from mindnlp.wizard.merge.io.loader import (
    TensorLoader,
    LazyCkptLoader,
    DumbCkptLoader,
    SafetensorsLoader,
)
from mindnlp.wizard.merge.options import MergeOptions, VALID_OUTPUT_FORMATS


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_tensors() -> Dict[str, mindspore.Tensor]:
    return {
        "layer.0.weight": mindspore.Tensor(np.random.randn(64, 128).astype(np.float32)),
        "layer.0.bias": mindspore.Tensor(np.random.randn(64).astype(np.float32)),
        "layer.1.weight": mindspore.Tensor(np.random.randn(32, 64).astype(np.float32)),
    }


def _write_and_read_round_trip(
    output_format: str,
    tensors: Dict[str, mindspore.Tensor],
    max_shard_size: int = -1,
    lazy_loader: bool = True,
) -> Dict[str, mindspore.Tensor]:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorWriter(
            tmpdir,
            max_shard_size=max_shard_size,
            output_format=output_format,
        )
        for name, tensor in tensors.items():
            writer.save_tensor(name, tensor)
        writer.finalize()

        loader = LazyTensorLoader.from_disk(tmpdir, lazy_loader=lazy_loader)
        result = {}
        for name in tensors:
            result[name] = loader.get_tensor(name)
        return result


# ── TensorWriter format tests ────────────────────────────────────────────

class TestTensorWriterFormats:
    def test_safetensors_single_shard(self):
        tensors = _make_tensors()
        result = _write_and_read_round_trip("safetensors", tensors)
        for name, original in tensors.items():
            np.testing.assert_allclose(
                result[name].asnumpy(), original.asnumpy(), rtol=0, atol=0,
            )

    def test_safetensors_multi_shard(self):
        tensors = _make_tensors()
        result = _write_and_read_round_trip("safetensors", tensors, max_shard_size=4096)
        for name, original in tensors.items():
            np.testing.assert_allclose(
                result[name].asnumpy(), original.asnumpy(), rtol=0, atol=0,
            )

    def test_ckpt_single_shard(self):
        tensors = _make_tensors()
        result = _write_and_read_round_trip("ckpt", tensors)
        for name, original in tensors.items():
            np.testing.assert_allclose(
                result[name].asnumpy(), original.asnumpy(), rtol=0, atol=0,
            )

    def test_ckpt_multi_shard(self):
        tensors = _make_tensors()
        result = _write_and_read_round_trip("ckpt", tensors, max_shard_size=4096)
        for name, original in tensors.items():
            np.testing.assert_allclose(
                result[name].asnumpy(), original.asnumpy(), rtol=0, atol=0,
            )

    def test_unsupported_format_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported output_format"):
                TensorWriter(tmpdir, output_format="bin")

    def test_ckpt_output_file_naming(self):
        """Single ckpt shard should be named mindspore_model.ckpt."""
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="ckpt")
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()
            assert os.path.exists(os.path.join(tmpdir, "mindspore_model.ckpt"))

    def test_safetensors_output_file_naming(self):
        """Single safetensors shard should be named model.safetensors."""
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="safetensors")
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))

    def test_ckpt_sharded_index_json(self):
        """Multiple ckpt shards should produce an index.json."""
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="ckpt", max_shard_size=4096)
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()

            index_path = os.path.join(tmpdir, "mindspore_model.ckpt.index.json")
            assert os.path.exists(index_path)
            with open(index_path) as f:
                idx = json.load(f)
            assert "weight_map" in idx
            for name in tensors:
                assert name in idx["weight_map"]


# ── ShardedTensorIndex detection tests ────────────────────────────────────

class TestShardedTensorIndex:
    def test_from_disk_detects_safetensors(self):
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="safetensors")
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()

            index = ShardedTensorIndex.from_disk(tmpdir)
            assert index.format == "safetensors"
            assert index.is_safetensors is True
            for name in tensors:
                assert name in index.tensor_paths

    def test_from_disk_detects_ckpt(self):
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="ckpt")
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()

            index = ShardedTensorIndex.from_disk(tmpdir)
            assert index.format == "ckpt"
            assert index.is_safetensors is False
            for name in tensors:
                assert name in index.tensor_paths

    def test_from_disk_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="Unable to find model"):
                ShardedTensorIndex.from_disk(tmpdir)


# ── Loader factory dispatch tests ─────────────────────────────────────────

class TestLoaderFactory:
    def test_dispatch_safetensors(self):
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="safetensors")
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()

            path = os.path.join(tmpdir, "model.safetensors")
            loader = TensorLoader.get(path)
            assert isinstance(loader, SafetensorsLoader)

    def test_dispatch_ckpt_lazy(self):
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="ckpt")
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()

            path = os.path.join(tmpdir, "mindspore_model.ckpt")
            loader = TensorLoader.get(path, use_lazy_loader=True)
            assert isinstance(loader, LazyCkptLoader)

    def test_dispatch_ckpt_eager(self):
        tensors = _make_tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format="ckpt")
            for name, tensor in tensors.items():
                writer.save_tensor(name, tensor)
            writer.finalize()

            path = os.path.join(tmpdir, "mindspore_model.ckpt")
            loader = TensorLoader.get(path, use_lazy_loader=False)
            assert isinstance(loader, DumbCkptLoader)


# ── MergeOptions tests ────────────────────────────────────────────────────

class TestMergeOptions:
    def test_output_format_default(self):
        opts = MergeOptions()
        assert opts.output_format == "safetensors"

    def test_output_format_ckpt(self):
        opts = MergeOptions(output_format="ckpt")
        assert opts.output_format == "ckpt"

    def test_output_format_bin_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            MergeOptions(output_format="bin")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid output_format"):
            MergeOptions(output_format="pickle")

    def test_lazy_loader_field(self):
        opts = MergeOptions(lazy_loader=True)
        assert opts.lazy_loader is True

    def test_ckpt_load_kwargs_passthrough(self):
        opts = MergeOptions(ckpt_load_kwargs={"dec_key": b"secret"})
        assert opts.ckpt_load_kwargs == {"dec_key": b"secret"}


# ── Dtype round-trip matrix ───────────────────────────────────────────────

_DTYPE_PAIRS = [
    ("float32", np.float32, mindspore.float32),
    ("float16", np.float16, mindspore.float16),
    ("int32", np.int32, mindspore.int32),
    ("int64", np.int64, mindspore.int64),
    ("int8", np.int8, mindspore.int8),
    ("uint8", np.uint8, mindspore.uint8),
]


class TestDtypeRoundTrip:
    @pytest.mark.parametrize("fmt", ["safetensors", "ckpt"])
    @pytest.mark.parametrize("dtype_name,np_dtype,ms_dtype", _DTYPE_PAIRS)
    def test_dtype_round_trip(self, fmt, dtype_name, np_dtype, ms_dtype):
        arr = np.array([1, 2, 3, 4, 5], dtype=np_dtype)
        tensor = mindspore.Tensor(arr)
        assert tensor.dtype == ms_dtype

        result = _write_and_read_round_trip(fmt, {"test_tensor": tensor})
        out = result["test_tensor"]
        np.testing.assert_array_equal(out.asnumpy(), arr)
