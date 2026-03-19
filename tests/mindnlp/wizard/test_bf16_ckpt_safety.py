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

"""P0: BFloat16 precision regression tests.

Verifies that bfloat16 tensors survive a write → read round trip
through both safetensors and ckpt formats *without* precision loss.
This is a direct regression test for the historical .bin bf16 bug
where raw bytes were misinterpreted as uint16.
"""

import tempfile

import mindspore
import numpy as np
import pytest

from mindnlp.wizard.merge.io.tensor_writer import TensorWriter
from mindnlp.wizard.merge.io.lazy_tensor_loader import LazyTensorLoader
from mindnlp.wizard.merge.io.loader import TensorLoader, LazyCkptLoader
from mindnlp.wizard.merge.dtype_policy import (
    mindspore_to_numpy,
    numpy_to_mindspore,
)


def _bf16_reference_values():
    """Known bf16 values whose byte patterns differ from uint16/float16."""
    try:
        import ml_dtypes
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    values = np.array(
        [0.1, -0.1, 1.5, -1.5, 3.14, 65504.0, 1e-7, -1e-7],
        dtype=np.float32,
    )
    bf16_arr = values.astype(ml_dtypes.bfloat16)
    return bf16_arr


class TestBF16PrecisionSafety:
    """BF16 round-trip through safetensors and ckpt must be bit-exact."""

    def _round_trip_bf16(self, output_format: str, lazy_loader: bool = True):
        try:
            import ml_dtypes  # noqa: F401
        except ImportError:
            pytest.skip("ml_dtypes not installed")

        bf16_arr = _bf16_reference_values()
        tensor = numpy_to_mindspore(bf16_arr)
        assert tensor.dtype == mindspore.bfloat16

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TensorWriter(tmpdir, output_format=output_format)
            writer.save_tensor("bf16_test", tensor)
            writer.finalize()

            loader = LazyTensorLoader.from_disk(tmpdir, lazy_loader=lazy_loader)
            restored = loader.get_tensor("bf16_test")

        assert restored.dtype == mindspore.bfloat16

        original_bytes = mindspore_to_numpy(tensor).tobytes()
        restored_bytes = mindspore_to_numpy(restored).tobytes()
        assert original_bytes == restored_bytes, (
            "BF16 byte-level mismatch detected — precision loss in round trip"
        )

        original_f32 = mindspore_to_numpy(tensor).astype(np.float32)
        restored_f32 = mindspore_to_numpy(restored).astype(np.float32)
        np.testing.assert_array_equal(original_f32, restored_f32)

    @pytest.mark.parametrize("fmt", ["safetensors", "ckpt"])
    def test_bf16_round_trip_lazy(self, fmt):
        self._round_trip_bf16(fmt, lazy_loader=True)

    @pytest.mark.parametrize("fmt", ["safetensors", "ckpt"])
    def test_bf16_round_trip_eager(self, fmt):
        self._round_trip_bf16(fmt, lazy_loader=False)

    def test_bf16_numpy_mindspore_conversion_identity(self):
        """numpy_to_mindspore(mindspore_to_numpy(t)) must be bit-exact for bf16."""
        try:
            import ml_dtypes
        except ImportError:
            pytest.skip("ml_dtypes not installed")

        bf16_arr = _bf16_reference_values()
        tensor = numpy_to_mindspore(bf16_arr)
        arr_back = mindspore_to_numpy(tensor)

        assert arr_back.dtype == ml_dtypes.bfloat16
        np.testing.assert_array_equal(
            bf16_arr.view(np.uint16),
            arr_back.view(np.uint16),
        )

    def test_bf16_not_misinterpreted_as_uint16(self):
        """Guard against the specific historical bug: treating bf16 bytes as uint16."""
        try:
            import ml_dtypes
        except ImportError:
            pytest.skip("ml_dtypes not installed")

        bf16_arr = np.array([3.14], dtype=ml_dtypes.bfloat16)
        raw_bytes = bf16_arr.tobytes()

        correct = np.frombuffer(raw_bytes, dtype=ml_dtypes.bfloat16)
        wrong = np.frombuffer(raw_bytes, dtype=np.uint16)

        correct_f32 = float(correct[0])
        wrong_f32 = float(wrong[0])
        assert abs(correct_f32 - 3.14) < 0.1
        assert abs(wrong_f32 - 3.14) > 1.0, (
            "Test sanity check failed — uint16 interpretation should differ"
        )
