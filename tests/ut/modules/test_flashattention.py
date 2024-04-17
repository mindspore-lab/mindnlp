# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test FlashAttention-v2-FP32"""
import unittest
import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, Tensor
from mindnlp.transformers.kernel_utils import compile_kernel
from mindnlp.utils.testing_utils import require_mindspore_gpu


class TestFlashAttention(unittest.TestCase):
    r"""
    Test module flashattention
    """

    def manual_attn(self, query, key, value):
        r"""
        manual attention
        """
        embed_size = query.shape[-1]
        scaling_factor = ops.sqrt(ops.sqrt(Tensor(embed_size, ms.float32)))
        query = query / scaling_factor
        attn_mask = ops.ones((query.shape[-2], key.shape[-2]), ms.bool_).tril()
        attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = ops.softmax(attn, -1)
        output = ops.matmul(attn, value)
        return output

    @require_mindspore_gpu
    def test_flashattention_forward_FP32(self):
        r"""
        Unit test for flashattention forward.
        """
        # 加载flash cuda kernel
        device_target = ms.get_context("device_target")
        if device_target != "GPU":
            raise RuntimeError("FlashAttention operator only support GPU currently.")

        so_path = compile_kernel(kernel_name="flash", Tmax=1024)
        flash_1_op = ops.Custom(
            f"{str(so_path)}:flash_attn_1_fwd_f32",
            out_shape=lambda q, k, v, l, m: q,
            out_dtype=lambda q, k, v, l, m: q,
            func_type="aot",
        )
        flash_1_op.add_prim_attr("primitive_target", device_target)

        flash_2_op = ops.Custom(
            f"{str(so_path)}:flash_attn_2_fwd_f32",
            out_shape=lambda q, k, v, l: q,
            out_dtype=lambda q, k, v, l: q,
            func_type="aot",
        )
        flash_2_op.add_prim_attr("primitive_target", device_target)
        profiler = ms.Profiler()

        # seq_len must be multiple of Br
        Q = np.random.randn(8, 12, 1024, 64).astype(np.float32)
        K = np.random.randn(8, 12, 1024, 64).astype(np.float32)
        V = np.random.randn(8, 12, 1024, 64).astype(np.float32)
        l = np.zeros((8, 12, 1024), dtype=np.float32)
        m = np.ones((8, 12, 1024), dtype=np.float32) * (-np.inf)
        print("=== profiling MindSpore manual-attention === ")
        output_manual = self.manual_attn(ms.Tensor(Q), ms.Tensor(K), ms.Tensor(V))
        print("=== profiling MindSpore flash-attention-v1 === ")
        output1 = flash_1_op(
            ms.Tensor(Q),
            ms.Tensor(K),
            ms.Tensor(V),mind
            ms.Tensor(l),
            ms.Tensor(m),
        )
        print("=== profiling MindSpore flash-attention-v2 === ")
        output2 = flash_2_op(
            ms.Tensor(Q),
            ms.Tensor(K),
            ms.Tensor(V),
            ms.Tensor(l),
        )
        profiler.analyse()
        # print("manual_out:\n", output_manual[0].asnumpy())
        # print("flash1_out:\n", output1[0].asnumpy())
        # print("flash2_out:\n", output2[0].asnumpy())
        assert np.allclose(output1[0].asnumpy(), output_manual[0].asnumpy(), atol=1e-02)
        assert np.allclose(output2[0].asnumpy(), output_manual[0].asnumpy(), atol=1e-02)
        print("=== flash_attn test pass === ")
