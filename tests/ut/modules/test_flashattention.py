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
from math import sqrt
import unittest
import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, Tensor
from mindnlp.transformers.kernel_utils import compile_kernel


class TestFlashAttention(unittest.TestCase):
    r"""
    Test module flashattention
    """

    def load_flash_cuda_kernel(self, func_name, context_length):
        """load flash cuda kernel"""
        device_target = ms.get_context("device_target")
        if device_target != "GPU":
            raise RuntimeError("FlashAttention operator only support GPU currently.")

        so_path = compile_kernel(kernel_name="flash", Tmax=context_length)
        flash_op = ops.Custom(
            f"{str(so_path)}:{func_name}",
            out_shape=lambda q, k, v, l, m: q,
            out_dtype=lambda q, k, v, l, m: q,
            func_type="aot",
        )
        flash_op.add_prim_attr("primitive_target", device_target)
        return flash_op

    def manual_attn(self, query, key, value):
        r"""
        manual attention
        """
        embed_size = query.shape[-1]
        scaling_factor = sqrt(sqrt(Tensor(embed_size, ms.float32)))
        query = query / scaling_factor
        attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)
        attn = ops.softmax(attn, -1)
        output = ops.matmul(attn, value)
        return output

    @pytest.mark.gpu_only
    def test_flashattention2_forward_FP32(self):
        r"""
        Unit test for flashattention forward.
        """
        # 加载flash cuda kernel
        op = self.load_flash_cuda_kernel("flash_forward", 512)

        profiler = ms.Profiler()

        # 定义输入数据
        Q = np.random.randn(16, 12, 64, 64).astype(np.float32)
        K = np.random.randn(16, 12, 64, 64).astype(np.float32)
        V = np.random.randn(16, 12, 64, 64).astype(np.float32)
        l = np.zeros((16, 12, 64), dtype=np.float32)
        m = np.ones((16, 12, 64), dtype=np.float32) * (-np.inf)
        print("=== profiling MindSpore manual-attention === ")
        output_manual = self.manual_attn(ms.Tensor(Q), ms.Tensor(K), ms.Tensor(V))
        # profiler.analyse()
        print("=== profiling MindSpore flash-attention === ")
        output = op(
            ms.Tensor(Q), ms.Tensor(K), ms.Tensor(V), ms.Tensor(l), ms.Tensor(m)
        )
        profiler.analyse()
        # print(output)
        assert np.allclose(output[0].asnumpy(), output_manual[0].asnumpy(), atol=1e-03)
