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

    def manual_attn_forward(self, query, key, value):
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

    def manual_attn_backward(self, query, key, value):
        return ms.grad(self.manual_attn_forward, grad_position=(0, 1, 2))(
            query, key, value
        )

    @require_mindspore_gpu
    def test_flashattention2_fp32(self):
        r"""
        Unit test for flashattention forward.
        """
        # 加载flash cuda kernel
        device_target = ms.get_context("device_target")
        if device_target != "GPU":
            raise RuntimeError("FlashAttention operator only support GPU currently.")

        so_path = compile_kernel(kernel_name="flash", Tmax=1024)
        flash_1_fwd_op = ops.Custom(
            f"{str(so_path)}:flash_attn_1_fwd_f32",
            out_shape=lambda q, k, v: (q, (q[0], q[1], q[2]), (q[0], q[1], q[2])),
            out_dtype=lambda q, k, v: (q, q, q),
            func_type="aot",
        )
        flash_1_fwd_op.add_prim_attr("primitive_target", device_target)

        flash_2_fwd_op = ops.Custom(
            f"{str(so_path)}:flash_attn_2_fwd_f32",
            out_shape=lambda q, k, v,: (q, (q[0], q[1], q[2])),
            out_dtype=lambda q, k, v,: (q, q),
            func_type="aot",
        )
        flash_2_fwd_op.add_prim_attr("primitive_target", device_target)
        profiler = ms.Profiler()

        # seq_len must be multiple of Br
        Q = np.random.randn(8, 12, 1024, 64).astype(np.float32)
        K = np.random.randn(8, 12, 1024, 64).astype(np.float32)
        V = np.random.randn(8, 12, 1024, 64).astype(np.float32)
        print("====== profiling forward pass ======")
        print("=== profiling MindSpore manual-attention(forward) === ")
        output_manual_fwd = self.manual_attn_forward(
            ms.Tensor(Q), ms.Tensor(K), ms.Tensor(V)
        )
        print("=== profiling MindSpore flash-attention-v1(forward) === ")
        output1_fwd, l, m = flash_1_fwd_op(
            ms.Tensor(Q),
            ms.Tensor(K),
            ms.Tensor(V),
        )
        print(output1_fwd)
        assert np.allclose(
            output1_fwd[0].asnumpy(), output_manual_fwd[0].asnumpy(), atol=1e-02
        )
        print("=== profiling MindSpore flash-attention-v2(forward) === ")
        output2_fwd, L = flash_2_fwd_op(
            ms.Tensor(Q),
            ms.Tensor(K),
            ms.Tensor(V),
        )

        assert np.allclose(
            output2_fwd[0].asnumpy(), output_manual_fwd[0].asnumpy(), atol=1e-02
        )
        print("=== MindSpore flash-attention(forward) pass === \n")

        print("====== profiling backward pass ======")

        flash_1_bwd_op = ops.Custom(
            f"{str(so_path)}:flash_attn_1_bwd_f32",
            out_shape=lambda q, k, v, do, o, l, m: (q, q, q),
            out_dtype=lambda q, k, v, do, o, l, m: (q, q, q),
            func_type="aot",
        )
        flash_1_bwd_op.add_prim_attr("primitive_target", device_target)

        flash_2_bwd_op = ops.Custom(
            f"{str(so_path)}:flash_attn_2_bwd_f32",
            out_shape=lambda q, k, v, o, do, l: (q, q, q),
            out_dtype=lambda q, k, v, o, do, l: (q, q, q),
            func_type="aot",
        )
        flash_2_bwd_op.add_prim_attr("primitive_target", device_target)

        y_grad = np.ones_like(output_manual_fwd.asnumpy(), dtype=np.float32)

        print("=== profiling MindSpore manual-attention(backward) === ")
        (
            output_manual_bwd_dq,
            output_manual_bwd_dk,
            output_manual_bwd_dv,
        ) = self.manual_attn_backward(
            ms.Tensor(Q),
            ms.Tensor(K),
            ms.Tensor(V),
        )
        print("=== profiling MindSpore flash-attention-v1(backward) === ")
        (
            output1_bwd_dq,
            output1_bwd_dk,
            output1_bwd_dv,
        ) = flash_1_bwd_op(
            ms.Tensor(Q),
            ms.Tensor(K),
            ms.Tensor(V),
            ms.Tensor(output1_fwd),
            ms.Tensor(y_grad),
            ms.Tensor(l),
            ms.Tensor(m),
        )
        assert np.allclose(
            output1_bwd_dq[0].asnumpy(), output_manual_bwd_dq[0].asnumpy(), atol=1e-02
        )
        print("=== profiling MindSpore flash-attention-v2(backward) === ")
        (
            output2_bwd_dq,
            output2_bwd_dk,
            output2_bwd_dv,
        ) = flash_2_bwd_op(
            ms.Tensor(Q),
            ms.Tensor(K),
            ms.Tensor(V),
            ms.Tensor(output2_fwd),
            ms.Tensor(y_grad),
            ms.Tensor(L),
        )
        profiler.analyse()
        assert np.allclose(
            output2_bwd_dq[0].asnumpy(), output_manual_bwd_dq[0].asnumpy(), atol=1e-02
        )
        print("=== MindSpore flash-attention(backward) pass === ")
