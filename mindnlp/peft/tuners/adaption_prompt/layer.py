# Copyright 2023-present the HuggingFace Inc. team.
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
"""layer for adaption prompt tuners."""
import math
import numpy as np
from mindspore import nn, ops, Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from .config import TRANSFORMERS_MODEL_CONFIG


class AdaptedAttention(nn.Cell):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: str, adapter_len: int, model):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        super(AdaptedAttention, self).__init__()
        self.model_type = model_type
        self.model = model
        self.adapter_len = adapter_len

        # 正确的初始化和使用 Normal 初始化器
        normal_values = np.random.normal(loc=0.0, scale=1.0, size=(adapter_len, self.model.hidden_size)).astype(
            np.float32)
        self.adaption_prompt = Parameter(Tensor(normal_values, dtype=mstype.float32), name="adaption_prompt")

        # 使用零初始化器初始化门控参数
        zero_values = np.zeros((1,), dtype=np.float32)
        self.adaption_gate = Parameter(Tensor(zero_values, dtype=mstype.float32), name="adaption_gate")
    def construct(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.
        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")

        output, _, past_key_value = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
        factor = (
                self.model.k_proj.in_features // self.model.k_proj.out_features
        )


        if k_proj_layer == v_proj_layer:
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, axis=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)

        # Operations are similar to PyTorch but using MindSpore operations
        transpose_op = P.Transpose()
        adapter_k = key.view(1, self.adapter_len, (self.model.num_heads // factor), self.model.head_dim)
        adapter_k = ops.tile(adapter_k, (bsz, 1, 1, 1))
        adapter_k = transpose_op(adapter_k, (0, 2, 1, 3))

        adapter_v = value.view(1, self.adapter_len, (self.model.num_heads // factor), self.model.head_dim)
        adapter_v = ops.tile(adapter_v, (bsz, 1, 1, 1))
        adapter_v = transpose_op(adapter_v, (0, 2, 1, 3))

        # Repeat interleave functionality
        repeat_interleave = P.Tile()
        adapter_k = repeat_interleave(adapter_k, (1, factor, 1, 1))
        adapter_v = repeat_interleave(adapter_v, (1, factor, 1, 1))

        # Recompute query states
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        query_states = compute_query_states(model=self.model, **kwargs)

        previous_dtype = query_states.dtype

        # Dot product and softmax operations
        matmul = P.BatchMatMul()
        scores = matmul(query_states, adapter_k.transpose(2, 3))
        scores /= math.sqrt(self.model.head_dim)

        softmax = nn.Softmax(axis=-1)
        scores = softmax(scores).astype(mstype.float32)  # upcasting to fp32
        scores *= self.adaption_gate

        adapter_output = matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)

        # Projection layer if exists
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        # Combine outputs
        output = output + adapter_output
        output = output.astype(previous_dtype)  # restore dtype if necessary

        return output, None, past_key_value
