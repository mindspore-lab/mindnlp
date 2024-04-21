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


import math
import mindspore
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import Normal
from .config import TRANSFORMERS_MODEL_CONFIG


class AdaptedAttention(nn.Cell):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: str, adapter_len: int, model):
        super(AdaptedAttention, self).__init__()
        assert not isinstance(model, AdaptedAttention)
        self.model_type = model_type
        self.model = model
        self.adapter_len = adapter_len
        self.hidden_size = model.hidden_size  # Assuming hidden_size is defined in model

        # 初始化adaption_prompt参数
        # MindSpore中使用mnp.empty创建未初始化的张量，然后使用Tensor正态分布初始化
        target_dtype = mstype.float32  # MindSpore默认使用float32，与PyTorch保持一致
        init_value = mnp.empty((1, adapter_len, self.hidden_size), dtype=target_dtype)
        init_value = Tensor(init_value, dtype=target_dtype).astype(target_dtype)  # 转换类型确保与target_dtype一致
        self.adaption_prompt = Parameter(init_value)
        self.adaption_prompt.set_data(init_value.normal_())  # 使用正态分布填充数据

        # 初始化gate参数
        self.adaption_gate = Parameter(Tensor(mnp.zeros(1), dtype=target_dtype))
    def construct(self, **kwargs):
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")

        output, _, past_key_value = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]

        # Get projection layers info from config
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type]['k_proj_layer']
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type]['v_proj_layer']
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type]['o_proj_layer']

        key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
        value = getattr(self.model, v_proj_layer)(self.adaption_prompt)

        adapter_k = ops.transpose(
            key.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim).repeat(bsz, 1, 1, 1), (0, 2, 1, 3))
        adapter_v = ops.transpose(
            value.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim).repeat(bsz, 1, 1, 1),
            (0, 2, 1, 3))

        query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type]['compute_query_states'](model=self.model, **kwargs)

        scores = ops.matmul(query_states, ops.transpose(adapter_k, (0, 1, 3, 2))) / math.sqrt(self.model.head_dim)
        scores = self.adaption_gate * ops.softmax(scores.astype(mindspore.float32), axis=-1).astype(query_states.dtype)

        adapter_output = ops.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)

        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        output = output + adapter_output

        return output, None, past_key_value
