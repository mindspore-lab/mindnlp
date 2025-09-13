# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Mistral模型配置文件."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class MoeConfig:
    """混合专家(MoE)层的配置类."""
    num_experts: int = 8  # 专家网络总数
    num_experts_per_tok: int = 2  # 每个token激活的专家数量


@dataclass
class MistralConfig:
    """
    存储Mistral模型配置的配置类.
    
    参数:
        vocab_size (int): Mistral模型的词汇表大小. 默认为32000.
        hidden_size (int): 隐藏表示的维度. 默认为4096.
        intermediate_size (int): MLP表示的维度. 默认为14336.
        num_hidden_layers (int): Transformer编码器中隐藏层的数量. 默认为32.
        num_attention_heads (int): 每个注意力层的注意力头数量. 默认为32.
        num_key_value_heads (int): 键值头的数量. 默认为8.
        head_dim (int): 注意力头的维度. 默认为128.
        hidden_act (str): 解码器中的非线性激活函数. 默认为"silu".
        max_position_embeddings (int): 最大序列长度. 默认为32768.
        initializer_range (float): 截断正态初始化器的标准差. 默认为0.02.
        rms_norm_eps (float): RMS归一化层使用的epsilon值. 默认为1e-05.
        use_cache (bool): 模型是否应该返回最后的键/值注意力. 默认为True.
        pad_token_id (int): 填充token的id. 默认为None.
        bos_token_id (int): 序列开始token的id. 默认为1.
        eos_token_id (int): 序列结束token的id. 默认为2.
        tie_word_embeddings (bool): 是否绑定词嵌入权重. 默认为False.
        rope_theta (float): RoPE嵌入的基础周期. 默认为10000.0.
        sliding_window (int): 滑动窗口注意力的窗口大小. 如果为None则无滑动窗口. 默认为4096.
        attention_dropout (float): 注意力概率的dropout比率. 默认为0.0.
        max_batch_size (int): 最大批次大小. 默认为1.
        output_attentions (bool): 是否输出注意力权重. 默认为False.
        output_hidden_states (bool): 是否输出隐藏状态. 默认为False.
        moe (MoeConfig): MoE层的配置. 如果为None则使用密集层. 默认为None.
    """
    
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = 4096
    attention_dropout: float = 0.0
    max_batch_size: int = 1
    # 输出控制配置
    output_attentions: bool = False
    output_hidden_states: bool = False
    # MoE specific
    moe: Optional[MoeConfig] = None
    
    @property
    def model_type(self):
        """返回模型类型标识符."""
        return "mistral"
    
    def to_dict(self):
        """将配置转换为字典格式."""
        output = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, MoeConfig):
                    output[key] = {"num_experts": value.num_experts, "num_experts_per_tok": value.num_experts_per_tok}
                else:
                    output[key] = value
        return output
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置对象."""
        moe_config = config_dict.get("moe")
        if moe_config:
            config_dict["moe"] = MoeConfig(**moe_config)
        return cls(**config_dict)
