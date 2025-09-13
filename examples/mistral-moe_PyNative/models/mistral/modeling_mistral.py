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
"""基于MindSpore的Mistral模型，支持混合专家(MoE)架构."""

import math
from typing import List, Optional, Tuple

import mindspore
import numpy as np
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from .configuration_mistral import MistralConfig, MoeConfig


class MistralEmbedding(nn.Cell):
    """
    MindSpore兼容的Embedding层
    类似于mistral-mindspore中的实现，确保有weight属性
    """
    def __init__(self, vocab_size, embedding_size, padding_idx=None, dtype=mindspore.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        self.dtype = dtype
        
        # 创建权重参数
        self.weight = Parameter(
            initializer(Normal(sigma=0.02), [vocab_size, embedding_size], dtype),
            name='weight'
        )
        
        # 如果有padding_idx，将其初始化为0
        if padding_idx is not None:
            self._init_padding_idx()
    
    def _init_padding_idx(self):
        """初始化padding索引为0"""
        if self.padding_idx is not None:
            # 将padding_idx对应的embedding设置为0
            self.weight.data[self.padding_idx] = 0
    
    def construct(self, input_ids):
        """
        前向传播
        
        参数:
            input_ids: 输入的token id张量
            
        返回:
            嵌入向量
        """
        # 获取输出形状
        out_shape = input_ids.shape + (self.embedding_size,)
        flat_ids = input_ids.reshape((-1,))
        
        # 使用gather操作获取嵌入
        output_for_reshape = ops.gather(self.weight, flat_ids, 0)
        output = output_for_reshape.reshape(out_shape)
        
        return output


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码(RoPE)的频率张量。
    
    参数:
        dim: 注意力头的维度
        end: 最大序列长度
        theta: 基础频率参数
    
    返回:
        freqs_cos: 余弦频率张量
        freqs_sin: 正弦频率张量
    """
    freqs = 1.0 / (theta ** (ops.arange(0, dim, 2, dtype=mindspore.float32)[: (dim // 2)] / dim))
    t = ops.arange(end, dtype=mindspore.float32)
    freqs = ops.outer(t, freqs)
    # 创建复数表示的实部和虚部
    freqs_cos = ops.cos(freqs)
    freqs_sin = ops.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(
    xq: mindspore.Tensor,
    xk: mindspore.Tensor,
    freqs_cos: mindspore.Tensor,
    freqs_sin: mindspore.Tensor
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    """
    对查询和键张量应用旋转位置编码。
    基于MindSpore 2.6 API实现，参考MindNLP标准。
    
    参数:
        xq: 查询张量 [bsz, num_heads, q_len, head_dim]
        xk: 键张量 [bsz, num_heads, q_len, head_dim]
        freqs_cos: 余弦频率张量 [q_len, head_dim//2]
        freqs_sin: 正弦频率张量 [q_len, head_dim//2]
    
    返回:
        应用RoPE后的查询和键张量
    """
    # 获取形状信息
    bsz, num_heads, q_len, head_dim = xq.shape
    
    # 重塑张量以进行复数乘法运算
    xq_r = xq[..., : head_dim // 2]  # [bsz, num_heads, q_len, head_dim//2]
    xq_i = xq[..., head_dim // 2 :]  # [bsz, num_heads, q_len, head_dim//2]
    xk_r = xk[..., : head_dim // 2]  # [bsz, num_heads, q_len, head_dim//2]
    xk_i = xk[..., head_dim // 2 :]  # [bsz, num_heads, q_len, head_dim//2]
    
    # 参考官方实现：freqs_cis[:, None, :]
    # freqs_cos/freqs_sin: [q_len, head_dim//2] -> [q_len, 1, head_dim//2]
    cos = freqs_cos[:, None, :]
    sin = freqs_sin[:, None, :]
    
    # 应用旋转变换，现在形状能正确广播
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos
    
    # 重新拼接实部和虚部
    xq_out = ops.concat([xq_out_r, xq_out_i], axis=-1)
    xk_out = ops.concat([xk_out_r, xk_out_i], axis=-1)
    
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """
    键值重复函数，等价于ops.repeat_interleave(x, axis=1, repeats=n_rep)。
    将隐藏状态从(batch, num_key_value_heads, seqlen, head_dim)
    转换为(batch, num_attention_heads, seqlen, head_dim)
    
    参数:
        hidden_states: 输入的键或值张量
        n_rep: 重复次数
    
    返回:
        重复后的张量
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 使用tile替代expand来避免兼容性问题
    hidden_states = hidden_states.unsqueeze(2)  # [batch, num_kv_heads, 1, slen, head_dim]
    hidden_states = ops.tile(hidden_states, (1, 1, n_rep, 1, 1))  # [batch, num_kv_heads, n_rep, slen, head_dim]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralRMSNorm(nn.Cell):
    """RMS归一化层，用于稳定训练过程。"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        前向传播函数。
        
        参数:
            hidden_states: 输入的隐藏状态张量
        
        返回:
            归一化后的张量
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class MistralMLP(nn.Cell):
    """Mistral多层感知机层，使用SwiGLU激活函数。"""
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)

    def silu(self, x):
        """SiLU激活函数实现: x * sigmoid(x)"""
        return x * ops.sigmoid(x)

    def construct(self, x):
        """
        前向传播，实现SwiGLU激活。
        
        参数:
            x: 输入张量
            
        返回:
            变换后的张量
        """
        gate_output = self.silu(self.gate_proj(x))  # 门控输出经过SiLU
        up_output = self.up_proj(x)  # 上投影输出
        return self.down_proj(gate_output * up_output)  # 门控与上投影相乘后下投影


class MistralAttention(nn.Cell):
    """
    Mistral多头注意力机制，基于'Attention Is All You Need'论文实现。
    支持分组查询注意力(GQA)和滑动窗口注意力。
    """
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 查询头数量
        self.head_dim = config.head_dim  # 每个头的维度
        self.num_key_value_heads = config.num_key_value_heads  # 键值头数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 分组数量
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        # 查询、键、值投影层
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=False)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)
        
        # 初始化旋转位置编码
        self._init_rope()

    def _init_rope(self):
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            self.head_dim,
            self.max_position_embeddings,
            self.rope_theta
        )

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        # 应用旋转位置编码
        if position_ids is None:
            position_ids = ops.arange(0, q_len, dtype=mindspore.int64)
        
        # 确保position_ids是1D的，避免产生额外维度
        if position_ids.ndim > 1:
            # 如果是多维的，取第一个batch的position_ids（通常都相同）
            position_ids = position_ids[0]
        
        # 确保position_ids长度与q_len匹配
        if position_ids.shape[0] != q_len:
            # 如果长度不匹配，重新生成position_ids
            position_ids = ops.arange(0, q_len, dtype=mindspore.int64)
            
        # 获取对应位置的频率，freqs_cos/sin的形状是[max_seq_len, head_dim//2]
        cos_cached = self.freqs_cos[position_ids]  # [q_len, head_dim//2]
        sin_cached = self.freqs_sin[position_ids]  # [q_len, head_dim//2]
        
        # 验证索引后的形状
        if cos_cached.ndim != 2 or sin_cached.ndim != 2:
            raise ValueError(f"频率张量应该是2维，但得到 cos: {cos_cached.shape}, sin: {sin_cached.shape}")
        
        # RoPE需要正确的维度来匹配query/key: [bsz, num_heads, q_len, head_dim//2]
        # 参考官方实现，简单添加维度让广播自动处理
        # cos_cached, sin_cached: [q_len, head_dim//2]
        
        # 使用MindSpore 2.6原生RoPE支持
        # 检查是否有原生的rotary_position_embedding
        try:
            # 尝试使用MindSpore原生RoPE实现
            query_states, key_states = ops.rotary_position_embedding(
                query_states, key_states, cos_cached, sin_cached, position_ids
            )
        except Exception:
            # 如果原生实现不可用，回退到手动实现
            # 将query/key转换为正确的形状用于RoPE计算
            query_states_reshaped = query_states.swapaxes(1, 2)  # [bsz, num_heads, q_len, head_dim]
            key_states_reshaped = key_states.swapaxes(1, 2)
            
            # 调用我们的apply_rotary_emb函数，它会处理广播
            query_states_reshaped, key_states_reshaped = apply_rotary_emb(
                query_states_reshaped, key_states_reshaped, cos_cached, sin_cached
            )
            
            # 转换回原来的形状
            query_states = query_states_reshaped.swapaxes(1, 2)
            key_states = key_states_reshaped.swapaxes(1, 2)

        if past_key_value is not None:
            # 与过去的键值连接
            key_states = ops.concat([past_key_value[0], key_states], axis=2)
            value_states = ops.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).astype(query_states.dtype)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2).reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralMoELayer(nn.Cell):
    """Mixture of Experts layer for Mistral."""
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.moe.num_experts
        self.num_experts_per_tok = config.moe.num_experts_per_tok
        self.hidden_size = config.hidden_size
        
        # Gate network
        self.gate = nn.Dense(self.hidden_size, self.num_experts, has_bias=False)
        
        # Expert networks
        self.experts = nn.CellList([MistralMLP(config) for _ in range(self.num_experts)])

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        MoE层的前向传播。
        
        参数:
            hidden_states: 输入的隐藏状态张量，形状为[batch_size, seq_len, hidden_dim]
        
        返回:
            经过MoE处理的输出张量，形状与输入相同
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
        
        # 计算路由器logits并选择top-k专家
        router_logits = self.gate(hidden_states_flat)
        routing_weights, selected_experts = ops.topk(router_logits, self.num_experts_per_tok)
        routing_weights = ops.softmax(routing_weights, axis=-1)
        
        # 初始化输出张量
        output = ops.zeros_like(hidden_states_flat)
        
        # 处理每个专家（简化的实现）
        for expert_idx in range(self.num_experts):
            # 找到分配给这个专家的tokens
            expert_mask = (selected_experts == expert_idx)
            token_mask = expert_mask.any(axis=-1)
            
            if token_mask.any():
                # 获取被选中的token索引
                token_indices = ops.nonzero(token_mask).squeeze(-1)
                if token_indices.numel() == 0:
                    continue
                    
                # 确保token_indices是一维的
                if token_indices.ndim == 0:
                    token_indices = token_indices.unsqueeze(0)
                
                # 计算这个专家的权重
                expert_weights = ops.zeros(token_indices.shape[0], dtype=routing_weights.dtype)
                for i, token_idx in enumerate(token_indices):
                    # 找到这个token在哪些位置选择了当前专家
                    positions = ops.nonzero(selected_experts[token_idx] == expert_idx).squeeze(-1)
                    if positions.numel() > 0:
                        if positions.ndim == 0:
                            positions = positions.unsqueeze(0)
                        expert_weights[i] = routing_weights[token_idx, positions].sum()
                
                # 通过专家处理tokens
                expert_input = hidden_states_flat[token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                
                # 添加加权的专家输出
                weighted_output = expert_weights.unsqueeze(-1) * expert_output
                output[token_indices] += weighted_output
        
        return output.reshape(batch_size, seq_len, hidden_dim)


class MistralDecoderLayer(nn.Cell):
    """Mistral decoder layer."""
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config=config)
        
        # Use MoE or standard MLP based on config
        if config.moe is not None:
            self.mlp = MistralMoELayer(config)
        else:
            self.mlp = MistralMLP(config)
            
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`mindspore.Tensor`, *optional*):
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MistralPreTrainedModel(nn.Cell):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]

    def __init__(self, config: MistralConfig, **kwargs):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Dense):
            # Initialize using normal distribution
            module.weight.set_data(initializer(Normal(self.config.initializer_range),
                                              module.weight.shape,
                                              module.weight.dtype))
            if module.bias is not None:
                module.bias.set_data(initializer('zeros', module.bias.shape, module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            module.weight.set_data(initializer(Normal(self.config.initializer_range),
                                              module.weight.shape,
                                              module.weight.dtype))


class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of `config.num_hidden_layers` layers.
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = MistralEmbedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList([MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = ops.ones((batch_size, seq_length), dtype=mindspore.bool_)

        # Expand attention mask
        if len(attention_mask.shape) == 2:
            # Create causal mask
            seq_length = attention_mask.shape[1]
            causal_mask = ops.tril(ops.ones((seq_length, seq_length), dtype=mindspore.bool_))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask * causal_mask
            # Convert to attention bias
            attention_mask = ops.where(attention_mask, 0.0, float("-inf"))

        # Embed positions
        if position_ids is None:
            position_ids = ops.arange(0, seq_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)
            # 使用tile替代expand以兼容MindSpore API
            position_ids = ops.tile(position_ids, (batch_size, 1))

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)


class MistralForCausalLM(MistralPreTrainedModel):
    """Mistral Model with a language modeling head on top for causal language modeling."""

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Decoder outputs consists of (hidden_states, past_key_values, hidden_states, attentions)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.astype(mindspore.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].reshape(-1)
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        return (loss, logits) + outputs[1:]
