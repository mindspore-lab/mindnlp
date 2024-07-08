# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
InternLM Model.
"""
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindspore import dtype as mstype

from mindnlp.utils import logging
from .configuration_internlm import InternLMConfig
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast
)

logger = logging.get_logger(__name__)

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: Union[tuple, list], dtype: mstype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full(
        (tgt_len, tgt_len),
        Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min, dtype),
    )
    mask_cond = ops.arange(mask.shape[-1])
    mask = ops.masked_fill(mask, Tensor(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1)), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.concat(
            [ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1
        )
    return ops.broadcast_to(
        mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length)
    )

def _expand_mask(mask: Tensor, dtype: mstype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = ops.broadcast_to(
        mask[:, None, None, :], (bsz, 1, tgt_len, src_len)
    ).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(mindspore.bool_),
        mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min),
    )

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->InternLM
class InternLMRMSNorm(nn.Cell):
    """
    RMSNorm
    """
    def __init__(self, hidden_size, epsilon=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size), 'weight')
        self.variance_epsilon = epsilon

    def construct(self, hidden_states):
        """
        Constructs the RMS normalization of hidden states.
        
        Args:
            self (InternLMRMSNorm): An instance of the InternLMRMSNorm class.
            hidden_states (Tensor): Tensor holding the hidden states. Should be of type mindspore.Tensor.
            
        Returns:
            None: This method modifies the input hidden states in-place.
        
        Raises:
            ValueError: If the input hidden_states are not of type mindspore.Tensor.
            TypeError: If the weight dtype is not mindspore.float16 or mindspore.bfloat16.
        """
        variance = hidden_states.to(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16, mindspore.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class InternLMRotaryEmbedding(nn.Cell):
    """
    RotaryEmbedding
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        __init__ method in the InternLMRotaryEmbedding class.
        
        Args:
            self: The instance of the class.
            dim (int): The dimension of the input embeddings.
            max_position_embeddings (int, optional): The maximum position embeddings. Defaults to 2048.
            base (int, optional): The base value for calculations. Defaults to 10000.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2).float() / dim))

        self.max_seq_len_cached = max_position_embeddings
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper,
        # but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def construct(self, x, seq_len=None):
        '''
        This method constructs the rotary embeddings for the input sequence.
        
        Args:
            self (InternLMRotaryEmbedding): The instance of the InternLMRotaryEmbedding class.
            x:
                The input tensor for which the rotary embeddings are to be constructed.

                - Type: tensor
                - Purpose: This parameter represents the input tensor for which the rotary embeddings are to be constructed.
            seq_len (int, optional):
                The length of the input sequence.

                - Type: int
                - Purpose: This parameter represents the length of the input sequence for which the rotary embeddings
                    are to be constructed. If not provided, it defaults to None.
        
        Returns:
            None.
        
        Raises:
            ValueError: If seq_len is greater than the maximum sequence length cached.
            TypeError: If the input parameters are not of the expected types.
        '''
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = ops.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = ops.cat((freqs, freqs), axis=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class InternLMDynamicNTKScalingRotaryEmbedding(InternLMRotaryEmbedding):

    """
    The `InternLMDynamicNTKScalingRotaryEmbedding` class is a Python class that represents a dynamic version of the
    Neural Tangent Kernel (NTK) Scaling Rotary Embedding used in the context of an InternLM model.
    
    This class inherits from the `InternLMRotaryEmbedding` class and provides additional functionality for dynamically
    adjusting the NTK scaling factor based on the sequence length. It calculates and caches the cosine and sine values
    necessary for the rotary embeddings.

    Attributes:
        scaling_factor (float): The scaling factor used for adjusting the NTK scaling based on sequence length.

    Methods:
        __init__:
            Initializes the `InternLMDynamicNTKScalingRotaryEmbedding` object with the specified dimensions,
            maximum position embeddings, base, and scaling factor. Calls the superclass initializer.

        _set_cos_sin_cache:
            Sets the cosine and sine cache based on the provided sequence length and data type. Calculates the NTK
            scaling factor, inverse frequencies, and caches the cosine and sine values.

    Note:
        This class assumes the existence of the `InternLMRotaryEmbedding` superclass.

    Example:
        ```python
        >>> # Create an instance of InternLMDynamicNTKScalingRotaryEmbedding
        >>> embedding = InternLMDynamicNTKScalingRotaryEmbedding(dim=512, max_position_embeddings=1024, base=20000, scaling_factor=0.8)
        ...
        >>> # Access the scaling factor attribute
        >>> scaling_factor = embedding.scaling_factor
        ...
        >>> # Call the _set_cos_sin_cache method
        >>> embedding._set_cos_sin_cache(seq_len=512, dtype=torch.float32)
        ```
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes an instance of the InternLMDynamicNTKScalingRotaryEmbedding class.

        Args:
            self (InternLMDynamicNTKScalingRotaryEmbedding): The instance of the class itself.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value used in positional encoding calculation. Defaults to 10000.
            scaling_factor (float, optional): The scaling factor applied to the embeddings. Defaults to 1.0.

        Returns:
            None.

        Raises:
            None.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        Method '_set_cos_sin_cache' in the class 'InternLMDynamicNTKScalingRotaryEmbedding'.

        This method initializes the cosine and sine cache based on the given sequence length and data type.

        Args:
            self: The instance of the class.
            seq_len (int): The length of the input sequence. Must be greater than 0.
            dtype: The data type for the calculations.
                Should be a valid data type compatible with the operations performed.

        Returns:
            None.

        Raises:
            ValueError: If the input sequence length 'seq_len' is not a positive integer.
            TypeError: If the provided data type 'dtype' is not valid or compatible with the operations.
        """
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2).float() / self.dim))
            self.inv_freq = inv_freq

        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return ops.cat((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary positional embeddings to input queries (q) and keys (k).
    """
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class InternLMMLP(nn.Cell):
    """
    MLP
    """
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        """
        Initializes the InternLMMLP class.

        Args:
            self: The instance of the class.
            hidden_size (int): The size of the hidden layer in the neural network.
            intermediate_size (int): The size of the intermediate layer in the neural network.
            hidden_act (str): The activation function for the hidden layer.
                It should be one of the supported activation functions.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the hidden_act parameter does not correspond to a supported activation function.
        """
        super().__init__()
        self.gate_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(intermediate_size, hidden_size, has_bias=False)
        self.up_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def construct(self, x):
        """
        Constructs the output of the InternLMMLP model.

        Args:
            self (InternLMMLP): An instance of the InternLMMLP class.
            x: The input data for the model (type: unspecified).

        Returns:
            None.

        Raises:
            None.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class InternLMAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: InternLMConfig):
        """
        Initializes an instance of the InternLMAttention class.

        Args:
            self: The instance of the class.
            config (InternLMConfig): An instance of the InternLMConfig class containing the configuration parameters.

        Returns:
            None

        Raises:
            ValueError: If `hidden_size` is not divisible by `num_heads`.

        This method initializes the InternLMAttention class by setting the instance variables and initializing the projection layers.

        The `config` parameter is an instance of the InternLMConfig class, which contains the following attributes:

        - `hidden_size` (int): The size of the hidden state.
        - `num_attention_heads` (int): The number of attention heads.
        - `max_position_embeddings` (int): The maximum number of position embeddings.
        - `bias` (bool): Whether to include bias in the projection layers.

        The method sets the following instance variables:

        - `config` (InternLMConfig): The configuration instance.
        - `hidden_size` (int): The size of the hidden state.
        - `num_heads` (int): The number of attention heads.
        - `head_dim` (int): The dimension of each attention head.
        - `max_position_embeddings` (int): The maximum number of position embeddings.

        The method also initializes the following projection layers:

        - `q_proj` (Dense): The projection layer for the query.
        - `k_proj` (Dense): The projection layer for the key.
        - `v_proj` (Dense): The projection layer for the value.
        - `o_proj` (Dense): The projection layer for the output.

        If the product of `head_dim` and `num_heads` is not equal to `hidden_size`, a ValueError is raised.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.bias)
        self.k_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.bias)
        self.v_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.bias)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=config.bias)
        self._init_rope()

    def _init_rope(self):
        """
        This method initializes the rotary embedding for the InternLMAttention class.

        Args:
            self: The instance of the InternLMAttention class.

        Returns:
            None: However, it initializes and assigns the rotary embedding to the instance variable 'rotary_emb'.

        Raises:
            ValueError: If the rotary embedding's type specified in the configuration is not one of ('origin', 'dynamic').
        """
        if self.config.rotary["type"] == "origin":
            self.rotary_emb = InternLMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rotary["base"],
            )
        elif self.config.rotary["type"] == "dynamic":
            self.rotary_emb = InternLMDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rotary["base"],
                scaling_factor=self.config.rotary.get("scaling_factor", 1.0),
            )
        else:
            raise ValueError("Currently we only support rotary embedding's type being one of ('origin', 'dynamic').")
        return self.rotary_emb

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor according to the specified dimensions for the InternLMAttention class.

        Args:
            self (InternLMAttention): An instance of the InternLMAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None: The method modifies the input tensor in-place.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        Constructs the attention mechanism for the InternLMAttention class.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states.
                Its shape is (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor.
                It has the same shape as `hidden_states`. Default is None.
            position_ids (Optional[mindspore.Tensor]): The position ids tensor.
                It has the same shape as `hidden_states`. Default is None.
            past_key_value (Optional[Tuple[mindspore.Tensor]]): The past key-value tuple. Default is None.
            output_attentions (bool): Whether to output attention weights. Default is False.
            use_cache (bool): Whether to use cache. Default is False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the attention output, attention weights, and the updated past key-value tuple.

                - attn_output (mindspore.Tensor): The output tensor of shape (batch_size, sequence_length, hidden_size).
                - attn_weights (Optional[mindspore.Tensor]): The attention weights tensor of shape
                (batch_size, num_heads, sequence_length, sequence_length). If `output_attentions` is False, it is set to None.
                - past_key_value (Optional[Tuple[mindspore.Tensor]]): The updated past key-value tuple.
                If `use_cache` is False, it is set to None.

        Raises:
            ValueError: If the shape of attention weights is not (batch_size, num_heads, sequence_length, sequence_length).
            ValueError: If the shape of attention mask is not (batch_size, 1, sequence_length, sequence_length).

        """
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None
        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )
        if attention_mask is not None:
            if attention_mask.shape!= (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = ops.maximum(attn_weights,Tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min))

        attn_weights = ops.softmax(attn_weights, axis=-1).astype(query_states.dtype)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


INTERNLM_ATTENTION_CLASSES = {
    "eager": InternLMAttention,
}
class InternLMDecoderLayer(nn.Cell):
    """
    DecoderLayer
    """
    def __init__(self, config: InternLMConfig):
        """Initialize an instance of the InternLMDecoderLayer class.

        Args:
            self (InternLMDecoderLayer): The instance of the class.
            config (InternLMConfig):
                The configuration object containing various settings for the decoder layer.

                - hidden_size (int): The size of the hidden states.
                - intermediate_size (int): The size of the intermediate layer in the MLP.
                - hidden_act (str): The activation function to be used in the MLP.
                - rms_norm_eps (float): The epsilon value used in the RMS normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = INTERNLM_ATTENTION_CLASSES['eager'](config=config)

        self.mlp = InternLMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = InternLMRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = InternLMRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            hidden_states (`Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(Tensor)`, *optional*): cached past key and value projection states
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


# Copied from transformers.models.llama.modeling_llama.LlamaPretrainedModel with Llama->InternLM
class InternLMPreTrainedModel(PreTrainedModel):

    """
    The 'InternLMPreTrainedModel' class represents a pre-trained language model for internal use.
    It inherits from the 'PreTrainedModel' class and includes methods for initializing weights and setting gradient
    checkpointing.

    Attributes:
        config: The configuration for the pre-trained model.

    Methods:
        _init_weights: Initializes the weights for the specified cell using the specified initializer range.
        _set_gradient_checkpointing: Sets the gradient checkpointing for the specified module to the specified value.

    """
    config_class = InternLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLMDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]
    #_skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """
        Initializes the weights of a given cell.

        Args:
            self (InternLMPreTrainedModel): The instance of the InternLMPreTrainedModel class.
            cell (nn.Module): The cell for which the weights need to be initialized.

        Returns:
            None: This method modifies the weights of the given cell in-place.

        Raises:
            None.

        This method initializes the weights of the given cell based on the configuration specified in `self.config`.
        It supports two types of cells: `nn.Dense` and `nn.Embedding`.

        For `nn.Dense` cells, the weights are initialized using a normal distribution with mean 0 and standard deviation
        `self.config.initializer_range`.
        The weights are set using the `set_data` method of the `weight` attribute of the cell.
        If the cell has a bias attribute (`cell.bias`), it is initialized with zeros using the `set_data` method as well.

        For `nn.Embedding` cells, the weights are initialized using a normal distribution with mean 0 and
        standard deviation `self.config.initializer_range`. The weights are randomly sampled using the `np.random.normal`
        function and set using the `set_data` method of the `weight` attribute of the cell.
        If the cell has a `padding_idx` attribute (`cell.padding_idx`), the weight at that index is set to 0.

        Note:
            This method modifies the weights of the cell in-place and does not return any value.
        """
        std = self.config.initializer_range
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(
                sigma=std, mean=0.0), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Sets the gradient checkpointing attribute of a given module.

        Args:
            self (InternLMPreTrainedModel): The instance of the InternLMPreTrainedModel class.
            module: The module for which the gradient checkpointing attribute needs to be set.
                Should be an instance of the InternLMModel class.
            value (bool): The value to set for the gradient checkpointing attribute. Default is False.

        Returns:
            None.

        Raises:
            None.

        This method sets the gradient_checkpointing attribute of the specified module to the given value.
        The gradient checkpointing attribute determines whether to enable gradient checkpointing during training.
        Gradient checkpointing is a technique used to reduce memory consumption during backward pass by trading off computation time.
        If the module is an instance of the InternLMModel class, the gradient checkpointing attribute is set to the specified value.
        """
        if isinstance(module, InternLMModel):
            module.gradient_checkpointing = value


class InternLMModel(InternLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: InternLMConfig
    """
    _auto_class = "AutoModel"

    def __init__(self, config: InternLMConfig):
        """
        Args:
            self (object): The instance of the InternLMModel class.
            config (InternLMConfig):
                An instance of the InternLMConfig class containing the configuration for the language model.
                It specifies the model's parameters such as vocabulary size, hidden size, number of hidden layers, etc.
                The config parameter is required and must be an instance of the InternLMConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)

        self.layers = nn.CellList([InternLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = InternLMRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the InternLMModel.

        Args:
            self (InternLMModel): The instance of the InternLMModel class.

        Returns:
            embed_tokens: This method returns the input embeddings from the InternLMModel.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the InternLMModel.

        Args:
            self (object): The instance of the InternLMModel class.
            value (object): The input embeddings value to be set for the model.
                It should be an object of appropriate type.

        Returns:
            None.

        Raises:
            None
        """
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        This method prepares the decoder attention mask for the InternLMModel.

        Args:
            self (object): The instance of the InternLMModel.
            attention_mask (torch.Tensor): The attention mask to be prepared.
                It should have the same shape as input_shape.
            input_shape (tuple): The shape of the input tensor.
            inputs_embeds (torch.Tensor): The input embeddings tensor.
            past_key_values_length (int): The length of past key values.

        Returns:
            combined_attention_mask (torch.Tensor): The combined attention mask prepared for the decoder.
                Returns None if the input_shape[-1] is less than or equal to 1.

        Raises:
            ValueError: If input_shape[-1] is less than or equal to 0.
            TypeError: If any of the input parameters are of incorrect type.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        '''
        Constructs the internal language model for the model.

        Args:
            self (object): The instance of the class.
            input_ids (Tensor, optional): The input tensor of token indices. Defaults to None.
            attention_mask (Tensor, optional): The mask tensor to avoid attention on padding tokens. Defaults to None.
            position_ids (Tensor, optional): The tensor of token positions. Defaults to None.
            past_key_values (List[Tensor], optional): List of tensors containing past key values. Defaults to None.
            inputs_embeds (Tensor, optional): The input embeddings tensor. Defaults to None.
            use_cache (bool, optional): Flag to use caching. Defaults to None.
            output_attentions (bool, optional): Flag to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Flag to output hidden states. Defaults to None.
            return_dict (bool, optional): Flag to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The output as a tuple or an instance of BaseModelOutputWithPast.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified, or if neither of them is specified.
            Warning: If `use_cache=True` is incompatible with gradient checkpointing.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #if self.config.attn_implementation == "flash_attention_2":
            #_import_flash_attn()
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = ops.ones(
                (batch_size, seq_length_with_past), dtype=mindspore.bool_
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # TODO: how checkpoint
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
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class InternLMForCausalLM(InternLMPreTrainedModel):

    """
    A class representing an InternLM model for causal language modeling.

    This class extends the InternLMPreTrainedModel class and provides additional functionality specific to
    causal language modeling tasks. It includes methods for initializing the model, setting and getting
    input and output embeddings, setting the decoder, constructing the model, and preparing inputs for generation.

    Attributes:
        model (InternLMModel): The underlying InternLM model.
        lm_head (nn.Dense): The linear layer for mapping hidden states to the vocabulary space.

    Methods:
        __init__: Initializes the InternLMForCausalLM instance.
        get_input_embeddings: Returns the input embeddings of the model.
        set_input_embeddings: Sets the input embeddings of the model.
        get_output_embeddings: Returns the output embeddings of the model.
        set_output_embeddings: Sets the output embeddings of the model.
        set_decoder: Sets the decoder for the model.
        get_decoder: Returns the decoder of the model.
        construct: Constructs the model and computes the masked language modeling loss.
        prepare_inputs_for_generation: Prepares inputs for generation by modifying the input_ids, attention_mask,
            and position_ids.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, InternLMForCausalLM
        ...
        >>> model = InternLMForCausalLM(config)
        >>> tokenizer = AutoTokenizer.from_pretrained(model)
        ...
        >>> # Access model attributes
        >>> input_embeddings = model.get_input_embeddings()
        >>> output_embeddings = model.get_output_embeddings()
        ...
        >>> # Modify model attributes
        >>> model.set_input_embeddings(new_input_embeddings)
        >>> model.set_output_embeddings(new_output_embeddings)
        ...
        >>> # Set decoder
        >>> model.set_decoder(decoder_model)
        ...
        >>> # Generate text
        >>> model.construct(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
        >>> generated_text = model.prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        ```
    """
    _auto_class = "AutoModelForCausalLM"
    def __init__(self, config, size=None):
        """
        Initializes a new instance of the InternLMForCausalLM class.

        Args:
            self: The instance of the class.
            config:
                The configuration for the language model.

                - Type: object
                - Purpose: Specifies the configuration parameters for the language model.
            size:
                The size of the language model input. (Optional)

                - Type: int
                - Purpose: Specifies the size of the language model input. If not provided, defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = InternLMModel(config)

        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the InternLMForCausalLM model.

        Args:
            self: An instance of the InternLMForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        This method is used to obtain the input embeddings from the model.
        The input embeddings are representations of the input tokens that the model uses to process the text.
        The embeddings capture the semantic meaning and contextual information of the tokens, which is crucial for
        the model's performance.

        Note:
            The 'embed_tokens' attribute of the 'self.model' object contains the input embeddings.
            This attribute should be accessed to retrieve the embeddings.

        Example:
            ```python
            >>> model = InternLMForCausalLM()
            >>> embeddings = model.get_input_embeddings()
            ```
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        '''
        Sets the input embeddings for the InternLMForCausalLM model.

        Args:
            self (InternLMForCausalLM): An instance of the InternLMForCausalLM class.
            value (object): The input embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.
        '''
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the InternLMForCausalLM model.

        Args:
            self: The instance of the InternLMForCausalLM class.

        Returns:
            The output embeddings of the model, represented by the 'lm_head' attribute.

        Raises:
            None.

        Note:
            The output embeddings are typically used to map the model's hidden state to a specific output vocabulary.
            These embeddings can be used for downstream tasks such as text generation or classification.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the InternLMForCausalLM model.

        Args:
            self (InternLMForCausalLM): The instance of the InternLMForCausalLM class.
            new_embeddings (Tensor): The new embeddings to be set for the output layer.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of type Tensor.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the InternLMForCausalLM class.

        Args:
            self (InternLMForCausalLM): The current instance of the InternLMForCausalLM class.
            decoder: The decoder object to be set for the InternLMForCausalLM instance.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Method to retrieve the decoder from the InternLMForCausalLM class.

        Args:
            self (object): The instance of the InternLMForCausalLM class.
                This parameter is required to access the model within the class.

        Returns:
            None:
                The method returns the decoder object associated with the InternLMForCausalLM instance.

        Raises:
            None.
        """
        return self.model

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, InternLMForCausalLM
            >>> model = InternLMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
            >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
            >>> prompt = "Hey, are you consciours? Can you talk to me?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = ops.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        This method prepares the inputs for the generation process in the InternLMForCausalLM class.

        Args:
            self (InternLMForCausalLM): The instance of the InternLMForCausalLM class.
            input_ids (torch.Tensor): The input tensor containing the tokenized input sequence.
            past_key_values (Optional[torch.Tensor]): The tensor of past key values for generation. Default is None.
            attention_mask (Optional[torch.Tensor]): The attention mask tensor. Default is None.
            inputs_embeds (Optional[torch.Tensor]): The tensor of embedded inputs. Default is None.

        Returns:
            model_inputs (dict): A dictionary containing the prepared model inputs for generation.
                It can have the following keys:

                - 'inputs_embeds' (torch.Tensor): The tensor of embedded inputs, if provided.
                - 'input_ids' (torch.Tensor): The tensor of tokenized input sequence.
                - 'position_ids' (torch.Tensor): The tensor of position IDs.
                - 'past_key_values' (torch.Tensor): The tensor of past key values for generation.
                - 'use_cache' (bool): A flag indicating whether to use cache or not.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.

        Raises:
            None.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache of past key values based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer.
                Each past key value is expected to be a tensor.
            beam_idx (tensor): A tensor representing the beam index.

        Returns:
            None.

        Raises:
            None
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class InternLMForSequenceClassification(InternLMPreTrainedModel):

    """
    This class represents an InternLM model for sequence classification tasks.
    It is a subclass of the InternLMPreTrainedModel class.

    The InternLMForSequenceClassification class is initialized with a configuration object, which includes the number
    of labels for the classification task. The model architecture consists of an InternLMModel and a score layer.

    The class provides methods for getting and setting the input embeddings of the model.
    The get_input_embeddings method returns the embedded tokens of the model, while the set_input_embeddings method
    allows for setting new input embeddings.

    The construct method is responsible for processing input data and generating classification outputs.
    It takes several optional parameters, including input_ids, attention_mask, position_ids, past_key_values,
    inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, and return_dict.
    The method returns either a tuple or a SequenceClassifierOutputWithPast object, depending on the value of the
    return_dict parameter.

    If labels are provided, the method computes the sequence classification loss based on the configured problem type.
    The problem type can be 'regression', 'single_label_classification', or 'multi_label_classification',
    depending on the number of labels and the data type of the labels.
    The loss is computed using various loss functions, such as mean squared error (MSE) loss, cross-entropy loss, or
    binary cross-entropy with logits loss.

    If the return_dict parameter is False, the method returns a tuple containing the pooled logits and
    other transformer outputs. If the loss is not None, it is included in the tuple. If the return_dict parameter is
    True, the method returns a SequenceClassifierOutputWithPast object, which includes the loss, pooled logits,
    past key values, hidden states, and attentions.

    Note:
        The class assumes that the batch size is 1 or that a padding token ID is defined.
        If the batch size is greater than 1 and no padding token ID is defined, a ValueError is raised.

    """
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the `InternLMForSequenceClassification` class.

        Args:
            self: The object itself.
            config: An instance of the `InternLMConfig` class containing the configuration parameters for the model.
                It includes the following attributes:

                - num_labels (int): The number of labels for classification.
                This value determines the size of the output layer.
                - hidden_size (int): The size of the hidden layers in the model.
                This value is used in the `nn.Dense` layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLMModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the InternLMForSequenceClassification model.

        Args:
            self: An instance of the InternLMForSequenceClassification class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the model's embed_tokens attribute.
        The input embeddings are used as the input to the model for sequence classification tasks.
        The method does not modify the input embeddings or perform any additional processing.
        The retrieved input embeddings can be used for further analysis or visualization, if needed.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        This method is a part of the 'InternLMForSequenceClassification' class and is used to set the
        input embeddings for the model.

        Args:
            self (object): The instance of the class.
            value (object): The input embeddings value to be set for the model.
                It can be of any valid type that represents the input embeddings.

        Returns:
            None: This method does not return any value explicitly, but it sets the input embeddings for the model.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = ops.equal(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(0,batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
__all__ = [
    "InternLMForCausalLM",
    "InternLMModel",
    "InternLMPreTrainedModel",
    "InternLMForSequenceClassification",
]
