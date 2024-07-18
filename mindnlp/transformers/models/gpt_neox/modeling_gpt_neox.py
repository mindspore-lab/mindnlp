# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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
""" MindNLP  GPTNeoX model."""

from typing import Optional, Tuple, Union

from functools import partial
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindnlp.utils import logging, get_default_dtype

from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from .configuration_gpt_neox import GPTNeoXConfig


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"


GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neox-20b",
    # See all GPTNeoX models at https://hf-mirror.com/models?filter=gpt_neox
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    Args:
        attention_mask (Tensor): A tensor representing the attention mask for the input data.
            It is used to mask padded tokens in the input sequence. Should be a 2D tensor with shape
            (batch_size, sequence_length) containing 0s for padded tokens and 1s for non-padded tokens.
    
    Returns:
        None: This function does not return any value.
            Instead, it computes and stores the necessary data in the form of indices, cumulative sequence lengths,
            and maximum sequence length.
    
    Raises:
        None
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = mindspore.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoXLayer"]
    _keys_to_ignore_on_load_unexpected = [r'masked_bias', r'attention.bias', r'inv_freq']
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(sigma=self.config.initializer_range, mean=0.0),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(sigma=self.config.initializer_range, mean=0.0),
                                 cell.weight.shape, cell.weight.dtype)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Sets the gradient checkpointing flag for the specified module.
        
        Args:
            self (GPTNeoXPreTrainedModel): The instance of the GPTNeoXPreTrainedModel class.
            module: The module for which the gradient checkpointing flag needs to be set.
            value (bool): The value to set for the gradient checkpointing flag.
        
        Returns:
            None.
        
        Raises:
            None.
        
        This method sets the gradient checkpointing flag to the specified value for the given module.
        The gradient checkpointing flag determines whether gradient checkpointing is used during the forward pass
        of the module. Gradient checkpointing can be used to trade compute for memory, as it reduces the memory usage
        at the expense of additional compute. The flag is only set if the module is an instance of the GPTNeoXModel
        class.
        """
        if isinstance(module, GPTNeoXModel):
            module.gradient_checkpointing = value

    def _backward_compatibility_gradient_checkpointing(self):
        """
        Support gradient_checkpointing.
        """
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))


class GPTNeoXAttention(nn.Cell):
    """GPTNeoXAttention"""
    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoXAttention class.

        Args:
            self: The object instance itself.
            config:
                A configuration object containing various hyperparameters for the GPTNeoXAttention model.

                - Type: Any
                - Purpose: To store the configuration settings for the GPTNeoXAttention model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None

        Raises:
            ValueError:
                If the hidden size is not divisible by the number of attention heads specified in the configuration.
        """
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self._init_bias(config.max_position_embeddings)

        self.masked_bias = mindspore.tensor(-1e9)
        self._init_rope()

        self.norm_factor = self.head_size ** -0.5
        self.query_key_value = nn.Dense(config.hidden_size, 3 * config.hidden_size, has_bias=config.attention_bias)
        self.dense = nn.Dense(config.hidden_size, config.hidden_size, has_bias=config.attention_bias)
        self.attention_dropout = nn.Dropout(p=config.attention_dropout)
        self.is_causal = True

    def _init_bias(self, max_positions):
        """
        Initialize the bias matrix for GPTNeoXAttention.

        Args:
            self (object): The instance of the GPTNeoXAttention class.
            max_positions (int): The maximum number of positions for the bias matrix.
                It defines the size of the square matrix and must be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """
        self.bias = ops.tril(ops.ones((max_positions, max_positions))).view(
                1, 1, max_positions, max_positions).astype(mindspore.bool_)

    def _init_rope(self):
        """
        Initializes the routing position encoding (RoPE) for the GPTNeoXAttention class.

        Args:
            self: The instance of the GPTNeoXAttention class.

        Returns:
            None.

        Raises:
            ValueError: If the scaling_type provided in the configuration for RoPE is neither 'linear' nor 'dynamic'.
        """
        if self.config.rope_scaling is None:
            self.rotary_emb = GPTNeoXRotaryEmbedding(
                self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: mindspore.Tensor,
            position_ids: mindspore.Tensor,
            head_mask: Optional[mindspore.Tensor] = None,
            layer_past: Optional[Tuple[mindspore.Tensor]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ):
        '''
        Constructs the GPTNeoXAttention method.

        Args:
            self: The instance of the GPTNeoXAttention class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            attention_mask (mindspore.Tensor): The attention mask tensor to mask invalid positions in the input.
            position_ids (mindspore.Tensor): The tensor representing the position indices in the input sequence.
            head_mask (Optional[mindspore.Tensor]): An optional tensor to mask attention heads. Default is None.
            layer_past (Optional[Tuple[mindspore.Tensor]]): An optional tuple representing the cached layer past.
                Default is None.
            use_cache (Optional[bool]): An optional boolean flag indicating whether to use cached values.
                Default is False.
            output_attentions (Optional[bool]): An optional boolean flag indicating whether to output attentions.
                Default is False.

        Returns:
            None.

        Raises:
            None
        '''
        has_layer_past = layer_past is not None # Atte

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.shape[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].transpose(0, 2, 1, 3)
        key = qkv[..., self.head_size: 2 * self.head_size].transpose(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].transpose(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = ops.cat((query, query_pass.type_as(query)), axis=-1)
        key = ops.cat((key, key_pass.type_as(key)), axis=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _split_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.shape[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.transpose(0, 2, 1, 3)
        return tensor

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.transpose(0, 2, 1, 3)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.shape[0], tensor.shape[1], num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Performs attention mechanism on the given inputs.

        Args:
            self (GPTNeoXAttention): An instance of the GPTNeoXAttention class.
            query (Tensor): The query tensor of shape (batch_size, num_attention_heads, query_length, attn_head_size).
            key (Tensor): The key tensor of shape (batch_size, num_attention_heads, key_length, attn_head_size).
            value (Tensor): The value tensor of shape (batch_size, num_attention_heads, key_length, attn_head_size).
            attention_mask (Tensor, optional): An optional tensor of shape
                (batch_size, num_attention_heads, query_length, key_length). It is used to mask attention scores.
                Defaults to None.
            head_mask (Tensor, optional):
                An optional tensor of shape (num_attention_heads,) or (batch_size, num_attention_heads).
                It is used to mask attention weights. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the attention output tensor of shape
                (batch_size, num_attention_heads, query_length, attn_head_size) and the attention weights tensor of
                shape (batch_size, num_attention_heads, query_length, key_length).

        Raises:
            None.
        """
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.shape
        key_length = key.shape[-2]

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = ops.zeros((
            batch_size * num_attention_heads,
            query_length,
            key_length),
            dtype=query.dtype
        )
        attn_scores = ops.bmm(
            query,
            key.swapaxes(1, 2),
        ) * self.norm_factor
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = np.finfo(mindspore.dtype_to_nptype(attn_scores.dtype)).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = mindspore.tensor(mask_value, dtype=attn_scores.dtype)
        attn_scores = ops.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = ops.softmax(attn_scores, axis=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = ops.matmul(attn_weights, value)
        return attn_output, attn_weights


def attention_mask_func(attention_scores, ltor_mask):
    """attention mask function"""
    attention_scores = attention_scores.masked_fill(~ltor_mask, mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(attention_scores.dtype)).min))
    return attention_scores


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with LlamaRotary->GPTNeoXRotary
class GPTNeoXRotaryEmbedding(nn.Cell):
    """GPTNeoXRotaryEmbedding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes the GPTNeoXRotaryEmbedding class.

        Args:
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value for computing inverse frequencies. Defaults to 10000.

        Returns:
            None.

        Raises:
            TypeError: If the provided dimensions are not integers.
            ValueError: If max_position_embeddings or base is non-positive.
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        '''
        _set_cos_sin_cache(self, seq_len, dtype):
            Set the cached cosine and sine values for the GPTNeoXRotaryEmbedding layer.

            Args:
                self (GPTNeoXRotaryEmbedding): The instance of the GPTNeoXRotaryEmbedding class.
                seq_len (int): The length of the input sequence. It specifies the number of time steps in the sequence.
                dtype: The data type for the calculations. It should be compatible with the data type of self.inv_freq.

            Returns:
                None.

            Raises:
                None.
        '''
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype).type_as(self.inv_freq)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def construct(self, x, seq_len=None):
        """
        Constructs the rotary embeddings for the GPTNeoX model.

        Args:
            self (GPTNeoXRotaryEmbedding): The instance of the GPTNeoXRotaryEmbedding class.
            x (Tensor): The input tensor for which rotary embeddings are to be constructed.
            seq_len (int, optional): The length of the sequence. Defaults to None.

        Returns:
            The constructed cosine and sine embeddings for the input tensor.

        Raises:
            ValueError: If seq_len is greater than the maximum sequence length cached in the instance.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


# copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding.__init__
class GPTNeoXLinearScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes an instance of GPTNeoXLinearScalingRotaryEmbedding.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the embeddings.
            max_position_embeddings (int): The maximum number of position embeddings. Default is 2048.
            base (int): The base value used in calculations. Default is 10000.
            scaling_factor (float): The scaling factor applied to the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        Sets the cosine and sine caches for the GPTNeoXLinearScalingRotaryEmbedding class.

        Args:
            self (GPTNeoXLinearScalingRotaryEmbedding): An instance of the GPTNeoXLinearScalingRotaryEmbedding class.
            seq_len (int): The length of the sequence.
            dtype: The data type of the elements in the cache.

        Returns:
            None: This method modifies the state of the GPTNeoXLinearScalingRotaryEmbedding instance.

        Raises:
            None.

        Description:
            This method sets the cosine and sine caches for the GPTNeoXLinearScalingRotaryEmbedding instance based on
            the given sequence length and data type. The cosine and sine caches are used to store precalculated values
            for efficient computation during the forward pass of the GPTNeoX model.

            The parameters for this method are as follows:

            - self: This parameter refers to the instance of the GPTNeoXLinearScalingRotaryEmbedding class
            on which the method is called.
            - seq_len: This parameter specifies the length of the sequence. It is an integer value.
            - dtype: This parameter denotes the data type of the elements in the cache. The data type can be any valid
            data type supported by the underlying framework.

            The method first sets the maximum sequence length cached by assigning the value of seq_len to
            self.max_seq_len_cached.

            Next, it creates a tensor 't' using the 'ops.arange' function with the length of self.max_seq_len_cached
            and the specified data type. The 'type_as' method is used to ensure that 't' has the same data
            type as self.inv_freq.

            Then, 't' is divided by self.scaling_factor to scale the values.

            The 'ops.outer' function is used to calculate the outer product of 't' and self.inv_freq,
            resulting in a tensor 'freqs'.

            The 'ops.cat' function is called to concatenate 'freqs' with itself along the last dimension,
            creating a tensor 'emb'.

            Finally, 'emb.cos()' and 'emb.sin()' are called to compute the cosine and sine values of 'emb', respectively.
            The resulting cosine values are stored in self.cos_cached and sine values are stored in self.sin_cached.

            This method does not return any value but modifies the state of the GPTNeoXLinearScalingRotaryEmbedding
            instance.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=mindspore.float32).type_as(self.inv_freq)
        t = t / self.scaling_factor

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()


class GPTNeoXDynamicNTKScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        __init__

        Initializes a new instance of the GPTNeoXDynamicNTKScalingRotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Default is 2048.
            base (int, optional): The base value for position embedding calculations. Default is 10000.
            scaling_factor (float, optional): A scaling factor for the embeddings. Default is 1.0.

        Returns:
            None.

        Raises:
            TypeError: If the provided dimension, max_position_embeddings, base,
                or scaling_factor is not of the correct type.
            ValueError: If the provided dimension, max_position_embeddings, base,
                or scaling_factor does not meet specific criteria.
            NotImplementedError: If the method is not implemented for some reason.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        Method _set_cos_sin_cache in the class GPTNeoXDynamicNTKScalingRotaryEmbedding.

        Args:
            self (GPTNeoXDynamicNTKScalingRotaryEmbedding):
                The instance of the GPTNeoXDynamicNTKScalingRotaryEmbedding class.
            seq_len (int): The length of the sequence for which to set the cosine and sine cache.
            dtype (Type): The data type to be used for calculations.

        Returns:
            None.

        Raises:
            ValueError: If the sequence length 'seq_len' is less than or equal to 0.
            RuntimeError: If an error occurs during the computation of cosine and sine cache.
        """
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim))
            self.inv_freq = inv_freq

        t = ops.arange(self.max_seq_len_cached, dtype=mindspore.int64).type_as(self.inv_freq)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2:]
    x1, x2 = x.tensor_split(2, -1)
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply rotary positional embeddings to input queries (q) and keys (k)."""
    cos = cos[position_ids].expand_dims(unsqueeze_dim)
    sin = sin[position_ids].expand_dims(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GPTNeoXMLP(nn.Cell):
    """GPTNeoXMLP"""
    def __init__(self, config):
        """
        __init__ method in the GPTNeoXMLP class.

        This method initializes the GPTNeoXMLP class.

        Args:
            self: The instance of the GPTNeoXMLP class.
            config:
                An instance of the configuration class that contains the configuration parameters
                for the GPTNeoXMLP model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense_h_to_4h = nn.Dense(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Dense(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        """
        Constructs the hidden states using the specified operations.

        Args:
            self (GPTNeoXMLP): The instance of the GPTNeoXMLP class.
            hidden_states (tensor): The input hidden states to be processed.

        Returns:
            hidden_states: The processed hidden states are returned after applying the specified operations.

        Raises:
            None.
        """
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


GPT_NEOX_ATTENTION_CLASSES = {
    "eager": GPTNeoXAttention,
}


class GPTNeoXLayer(nn.Cell):
    """GPTNeoXLayer"""
    def __init__(self, config):
        """
        Initializes a GPTNeoXLayer instance.

        Args:
            self: The object instance itself.
            config:
                An instance of a configuration class containing the following attributes:

                - use_parallel_residual: A boolean flag indicating whether to use parallel residual connections.
                - hidden_size: An integer specifying the size of the hidden layers.
                - layer_norm_eps: A float representing the epsilon value for layer normalization.
                - hidden_dropout: A float indicating the dropout probability for hidden layers.

        Returns:
            None: This method initializes various components of the GPTNeoXLayer class including layer normalization,
                dropout layers, attention mechanism, and multi-layer perceptron (MLP).

        Raises:
            AttributeError: If the required attributes are missing in the 'config' parameter.
            TypeError: If the data types of the attributes in the 'config' parameter are incorrect.
            ValueError: If the values of the attributes in the 'config' parameter are invalid.
        """
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(p=config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(p=config.hidden_dropout)
        # self.attention = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation](config)
        self.attention = GPT_NEOX_ATTENTION_CLASSES["eager"](config)
        self.mlp = GPTNeoXMLP(config)

    def construct(
            self,
            hidden_states: Optional[mindspore.Tensor],
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = False,
            layer_past: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: Optional[bool] = False,
    ):
        """
        Constructs the GPTNeoXLayer.

        Args:
            self (GPTNeoXLayer): The instance of the GPTNeoXLayer class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (mindspore.Tensor, optional): The position IDs tensor. Defaults to None.
            head_mask (mindspore.Tensor, optional): The head mask tensor. Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to False.
            layer_past (Tuple[mindspore.Tensor], optional): The past layer tensor. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: The output tensor(s) of the GPTNeoXLayer.

        Raises:
            None
        """
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class GPTNeoXModel(GPTNeoXPreTrainedModel):
    """GPTNeoXModel"""
    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoXModel class.

        Args:
            self (GPTNeoXModel): The current instance of the GPTNeoXModel class.
            config (object): An object containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(p=config.hidden_dropout)
        self.layers = nn.CellList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        # Not support flash_attention_2
        # self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_flash_attention_2 = False

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the GPTNeoXModel.

        Args:
            self (object): The instance of the GPTNeoXModel class.
                This parameter is used to access the instance attributes and methods.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_in

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the GPTNeoXModel.

        Args:
            self (GPTNeoXModel): The instance of the GPTNeoXModel class.
            new_embeddings (object): The new input embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_in = new_embeddings

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4
                tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = ops.arange(past_length, seq_length + past_length, dtype=mindspore.int64)
            position_ids = position_ids.expand_dims(0)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    output_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    """GPTNeoXForCausalLM"""
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoXForCausalLM class.

        Args:
            self: GPTNeoXForCausalLM
                The instance of the GPTNeoXForCausalLM class.
            config: object
                The configuration object containing the settings for the GPTNeoXForCausalLM model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Method: get_output_embeddings

        Description:
        Returns the output embeddings for the GPTNeoXForCausalLM model.

        Args:
            self (GPTNeoXForCausalLM): The instance of the GPTNeoXForCausalLM class.

        Returns:
            The output embeddings.

        Raises:
            None
        """
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the GPTNeoXForCausalLM model.

        Args:
            self (GPTNeoXForCausalLM): The instance of the GPTNeoXForCausalLM class.
            new_embeddings (Any): The new embeddings to be set as the output embeddings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_out = new_embeddings

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
                or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
                only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
                `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :]
            labels = labels[:, 1:]
            lm_loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        This method prepares inputs for generation in the GPTNeoXForCausalLM class.

        Args:
            self: The instance of the class.
            input_ids (torch.Tensor):
                The input tensor containing the token IDs. Shape should be [batch_size, sequence_length].
            past_key_values (tuple of torch.Tensor, optional):
                The past key values for autoregressive generation. Default is None.
            attention_mask (torch.Tensor, optional):
                The attention mask tensor. Shape should be [batch_size, sequence_length].
            inputs_embeds (torch.Tensor, optional): The embedded input tensor. Default is None.

        Returns:
            dict: A dictionary containing the model inputs necessary for generation, including 'input_ids', 'attention_mask',
                'past_key_values', and 'position_ids'.

        Raises:
            TypeError: If the input_ids or attention_mask is not of type torch.Tensor.
            ValueError: If the past_key_values do not have the expected shape.
            RuntimeError: If an error occurs during the computation of position_ids.
        """
        input_shape = input_ids.shape
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def _reorder_cache(self, past, beam_idx):
        """
        Reorders the cache for the GPTNeoXForCausalLM model based on the given beam index.

        Args:
            self (GPTNeoXForCausalLM): The instance of the GPTNeoXForCausalLM class.
            past (Tuple): The past cache states to be reordered.
            beam_idx (Tensor): The indices of the beams to reorder the cache.

        Returns:
            Tuple: The reordered past cache states.

        Raises:
            ValueError: If the past cache states are not in the expected format.
            IndexError: If the beam index is out of range.
        """
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


class GPTNeoXForSequenceClassification(GPTNeoXPreTrainedModel):
    """GPTNeoXForSequenceClassification"""
    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoXForSequenceClassification class.

        Args:
            self: The instance of the class.
            config: An object containing configuration settings for the model.
                It should have attributes:

                - num_labels (int): The number of labels for classification.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided.
            ValueError: If the num_labels attribute is missing from the config object.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = ops.equal(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                # sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

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
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTNeoXForTokenClassification(GPTNeoXPreTrainedModel):
    """GPTNeoXForTokenClassification"""
    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoXForTokenClassification class.

        Args:
            self: The object itself.
            config (GPTNeoXConfig): The model configuration class that defines the model architecture and hyperparameters.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes the GPTNeoXForTokenClassification model with the provided configuration.
            It sets the number of labels for token classification based on the configuration. The GPTNeoXModel is
            instantiated with the provided configuration. Additionally, a dropout layer with a specified dropout rate
            is added, and a fully connected layer (classifier) is initialized with the hidden size and the number
            of labels from the configuration.
            Finally, the post_init() method is called for any post-initialization tasks.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTNeoXForQuestionAnswering(GPTNeoXPreTrainedModel):
    """GPTNeoXForQuestionAnswering"""
    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoXForQuestionAnswering class.
        
        Args:
            self (GPTNeoXForQuestionAnswering): The instance of the class itself.
            config: An instance of the configuration class containing the model configuration.
                This parameter is required to initialize the model and set various configuration options.
                It must contain the 'num_labels' attribute specifying the number of labels for the model.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the 'config' parameter is not provided or is not of the expected type.
            ValueError: If the 'num_labels' attribute is missing in the 'config' parameter.
            AttributeError: If any required attributes are missing during initialization.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            start_positions: Optional[mindspore.Tensor] = None,
            end_positions: Optional[mindspore.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.cross_entropy(start_logits, start_positions)
            end_loss = ops.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "GPTNeoXPreTrainedModel",
    "GPTNeoXLayer",
    "GPTNeoXModel",
    "GPTNeoXForCausalLM",
    "GPTNeoXForSequenceClassification",
    "GPTNeoXForTokenClassification",
    "GPTNeoXForQuestionAnswering",
]
