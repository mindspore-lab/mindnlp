# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# ============================================================================
""" MindSpore MiniCPM model."""
import re
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS

from .configuration_minicpm import MiniCPMConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniCPMConfig"

def rms_layernorm(hidden: mindspore.Tensor, weight: mindspore.Tensor, eps: float):
    """
    Args:
        hidden (mindspore.Tensor): The input tensor to be normalized.
        weight (mindspore.Tensor): The weight tensor applied to the normalized input.
        eps (float): A small value added to the variance to avoid division by zero.
    
    Returns:
        None: This function does not return a value. It operates in place on the 'hidden' tensor.
    
    Raises:
        ValueError: If the 'hidden' tensor or 'weight' tensor is not of type mindspore.Tensor.
        TypeError: If the 'eps' parameter is not of type float.
    """
    old_dtype = hidden.dtype
    variance = hidden.to(mindspore.float32).pow(2).mean(axis=-1, keep_dims=True)
    hidden = (hidden * ops.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):

    """
    MiniCPMRMSNorm is a custom layer normalization module designed to mimic the functionality of T5LayerNorm. 
    It performs RMS-based layer normalization on the input hidden states using the provided weight and epsilon value.
    
    Parameters:
        hidden_size (int): The size of the hidden states being normalized.
        eps (float, optional): A small value added to the variance to prevent division by zero. Default is 1e-06.

    Inherits From:
        nn.Module

    Attributes:
        weight (Parameter): The weight parameter used for normalization.
        variance_epsilon (float): The epsilon value added to the variance.

    Methods:
        __init__: Initializes the MiniCPMRMSNorm instance with the given hidden size and epsilon.
        forward: Applies RMS-based layer normalization on the input hidden states using the weight and epsilon.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Constructs a MiniCPMRMSNorm object.

        Args:
            self (MiniCPMRMSNorm): The instance of the MiniCPMRMSNorm class.
            hidden_states (tensor): The input hidden states to be normalized.

        Returns:
            None.

        Raises:
            TypeError: If the input hidden_states is not a valid tensor.
            ValueError: If the weight or variance_epsilon attributes are not set in the MiniCPMRMSNorm object.
        """
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Module):

    """
    MiniCPMRotaryEmbedding is a class that represents a rotary positional embedding layer for neural networks.
    It inherits from nn.Module and provides methods for initializing the embedding layer, setting cosine and sine cache,
    and forwarding the embeddings based on input data.
    The class allows for dynamic caching of positional embeddings up to a specified maximum sequence length.
    The rotary embeddings are computed based on the provided dimensions, maximum position embeddings, and base values.
    The forwardor initializes the necessary attributes, while the _set_cos_sin_cache method precomputes and caches
    cosine and sine values for positional embeddings.
    The forward method generates the positional embeddings based on the input data and the specified sequence length.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes a new instance of the MiniCPMRotaryEmbedding class.

        Args:
            self (MiniCPMRotaryEmbedding): The instance of the class.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value used for calculating the inverse frequency. Defaults to 10000.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            # seq_len=max_position_embeddings, dtype=torch.get_default_dtype()
            seq_len=max_position_embeddings, dtype=mindspore.float32
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        Method to calculate and cache the cosine and sine values for rotary embeddings.

        Args:
            self: Instance of MiniCPMRotaryEmbedding class.
            seq_len (int): The length of the sequence for which to calculate the cosine and sine values.
            dtype: Data type to which the cosine and sine values should be converted.

        Returns:
            None: This method does not return any value. It caches the cosine and sine values internally.

        Raises:
            None.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        """
        Construct a rotary embedding for a MiniCPM model.

        Args:
            self (MiniCPMRotaryEmbedding): The instance of the MiniCPMRotaryEmbedding class.
            x (Tensor): The input tensor for which the rotary embedding needs to be forwarded.
            seq_len (int, optional): The length of the sequence. If not provided, the default value is None.
                Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing two tensors, cosine and sine values of the rotary embedding,
                both of the same dtype as input tensor x.

        Raises:
            ValueError: If seq_len is greater than the maximum sequence length cached in the instance.
            TypeError: If the input dtype is not supported for the cosine and sine caches.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes an instance of MiniCPMLinearScalingRotaryEmbedding.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int): The maximum number of position embeddings.
            base (int): The base value used in calculations.
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
        Sets the cosine and sine cache for the MiniCPMLinearScalingRotaryEmbedding class.

        Args:
            self (MiniCPMLinearScalingRotaryEmbedding): An instance of the MiniCPMLinearScalingRotaryEmbedding class.
            seq_len (int): The length of the sequence for which to set the cache.
            dtype: The desired data type for the cache.

        Returns:
            None.

        Raises:
            None.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes a new instance of the MiniCPMDynamicNTKScalingRotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimensionality of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value. Defaults to 10000.
            scaling_factor (float, optional): The scaling factor. Defaults to 1.0.

        Returns:
            None.

        Raises:
            None.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        This method '_set_cos_sin_cache' is defined in the class 'MiniCPMDynamicNTKScalingRotaryEmbedding'.
        It initializes the cosine and sine caches based on the given sequence length and data type.

        Args:
            self (object): The instance of the MiniCPMDynamicNTKScalingRotaryEmbedding class.
            seq_len (int): The length of the sequence for which cosine and sine caches need to be computed.
            dtype (dtype): The data type to be used for computation. Typically, this should be a floating-point data type.

        Returns:
            None: This method does not return any value explicitly. It updates the 'cos_cached' and 'sin_cached'
                attributes of the class instance.

        Raises:
            ValueError: If the 'seq_len' provided is less than or equal to 0.
            RuntimeError: If an error occurs during the computation of cosine and sine caches.
        """
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2).float() / self.dim))
            self.inv_freq = inv_freq

        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1, x2 = x.tensor_split(2, -1)
    return ops.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`mindspore.Tensor`): The query tensor.
        k (`mindspore.Tensor`): The key tensor.
        cos (`mindspore.Tensor`): The cosine part of the rotary embedding.
        sin (`mindspore.Tensor`): The sine part of the rotary embedding.
        position_ids (`mindspore.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(mindspore.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=mindspore.float32)
    k_fp32 = k.to(dtype=mindspore.float32)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)

class MiniCPMMLP(nn.Module):

    """
    MiniCPMMLP is a neural network model that implements a specific variant of a Multi-Layer Perceptron (MLP)
    architecture for deep learning tasks.
    This class inherits from nn.Module and includes methods for initializing the model's parameters and forwarding
    the forward pass computation.

    Attributes:
        config: A configuration object containing parameters such as hidden_size, intermediate_size,
            hidden activation function, and pretraining_tp.
        hidden_size: The size of the hidden layers in the MLP.
        intermediate_size: The size of the intermediate layers in the MLP.
        gate_proj: A dense layer for projecting input to intermediate size with no bias.
        up_proj: A dense layer for projecting input to intermediate size with no bias.
        down_proj: A dense layer for projecting intermediate size to hidden size with no bias.
        act_fn: The activation function applied to the hidden layers based on the specified configuration.

    Methods:
        __init__: Initializes the MiniCPMMLP instance with the provided configuration.
        forward: Constructs the forward pass computation of the MiniCPMMLP model based on the input tensor x.
            If pretraining_tp > 1, it performs a segmented computation using the specified number of segments.
            Otherwise, it computes the forward pass in a single step.

    Returns:
        down_proj: The output tensor resulting from the forward pass computation of the MiniCPMMLP model.
    """
    def __init__(self, config):
        """
        Initializes a MiniCPMMLP object with the provided configuration.

        Args:
            self (MiniCPMMLP): The MiniCPMMLP object instance.
            config: Configuration object containing parameters for the MiniCPMMLP model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Constructs the intermediate states of the MiniCPMMLP model based on the input tensor x.

        Args:
            self (MiniCPMMLP): An instance of the MiniCPMMLP class.
            x (tensor): The input tensor for forwarding the intermediate states.

        Returns:
            None. The method forwards the intermediate states of the model.

        Raises:
            None.
        """
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, axis=0)
            up_proj_slices = self.up_proj.weight.split(slice, axis=0)
            down_proj_slices = self.down_proj.weight.split(slice, axis=1)

            gate_proj = ops.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = ops.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, axis=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MiniCPMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        """
        Initializes an instance of the MiniCPMAttention class.

        Args:
            self: The instance of the class.
            config (MiniCPMConfig):
                The configuration object for MiniCPMAttention.

                - `config` contains various attributes that define the behavior of the attention mechanism.
                - It is an instance of the MiniCPMConfig class.
            layer_idx (Optional[int], default=None):
                The index of the layer.

                - This parameter is optional and can be omitted.
                - If provided, it helps in caching during the forward call.
                - Not providing `layer_idx` is not recommended, as it may lead to errors if caching is used.
                - Please make sure to provide a valid `layer_idx` when creating an instance of this class.

        Returns:
            None.

        Raises:
            ValueError:
                If `hidden_size` is not divisible by `num_heads`.

                - This exception is raised when the condition `hidden_size % num_heads != 0` is not satisfied.
                - `hidden_size` must be divisible by `num_heads` for the attention mechanism to work correctly.

            Warning:
                If `layer_idx` is not provided, a warning is issued.

                - The warning message suggests that not providing `layer_idx` is not recommended.
                - It also highlights that errors may occur during the forward call if caching is used.
                - The user is advised to provide a valid `layer_idx` when creating an instance of this class.

        Note:
            The method initializes the MiniCPMAttention instance by assigning values to various attributes.
            It performs several checks to ensure the correctness of the provided configuration.
            The method also initializes the projection layers and sets up the required variables
            for the attention mechanism.
            Additionally, it initializes the rope mechanism by calling the `_init_rope` method.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        """
        This method initializes the Rotary Positional Encoding (RoPE) for the MiniCPMAttention class.

        Args:
            self: MiniCPMAttention
                The instance of the MiniCPMAttention class.

        Returns:
            None.

        Raises:
            ValueError:
                If the scaling_type provided in the configuration for RoPE is not 'linear' or 'dynamic'.
        """
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method is responsible for shaping the input tensor to prepare it for MiniCPMAttention computation.

        Args:
            tensor (mindspore.Tensor): The input tensor to be reshaped.
                It should be of shape (seq_len * bsz, num_heads * head_dim).
            seq_len (int): The length of the input sequence.
            bsz (int): The batch size.

        Returns:
            None: This method returns None as it directly modifies the input tensor in place.

        Raises:
            ValueError: If the shape of the input tensor is not compatible with the reshaping operation.
            TypeError: If the input tensor is not of type mindspore.Tensor.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        '''
        This method forwards the MiniCPMAttention layer.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states with shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): Optional tensor with shape
                (batch_size, 1, sequence_length, sequence_length) representing the attention mask.
            position_ids (Optional[mindspore.Tensor]): Optional tensor with shape (batch_size, sequence_length)
                representing the position indices of input tokens.
            past_key_value (Optional[Cache]): Optional cache for past key-value pairs.
            output_attentions (bool): Flag indicating whether to return the attention weights.
            use_cache (bool): Flag indicating whether to use cache for key-value pairs.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the output tensor of shape (batch_size, sequence_length, hidden_size),
                optional attention weights tensor, and optional updated past key-value pairs.

        Raises:
            ValueError: If the attention weights or attention mask have invalid shapes.
            ValueError: If the output tensor 'attn_output' has an unexpected shape.
            ValueError: If the cache structure has changed since version v4.36 and the layer index is not
                initialized for auto-regressive decoding.
        '''
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.shape

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, axis=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = ops.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = ops.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = ops.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states.to(mindspore.float32), seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, dim=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, axis=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, axis=1)
            attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
}


class MiniCPMDecoderLayer(nn.Module):

    """
    MiniCPMDecoderLayer represents a single layer of the MiniCPM (Minimalist Conditional Pretrained Model) decoder.
    This class is responsible for processing input hidden states through self-attention mechanism and MLP
    (Multi-Layer Perceptron) for decoding tasks.

    Attributes:
        hidden_size (int): Size of the hidden states.
        self_attn (MINICPM_ATTENTION_CLASSES): Instance of the attention mechanism used in the layer.
        mlp (MiniCPMMLP): Instance of the MLP network.
        input_layernorm (MiniCPMRMSNorm): Layer normalization applied to the input hidden states.
        post_attention_layernorm (MiniCPMRMSNorm): Layer normalization applied after the self-attention mechanism.
        scale_depth (int): Scaling factor applied to the hidden states.
        num_hidden_layers (int): Number of hidden layers in the model.

    Methods:
        forward:
            Processes the input hidden states through the layer.

            Args:

            - hidden_states (mindspore.Tensor): Input to the layer of shape (batch, seq_len, embed_dim).
            - attention_mask (mindspore.Tensor, optional): Attention mask used for masking certain positions in the input.
            - position_ids (mindspore.Tensor, optional): Tensor representing the position ids of each token.
            - past_key_value (Tuple[mindspore.Tensor], optional): Cached past key and value projection states.
            - output_attentions (bool, optional): Whether to return attention tensors of all attention layers.
            - use_cache (bool, optional): If True, past key-value states are returned for speeding up decoding.
            - kwargs: Additional keyword arguments.

            Returns:

            - Tuple containing the processed hidden states and optionally attentions and present key values.

    Note:
        If 'padding_mask' is passed as a keyword argument in kwargs, a deprecation warning will be issued.
        It is recommended to use 'attention_mask' instead.
    """
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        """
        Initializes a new instance of MiniCPMDecoderLayer.

        Args:
            self: The object instance.
            config (MiniCPMConfig): An instance of MiniCPMConfig containing the configuration settings
                for the decoder layer.
            layer_idx (int): The index of the layer within the decoder.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MiniCPMConfig.
            ValueError: If the layer_idx parameter is not a non-negative integer.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MINICPM_ATTENTION_CLASSES['eager'](config=config, layer_idx=layer_idx)

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated.37. Please make sure use `attention_mask` instead.`"
            )

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
            **kwargs,
        )

        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MiniCPMPreTrainedModel(PreTrainedModel):

    """
    Represents a pre-trained mini version of CPM (Code-PM) model for various NLP tasks.
    This class inherits from PreTrainedModel and provides functionality to initialize weights for different types
    of cells.

    The _init_weights method initializes the weights of the given cell based on the specified configuration.
    It sets the weights using either a normal distribution with the specified standard deviation or zeros for bias,
    depending on the type of the cell. For Dense cells, it initializes both weights and biases, while for Embedding cells,
    it initializes weights with random values and sets a specific padding index to zero if provided.

    Parameters:
        cell: The cell for which weights need to be initialized.

    Returns:
        None
    """
    config_class = MiniCPMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, cell):
        """
        Initializes the weights of the given cell.

        Args:
            self (MiniCPMPreTrainedModel): The instance of the MiniCPMPreTrainedModel class.
            cell: The cell whose weights need to be initialized.

        Returns:
            None. This method initializes the weights of the cell in-place.

        Raises:
            None.
        """
        std = self.config.initializer_range
        if isinstance(cell, nn.Linear):
            cell.weight.set_data(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class MiniCPMModel(MiniCPMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]

    Args:
        config: MiniCPMConfig
    """
    def __init__(self, config: MiniCPMConfig):
        """
        Initializes a MiniCPMModel instance with the provided configuration.

        Args:
            self (MiniCPMModel): The instance of MiniCPMModel.
            config (MiniCPMConfig):
                The configuration object containing various settings for the model.

                - config.pad_token_id (int): The token ID used for padding sequences.
                - config.vocab_size (int): The size of the vocabulary.
                - config.hidden_size (int): The dimension of the hidden layers.
                - config.num_hidden_layers (int): The number of hidden layers in the model.
                - config.rms_norm_eps (float): The epsilon value for RMS normalization.

        Returns:
            None.

        Raises:
            ValueError: If the configuration object is missing required attributes.
            TypeError: If the configuration attributes are of incorrect types.
            RuntimeError: If there is an issue during the initialization process.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings for the MiniCPMModel.

        Args:
            self (MiniCPMModel): An instance of the MiniCPMModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the MiniCPMModel.

        Args:
            self (MiniCPMModel): The instance of the MiniCPMModel class.
            new_embeddings (object): The new embeddings to be set for self.embed_tokens.

        Returns:
            None.

        Raises:
            None.

        This method allows the user to set the input embeddings for the MiniCPMModel by replacing the current embeddings
        with the provided new_embeddings. The new_embeddings can be of any type or format, as long as it is compatible
        with the self.embed_tokens attribute. After calling this method, the MiniCPMModel instance will use the
        new embeddings for further processing.

        Note:
            The new_embeddings should be compatible with the existing self.embed_tokens attribute to ensure proper
            functioning of the MiniCPMModel.
        """
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the MiniCPMModel.

        Args:
            self (object): The instance of the MiniCPMModel class.
            input_ids (mindspore.Tensor): The input tensor containing the token IDs. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor. Default is None.
            position_ids (Optional[mindspore.Tensor]): The tensor containing the position IDs. Default is None.
            past_key_values (Optional[List[mindspore.Tensor]]): List of tensors representing past key values. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The tensor containing the embeddings of input tokens. Default is None.
            use_cache (Optional[bool]): Flag indicating whether to use cache. Default is None.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Flag indicating whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                A tuple containing the hidden states, next_cache, all_hidden_states, and all_self_attns if not None;
                or a BaseModelOutputWithPast instance containing the last hidden state, past key values, hidden states,
                and attentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified simultaneously, or if neither input_ids nor
                inputs_embeds are specified.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPMForCausalLM(MiniCPMPreTrainedModel):
    r"""
    This class represents the MiniCPM model for causal language modeling. It is specifically designed for generating
    text based on given input prompts. The model is initialized with a configuration and consists of a MiniCPM model,
    an embedding layer, and a linear layer for predicting the next token in the sequence.

    Attributes:
        model (MiniCPMModel): The underlying MiniCPM model.
        vocab_size (int): The size of the vocabulary.
        lm_head (nn.Linear): The linear layer for predicting the next token.

    Methods:
        __init__: Initializes the MiniCPMForCausalLM model.
        get_input_embeddings: Returns the input embeddings of the model.
        set_input_embeddings: Sets the input embeddings of the model.
        get_output_embeddings: Returns the output embeddings of the model.
        set_output_embeddings: Sets the output embeddings of the model.
        set_decoder: Sets the decoder of the model.
        get_decoder: Returns the decoder of the model.
        forward: Constructs the MiniCPM model and computes the language modeling loss.
        prepare_inputs_for_generation: Prepares the inputs for text generation.
        _reorder_cache: Reorders the cache for beam search.
        chat: Generates a response to a given query using the MiniCPM model.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, MiniCPMForCausalLM
        ...
        >>> model = MiniCPMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        ...
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        ...
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the MiniCPMForCausalLM class.

        Args:
            self (MiniCPMForCausalLM): The object instance.
            config: The configuration object containing the model's settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings from the MiniCPMForCausalLM model.

        Args:
            self: The instance of the MiniCPMForCausalLM class.

        Returns:
            The input embeddings from the model.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Method to set new input embeddings for the MiniCPMForCausalLM model.

        Args:
            self (MiniCPMForCausalLM): The instance of MiniCPMForCausalLM class.
            new_embeddings (object): The new embeddings to be set for the model.
                Should be compatible with the model's embed_tokens attribute.

        Returns:
            The input embeddings for the model.

        Raises:
            None.
        """
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the MiniCPMForCausalLM model.

        Args:
            self (MiniCPMForCausalLM): The instance of the MiniCPMForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the output embeddings of the MiniCPMForCausalLM model.
        The output embeddings are computed by the 'lm_head' layer of the model.

        Note:
            The 'lm_head' layer is a linear transformation layer that maps the final hidden states of the model to
            the vocabulary size. It is responsible for generating the output probabilities for each token
            in the sequence.

        Example:
            ```python
            >>> model = MiniCPMForCausalLM()
            >>> embeddings = model.get_output_embeddings()
            ```
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new embeddings for the output layer of the MiniCPMForCausalLM model.

        Args:
            self (MiniCPMForCausalLM): The instance of the MiniCPMForCausalLM class.
                This parameter is used to reference the current instance of the MiniCPMForCausalLM model.
            new_embeddings (any): The new embeddings to be set as the output embeddings.
                This parameter represents the new embeddings that will replace the current output embeddings.
                It can be of any data type.

        Returns:
            None: This method does not return any value. It sets the 'lm_head' attribute of the MiniCPMForCausalLM
                instance to the new_embeddings.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        This method sets the decoder for the MiniCPMForCausalLM model.

        Args:
            self (object): The instance of the MiniCPMForCausalLM class.
            decoder (object): The decoder object to be set for the model. It should be an instance of a decoder class.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Retrieves the decoder model used for the MiniCPMForCausalLM class.

        Args:
            self: An instance of the MiniCPMForCausalLM class.

        Returns:
            The decoder model object.

        Raises:
            None.

        This method returns the decoder model object associated with the MiniCPMForCausalLM instance.
        The decoder model is an essential component of the MiniCPMForCausalLM class and is used for generating
        predictions based on the input data. The decoder model object is returned as the result of this method.
        """
        return self.model

    def forward(
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            `Union[Tuple, CausalLMOutputWithPast]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MiniCPMForCausalLM
            ...
            >>> model = MiniCPMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
            >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
            ...
            >>> prompt = "Hey, are you conscious? Can you talk to me?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
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
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, axis=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = ops.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = F.cross_entropy(shift_logits, shift_labels)

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

        Args:
            self (MiniCPMForCausalLM): The instance of the MiniCPMForCausalLM class.
            input_ids (torch.Tensor): The input tensor of token indices. Shape: [batch_size, sequence_length].
            past_key_values (Cache or Tuple[torch.Tensor, torch.Tensor] or None): The past key values used for
                efficient generation. If Cache object or Tuple is provided, it contains the cached key and value
                tensors. If None, no past key values are used.
            attention_mask (torch.Tensor or None): The attention mask tensor to mask padded tokens.
                Shape: [batch_size, sequence_length].
            inputs_embeds (torch.Tensor or None): The tensor of embeddings for input tokens.
                Shape: [batch_size, sequence_length, embedding_dim].

        Returns:
            dict: A dictionary containing the model inputs including either 'input_ids' or 'inputs_embeds',
                'position_ids', 'past_key_values', 'use_cache', and 'attention_mask'.

        Raises:
            TypeError: If the input_ids, past_key_values, attention_mask, or inputs_embeds have invalid types.
            ValueError: If the input_ids and attention_mask shapes are incompatible or
                if cache_length + input_ids.shape[1] > max_cache_length.
        """
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

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
        Reorders the past key values based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing past key values from the model layers.
            beam_idx (Tensor): A tensor representing the beam index used for reordering.

        Returns:
            tuple: A tuple of reordered past key values based on the provided beam index.

        Raises:
            None
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 4096, num_beams=1, do_sample=True, top_p=0.8, temperature=0.3, logits_processor=None,
             **kwargs):
        """
        Chat method for MiniCPMForCausalLM class.

        This method facilitates a conversation by generating responses based on the given query and history.
        It utilizes a tokenizer to convert text into tokens and a language model to generate responses.

        Args:
            self (MiniCPMForCausalLM): An instance of the MiniCPMForCausalLM class.
            tokenizer: The tokenizer object used to tokenize the input text.
            query (str): The user's query as a string.
            history (List[Dict], optional): A list of dictionaries representing the conversation history.
                Each dictionary contains the role (e.g., 'user' or 'assistant') and the content of the message.
                Defaults to None.
            role (str, optional): The role of the current message. Defaults to 'user'.
            max_length (int, optional): The maximum length of the generated response. Defaults to 4096.
            num_beams (int, optional): The number of beams to be used during generation. Defaults to 1.
            do_sample (bool, optional): Whether to use sampling during generation. Defaults to True.
            top_p (float, optional): The cumulative probability for top-p sampling. Defaults to 0.8.
            temperature (float, optional): The temperature value for generation. Defaults to 0.3.
            logits_processor: An optional logits_processor object to be used during generation. Defaults to None.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            tuple: A tuple containing the generated response (str) and the updated conversation history (List[Dict]).

        Raises:
            None.
        """
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        else:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(history_str, return_tensors='ms')
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return response, history


class MiniCPMForSequenceClassification(MiniCPMPreTrainedModel):

    """
    MiniCPMForSequenceClassification is a Python class that represents a fine-tuning model for sequence classification
    tasks based on the MiniCPM architecture. It inherits from the MiniCPMPreTrainedModel class and provides methods for
    initializing the model, getting and setting input embeddings, and forwarding the sequence classification model.

    Attributes:
        num_labels (int): The number of labels for sequence classification.
        model (MiniCPMModel): The MiniCPM model used for sequence classification.
        score (nn.Linear): The layer for scoring sequence classification logits.

    Methods:
        __init__: Initializes the MiniCPMForSequenceClassification instance with the provided configuration.
        get_input_embeddings: Returns the input embeddings from the MiniCPM model.
        set_input_embeddings: Sets new input embeddings for the MiniCPM model.
        forward: Constructs the sequence classification model based on the provided input arguments.

    Args:
        input_ids (mindspore.Tensor, optional): The input token IDs for the sequence.
        attention_mask (mindspore.Tensor, optional): The attention mask for the input sequence.
        position_ids (mindspore.Tensor, optional): The position IDs for the input tokens.
        past_key_values (List[mindspore.Tensor], optional): The past key values for autoregressive decoding.
        inputs_embeds (mindspore.Tensor, optional): The input embeddings for the sequence.
        labels (mindspore.Tensor, optional): The labels for computing the sequence classification/regression loss.
        use_cache (bool, optional): Whether to use cache for autoregressive decoding.
        output_attentions (bool, optional): Whether to output attentions in the model.
        output_hidden_states (bool, optional): Whether to output hidden states in the model.
        return_dict (bool, optional): Whether to return the model outputs as a dictionary.

    Returns:
        Union[Tuple, SequenceClassifierOutputWithPast]: The forwarded model outputs, including the loss, logits,
            past key values, hidden states, and attentions.

    Raises:
        ValueError: If the batch size is greater than 1 and no padding token is defined.

    Note:
        This class inherits from MiniCPMPreTrainedModel and extends its functionality to support sequence
        classification tasks.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the MiniCPMForSequenceClassification class.

        Args:
            self (MiniCPMForSequenceClassification): The current instance of the class.
            config: An instance of the configuration class specifying the model's hyperparameters and settings.

        Returns:
            None

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the model.

        Args:
            self (MiniCPMForSequenceClassification): The instance of the MiniCPMForSequenceClassification class.
                This parameter is used to access the model's embed_tokens attribute.

        Returns:
            None: This method returns None as it simply retrieves the input embeddings from the model.

        Raises:
            None
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Method to set new input embeddings for the MiniCPMForSequenceClassification model.

        Args:
            self (MiniCPMForSequenceClassification): Instance of the MiniCPMForSequenceClassification class.
            new_embeddings (object): New embeddings to be set for the model.
                Should be compatible with the model's input embedding format.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = new_embeddings

    def forward(
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
                sequence_lengths = ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            else:
                sequence_lengths = -1

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
                    loss = F.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(pooled_logits, labels)
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
    'MiniCPMModel',
    'MiniCPMPreTrainedModel',
    'MiniCPMForCausalLM',
    'MiniCPMForSequenceClassification'
]
