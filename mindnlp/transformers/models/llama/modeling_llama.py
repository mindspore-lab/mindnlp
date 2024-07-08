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
""" MindSpore LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import mindspore
from mindspore import ops, nn, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


class LlamaRMSNorm(nn.Cell):

    """
    LlamaRMSNorm is a class that represents a normalization layer, equivalent to T5LayerNorm,
    used in deep learning models. It inherits from the nn.Cell class.
    
    This class provides methods to initialize and apply RMS normalization to the input hidden states.
    The RMS normalization is calculated based on the variance of the hidden states and a weight parameter.
    The normalized hidden states are then multiplied by the weight parameter to obtain the final output.

    Attributes:
        weight (mindspore.Parameter): The weight parameter used in the RMS normalization.
        variance_epsilon (float): The epsilon value added to the variance to avoid division by zero.

    Methods:
        __init__: Initializes a new instance of the LlamaRMSNorm class.
        construct: Applies RMS normalization to the input hidden states.

    Example:
        ```python
        >>> # Create an instance of LlamaRMSNorm
        >>> norm = LlamaRMSNorm(hidden_size=256)
        ...
        >>> # Apply RMS normalization to hidden states
        >>> output = norm.construct(hidden_states)
        ```
    Please note that the LlamaRMSNorm class is designed to be used as part of a neural network model and requires the
    MindSpore library for execution."""
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size), 'weight')
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """Constructs the RMS normalization of the hidden states.

        Args:
            self (LlamaRMSNorm): The instance of the LlamaRMSNorm class.
            hidden_states (Union[Tensor, ndarray]): The input hidden states to be normalized.
                Should be a tensor or numpy array of any shape.

        Returns:
            None: This method does not return any value. The normalization is applied in place.

        Raises:
            ValueError: If the input hidden_states is not a valid tensor or numpy array.
            RuntimeError: If an error occurs during the normalization process.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight.astype(input_dtype) * hidden_states.astype(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Cell):

    """
    The `LlamaRotaryEmbedding` class represents a rotary positional embedding layer that can be used in
    neural network models. It inherits from the `nn.Cell` class.

    Attributes:
        dim (int): The dimension of the embedding.
        max_position_embeddings (int): The maximum number of position embeddings.
        base (int): The base value used for calculating inverse frequencies.
        inv_freq (Tensor): The tensor containing the inverse frequencies calculated based on the `dim` and `base` values.
        max_seq_len_cached (int): The maximum sequence length for which cosine and sine values are cached.
        cos_cached (Tensor): The cached cosine values for the positional embeddings.
        sin_cached (Tensor): The cached sine values for the positional embeddings.

    Methods:
        __init__:
            Initializes a new instance of the `LlamaRotaryEmbedding` class.

        _set_cos_sin_cache:
            Sets up the cosine and sine cache for a given sequence length and data type.

        construct:
            Constructs the positional embedding for the input tensor `x`.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes a new instance of the LlamaRotaryEmbedding class.

        Args:
            self: The LlamaRotaryEmbedding object itself.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value for calculating the inverse frequency. Defaults to 10000.

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

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=mindspore.float32
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        Sets the cosine and sine caches for LlamaRotaryEmbedding.

        Args:
            self (LlamaRotaryEmbedding): An instance of the LlamaRotaryEmbedding class.
            seq_len (int): The length of the sequence.
            dtype: The data type of the cache.

        Returns:
            None: The method updates the 'cos_cached' and 'sin_cached' attributes of the LlamaRotaryEmbedding instance.

        Raises:
            None.

        Description:
            This method sets the cosine and sine caches for LlamaRotaryEmbedding.
            The caches are used in the forward pass of the neural network for efficient calculation of rotary
            position embeddings.

            The method first sets the maximum sequence length cached to the given sequence length.
            It then creates a tensor 't' using the 'arange' operation from the 'ops' module, with the same data type as
            'inv_freq'.

            Next, it calculates the element-wise product of 't' and 'inv_freq' using 'einsum' operation
            from the 'ops' module. The result is a tensor 'freqs' which represents the frequencies for each position in
            the sequence.

            To create the cache tensor, 'freqs' is concatenated with itself along the last axis using the 'cat'
            operation from the 'ops' module. The resulting tensor 'emb' has shape (seq_len, 2 * frequency_dim),
            where frequency_dim is the dimension of the 'inv_freq' tensor.

            Finally, the 'cos_cached' and 'sin_cached' attributes are updated by calculating the cosine and sine of
            each element in 'emb', respectively. The resulting tensors are converted to the given data type
            'dtype' using the 'to' method.

        Note:
            It is assumed that the 'inv_freq' attribute of the LlamaRotaryEmbedding instance has been initialized
            prior to calling this method.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def construct(self, x, seq_len=None):
        """
        Constructs a subset of the cached cosine and sine values based on the given sequence length.

        Args:
            self (LlamaRotaryEmbedding): An instance of the LlamaRotaryEmbedding class.
            x: The input tensor.
            seq_len (int, optional): The length of the desired subset. Defaults to None.

        Returns:
            tuple: A tuple containing two tensors. The first tensor represents the subset of cached cosine values,
                and the second tensor represents the subset of cached sine values. Both tensors are of the
                same dtype as x.

        Raises:
            TypeError: If seq_len is not an integer or None.
            ValueError: If seq_len is less than or equal to 0.
            AttributeError: If seq_len exceeds the maximum sequence length that has been cached.

        Note:
            The returned subset will include elements up to the index 'seq_len - 1' from the cached cosine and sine values.
            If seq_len is None or not provided, the entire cached cosine and sine values will be returned.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes a new instance of the LlamaLinearScalingRotaryEmbedding class.

        Args:
            self (LlamaLinearScalingRotaryEmbedding): The current instance of the class.
            dim (int): The dimensionality of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Default is 2048.
            base (int, optional): The base value used for scaling. Default is 10000.
            scaling_factor (float, optional): The scaling factor applied to the embeddings. Default is 1.0.

        Returns:
            None.

        Raises:
            None.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        Sets the cosine and sine caches for the LlamaLinearScalingRotaryEmbedding class.

        Args:
            self (LlamaLinearScalingRotaryEmbedding): The instance of the LlamaLinearScalingRotaryEmbedding class.
            seq_len (int): The length of the sequence.
            dtype: The desired data type for the cache.

        Returns:
            None.

        Raises:
            None.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes an instance of the LlamaDynamicNTKScalingRotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int): The maximum number of position embeddings to be considered. Default is 2048.
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
        Method to set the cosine and sine cache for dynamic NTK scaling rotary embedding in the
        LlamaDynamicNTKScalingRotaryEmbedding class.

        Args:
            self: Instance of the LlamaDynamicNTKScalingRotaryEmbedding class.
            seq_len: Integer representing the length of the sequence for which the cosine and sine cache is being set.
            dtype: Data type of the elements in the cache.

        Returns:
            None: This method updates the cosine and sine cache attributes of the instance.

        Raises:
            ValueError: If the input sequence length 'seq_len' is not a positive integer.
            ValueError: If the input data type 'dtype' is not a valid data type.
            RuntimeError: If an error occurs during the calculation of the cosine and sine cache.
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
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    (x1, x2) = x.tensor_split(2, axis=-1)
    return ops.cat((-x2, x1), axis=x.ndim-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.

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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Cell):

    """
    This class represents a multi-layer perceptron (MLP) model called LlamaMLP.

    LlamaMLP inherits from the nn.Cell class and is designed for deep learning tasks.
    It consists of multiple layers, including gate projection, up projection, and down projection layers,
    which are used to transform the input data and produce the final output.

    Attributes:
        config (object): The configuration object that stores the hyperparameters of the LlamaMLP model.
        hidden_size (int): The size of the hidden layer in the LlamaMLP model.
        intermediate_size (int): The size of the intermediate layer in the LlamaMLP model.
        gate_proj (object): The dense layer responsible for the gate projection in the LlamaMLP model.
        up_proj (object): The dense layer responsible for the up projection in the LlamaMLP model.
        down_proj (object): The dense layer responsible for the down projection in the LlamaMLP model.
        act_fn (function): The activation function used in the LlamaMLP model.

    Methods:
        __init__:
            Initializes a new instance of the LlamaMLP class.

        construct:
            Constructs the LlamaMLP model by applying the necessary transformations on the input data.
            This method returns the final output of the LlamaMLP model.

    Note:
        The LlamaMLP model supports pretraining when the 'pretraining_tp' hyperparameter is greater than 1.
        In this case, the input data is split into slices to perform parallel computations. Otherwise, the
        computations are performed in a single path.
    """
    def __init__(self, config):
        """
        Initializes an instance of the LlamaMLP class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config' containing the configuration settings for the MLP.
                The 'Config' object should have the following properties:

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str): The activation function for the hidden layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        """
        Constructs the output of the LlamaMLP model based on the input and configuration settings.

        Args:
            self (LlamaMLP): The instance of the LlamaMLP class.
            x (tensor): The input tensor to be processed by the model.

        Returns:
            None.

        Raises:
            ValueError: If the value of 'pretraining_tp' in the configuration is less than or equal to 1.
            TypeError: If any of the operations cannot be performed due to data type mismatch or other reasons.
            IndexError: If any index used for slicing or accessing tensors is out of bounds.
        """
        if self.config.pretraining_tp > 1:
            slices = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slices, axis=0)
            up_proj_slices = self.up_proj.weight.split(slices, axis=0)
            down_proj_slices = self.down_proj.weight.split(slices, axis=1)

            gate_proj = ops.cat(
                [ops.dense(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], axis=-1
            )
            up_proj = ops.cat([ops.dense(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], axis=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, axis=2)
            down_proj = [
                ops.dense(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
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
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: LlamaConfig):
        """
        Initializes an instance of the LlamaAttention class.

        Args:
            self: The instance of the LlamaAttention class.
            config (LlamaConfig):
                The configuration object that holds various parameters for the attention mechanism.

                - config.attention_dropout (float): The dropout rate for attention weights.
                - config.hidden_size (int): The size of the hidden state.
                - config.num_attention_heads (int): The number of attention heads.
                - config.num_key_value_heads (int): The number of key-value attention heads.
                - config.max_position_embeddings (int): The maximum number of position embeddings.
                - config.rope_theta (float): The rope theta value.
                - config.attention_bias (bool): Specifies whether to use bias in attention projections.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size is not divisible by num_heads.

        Note:
            This method initializes various attributes of the LlamaAttention object, such as attention_dropout, hidden_size,
            num_heads, head_dim, num_key_value_heads, num_key_value_groups, max_position_embeddings, rope_theta, and is_causal.
            It also initializes the projection layers q_proj, k_proj, v_proj, and o_proj.
            Additionally, it initializes the rope (a positional encoding used in the attention mechanism) using _init_rope method.
        """
        super().__init__()
        self.config = config
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

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.attention_bias)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        """
        Initializes the Rotary Positional Encoding (RoPE) based on the provided configuration.

        Args:
            self (LlamaAttention): The instance of the LlamaAttention class.

        Returns:
            None.

        Raises:
            ValueError: If the 'type' of RoPE scaling provided in the configuration is not recognized or supported.
        """
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor according to the specified dimensions for the LlamaAttention class.

        Args:
            self (LlamaAttention): The instance of the LlamaAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size of the tensor.

        Returns:
            None

        Raises:
            None
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
        This method constructs the LlamaAttention layer.

        Args:
            self: The instance of the LlamaAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional tensor of shape
                (batch_size, 1, sequence_length, sequence_length) representing the attention mask.
            position_ids (Optional[mindspore.Tensor]): An optional tensor of shape
                (batch_size, sequence_length) representing the position ids.
            past_key_value (Optional[Tuple[mindspore.Tensor]]): An optional tuple containing the past key and value states.
            output_attentions (bool): A flag indicating whether to output attention weights.
            use_cache (bool): A flag indicating whether to use cache for past key-value states.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]: A tuple containing
                the attention output tensor of shape (batch_size, sequence_length, hidden_size),
                optional attention weights tensor, and optional updated past key-value states tuple.

        Raises:
            ValueError: If the shape of attention weights or attention mask is not as expected.
        """
        bsz, q_len, _ = hidden_states.shape

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, axis=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [ops.dense(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = ops.cat(query_states, axis=-1)

            key_states = [ops.dense(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = ops.cat(key_states, axis=-1)

            value_states = [ops.dense(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = ops.cat(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

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
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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
            attn_output = sum(ops.dense(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayer(nn.Cell):

    """
    The `LlamaDecoderLayer` class represents a layer of the Llama decoder in the Llama model.
    It inherits from the `nn.Cell` class.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        self_attn (`LlamaAttention`): The attention mechanism used in the layer.
        mlp (`LlamaMLP`): The multi-layer perceptron used in the layer.
        input_layernorm (`LlamaRMSNorm`): The input layer normalization module.
        post_attention_layernorm (`LlamaRMSNorm`): The layer normalization module applied after the attention mechanism.

    Methods:
        construct:
            Applies the Llama decoder layer to the input hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input to the layer of shape `(batch, seq_len, embed_dim)`.
            - attention_mask (mindspore.Tensor, optional): The attention mask. Its shape depends on the attention
            mechanism used. For flash attention, it has a shape of `(batch_size, sequence_length)`, and
            for default attention, it has a shape of `(batch_size, 1, query_sequence_length, key_sequence_length)`.
            - position_ids (mindspore.Tensor, optional): The position ids tensor.
            - past_key_value (Tuple[mindspore.Tensor], optional): The cached past key and value projection states.
            - output_attentions (bool, optional): Whether or not to return the attention tensors of all attention layers.
            See the `attentions` under the returned tensors for more detail.
            - use_cache (bool, optional): If set to True, the `past_key_values` key value states are returned and can be
            used to speed up decoding. See `past_key_values` for more information.
            - kwargs: Additional keyword arguments.

            Returns:

            - Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
            The output tensor of shape `(batch, seq_len, embed_dim)`.
            If `output_attentions` is True, the tuple also includes
            - the attention weights tensor.
            If `use_cache` is True, the tuple also includes the present key and value projection states.

    Note:
        The `LlamaDecoderLayer` class assumes that the `LlamaConfig` instance is already defined and passed as
        an argument to the constructor.

    Example:
        ```python
        >>> # Create a LlamaDecoderLayer instance
        >>> config = LlamaConfig(hidden_size=512)
        >>> decoder_layer = LlamaDecoderLayer(config)
        ...
        >>> # Apply the Llama decoder layer to the hidden states
        >>> hidden_states = ...
        >>> attention_mask = ...
        >>> output = decoder_layer.construct(hidden_states, attention_mask)
        ```
    """
    def __init__(self, config: LlamaConfig):
        """
        Initializes a LlamaDecoderLayer instance.

        Args:
            self (LlamaDecoderLayer): The current instance of the LlamaDecoderLayer class.
            config (LlamaConfig): An object of type LlamaConfig containing configuration parameters for the decoder layer.
                The config object must have the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - rms_norm_eps (float): The epsilon value for RMS normalization.

        Returns:
            None.

        Raises:
            TypeError: If config is not an instance of LlamaConfig.
            ValueError: If config is missing any required attribute.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
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


class LlamaPreTrainedModel(PreTrainedModel):

    """
    LlamaPreTrainedModel is a Python class representing a pre-trained model for llama-based machine learning tasks.
    This class inherits from PreTrainedModel and provides methods for initializing weights.

    The _init_weights method initializes the weights for the given cell.
    If the cell is of type nn.Dense, the weight is initialized using the Normal initializer within the specified range.
    If the cell has bias, it is initialized with zeros.
    If the cell is of type nn.Embedding, the weight is initialized with random normal values within the specified range,
    and the padding index is set to 0 if provided.

    Parameters:
        cell: The cell for which the weights need to be initialized.

    Returns:
        None
    """
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """
    def __init__(self, config: LlamaConfig):
        """
        Initializes a new instance of the LlamaModel class.

        Args:
            self: The object instance.
            config (LlamaConfig): The configuration object for the LlamaModel.
                This parameter specifies the configuration settings for the model.
                It should be an instance of the LlamaConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
            This method retrieves the input embeddings from the LlamaModel instance.

        Args:
            self (LlamaModel): The LlamaModel instance that this method is called on.

        Returns:
            None: This method returns the embed_tokens attribute of the LlamaModel instance,
                which represents the input embeddings. The return value is of type None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the LlamaModel instance.

        Args:
            self (LlamaModel): The LlamaModel instance.
            value (torch.Tensor): The input embeddings to be set.
                It should be a tensor of shape (num_embeddings, embedding_dim).

        Returns:
            None

        Raises:
            TypeError: If the input value is not a tensor.
            ValueError: If the input tensor shape is invalid.
        """
        self.embed_tokens = value

    def construct(
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
        Constructs the LlamaModel.

        Args:
            self (LlamaModel): The instance of the LlamaModel class.
            input_ids (mindspore.Tensor, optional): The input IDs tensor. Default is None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. Default is None.
            position_ids (mindspore.Tensor, optional): The position IDs tensor. Default is None.
            past_key_values (List[mindspore.Tensor], optional): The list of past key values. Default is None.
            inputs_embeds (mindspore.Tensor, optional): The input embeddings tensor. Default is None.
            use_cache (bool, optional): Whether to use cache. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The output of the LlamaModel.
                It can be a tuple containing hidden states, next cache, all hidden states, and all self attentions;
                or an instance of BaseModelOutputWithPast.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
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
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

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


class LlamaForCausalLM(LlamaPreTrainedModel):
    r"""
    This class represents a Llama model for Causal Language Modeling (LM) tasks.
    It includes methods for setting and getting input and output embeddings, setting and getting the decoder,
    as well as methods for model construction and preparing inputs for generation.
    The class inherits from LlamaPreTrainedModel and implements the necessary functionalities for generating text
    based on a given prompt.

    Attributes:
        model: Instance of LlamaModel used for the LM task.
        vocab_size: Size of the vocabulary used in the LM task.
        lm_head: Neural network layer for LM head.

    Methods:
        get_input_embeddings(): Retrieve the input embeddings from the model.
        set_input_embeddings(value): Set new input embeddings for the model.
        get_output_embeddings(): Get the output embeddings for the LM task.
        set_output_embeddings(new_embeddings): Set new output embeddings.
        set_decoder(decoder): Set a new decoder for the model.
        get_decoder(): Get the current decoder used in the model.
        construct(): Construct the model for the LM task with specified inputs and return the outputs.
        prepare_inputs_for_generation(): Prepare input data for text generation based on past key values and attention mask.
        _reorder_cache(past_key_values, beam_idx): Reorder cache elements based on beam index for efficient generation.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        ...
        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
        r"""
        Initializes an instance of the LlamaForCausalLM class.

        Args:
            self (LlamaForCausalLM): The instance of the LlamaForCausalLM class.
            config (dict): The configuration dictionary containing parameters for model initialization.
                Must include the following keys:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers in the model.

        Returns:
            None.

        Raises:
            ValueError: If the 'config' dictionary is missing required keys or if the values are of incorrect types.
            TypeError: If 'config' is not a dictionary or if any of the values in the 'config'
                dictionary are of incorrect types.
            RuntimeError: If an error occurs during model initialization.
        """
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve input embeddings from the 'LlamaForCausalLM' class model.

        Args:
            self (LlamaForCausalLM): The instance of the 'LlamaForCausalLM' class.
                This parameter is used to access the model's embed tokens for input embeddings.

        Returns:
            None: This method returns None as it directly retrieves and returns the input embeddings from the model.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method: set_input_embeddings

        Description: Sets the input embeddings of the LlamaForCausalLM model.

        Args:
            self (LlamaForCausalLM):
                The instance of the LlamaForCausalLM class.

                - Type: LlamaForCausalLM
                - Purpose: Represents the current instance of the LlamaForCausalLM class.
                - Restrictions: Must be an instance of the LlamaForCausalLM class.

            value:
                The input embeddings to be set for the model.

                - Type: Any
                - Purpose: Represents the new input embeddings to be assigned to the model.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Retrieve the output embeddings from the LlamaForCausalLM model.

        Args:
            self: An instance of the LlamaForCausalLM class.

        Returns:
            lm_head: This method returns the output embeddings from the lm_head layer of the LlamaForCausalLM model.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the LlamaForCausalLM model.

        Args:
            self (LlamaForCausalLM): The instance of the LlamaForCausalLM class.
            new_embeddings (Tensor): The new embeddings to be set for the model's lm_head.

        Returns:
            None.

        Raises:
            None.

        This method allows the user to update the output embeddings of the LlamaForCausalLM model by replacing the
        current embeddings with the provided new_embeddings.
        The new_embeddings should be a tensor of the same shape and size as the current embeddings.
        This method is useful in fine-tuning the model with custom embeddings or when transferring the model to
        a different task that requires different output embeddings.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the LlamaForCausalLM model.

        Args:
            self (LlamaForCausalLM): The instance of the LlamaForCausalLM class.
            decoder: The decoder object to be set for the model.

        Returns:
            None.

        Raises:
            None.

        This method sets the decoder object provided as an argument to the 'model' attribute of the
        LlamaForCausalLM instance.
        The 'model' attribute represents the decoder used for the causal language modeling task.
        """
        self.model = decoder

    def get_decoder(self):
        """
        This method returns the decoder model used for the LlamaForCausalLM class.

        Args:
            self: The instance of the LlamaForCausalLM class.

        Returns:
            None: This method returns the decoder model associated with the LlamaForCausalLM instance.

        Raises:
            None.
        """
        return self.model

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, LlamaForCausalLM
            ...
            >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [ops.dense(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = ops.cat(logits, axis=-1)
        else:
            logits = self.lm_head(hidden_states)
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
        Method to prepare inputs for generation in the LlamaForCausalLM class.

        Args:
            self (object): The instance of the class.
            input_ids (torch.Tensor): The input tensor representing tokenized input sequence.
            past_key_values (tuple, optional): Tuple containing past key values for autoregressive generation.
                Default is None.
            attention_mask (torch.Tensor, optional): Mask tensor indicating attention areas. Default is None.
            inputs_embeds (torch.Tensor, optional): Embedding tensor for the input tokens. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared model inputs including 'input_ids', 'position_ids',
                'past_key_values', 'use_cache', and 'attention_mask'.

        Raises:
            ValueError: If the input_ids shape is incorrect or if attention_mask is not provided.
            TypeError: If the position_ids are not of type torch.Tensor.
            RuntimeError: If an unexpected error occurs during position_ids calculation.
        """
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
        Reorders the cache of past key values for a given beam index in the LlamaForCausalLM class.

        Args:
            past_key_values (tuple): A tuple containing the cache of past key values.
                Each element in the tuple represents the cache for a particular layer.
            beam_idx (int): The index of the beam for which the cache is to be reordered.

        Returns:
            None: This method modifies the existing cache in-place.

        Raises:
            None.

        This static method reorders the cache of past key values for a specific beam index in the LlamaForCausalLM class.
        The method iterates over each layer's cache and reorders the past states based on the provided beam index.
        The reordered cache is then returned as a tuple of past key values. The original cache is modified in-place
        and no new objects are created.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class LlamaForSequenceClassification(LlamaPreTrainedModel):

    """
    LlamaForSequenceClassification

    This class is a sequence classification model based on the Llama architecture. It inherits from the LlamaPreTrainedModel class.

    Attributes:
        num_labels (int): The number of labels for the sequence classification task.
        model (LlamaModel): The LlamaModel instance used for the sequence classification.
        score (nn.Dense): The final layer that computes the logits for the classification.

    Methods:
        __init__:
            Initializes a new instance of the LlamaForSequenceClassification class.

        get_input_embeddings:
            Retrieves the input embeddings from the LlamaModel.

        set_input_embeddings:
            Sets the input embeddings in the LlamaModel.

        construct:
            Constructs the sequence classification model.

            Parameters:

            - input_ids (mindspore.Tensor): The input tensor of shape `(batch_size, sequence_length)`.
            - attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape
            `(batch_size, sequence_length)`.
            - position_ids (Optional[mindspore.Tensor]): The position IDs tensor of shape
            `(batch_size, sequence_length)`.
            - past_key_values (Optional[List[mindspore.Tensor]]): The list of past key-value tensors.
            - inputs_embeds (Optional[mindspore.Tensor]): The input embeddings tensor of shape
            `(batch_size, sequence_length, hidden_size)`.
            - labels (Optional[mindspore.Tensor]): The labels tensor of shape `(batch_size,)`.
            - use_cache (Optional[bool]): Whether to use cache for the model.
            - output_attentions (Optional[bool]): Whether to output attention tensors.
            - output_hidden_states (Optional[bool]): Whether to output hidden state tensors.
            - return_dict (Optional[bool]): Whether to return a SequenceClassifierOutputWithPast object.

            Returns:

            - Union[Tuple, SequenceClassifierOutputWithPast]:
            The output tuple or a SequenceClassifierOutputWithPast object.
    """
    def __init__(self, config):
        """
        Initializes an instance of the LlamaForSequenceClassification class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config',
                containing the configuration parameters for the model.

                - Type: 'Config' object
                - Purpose: The configuration parameters for the model
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the given sequence for the LlamaForSequenceClassification model.

        Args:
            self: An instance of the LlamaForSequenceClassification class.

        Returns:
            embed_tokens: The method returns a value of type 'None'.

        Raises:
            None.

        This method retrieves the input embeddings for the given sequence from the LlamaForSequenceClassification model.
        Input embeddings are the vector representations of the input tokens in the sequence that the model uses for
        further processing. These embeddings capture the contextual information of the tokens and are essential for
        downstream tasks such as sequence classification.

        Note:
            The input embeddings are obtained by calling the 'embed_tokens' method of the model instance.

        Example:
            ```python
            >>> llama_classifier = LlamaForSequenceClassification()
            >>> embeddings = llama_classifier.get_input_embeddings()
            ```
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Set the embedding layer of the LlamaForSequenceClassification model with a specified value.

        Args:
            self (LlamaForSequenceClassification): An instance of the LlamaForSequenceClassification class.
            value (torch.nn.Embedding): The embedding layer to be set in the model.

        Returns:
            None.

        Raises:
            TypeError: If the value parameter is not an instance of torch.nn.Embedding.
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
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
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
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
]
