# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore CPMAnt"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
import mindnlp.modules.functional as F
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from .configuration_cpmant import CpmAntConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openbmb/cpm-ant-10b"
_CONFIG_FOR_DOC = "CpmAntConfig"

CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openbmb/cpm-ant-10b",
    # See all CPMAnt models at https://hf-mirror.com/models?filter=cpmant
]


class CpmAntLayerNorm(nn.Cell):
    """
    We use Root Mean Square (RMS) Layer Normalization, please see https://arxiv.org/abs/1910.07467 for details."
    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes a new instance of the CpmAntLayerNorm class.
        
        Args:
            self: The object that the method belongs to.
            config (CpmAntConfig): The configuration object used to initialize the instance.
                The config parameter is of type CpmAntConfig and is required to initialize the instance.
                It contains the following attributes:

                - eps: A float value representing the epsilon value used in layer normalization.
                - hidden_size: An integer specifying the size of the hidden layer.
                
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()

        self.eps = config.eps
        self.dim_norm = config.hidden_size
        self.weight = Parameter(ops.zeros(config.hidden_size))

    def construct(self, hidden_states: mindspore.Tensor):
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        if hidden_states.shape[-1] != self.dim_norm:
            raise AssertionError("hidden_states.shape[-1] != self.dim_norm")
        old_dtype = hidden_states.dtype
        variance = hidden_states.to(mindspore.float32).pow(2).mean(axis=-1, keep_dims=True)
        hidden_states = (hidden_states * ops.rsqrt(variance + self.eps)).to(old_dtype) * self.weight
        return hidden_states


class CpmAntAttention(nn.Cell):

    """
    This class represents the CpmAntAttention module, which is a component of the CpmAnt model.
    It performs the self-attention mechanism in the transformer block.
    
    The CpmAntAttention module inherits from the nn.Cell class and initializes with a config object of type CpmAntConfig.
    
    Attributes:
        dim_model (int): The hidden size of the model.
        num_heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        project_q (nn.Dense): The linear transformation layer for query projection.
        project_k (nn.Dense): The linear transformation layer for key projection.
        project_v (nn.Dense): The linear transformation layer for value projection.
        attention_out (nn.Dense): The linear transformation layer for output projection.
        softmax (nn.Softmax): The softmax activation function for attention scores.
        dropout (nn.Dropout): The dropout layer, if configured.

    Methods:
        construct(hidden_q, hidden_kv, attention_mask, position_bias, output_attentions, past_key_values, use_cache):
            Constructs the self-attention block of the transformer.

            Args:

            - hidden_q (mindspore.Tensor): The input tensor for the self-attention block.
            - hidden_kv (mindspore.Tensor): The tensor for key-value projection.
            - attention_mask (mindspore.Tensor): The mask tensor to avoid invalid areas in self-attention.
            - position_bias (mindspore.Tensor): The positional information tensor for self-attention.
            - output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
            - past_key_values (Tuple[mindspore.Tensor, mindspore.Tensor], optional): Cached past key and value projection states.
            - use_cache (bool, optional): Whether to use cached key-value states to speed up decoding.

            Returns:

            - score (mindspore.Tensor): The output attention score tensor.
            - attn_weights (mindspore.Tensor): The attention weights tensor, if output_attentions is set to True.
            - past_key_values (Tuple[mindspore.Tensor, mindspore.Tensor]): The cached key-value states, if use_cache is set to True.
    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes an instance of CpmAntAttention.

        Args:
            self: The instance of the class.
            config (CpmAntConfig):
                An instance of CpmAntConfig containing configuration parameters.

                - hidden_size (int): The dimension size of the model.
                - num_attention_heads (int): The number of attention heads.
                - dim_head (int): The dimension of each attention head.
                - dropout_p (float, optional): The dropout probability. Default is None.

        Returns:
            None: This method initializes the CpmAntAttention instance with the provided configuration parameters.

        Raises:
            None.
        """
        super().__init__()
        self.dim_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dim_head = config.dim_head

        self.project_q = nn.Dense(self.dim_model, self.num_heads * self.dim_head, has_bias=False)
        self.project_k = nn.Dense(self.dim_model, self.num_heads * self.dim_head, has_bias=False)
        self.project_v = nn.Dense(self.dim_model, self.num_heads * self.dim_head, has_bias=False)

        self.attention_out = nn.Dense(self.num_heads * self.dim_head, self.dim_model, has_bias=False)

        self.softmax = nn.Softmax(axis=-1)

        if config.dropout_p is not None:
            self.dropout = nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

    def construct(
        self,
        hidden_q: mindspore.Tensor,
        hidden_kv: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        position_bias: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_q (`mindspore.Tensor`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            hidden_kv (`mindspore.Tensor` of shape `(batch, len_k, dim_model)`)):
                Tensor *key_value* and *query* of shape `(batch, len_k, dim_model)`
            attention_mask (`mindspore.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`mindspore.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple[mindspore.Tensor, mindspore.Tensor]`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        batch_size = hidden_q.shape[0]
        len_q = hidden_q.shape[1]
        len_k = hidden_kv.shape[1]

        query = self.project_q(hidden_q)
        key = self.project_k(hidden_kv)
        value = self.project_v(hidden_kv)

        query = query.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        if past_key_values is not None:
            key = ops.cat([past_key_values[0], key], axis=-2)
            value = ops.cat([past_key_values[1], value], axis=-2)
            len_k = key.shape[-2]

        # (batch_size, num_heads, len_q, dim_head) @ (batch_size, num_heads, dim_head, len_k) -> (batch_size, num_heads, len_q, len_k)
        score = ops.matmul(query, key.swapaxes(-1, -2)) / math.sqrt(self.dim_head)
        score = score + position_bias

        score = ops.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == mindspore.Tensor(False),
            ops.scalar_to_tensor(float("-inf"), dtype=score.dtype),
        )
        score = self.softmax(score)

        score = ops.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == mindspore.Tensor(False),
            ops.scalar_to_tensor(0, dtype=score.dtype),
        )
        if output_attentions:
            attn_weights = score
        else:
            attn_weights = None

        if self.dropout is not None:
            score = self.dropout(score)

        # (batch_size, num_heads, len_q, len_k) @ (batch_size, num_heads, len_k, dim_head) -> (batch_size, num_heads, len_q, dim_head)
        score = ops.matmul(score, value)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
        score = score.view(batch_size, len_q, self.num_heads * self.dim_head)

        score = self.attention_out(score)

        past_key_values = None
        if use_cache:
            past_key_values = (key, value)

        return score, attn_weights, past_key_values


class CpmAntSelfAttentionBlock(nn.Cell):

    """
    This class represents a self-attention block used in the CpmAnt model. It is a subclass of the nn.Cell class.

    Attributes:
        layernorm_before_attention (CpmAntLayerNorm):
            An instance of the CpmAntLayerNorm class that performs layer normalization before the self-attention operation.
        self_attention (CpmAntAttention):
            An instance of the CpmAntAttention class that performs the self-attention operation.
        dropout (nn.Dropout or None): An optional dropout layer. If configured, it applies dropout to the outputs.

    Methods:
        __init__: Initializes the CpmAntSelfAttentionBlock instance.

            Args:

            - config (CpmAntConfig): The configuration object for the CpmAnt model.

        construct: Applies the self-attention block to the given hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input tensor of shape `(batch, len_seq, dim_model)` representing the hidden states.
            - attention_mask (mindspore.Tensor): The attention mask tensor of shape `(batch, len_seq, len_seq)` that avoids invalid areas in the self-attention calculation.
            - position_bias (Optional[mindspore.Tensor]): An optional positional bias tensor of shape `(batch, len_seq, len_seq)` that provides positional information to the self-attention block.
            - output_attentions (Optional[bool]): Whether or not to return the attention tensors of all attention layers.
            - past_key_values (Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]): An optional tuple of past key and value projection states used for caching.
            - use_cache (Optional[bool]): If set to `True`, the past key and value states in `past_key_values` are returned and can be used to speed up decoding.

            Returns:

            - Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]: A tuple containing the updated hidden states, attention weights, and current key-value states.
    """
    def __init__(self, config: CpmAntConfig):
        """
        This method initializes a CpmAntSelfAttentionBlock instance.

        Args:
            self (CpmAntSelfAttentionBlock): The instance of the CpmAntSelfAttentionBlock class.
            config (CpmAntConfig): The configuration object containing settings for the self-attention block.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layernorm_before_attention = CpmAntLayerNorm(config)
        self.self_attention = CpmAntAttention(config)
        if config.dropout_p:
            self.dropout = nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        position_bias: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            attention_mask (`mindspore.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`mindspore.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        outputs = self.layernorm_before_attention(hidden_states)
        outputs = self.self_attention(
            outputs, outputs, attention_mask, position_bias, output_attentions, past_key_values, use_cache
        )

        outputs, attn_weights, current_key_value = outputs

        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs

        return hidden_states, attn_weights, current_key_value


class CpmAntDenseGatedACT(nn.Cell):

    """
    A class representing a dense gated activation layer for neural networks in the CPM-ANT model.

    This class inherits from nn.Cell and provides functionality to transform an input tensor from one feature space to another via a nonlinear operation. The transformation is performed using two dense layers
    with gated activation.

    Attributes:
        w_0 (nn.Dense): The first dense layer for the transformation.
        w_1 (nn.Dense): The second dense layer for the transformation.
        act (nn.GELU): The activation function to apply.

    Methods:
        __init__: Initializes the CpmAntDenseGatedACT instance.
        construct: Transforms an input tensor using the dense gated activation.

    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes an instance of the CpmAntDenseGatedACT class.

        Args:
            self: The object instance.
            config (CpmAntConfig):
                The configuration object that contains the required parameters for initialization.

                - `hidden_size` (int): The size of the hidden layer.
                - `dim_ff` (int): The dimension of the feed-forward layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.w_0 = nn.Dense(config.hidden_size, config.dim_ff, has_bias=False)
        self.w_1 = nn.Dense(config.hidden_size, config.dim_ff, has_bias=False)
        self.act = nn.GELU()

    def construct(self, hidden_states: mindspore.Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        gate_score = self.act(self.w_0(hidden_states))
        hidden_states = self.w_1(hidden_states)

        hidden_states = gate_score * hidden_states
        return hidden_states


class CpmAntFeedForward(nn.Cell):

    """
    CpmAntFeedForward represents a feedforward neural network component designed for the CpmAnt model architecture.
    This class inherits from nn.Cell and is used for processing hidden states through a series of transformations.

    Attributes:
        w_in (CpmAntDenseGatedACT): The first layer of the feedforward network for processing input hidden states.
        dropout (nn.Dropout or None): Dropout layer for regularization, initialized based on the configuration parameter.
        w_out (nn.Dense): The output layer of the feedforward network for producing final hidden states.

    Methods:
        __init__: Constructor method for initializing the CpmAntFeedForward instance with the given configuration.
        construct: Method for processing the input hidden states through the network layers.

    Args:
        config (CpmAntConfig): Configuration object containing settings for the feedforward network.
        hidden_states (mindspore.Tensor): Input tensor representing hidden states with shape (batch, seq_len, dim_in).

    Returns:
        mindspore.Tensor: Output tensor containing the processed hidden states after passing through the feedforward network.

    Usage:
        Instantiate an object of CpmAntFeedForward with a CpmAntConfig object and then call the construct method with input hidden_states
        to obtain the processed output hidden states.

    Note:
        - The dropout layer is optional based on the dropout probability specified in the configuration.
    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes an instance of the CpmAntFeedForward class.

        Args:
            self: The instance of the class.
            config (CpmAntConfig): An object of type CpmAntConfig containing configuration parameters.
                This parameter is required for configuring the feed-forward network.
                It should be an instance of CpmAntConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.w_in = CpmAntDenseGatedACT(config)
        if config.dropout_p is not None:
            self.dropout = nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

        self.w_out = nn.Dense(config.dim_ff, config.hidden_size, has_bias=False)

    def construct(self, hidden_states: mindspore.Tensor):
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        hidden_states = self.w_in(hidden_states)

        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_out(hidden_states)

        return hidden_states


class CpmAntFFNBlock(nn.Cell):

    """
    This class represents a feed-forward neural network block used in the CpmAnt model.
    It is a sub-module of the CpmAnt model and is responsible for applying feed-forward operations to the input hidden states.

    The CpmAntFFNBlock class inherits from the nn.Cell class, which is a base class for neural network cells in the MindSpore framework.

    Attributes:
        layernorm_before_ffn (CpmAntLayerNorm):
            An instance of the CpmAntLayerNorm class used for layer normalization before the feed-forward operation.
        ffn (CpmAntFeedForward):
            An instance of the CpmAntFeedForward class responsible for the actual feed-forward operation.
        dropout (nn.Dropout or None):
            An instance of the nn.Dropout class used for applying dropout regularization, if configured.
            If dropout probability is not specified, it is set to None.

    Methods:
        construct:
            Applies the feed-forward operations to the input hidden states and returns the updated hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states before the feed-forward layer.
            It has a shape of `(batch, len_seq, dim_model)`.

            Returns:

            - mindspore.Tensor: The updated hidden states after applying the feed-forward operations.

    Note:
        The CpmAntFFNBlock class is typically used as a building block within the CpmAnt model to process intermediate hidden states.
        It performs layer normalization, feed-forward operations, and optionally applies dropout regularization.

    Example:
        ```python
        >>> config = CpmAntConfig()
        >>> ffn_block = CpmAntFFNBlock(config)
        >>> hidden_states = mindspore.Tensor(np.random.randn(batch, len_seq, dim_model), dtype=mindspore.float32)
        >>> updated_hidden_states = ffn_block.construct(hidden_states)
        ```
    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes a new instance of the CpmAntFFNBlock class.

        Args:
            self: The instance of the class.
            config (CpmAntConfig):
                The configuration object for the CpmAntFFNBlock. It contains the parameters and settings for the block.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layernorm_before_ffn = CpmAntLayerNorm(config)
        self.ffn = CpmAntFeedForward(config)
        if config.dropout_p:
            self.dropout = nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

    def construct(
        self,
        hidden_states: mindspore.Tensor,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch, len_seq, dim_model)`):
                Hidden states before feed forward layer.
        """
        ln_outputs = self.layernorm_before_ffn(hidden_states)
        outputs = self.ffn(ln_outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs
        return hidden_states


class CpmAntTransformerBlock(nn.Cell):

    """
    This class represents a block of the CpmAntTransformer model, which is a type of transformer used for
    natural language processing tasks. It inherits from the nn.Cell class.

    Attributes:
        self_att (CpmAntSelfAttentionBlock): The self-attention block of the transformer.
        ffn (CpmAntFFNBlock): The feed-forward neural network block of the transformer.

    Methods:
        __init__: Initializes a new instance of the CpmAntTransformerBlock class.
        construct: Constructs the transformer block.

    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes a new instance of the CpmAntTransformerBlock class.

        Args:
            self: The current instance of the class.
            config (CpmAntConfig): The configuration object for the transformer block.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.self_att = CpmAntSelfAttentionBlock(config)
        self.ffn = CpmAntFFNBlock(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        position_bias: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                Input to the layer of shape `(batch, seq_len, dim_model)`
            attention_mask (`mindspore.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_len, seq_len)`
            position_bias (`mindspore.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple[mindspore.Tensor, mindspore.Tensor])`, *optional*):
                Cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        hidden_states = self.self_att(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        hidden_states, attn_weights, current_key_value = hidden_states

        hidden_states = self.ffn(hidden_states)

        return hidden_states, attn_weights, current_key_value


class CpmAntEncoder(nn.Cell):

    """
    The CpmAntEncoder class represents a transformer encoder for the CpmAntConfig model.
    It inherits from nn.Cell and contains methods for initializing the encoder and constructing the encoder layers.

    The __init__ method initializes the CpmAntEncoder with the provided CpmAntConfig,
    setting the number of layers and creating a list of transformer blocks for the encoder.

    The construct method takes input hidden_states, attention_mask, position_bias, and optional parameters
    to perform the encoding process. It iterates through the encoder layers, applying the attention
    mechanism and caching key and value projection states if specified.
    The method returns the final hidden_states, current_key_values, hidden_states of all layers, and attention weights
    of all layers as per the specified optional outputs.

    Args:
        hidden_states (mindspore.Tensor):
            Input to the layer of shape (batch, seq_len, dim_model)
        attention_mask (mindspore.Tensor):
            Avoid invalid areas to participate in the calculation of shape (batch, seq_len, seq_len)
        position_bias (mindspore.Tensor):
            Provides position information to attention mechanism of shape (num_heads, seq_len, seq_len)
        output_attentions (bool, optional):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (bool, optional):
            Whether or not to return the hidden states of all layers.
        past_key_values (Tuple[mindspore.Tensor, mindspore.Tensor], optional):
            Cached past key and value projection states
        use_cache (bool, optional):
            If set to True, past_key_values key value states are returned and can be used to speed up decoding (see past_key_values).

    Returns:
        tuple:
            Tuple of mindspore.Tensor, Tuple of mindspore.Tensor, Optional[Tuple[mindspore.Tensor]],
            Optional[Tuple[mindspore.Tensor]]:

            - hidden_states: Final hidden states of the encoder
            - current_key_values: Current key and value projection states
            - all_hidden_states: Hidden states of all layers (if output_hidden_states is True)
            - all_self_attns: Attention weights of all layers (if output_attentions is True)
    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes a new instance of the CpmAntEncoder class.

        Args:
            self: The instance of the class.
            config (CpmAntConfig):
                The configuration object for the encoder.

                - num_hidden_layers (int): The number of hidden layers.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.layers = nn.CellList([CpmAntTransformerBlock(config) for ith in range(self.num_layers)])

        self.output_layernorm = CpmAntLayerNorm(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        position_bias: mindspore.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                Input to the layer of shape `(batch, seq_len, dim_model)`
            attention_mask (`mindspore.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_len, seq_len)`
            position_bias (`mindspore.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            past_key_values (`Tuple[mindspore.Tensor, mindspore.Tensor])`, *optional*):
                Cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        current_key_values = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_bias,
                output_attentions=output_attentions,
                past_key_values=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
            )
            hidden_states, attn_weights, current_key_value = layer_outputs
            if output_attentions:
                all_self_attns += (attn_weights,)
            if current_key_value is not None:
                current_key_values = current_key_values + (current_key_value,)

        hidden_states = self.output_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, current_key_values, all_hidden_states, all_self_attns


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->CPMAnt
class CpmAntIntermediate(nn.Cell):

    """
    The CpmAntIntermediate class represents an intermediate layer for the CpmAnt model.
    This class inherits from nn.Cell and is used to perform operations on hidden states,
    including dense transformations and activation functions.

    Attributes:
        dense (nn.Dense): A dense layer used for transforming hidden states.
        intermediate_act_fn (function): The activation function applied to the hidden states.

    Methods:
        __init__: Initializes the CpmAntIntermediate instance with the provided configuration.
        construct: Applies dense transformation and activation function to the input hidden states.
    """
    def __init__(self, config):
        """
        Initializes an instance of the CpmAntIntermediate class.

        Args:
            self: An instance of the CpmAntIntermediate class.
            config:
                An object of type 'config' containing the configuration parameters for the model.

                - Type: 'config'
                - Purpose: The configuration parameters for the model.
                - Restrictions: None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Docstring for method 'construct' in class 'CpmAntIntermediate':

        Args:
            self (CpmAntIntermediate): The instance of the class CpmAntIntermediate.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states data to be processed.
                It should be compatible with the operations performed by the method.

        Returns:
            mindspore.Tensor: A tensor representing the processed hidden states data.
                This tensor is the result of applying the dense layer and intermediate activation function.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CpmAntSegmentPositionEmbedding(nn.Cell):

    """
    This class represents a segment position embedding module for the CPM-ANT model.
    It is used to generate embeddings that encode the relative positions of segments in the input tensors.

    The class inherits from the nn.Cell class.

    Attributes:
        num_heads (int): The number of attention heads in the model.
        num_buckets (int): The number of buckets used for segment relative positions.
        max_distance (int): The maximum distance allowed for segment relative positions.
        num_segments (int): The number of segment types in the model.
        relative_attention_bias (mindspore.Parameter): The parameter used to compute the relative attention bias.

    Methods:
        __init__: Initializes the CpmAntSegmentPositionEmbedding instance with the provided configuration.
        construct: Constructs the segment position embeddings based on the input key and query positions and segments.
        _segment_relative_position_bucket: Computes the segment relative position bucket.
        _position_bucket: Computes the position bucket.

    Detailed Description:
        The CpmAntSegmentPositionEmbedding class is used to compute segment position embeddings for the CPM-ANT model.
        These embeddings encode the relative positions between different segments in the input tensors.

        The class takes a configuration object (CpmAntConfig) as input during initialization.
        This configuration object contains various parameters such as the number of attention heads, the number of buckets for
        segment relative positions, the maximum distance allowed for segment relative positions, and the number of segment types in the model.

        The construct method is the main function of this class.
        It takes four input tensors: key_pos, query_pos, key_segment, and query_segment.
        These tensors represent the positions and segments of the key and query elements.
        The method checks the shapes of the input tensors and raises an AssertionError if they are not compatible.
        It then performs various operations to compute the relative position bucket and the  position bucket.
        Finally, it uses the computed embeddings to generate the segment position embeddings.

        The _segment_relative_position_bucket method computes the segment relative position bucket based on the query and key segments.

        The _position_bucket method computes the position bucket based on the relative position, the number of buckets, and the maximum distance.

    Note:
        This class assumes the availability of the following modules: mindspore, math.

    Example:
        ```python
        >>> config = CpmAntConfig()
        >>> segment_embedding = CpmAntSegmentPositionEmbedding(config)
        >>> key_pos = mindspore.Tensor(...)
        >>> query_pos = mindspore.Tensor(...)
        >>> key_segment = mindspore.Tensor(...)
        >>> query_segment = mindspore.Tensor(...)
        >>> embeddings = segment_embedding.construct(key_pos, query_pos, key_segment, query_segment)
        ```
    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes an instance of the CpmAntSegmentPositionEmbedding class.

        Args:
            self: The instance of the class.
            config (CpmAntConfig):
                The configuration object containing the parameters for the segment position embedding.

                - num_heads (int): The number of attention heads.
                - num_buckets (int): The number of buckets for the position bias.
                - max_distance (int): The maximum distance for the position bias.
                - num_segments (int): The number of segment types.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_buckets = config.position_bias_num_buckets
        self.max_distance = config.position_bias_max_distance
        self.num_segments = config.segment_types

        self.relative_attention_bias = Parameter(
            ops.zeros(
                config.segment_types * config.segment_types + config.position_bias_num_buckets,
                config.num_attention_heads,
            )
        )

    def construct(
        self,
        key_pos: mindspore.Tensor,
        query_pos: mindspore.Tensor,
        key_segment: mindspore.Tensor,
        query_segment: mindspore.Tensor,
    ):
        """
        Constructs the segment position embedding for the CpmAntSegmentPositionEmbedding class.

        Args:
            self: An instance of the CpmAntSegmentPositionEmbedding class.
            key_pos (mindspore.Tensor): A tensor representing the positions of the keys. Its shape is (batch, keylen).
            query_pos (mindspore.Tensor): A tensor representing the positions of the queries. Its shape is (batch, querylen).
            key_segment (mindspore.Tensor): A tensor representing the segments of the keys. Its shape is (batch, keylen).
            query_segment (mindspore.Tensor): A tensor representing the segments of the queries. Its shape is (batch, querylen).

        Returns:
            None.

        Raises:
            AssertionError: If key_pos.shape[0] is not equal to query_pos.shape[0].
            AssertionError: If keylen is not equal to key_segment.shape[1] or querylen is not equal to query_segment.shape[1].
            AssertionError: If querylen is not equal to query_segment.shape[1].
        """
        batch = key_pos.shape[0]
        keylen = key_pos.shape[1]
        querylen = query_pos.shape[1]

        if key_pos.shape[0] != query_pos.shape[0]:
            raise AssertionError(
                f"key_pos.shape[0] should be equal to query_pos.shape[0], but got {key_pos.shape[0]} and {query_pos.shape[0]}!"
            )
        if keylen != key_segment.shape[1] or querylen != query_segment.shape[1]:
            raise AssertionError(
                f"keylen should be equal to key_segment.shape[1], but got {keylen} and {key_segment.shape[1]}!"
            )
        if querylen != query_segment.shape[1]:
            raise AssertionError(
                f"querylen should be equal to query_segment.shape[1], but got {querylen} and {query_segment.szie(1)}!"
            )

        key_pos = key_pos.view(batch, -1, keylen)
        query_pos = query_pos.view(batch, querylen, -1)
        key_segment = key_segment.view(batch, -1, keylen)
        query_segment = query_segment.view(batch, querylen, -1)

        relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
        relative_position_bucket = relative_position_bucket + self.num_buckets

        # (batch, len_q, len_k)
        absolute_position_bucket = self._position_bucket(
            ops.arange(keylen, dtype=mindspore.int32)[None, :]
            - ops.arange(querylen, dtype=mindspore.int32)[:, None],
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_position_bucket = ops.where(
            (key_segment == query_segment),
            absolute_position_bucket[None, :, :],
            relative_position_bucket,
        )

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2)
        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        """
        Method to calculate the relative position bucket between a query segment and a key segment.

        Args:
            self (CpmAntSegmentPositionEmbedding): An instance of the CpmAntSegmentPositionEmbedding class.
            query_segment (int): The segment index of the query.
            key_segment (int): The segment index of the key.

        Returns:
            None: This method does not return any value.

        Raises:
            None: This method does not raise any exceptions.
        """
        return query_segment * self.num_segments + key_segment

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        """
        Position bucket calculation.

        Args:
            self (CpmAntSegmentPositionEmbedding): The instance of the CpmAntSegmentPositionEmbedding class.
            relative_position (Tensor): The relative position for which the bucket is calculated.
            num_buckets (int): The total number of buckets to be used for bucketing the relative positions. Default is 32.
            max_distance (int): The maximum distance considered for bucketing. Default is 128.

        Returns:
            Tensor: The calculated relative bucket positions.

        Raises:
            ValueError: If the relative_position tensor is not valid or if any of the input parameters are invalid.
            TypeError: If the input parameters are not of the expected types.
            RuntimeError: If there is a runtime error during the bucket calculation process.
        """
        relative_buckets = 0
        # always bidirectional in CPMAnt
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(mindspore.int32) * num_buckets
        relative_position = ops.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.int32)
        relative_postion_if_large = ops.minimum(
            relative_postion_if_large,
            ops.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += ops.where(is_small, relative_position.to(mindspore.int32), relative_postion_if_large)
        return relative_buckets


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->CPMAnt
class CpmAntOutput(nn.Cell):

    """
    CpmAntOutput represents a custom module for processing hidden states and input tensors in a CpmAnt model.

    This class inherits from nn.Cell and includes methods for initializing the module and constructing the output tensor.

    Attributes:
        dense (nn.Dense): A dense layer for processing hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module for normalizing hidden states.
        dropout (nn.Dropout): A dropout module for applying dropout to hidden states.

    Methods:
        __init__(config): Initializes the CpmAntOutput module with the provided configuration.
        construct(hidden_states, input_tensor): Constructs the output tensor based on the given hidden states and input tensor.

    Example:
        ```python
        >>> config = Config(intermediate_size=256, hidden_size=512, layer_norm_eps=1e-6)
        >>> model = CpmAntOutput(config)
        >>> output = model.construct(hidden_states, input_tensor)
        ```
    """
    def __init__(self, config):
        """
        Initializes a new instance of the CpmAntOutput class.

        Args:
            self: The object itself.
            config: An instance of the configuration class containing the model configuration parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the CpmAntOutput by processing the given hidden states and input tensor.

        Args:
            self (CpmAntOutput): An instance of the CpmAntOutput class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
                Shape: (batch_size, sequence_length, hidden_size)
                The hidden states represent the intermediate outputs of the model.
            input_tensor (mindspore.Tensor): A tensor containing the input values.
                Shape: (batch_size, sequence_length, hidden_size)
                The input tensor is added to the hidden states after passing through the dense, dropout, and LayerNorm layers.

        Returns:
            mindspore.Tensor: A tensor representing the processed hidden states.
                Shape: (batch_size, sequence_length, hidden_size)
                The processed hidden states are obtained by passing the hidden states through the dense, dropout,
                and LayerNorm layers, and then adding the input tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CpmAntPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CpmAntConfig
    base_model_prefix = "cpmant"

    def _init_weights(self, cell):
        """Initialize the weights"""
        std = self.config.init_std
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, CpmAntLayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, CpmAntSegmentPositionEmbedding):
            cell.relative_attention_bias.set_data(initializer(
                Normal(std), cell.relative_attention_bias.shape, cell.relative_attention_bias.dtype))


class CpmAntModel(CpmAntPreTrainedModel):

    """
    CpmAntModel is a class that represents a model for CPM-ANT (Antecedent-Conditioned Prompting) tasks.
    It inherits from CpmAntPreTrainedModel and includes methods for initializing the model, preparing
    attention masks, and constructing the model output based on input tensors.

    Attributes:
        encoder: CpmAntEncoder object for encoding input data
        segment_embedding: nn.Embedding object for segment embeddings
        input_embedding: nn.Embedding object for input embeddings
        position_bias: CpmAntSegmentPositionEmbedding object for position bias calculations
        prompt_length: Length of the prompt in the input data
        vocab_size: Size of the vocabulary in the input data

    Methods:
        __init__: Initializes the model with the given configuration
        get_input_embeddings: Returns the input embeddings
        set_input_embeddings: Sets the input embeddings to the given value
        _prepare_attention_mask: Prepares the attention mask for the input data
        construct: Constructs the model output based on input tensors and optional configurations

    This class provides functionality for processing input data, calculating attention masks,
    and generating model outputs for CPM-ANT tasks.
    """
    def __init__(self, config: CpmAntConfig):
        """
        Initializes a new instance of the CpmAntModel class.

        Args:
            self: The object instance itself.
            config (CpmAntConfig): An instance of CpmAntConfig containing configuration parameters for the model.
                It specifies the configuration settings required for initializing the model.
                This parameter is mandatory and must be an instance of CpmAntConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.encoder = CpmAntEncoder(config)
        self.segment_embedding = nn.Embedding(config.segment_types, config.hidden_size)
        self.input_embedding = nn.Embedding(
            config.vocab_size + config.prompt_types * config.prompt_length, config.hidden_size
        )
        self.position_bias = CpmAntSegmentPositionEmbedding(config)
        self.prompt_length = config.prompt_length
        self.vocab_size = config.vocab_size

        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the CpmAntModel.

        Args:
            self: CpmAntModel - The instance of the CpmAntModel class.

        Returns:
            None:
                This method returns the input embeddings as an instance of the input_embedding attribute
                from the CpmAntModel.

        Raises:
            This method does not raise any exceptions.
        """
        return self.input_embedding

    def set_input_embeddings(self, embeddings, **kwargs):
        """
        Method to set input embeddings for the CpmAntModel.

        Args:
            self (CpmAntModel): The instance of the CpmAntModel class.
            embeddings:
                The input embeddings to be set for the model.

                - Type: Any
                - Purpose: Represents the embeddings to be assigned to the input_embedding attribute of
                the CpmAntModel instance.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        self.input_embedding = embeddings

    def _prepare_attention_mask(self, input_ids, span, context, length):
        """
        Prepare attention mask for the CpmAntModel.

        Args:
            self (CpmAntModel): The instance of the CpmAntModel class.
            input_ids (Tensor): The input tensor containing tokenized input IDs.
            span (Tensor): The tensor containing span information.
            context (Tensor): The tensor containing context information.
            length (Tensor): The tensor containing the length information.

        Returns:
            Tensor: The attention mask tensor prepared for the CpmAntModel.

        Raises:
            ValueError: If the input_ids, span, context, or length tensors are not provided.
            RuntimeError: If there is an issue during the preparation of the attention mask.
        """
        batch = input_ids.shape[0]
        seqlen = input_ids.shape[1]
        directional_mask_2d = ops.arange(seqlen) <= ops.arange(seqlen).view(-1, 1)
        attention_mask = context[:, None, :] | (
            context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
        )
        attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
        # mask for left padding
        mask_1d = (
            mindspore.Tensor(list(range(seqlen - self.prompt_length))[::-1])[None, :].repeat(batch, 1)
            < length[:, None]
        )
        mask_1d = ops.cat((ops.ones(batch, self.prompt_length).bool(), mask_1d), axis=1)
        attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
        return attention_mask

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPast]:
        """
        Constructs the CpmAntModel.

        This method initializes and constructs the CpmAntModel. It takes the following parameters:

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor of shape [batch_size, seq_length]. It represents the input IDs for the model.
                Defaults to None.
            output_attentions (Optional[bool]):
                Whether to output attentions. If set to True, the attentions will be returned. Defaults to None.
            output_hidden_states (Optional[bool]):
                Whether to output hidden states. If set to True, the hidden states will be returned. Defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                The past key values. Defaults to None.
            use_cache (Optional[bool]): Whether to use cache. Defaults to None.
            return_dict (Optional[bool]):
                Whether to return the output as a dictionary.
                If set to True, the output will be returned as a dictionary. Defaults to None.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPast]:
                The output of the model.

                - If return_dict is set to False, a tuple of outputs will be returned, including hidden_states,
                present_key_values, all_hidden_states, and all_attentions.
                - If return_dict is set to True, an instance of BaseModelOutputWithPast will be returned, containing
                the last_hidden_state, past_key_values, hidden_states, and attentions.

        Raises:
            None.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # add prompts ahead
        if input_ids.dtype != mindspore.int32:
            input_ids = input_ids.to(mindspore.int32)
        dtype = input_ids.dtype
        segment = ops.where(input_ids != 0, mindspore.tensor(2), 0).to(dtype=dtype)
        length = (segment != 0).sum(-1).to(dtype=dtype)
        input_ids = ops.cat(
            (
                ops.arange(
                    self.prompt_length * 2 + self.vocab_size,
                    self.prompt_length * 3 + self.vocab_size,
                    dtype=dtype,
                ).tile((input_ids.shape[0], 1)),
                input_ids,
            ),
            axis=1,
        )
        batch, seq_length = input_ids.shape
        segment = ops.cat((ops.zeros(batch, self.prompt_length, dtype=dtype), segment), axis=1)
        context = ops.full((batch, seq_length), 1, dtype=dtype)
        position = ops.arange(seq_length, dtype=dtype).repeat(batch, 1)
        span = ops.full((batch, seq_length), 0, dtype=dtype)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.encoder.num_layers)
            hidden_states = self.input_embedding(input_ids)
            segment_states = self.segment_embedding(segment)
            hidden_states = hidden_states + segment_states
        else:
            past_length = past_key_values[0][0].shape[-2]
            segment_states = self.segment_embedding(segment)
            hidden_states = self.input_embedding(input_ids) + segment_states[:, -1:, :]

        attention_mask = self._prepare_attention_mask(input_ids, span, context, length)
        position_bias = self.position_bias(position, position, segment, segment)

        attention_mask = attention_mask[:, past_length:, :]
        position_bias = position_bias[:, :, past_length:, :]
        hidden_states = hidden_states[:, past_length:, :]

        hidden_states, present_key_values, all_hidden_states, all_attentions = self.encoder(
            hidden_states,
            attention_mask,
            position_bias,
            output_attentions,
            output_hidden_states,
            past_key_values,
            use_cache,
        )

        if past_length == 0:
            hidden_states = hidden_states[:, self.prompt_length :, :]
            # drop the prompt
            if all_attentions is not None:
                new_attentions = ()
                for attention in all_attentions:
                    new_attentions += (attention[:, :, self.prompt_length :, self.prompt_length :],)
                all_attentions = new_attentions
            if all_hidden_states is not None:
                new_hidden_states = ()
                for hidden_state in all_hidden_states:
                    new_hidden_states += (hidden_state[:, self.prompt_length :, :],)
                all_hidden_states = new_hidden_states

        if not return_dict:
            return tuple(
                v for v in [hidden_states, present_key_values, all_hidden_states, all_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CpmAntForCausalLM(CpmAntPreTrainedModel):

    """
    CpmAntForCausalLM is a class representing a Causal Language Model based on the CPMAnt model for text generation tasks.
    This class extends the functionality of CpmAntPreTrainedModel and provides methods for model initialization,
    text generation, and handling embeddings.

    The CpmAntForCausalLM class includes methods for model initialization, generating text based on input sequences,
    accessing and setting input and output embeddings,
    preparing inputs for text generation, and reordering cache for beam search decoding.

    Example:
        Text Generation with CpmAntForCausalLM:
        ```python
        >>> from transformers import CPMAntTokenizer, CpmAntForCausalLM
        ...
        >>> texts = "Today is a beautiful day, "
        >>> model = CpmAntForCausalLM.from_pretrained("openbmb/cpm-ant-10b")
        >>> tokenizer = CPMAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
        >>> input_ids = tokenizer(texts, return_tensors="pt")
        >>> outputs = model.generate(**input_ids)
        >>> output_texts = tokenizer.batch_decode(outputs)
        >>> print(output_texts)
        ['Today is a beautiful day, the sun is shining, and the birds are singing.']
        ```

    Methods:
        __init__: Initializes the CpmAntForCausalLM model with the provided configuration.
        construct: Constructs the model for text generation based on the input arguments and returns output in the specified format.
        get_input_embeddings: Retrieves the input embeddings of the model.
        set_input_embeddings: Sets new input embeddings for the model.
        get_output_embeddings: Retrieves the output embeddings of the model.
        set_output_embeddings: Sets new output embeddings for the model.
        prepare_inputs_for_generation: Prepares inputs for text generation based on the provided input_ids and keyword arguments.
        _reorder_cache: Reorders the cache for beam search decoding.

    Args:
        input_ids (mindspore.Tensor): Indices of input sequence tokens in the vocabulary.
        past_key_values (List[Tuple[mindspore.Tensor, mindspore.Tensor]]): Pre-computed hidden states for sequential decoding.
        use_cache (bool): Flag to determine if cache should be used for decoding.
        output_attentions (bool): Flag to include attention tensors in the output.
        output_hidden_states (bool): Flag to include hidden states of all layers in the output.
        labels (mindspore.Tensor): Labels for computing the masked language modeling loss.
        return_dict (bool): Flag to determine the format of the output.
        attention_mask (mindspore.Tensor): Dummy parameter for text-generation pipeline.

    Returns:
        Union[Tuple, CausalLMOutputWithPast]: Tuple or CausalLMOutputWithPast object containing model outputs and past key values.

    Raises:
        NotImplementedError: If a method is not implemented in the subclass.

    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: CpmAntConfig):
        """
        Initializes an instance of the CpmAntForCausalLM class.

        Args:
            self: The instance of the class.
            config (CpmAntConfig): The configuration object for the CpmAnt model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.cpmant = CpmAntModel(config)

        # lm_head.weight is tied to cpmant.input_embedding.weight
        self.lm_head = nn.Dense(
            config.hidden_size, config.vocab_size + config.prompt_types * config.prompt_length, has_bias=False
        )
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[Tuple[mindspore.Tensor, mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
        attention_mask: Optional[mindspore.Tensor] = None,  # dummy parameter for text-generation pipeline
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, seq_len)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                CPMAnt will process attention mask automatically, this parameter is a dummy parameter for
                text-generation pipeline.

        Example:
            Text Generation with CpmAntForCausalLM.
            ```python
            >>> from transformers import CPMAntTokenizer, CpmAntForCausalLM
            ...
            >>> texts = "今天天气不错，"
            >>> model = CpmAntForCausalLM.from_pretrained("openbmb/cpm-ant-10b")
            >>> tokenizer = CPMAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
            >>> input_ids = tokenizer(texts, return_tensors="pt")
            >>> outputs = model.generate(**input_ids)
            >>> output_texts = tokenizer.batch_decode(outputs)
            >>> print(output_texts)
            ['今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的']
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.cpmant(
            input_ids, output_attentions, output_hidden_states, past_key_values, use_cache, return_dict
        )
        hidden_states = model_output.last_hidden_state if return_dict else model_output[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

        if not return_dict:
            output = (logits,) + model_output[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_output.past_key_values,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings used by the CpmAntForCausalLM model.

        Args:
            self (CpmAntForCausalLM): The instance of the CpmAntForCausalLM class.
                This parameter is required to access the input embeddings specific to this instance.

        Returns:
            None: This method returns the input embeddings associated with the CpmAntForCausalLM model.
                The input embeddings are used for processing input data within the model.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self.cpmant.input_embedding

    def set_input_embeddings(self, embeddings):
        """
        Set the input embeddings for the CpmAntForCausalLM model.

        Args:
            self (CpmAntForCausalLM): The instance of the CpmAntForCausalLM class.
            embeddings: The input embeddings to be set for the model.
                This parameter should be a valid embeddings object that can be assigned to the input_embedding attribute of the CpmAntForCausalLM instance.

        Returns:
            None.

        Raises:
            None.
        """
        self.cpmant.input_embedding = embeddings

    def get_output_embeddings(self):
        """
        Retrieves the output embeddings of the language model head.

        Args:
            self: An instance of the CpmAntForCausalLM class.

        Returns:
            lm_head: The method returns the output embeddings of the language model head.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the CpmAntForCausalLM model.

        Args:
            self (CpmAntForCausalLM): The instance of the CpmAntForCausalLM class.
            new_embeddings (torch.nn.Module): The new embeddings to be set as the output embeddings of the model.

        Returns:
            None

        Raises:
            None

        This method sets the output embeddings of the CpmAntForCausalLM model to the provided new embeddings.
        The new embeddings should be an instance of torch.nn.Module.

        Example:
            ```python
            >>> model = CpmAntForCausalLM()
            >>> new_embeddings = nn.Embedding(1000, 768)
            >>> model.set_output_embeddings(new_embeddings)
            ```
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for generation.

        This method takes in two parameters: self and input_ids.
        It modifies the input_ids and returns a dictionary containing the modified input_ids, use_cache, and past_key_values.

        Args:
            self: The instance of the CpmAntForCausalLM class.
            input_ids (tensor): The input tensor containing the tokenized input sequence.

        Returns:
            dict:
                A dictionary with the following keys:

                - input_ids (tensor): The modified input tensor.
                - use_cache (bool): The value of the use_cache parameter from kwargs.
                - past_key_values (tensor or None): The value of the past_key_values parameter from kwargs,
                or None if not provided.

        Raises:
            None.

        Note:
            - The input_ids parameter is cast to int.
            - If the 'attention_mask' key is present in kwargs, its value is replaced with a zero tensor of shape (1, 1).
        """
        input_ids = input_ids.int()
        # save the memory usage of dummy attention mask
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = ops.zeros(1, 1)

        return {
            "input_ids": input_ids,
            "use_cache": kwargs["use_cache"],
            "past_key_values": kwargs.get("past_key_values", None),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache for the specified beam index.
        
        Args:
            self (CpmAntForCausalLM): An instance of the CpmAntForCausalLM class.
            past_key_values (list): A list of past key values. Each element in the list represents a key-value layer 
                and is a list containing two elements: the key and the value. If a key-value layer
                is None, it will be preserved as None.
            beam_idx (int): The index of the beam for which the cache needs to be reordered.
        
        Returns:
            list: The reordered cache represented as a list of past key values. Each element in the list is a key-value 
                layer, and each key-value layer is a list containing two elements: the key and the value.
        
        Raises:
            None
        
        """
        past_key_values = [list(each) if each is not None else each for each in past_key_values]
        for key_value_layer in past_key_values:
            key_value_layer[0] = key_value_layer[0][beam_idx]
            key_value_layer[1] = key_value_layer[1][beam_idx]
        return past_key_values

__all__ = [
    "CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "CpmAntForCausalLM",
    "CpmAntModel",
    "CpmAntPreTrainedModel",
]
