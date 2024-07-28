# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
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
"""PyTorch MPT model."""
# pylint: disable=W0235
# pylint: disable=E1123

import math
from typing import Optional, Tuple, Union
import numpy as np

import mindspore
from mindnlp.core import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindnlp.utils import logging
from ...modeling_utils import PreTrainedModel
from .configuration_mpt import MptConfig, MPT_PRETRAINED_MODEL_ARCHIVE_LIST
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mosaicml/mpt-7b"
_CONFIG_FOR_DOC = "MptConfig"


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8):
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    alibi = ops.arange(1 - sequence_length, 1, dtype=mindspore.int32).view(1, 1, 1, sequence_length)
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = ops.arange(1, num_heads_power_of_2 + 1, dtype=mindspore.int64).float()
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / ops.pow(2, base)
    slopes = slopes.view(1, num_heads_power_of_2, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = ops.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], axis=1)[:, :num_heads, ...]

    alibi = alibi * slopes
    return alibi.squeeze(0)


class MptAttention(nn.Module):
    """Multi-head self attention.
    Using torch or triton attention implemetation enables user to also use additive bias.
    """
    def __init__(self, config: MptConfig):
        """
        Initializes an instance of the MptAttention class.
        
        Args:
            self: The object instance.
            config (MptConfig):
                An object of type MptConfig that contains the configuration parameters for the MptAttention.
            
        Returns:
            None
            
        Raises:
            None
        
        Description:
            This method is used to initialize the MptAttention class. It sets various attributes of the class instance
            based on the provided configuration.

            The 'self' parameter refers to the object instance itself. It is automatically passed when calling the method.

            The 'config' parameter is an instance of the MptConfig class which contains the following attributes:

            - hidden_size (int): The size of the hidden state.
            - n_heads (int): The number of attention heads.
            - max_seq_length (int): The maximum sequence length.
            - attn_config (AttnConfig): An object of type AttnConfig that contains attention configuration parameters.

                - softmax_scale (float): The scale factor for the softmax function.
                If not provided, it is set to 1 / sqrt(hidden_size / n_heads).
                - attn_pdrop (float): The dropout probability for attention weights.

            The method performs the following steps:

            1. Calls the __init__ method of the parent class using the 'super()' function.
            2. Sets the 'hidden_size', 'n_heads', 'max_seq_length', and 'head_dim' attributes based on the values
            from the 'config' parameter.
            3. Sets the 'softmax_scale' attribute based on the 'softmax_scale' attribute of the 'attn_config' attribute
            from the 'config' parameter. If not provided, it is calculated as 1 / sqrt(hidden_size / n_heads).
            4. Sets the 'attn_dropout_p' attribute based on the 'attn_pdrop' attribute of the 'attn_config' attribute
            from the 'config' parameter.
            5. Initializes the 'Wqkv' attribute as a dense layer with input size 'hidden_size' and output size
            '3 * hidden_size', with no bias.
            6. Initializes the 'out_proj' attribute as a dense layer with input size 'hidden_size' and output size
            'hidden_size', with no bias.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_seq_length = config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attn_dropout_p = config.attn_config.attn_pdrop
        self.Wqkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        position_bias: mindspore.Tensor,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
    ):
        """
        Constructs the attention mechanism.

        Args:
            self (MptAttention): The instance of the MptAttention class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, seq_length, hidden_size)
                representing the hidden states.
            position_bias (mindspore.Tensor): The input tensor of shape (batch_size, seq_length, seq_length)
                representing the position bias.
            past_key_value (Optional[Tuple[mindspore.Tensor]]): The optional tuple of past key-value states.
                Default is None.
            attention_mask (Optional[mindspore.Tensor]): The optional input tensor of shape (batch_size, seq_length)
                representing the attention mask. Default is None.

        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor, Tuple[mindspore.Tensor]]: A tuple containing the attention
                output tensor of shape (batch_size, seq_length, hidden_size), the attention weights tensor of
                shape (batch_size, num_heads, seq_length, seq_length), and the updated past key-value states tuple.

        Raises:
            ValueError: If the position_bias tensor does not have 3 dimensions.
        """
        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = mixed_qkv.chunk(3, axis=2)
        query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).swapaxes(1, 2)

        if past_key_value is not None:
            if len(past_key_value) != 0:
                key_states = ops.cat([past_key_value[0], key_states], axis=2)
                value_states = ops.cat([past_key_value[1], value_states], axis=2)
            past_key_value = (key_states, value_states)
        else:
            past_key_value = (key_states, value_states)

        attention_scores = ops.matmul(query_states, key_states.swapaxes(-1, -2)) * self.softmax_scale

        query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

        if position_bias is not None:
            if len(position_bias.shape) != 3:
                raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
            key_length = key_states.shape[-2]

            position_bias_query_index = max(0, position_bias.shape[1] - query_length)
            position_bias_key_index = max(0, position_bias.shape[2] - key_length)

            position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

            attention_scores = attention_scores + position_bias

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, finfo(query_states.dtype, 'min'))

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = ops.softmax(attention_scores.float(), axis=-1).to(dtype=value_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

        context_states = ops.matmul(attn_weights, value_states)
        context_states = context_states.permute(0, 2, 1, 3).view(batch_size, seq_length, -1)
        attn_output = self.out_proj(context_states)

        return attn_output, attn_weights, past_key_value


class MptMLP(nn.Module):

    """
    Class representing a Multi-Layer Perceptron (MLP) for Mpt models.

    This class defines the architecture of a Multi-Layer Perceptron for Mpt models.
    It consists of an up projection layer, activation function (GELU), down projection layer, hidden dropout layer,
    and a forward method to process hidden states and residuals.

    Inherits from nn.Module.
    """
    def __init__(self, config: MptConfig):
        """
        Initializes an instance of the MptMLP class.

        Args:
            self: The instance of the class.
            config (MptConfig): An object of type MptConfig containing configuration parameters.
                The config parameter is used to specify the hidden size for the MLP model.
                It is expected to have the following attributes:

                - hidden_size: An integer specifying the size of the hidden layer.
                - attn_config: An object containing attention configuration parameters.

                    - attn_pdrop: A float specifying the dropout probability for attention.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        hidden_size = config.hidden_size

        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.act = nn.GELU(approximate=False)
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)
        self.hidden_dropout = config.attn_config.attn_pdrop

    def forward(self, hidden_states: mindspore.Tensor, residual: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a multi-layer perception (MLP) module.

        Args:
            self (MptMLP): An instance of the MptMLP class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape (batch_size, hidden_size).
            residual (mindspore.Tensor): The residual tensor of shape (batch_size, hidden_size).

        Returns:
            mindspore.Tensor: The output tensor of shape (batch_size, hidden_size).

        Raises:
            TypeError: If the input hidden_states or residual is not a mindspore.Tensor.
            ValueError: If the shapes of hidden_states and residual do not match.
        """
        hidden_states = self.act(self.up_proj(hidden_states))

        intermediate_output = self.down_proj(hidden_states)

        output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
        output = output + residual

        return output


class MptBlock(nn.Module):

    """
    MptBlock represents a block within a Multi-Head Transformer model. This block consists of layers for
    self-attention and feed-forward networks. Inherits from nn.Module.

    Attributes:
        config: MptConfig object containing configuration parameters for the block.

    Methods:
        __init__: Initializes the MptBlock with the provided configuration.
        forward: Constructs the block by applying self-attention and feed-forward operations on the input hidden states.

    Example:
        ```python
        >>> config = MptConfig(...)
        >>> block = MptBlock(config)
        >>> outputs = block.forward(hidden_states, position_bias, attention_mask)
        ```
    """
    def __init__(self, config: MptConfig):
        """
        Initializes an instance of the MptBlock class.

        Args:
            self: The object itself.
            config (MptConfig): An instance of the MptConfig class representing the configuration settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        hidden_size = config.hidden_size

        self.norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon, elementwise_affine=False)
        # backward compatibility with weights on the Hub
        self.norm_1.bias = None

        self.num_heads = config.n_heads
        self.attn = MptAttention(config)

        self.norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon, elementwise_affine=False)
        # backward compatibility with weights on the Hub
        self.norm_2.bias = None

        self.ffn = MptMLP(config)

        self.dropout_rate = config.attn_config.attn_pdrop
        self.resid_attn_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        position_bias: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        This method forwards a multi-head self-attention block within the MptBlock class.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            position_bias (mindspore.Tensor): The tensor containing positional bias information.
            attention_mask (mindspore.Tensor): The tensor used for masking the attention scores.
            layer_past (Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]):
                A tuple containing the past key and value tensors, default is None.
            use_cache (bool): A boolean flag indicating whether to use caching, default is False.
            output_attentions (bool): A boolean flag indicating whether to output attention weights, default is False.

        Returns:
            tuple:
                A tuple containing the output tensor of the multi-head self-attention block.

                - If 'use_cache' is True, the tuple also includes the past key and value tensors.
                - If 'output_attentions' is True, the tuple additionally includes the attention weights tensor.

        Raises:
            None
        """
        # hidden_states: [batch_size, seq_length, hidden_size]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.norm_1(hidden_states)

        residual = hidden_states

        # Self attention.
        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs) + residual

        layernorm_output = self.norm_2(hidden_states)

        # Get residual
        residual = hidden_states

        # MLP.
        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        if use_cache:
            outputs += (past_key_value,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # hidden_states, present, attentions


class MptPreTrainedModel(PreTrainedModel):

    """
    MptPreTrainedModel is a Python class representing a pre-trained model for Mpt (MindSpore Transformer) models.
    It provides methods for initializing weights of different types of neural network cells and converting the cache
    to the format expected by Mpt.

    The class includes an __init__ method for initializing the model, an _init_weights method for weight initialization
    of different cell types, and a static method _convert_to_mpt_cache for converting the cache format to be compatible
    with Mpt.

    The _init_weights method initializes weights based on the type of neural network cell, such as nn.Linear,
    nn.Embedding, and nn.LayerNorm.
    The method sets the weights and biases of the cells according to specific initializations.

    The _convert_to_mpt_cache static method takes a past_key_value tuple and converts it to the format expected by Mpt,
    reshaping the tensors to match the batch size, number of heads, head dimension, and sequence length.

    Note:
        This class inherits from PreTrainedModel.
    """
    config_class = MptConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MptBlock"]
    _keys_to_ignore_on_load_missing = [r"lm_head.*."]

    def __init__(self, *inputs, **kwargs):
        """
        Initializes a new instance of the MptPreTrainedModel class.

        Args:
            self:
                The object itself.

                - Type: MptPreTrainedModel
                - Purpose: Represents the current instance of the MptPreTrainedModel class.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(mindspore.Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    @staticmethod
    def _convert_to_mpt_cache(
        past_key_value: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]],
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Converts the cache to the format expected by Mpt, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


class MptModel(MptPreTrainedModel):

    """
    This class represents a modified pre-trained transformer model (MptModel) for natural language processing tasks.
    It inherits from MptPreTrainedModel and includes methods for initializing the model, handling input embeddings,
    forwarding the model with various optional parameters, and building the multi-head positional tensor alibi.
    The model consists of multiple MptBlocks organized in a sequence. It provides functionality for processing input
    data, managing past key values, applying attention masks, and computing hidden states. Additionally, the model supports
    gradient checkpointing for efficient training. The MptModel class encapsulates the core functionality required for
    performing transformer-based operations on text data.
    """
    def __init__(self, config: MptConfig):
        """
        Initializes an instance of MptModel.

        Args:
            self: The object instance itself.
            config (MptConfig):
                An instance of MptConfig containing the configuration parameters for the model.

                - Type: MptConfig
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: Must be an instance of MptConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads

        # Embedding + LN Embedding
        self.wte = nn.Embedding(config.vocab_size, self.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([MptBlock(config) for _ in range(config.n_layers)])

        # Final Layer Norm
        self.norm_f = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon, elementwise_affine=False)
        # backward compatibility with weights on the Hub
        self.norm_f.bias = None

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method is part of the MptModel class and is used to retrieve the input embeddings.

        Args:
            self: An instance of the MptModel class.

        Returns:
            None: This method returns None, as it simply retrieves the input embeddings without any additional processing.

        Raises:
            None.
        """
        return self.wte

    def build_mpt_alibi_tensor(self, num_heads, sequence_length, alibi_bias_max=8):
        """
        This method builds a multi-head attention (MPT) alibi tensor.

        Args:
            self (MptModel): The instance of the MptModel class.
            num_heads (int): The number of attention heads to be used in the tensor.
            sequence_length (int): The length of the input sequence.
            alibi_bias_max (int, optional): The maximum value for the alibi bias. Defaults to 8.

        Returns:
            None.

        Raises:
            ValueError: If num_heads or sequence_length is not a positive integer.
            TypeError: If num_heads, sequence_length, or alibi_bias_max is not of type int.
        """
        return build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max)

    def set_input_embeddings(self, new_embeddings: mindspore.Tensor):
        """
        Sets the input embeddings for the MptModel.

        Args:
            self (MptModel): The instance of the MptModel class.
            new_embeddings (mindspore.Tensor): The new embeddings to be set as input.

        Returns:
            None.

        Raises:
            None.
        """
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the MptModel.

        Args:
            self: The MptModel instance.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, seq_length) containing the input IDs.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]]):
                The past key values tensor of shape (batch_size, past_seq_length, hidden_size) containing
                the past key values.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape (batch_size, seq_length)
                containing the attention mask values.
            inputs_embeds (Optional[mindspore.Tensor]): The inputs embeddings tensor of shape
                (batch_size, seq_length, hidden_size) containing the input embeddings.
            use_cache (Optional[bool]): Flag indicating whether to use cache. Default is None.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Flag indicating whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
                A tuple containing the hidden states, presents, all hidden states, and all self attentions.
                The hidden states tensor has shape (batch_size, seq_length, hidden_size). The presents tensor has shape
                (batch_size, past_seq_length + seq_length, hidden_size). The all hidden states tensor is a tuple of
                hidden states tensors at each layer. The all self attentions tensor is a tuple of self attention tensors
                at each layer.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified simultaneously.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            Warning: If use_cache is set to True and gradient checkpointing is enabled, as they are incompatible.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = ops.ones((batch_size, seq_length_with_past))
        else:
            pass

        alibi = self.build_mpt_alibi_tensor(self.num_heads, self.config.max_seq_len)

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        causal_mask = causal_mask.bool()

        for block, layer_past in zip(self.blocks, past_key_values):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_bias=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MptForCausalLM(MptPreTrainedModel):

    """
    This class represents a model for Causal Language Modeling based on the Mpt architecture.
    It provides methods for preparing inputs, generating outputs, and reordering cache for beam search.

    The class includes methods such as initializing the model, getting and setting output embeddings, preparing inputs
    for generation, forwarding the model, and reordering cache for beam search.

    The 'MptForCausalLM' class inherits from 'MptPreTrainedModel' and utilizes a transformer model along with
    specific configurations for language modeling tasks.

    Key methods:

    - __init__: Initialize the model with the provided configuration.
    - get_output_embeddings: Get the output embeddings of the model.
    - set_output_embeddings: Set new output embeddings for the model.
    - prepare_inputs_for_generation: Prepare inputs for text generation.
    - forward: Construct the model for language modeling.
    - _reorder_cache: Reorder the cache for beam search operations.

    The 'MptForCausalLM' class is designed to facilitate language modeling tasks with a focus on generating coherent
    text sequences in a causal manner.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MptConfig):
        """
        Initializes an instance of the MptForCausalLM class.

        Args:
            self: The instance of the MptForCausalLM class.
            config (MptConfig): An instance of the MptConfig class containing configuration parameters.
                This parameter is required for initializing various components within the MptForCausalLM instance.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MptConfig.
            ValueError: If any required configuration parameter is missing or invalid.
        """
        super().__init__(config)
        self.transformer = MptModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the MptForCausalLM model.

        Args:
            self: An instance of the MptForCausalLM class.

        Returns:
            embeddings: The output embeddings which are stored in the `lm_head` attribute of the MptForCausalLM instance.

        Raises:
            None.

        This method retrieves and returns the output embeddings of the MptForCausalLM model.
        The output embeddings are stored in the `lm_head` attribute of the instance.
        The `lm_head` attribute represents the final layer of the language model, responsible for generating the
        output predictions.

        Note that the output embeddings are specific to the MptForCausalLM model and are derived from the language
        model's internal representation. The embeddings capture the semantic meaning and contextual information of the
        input text, enabling downstream tasks such as text generation, completion, or classification.

        Example:
            ```python
            >>> model = MptForCausalLM()
            >>> output_embeddings = model.get_output_embeddings()
            ```
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: mindspore.Tensor):
        """
        Method to set new output embeddings for the language model.

        Args:
            self (MptForCausalLM): The instance of the MptForCausalLM class.
            new_embeddings (mindspore.Tensor): The new embeddings to be set as the output embeddings for the language model.
                It should be a Tensor object containing the embeddings to be used for the output layer.

        Returns:
            None:
                This method updates the 'lm_head' attribute of the MptForCausalLM instance with the new embeddings
                provided.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: mindspore.Tensor,
        past_key_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """
        Prepares inputs for generation in the MptForCausalLM class.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor): The input tensor containing the tokenized input sequence.
            past_key_values (Optional[mindspore.Tensor]): The tensor containing the past key values for
                auto-regressive generation. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The tensor specifying which tokens should be attended to.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The tensor containing the embeddings of the input sequence.
                Default is None.
            use_cache (Optional[bool]): Specifies whether to use cache for faster generation. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared model inputs, including 'input_ids', 'inputs_embeds',
                'past_key_values', 'use_cache', and 'attention_mask'.

        Raises:
            None.
        """
        # only last tokens for input_ids if past is not None
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,  # NITS should it be layer_past?
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...], beam_idx: mindspore.Tensor
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        # Get a copy of `beam_idx` on all the devices where we need those indices.
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, beam_idx),
                layer_past[1].index_select(0, beam_idx),
            )
            for layer_past in past
        )

        return reordered_past


class MptForSequenceClassification(MptPreTrainedModel):

    """
    This class represents a sequence classification model based on the MptPreTrainedModel architecture.

    The MptForSequenceClassification class is a subclass of MptPreTrainedModel and is designed for sequence 
    classification tasks. It includes methods for initializing the model, forwarding the model, and
    generating sequence classification outputs.

    Attributes:
        num_labels (int): The number of labels for the sequence classification task.
        transformer (MptModel): The transformer model used for sequence encoding.
        score (nn.Linear): The linear layer for generating logits from the hidden states.

    Methods:
        __init__: Initializes the MptForSequenceClassification instance with a configuration object.
        forward: Constructs the sequence classification model and returns the classification outputs.

    The MptForSequenceClassification class inherits from the MptPreTrainedModel class and extends its functionality 
    specifically for sequence classification tasks. It utilizes a transformer model for encoding the input sequences 
    and a linear layer for generating logits from the hidden states. The class provides a method for forwarding the 
    model and returning the classification outputs.

    Note:
        This class assumes that the input sequences are tokenized and encoded as input_ids, and the labels are 
        provided for computing the sequence classification/regression loss. The number of labels should be in the range 
        of [0, config.num_labels - 1]. If config.num_labels == 1, a regression loss is computed using Mean-Square loss. 
        If config.num_labels > 1, a classification loss is computed using Cross-Entropy.
    """
    def __init__(self, config: MptConfig):
        """
        Initializes an instance of MptForSequenceClassification.

        Args:
            self: The instance of the class.
            config (MptConfig): An instance of MptConfig containing configuration parameters.
                It specifies the configuration settings for the model.
                Must be of type MptConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = MptModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
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

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
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
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
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
                elif self.num_labels > 1 and labels.dtype in (mindspore.long64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(pooled_logits, labels)
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


class MptForTokenClassification(MptPreTrainedModel):

    """
    MptForTokenClassification represents a model for token classification tasks, inheriting from MptPreTrainedModel.
    It includes methods for initializing the model and forwarding the forward pass for token classification.

    The __init__ method initializes the model parameters and components such as the transformer and classifier layers,
    with optional dropout specified in the config.

    The forward method processes the input data through the transformer, applies dropout, computes logits using the
    classifier, calculates loss if labels are provided, and returns the output in the specified format.

    Parameters:
        config (MptConfig): Configuration object containing model settings.
        input_ids (Optional[mindspore.Tensor]): Input token IDs.
        past_key_values (Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]]):
            Past key values for attention mechanisms.
        attention_mask (Optional[mindspore.Tensor]): Mask for attention scores.
        inputs_embeds (Optional[mindspore.Tensor]): Embedded input tokens.
        labels (Optional[mindspore.Tensor]): Target labels for classification/regression.
        use_cache (Optional[bool]): Flag for using cache in the transformer.
        output_attentions (Optional[bool]): Flag for outputting attentions.
        output_hidden_states (Optional[bool]): Flag for outputting hidden states.
        return_dict (Optional[bool]): Flag for returning output as a dictionary.

    Returns:
        Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
            Tuple containing the loss and output based on return format.

    Note:
        - Labels should be indices in [0, ..., config.num_labels - 1].
        - For single label regression, config.num_labels should be 1.
        - For multiple labels classification, config.num_labels > 1.
        - Cross-entropy loss is computed for classification tasks.

    For detailed information on methods and attributes, please refer to the method implementations in the class.
    """
    def __init__(self, config: MptConfig):
        """
        Initializes an instance of the MptForTokenClassification class.

        Args:
            self: The instance of the MptForTokenClassification class.
            config (MptConfig):
                An object of type MptConfig containing configuration parameters for the model.

                - num_labels (int): The number of labels for token classification.
                - classifier_dropout (float, optional): The dropout probability for the classifier layer.
                - hidden_dropout (float, optional): The dropout probability for hidden layers.

        Returns:
            None:
                This method initializes the MptForTokenClassification instance with the provided configuration.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = MptModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            batch_size, seq_length = labels.shape
            loss = F.cross_entropy(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class MptForQuestionAnswering(MptPreTrainedModel):

    """
    MptForQuestionAnswering is a class representing a model for question answering. It inherits from MptPreTrainedModel
    and provides methods for forwarding a question answering model.

    The class includes an initializer method that takes a 'config' parameter and initializes the transformer and
    qa_outputs attributes. It also provides a 'forward' method for forwarding the question answering model,
    which takes input_ids, attention_mask, inputs_embeds, start_positions, end_positions, output_attentions,
    output_hidden_states, and return_dict as optional parameters and returns a QuestionAnsweringModelOutput.

    The 'forward' method computes the start and end positions for the labelled span, computes the token classification
    loss, and returns the total loss along with start_logits, end_logits, hidden_states, and attentions if return_dict
    is False. If return_dict is True, it returns a QuestionAnsweringModelOutput containing the loss, start_logits,
    end_logits, hidden_states, and attentions.
    
    """
    def __init__(self, config):
        """
        Initializes an instance of MptForQuestionAnswering class.
        
        Args:
            self (MptForQuestionAnswering): The instance itself.
            config: A dictionary containing configuration parameters for the model.
        
        Returns:
            None.
        
        Raises:
            None
        """
        super().__init__(config)
        self.transformer = MptModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
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

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
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

            start_loss =F.cross_entropy(start_logits,start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions,ignore_index=ignored_index)
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
    "MPT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MptForCausalLM",
    "MptModel",
    "MptPreTrainedModel",
    "MptForSequenceClassification",
    "MptForTokenClassification",
    "MptForQuestionAnswering",
]
