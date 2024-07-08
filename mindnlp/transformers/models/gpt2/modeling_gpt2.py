# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""MindSpore GPT-2 model."""

import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import mindspore
from mindspore import ops, nn
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import (
    ModelOutput,
    logging,
)
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...ms_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from .configuration_gpt2 import GPT2Config


logger = logging.get_logger(__name__)


GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://hf-mirror.com/models?filter=gpt2
]


class GPT2Attention(nn.Cell):

    """
    The `GPT2Attention` class represents the attention mechanism used in the GPT-2 model.
    It is a subclass of the `nn.Cell` class.
    
    Summary:
        This class implements the attention mechanism in GPT-2, which is used for self-attention within the model
        or cross-attention between the model and an encoder.
    
    Attributes:
        `config`: The configuration object containing various hyperparameters for the attention mechanism.
        `is_cross_attention`: A boolean flag indicating whether the attention is for cross-attention or self-attention.
        `layer_idx`: An optional integer representing the index of the layer.
        `bias`: A tensor representing the bias used in attention calculations.
        `masked_bias`: A tensor representing the bias used in attention calculations for masking.
        `embed_dim`: An integer representing the embedding dimension of the attention mechanism.
        `num_heads`: An integer representing the number of attention heads.
        `head_dim`: An integer representing the dimension of each attention head.
        `split_size`: An integer representing the size of split tensors.
        `scale_attn_weights`: A boolean flag indicating whether to scale the attention weights.
        `scale_attn_by_inverse_layer_idx`: A boolean flag indicating whether to scale the attention weights
            by the inverse of the layer index.
        `reorder_and_upcast_attn`: A boolean flag indicating whether to reorder and upcast the attention weights.
        `c_attn`: The convolutional layer for attention calculations.
        `q_attn`: The convolutional layer for calculating queries (only used for cross-attention).
        `c_proj`: The convolutional layer for projecting the attention output.
        `attn_dropout`: The dropout layer applied to the attention weights.
        `resid_dropout`: The dropout layer applied to the attention output.
        `pruned_heads`: A set containing the indices of pruned attention heads.

    Methods:
        `prune_heads`: Prunes the specified attention heads.
        `_attn`: Performs attention calculations for self-attention.
        `_upcast_and_reordered_attn`: Performs attention calculations for cross-attention.
        `_split_heads`: Splits the `hidden_size` dimension into `attn_head_size` and `num_heads`.
        `_merge_heads`: Merges the `attn_head_size` and `num_heads` dimensions into `hidden_size`.
        `construct`: Constructs the attention mechanism.

    Please note that this class does not include method signatures or any other code.
    The provided information is a summary of the class and its attributes and methods.
    """
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        """
        Initializes an instance of the GPT2Attention class.

        Args:
            self: The object itself.
            config (object): An object containing the configuration parameters.
            is_cross_attention (bool, optional): Indicates whether the attention is cross-attention or not.
                Defaults to False.
            layer_idx (int, optional): The index of the layer. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.

        """
        super().__init__()

        max_positions = config.max_position_embeddings
        self.bias = ops.tril(ops.ones((max_positions, max_positions), dtype=mindspore.bool_)).view(
                1, 1, max_positions, max_positions
            )
        self.masked_bias = mindspore.Tensor(-1e4)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method is part of the GPT2Attention class and is named prune_heads.

        Args:
            self: GPT2Attention object. Represents an instance of the GPT2Attention class.

            heads: List of integers. The list of head indices to be pruned from the attention mechanism.
                It identifies the specific heads to be pruned from the attention mechanism.

        Returns:
            None: This method does not return any value explicitly.
                It modifies the internal state of the GPT2Attention object.

        Raises:
            None: However, depending on the implementation of the helper functions find_pruneable_heads_and_indices,
                ops.cat, and prune_conv1d_layer, potential exceptions related to these functions may be raised during
                the execution of prune_heads method.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = ops.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, axis=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, axis=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Performs attention computation for the GPT2 model.

        Args:
            self (GPT2Attention): The GPT2Attention instance to which this method belongs.
            query (Tensor): The query tensor for attention computation.
            key (Tensor): The key tensor for attention computation.
            value (Tensor): The value tensor for attention computation.
            attention_mask (Tensor, optional): An optional tensor for masking the attention weights.
            head_mask (Tensor, optional): An optional tensor for masking specific attention heads.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the attention output tensor and the attention weights tensor.

        Raises:
            ValueError: If the dimensions of the input tensors are incompatible for matrix multiplication.
            TypeError: If the input tensors are not of type Tensor.
            RuntimeError: If there is a runtime issue during the computation.
        """
        attn_weights = ops.matmul(query, key.swapaxes(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / ops.full(
                [], value.shape[-1] ** 0.5, dtype=attn_weights.dtype
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.shape[-2], key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = float(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min)
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            mask_value = ops.full([], mask_value, dtype=attn_weights.dtype)
            attn_weights = ops.where(causal_mask, attn_weights.astype(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        This method _upcast_and_reordered_attn in the class GPT2Attention performs upcasting and reordering operations
        for the attention mechanism in a GPT-2 model.

        Args:
            self (GPT2Attention): The instance of the GPT2Attention class.
            query (Tensor): The input query tensor with shape (batch_size, num_heads, query_sequence_length, depth).
            key (Tensor): The input key tensor with shape (batch_size, num_heads, key_sequence_length, depth).
            value (Tensor): The input value tensor with shape (batch_size, num_heads, key_sequence_length, depth).
            attention_mask (Tensor, optional): An optional tensor defining additional attention masks with shape
                (batch_size, num_heads, query_sequence_length, key_sequence_length).
            head_mask (Tensor, optional): An optional tensor that masks specific heads of the attention mechanism.

        Returns:
            The computed attention output and attention weights.

        Raises:
            RuntimeError: Raised if there is an error during upcasting and the resulting attention weights
                do not have the expected data type 'mindspore.float32'.
        """
        bsz, num_heads, q_seq_len, dk = query.shape
        _, _, k_seq_len, _ = key.shape

        # Preallocate attn_weights for `baddbmm`
        attn_weights = ops.zeros((bsz * num_heads, q_seq_len, k_seq_len), dtype=mindspore.float32)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.shape[-1]) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        q, k = query.reshape(-1, q_seq_len, dk), key.swapaxes(-1, -2).reshape(-1, dk, k_seq_len)
        attn_weights = ops.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.shape[-2], key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = float(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min)
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            mask_value = mindspore.Tensor(mask_value, dtype=attn_weights.dtype)
            attn_weights = ops.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != mindspore.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype mindspore.float32")
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3)
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def construct(
        self,
        hidden_states: Optional[Tuple[mindspore.Tensor]],
        layer_past: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[mindspore.Tensor, Tuple[mindspore.Tensor]], ...]:
        """
        This method 'construct' in the class 'GPT2Attention' is responsible for constructing the attention mechanism
        for GPT-2 model.

        Args:
            self: The instance of the class.
            hidden_states (Optional[Tuple[mindspore.Tensor]]): The input hidden states for the attention mechanism.
            layer_past (Optional[Tuple[mindspore.Tensor]]): The past layer key and value tensors for optimization.
            attention_mask (Optional[mindspore.Tensor]): Masking tensor to prevent attention to certain positions.
            head_mask (Optional[mindspore.Tensor]): Masking tensor to prevent attention in specific heads.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states from the encoder for cross-attention.
            encoder_attention_mask (Optional[mindspore.Tensor]): Masking tensor for encoder attention.
            use_cache (Optional[bool]): Flag to use caching for optimization.
            output_attentions (Optional[bool]): Flag to output attention weights.

        Returns:
            Tuple[Union[mindspore.Tensor, Tuple[mindspore.Tensor]], ...]:
                A tuple containing the output tensor from attention mechanism and present states for caching.

        Raises:
            ValueError: If 'encoder_hidden_states' is provided without 'q_attn' weights defined for cross-attention.
        """
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, axis=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, axis=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Cell):

    """
    This class represents a multi-layer perceptron (MLP) component of the GPT-2 model.
    It is used to process the hidden states in the model architecture.

    The GPT2MLP class inherits from the nn.Cell class and contains methods for initializing the MLP and
    constructing the hidden states.

    Attributes:
        c_fc (Conv1D): A 1D convolutional layer used for intermediate processing of the hidden states.
        c_proj (Conv1D): A 1D convolutional layer used for final projection of the hidden states.
        act (activation function): The activation function used in the MLP.
        dropout (Dropout): A dropout layer used for regularization.

    Methods:
        __init__: Initializes the GPT2MLP with the given intermediate size and configuration.
        construct: Constructs the hidden states by applying the specified operations on the input hidden states.

    """
    def __init__(self, intermediate_size, config):
        """Initializes an instance of the GPT2MLP class.

        Args:
            self (GPT2MLP): The GPT2MLP object being initialized.
            intermediate_size (int): The size of the intermediate layer.
            config (object): The configuration object containing various settings.
                This object is expected to have the following attributes:

                - hidden_size (int): The size of the embedding dimension.
                - activation_function (str): The name of the activation function to use.
                - resid_pdrop (float): The dropout rate for residual connections.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def construct(self, hidden_states: Optional[Tuple[mindspore.Tensor]]) -> mindspore.Tensor:
        """
        Constructs a GPT2MLP model by applying a series of operations on the input hidden states.

        Args:
            self: An instance of the GPT2MLP class.
            hidden_states (Optional[Tuple[mindspore.Tensor]]): The input hidden states.
                It is an optional parameter and defaults to None.

        Returns:
            mindspore.Tensor: The output hidden states after applying the operations.

        Raises:
            None.

        Note:
            The `hidden_states` parameter should be a tuple of mindspore.Tensor objects representing the hidden states
            of the model.
            The `hidden_states` parameter can be None, in which case it will be ignored and not used in the operations.
            The output hidden states will be of type mindspore.Tensor.

        Example:
            ```python
            >>> model = GPT2MLP()
            >>> hidden_states = (tensor1, tensor2)
            >>> output = model.construct(hidden_states)
            ```
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Cell):

    """
    This class represents a single block of the GPT2 (Generative Pretrained Transformer 2) model.

    GPT2Block is a subclass of nn.Cell and contains the following attributes:

    - ln_1: A LayerNorm module for layer normalization.
    - attn: An instance of the GPT2Attention class for self-attention mechanism.
    - ln_2: A LayerNorm module for layer normalization.
    - crossattention: An instance of the GPT2Attention class for cross-attention mechanism
    (optional, if `encoder_hidden_states` are passed).
    - ln_cross_attn: A LayerNorm module for layer normalization in cross-attention mechanism
    (optional, if `encoder_hidden_states` are passed).
    - mlp: An instance of the GPT2MLP class for the feed-forward neural network.

    Methods:
        __init__: Initializes the GPT2Block instance with the given configuration and optional layer index.
        construct:
            Performs the forward pass of the GPT2Block.

            Parameters:

            - hidden_states: An optional tuple of tensors representing the input hidden states.
            - layer_past: An optional tuple of tensors representing the past hidden states (default: None).
            - attention_mask: An optional tensor representing the attention mask (default: None).
            - head_mask: An optional tensor representing the head mask (default: None).
            - encoder_hidden_states: An optional tensor representing the hidden states of the encoder (default: None).
            - encoder_attention_mask: An optional tensor representing the attention mask for the encoder (default: None).
            - use_cache: A boolean indicating whether to use cache for faster decoding (default: False).
            - output_attentions: A boolean indicating whether to output attentions weights (default: False).

            Returns:

            - A tuple of tensors representing the outputs of the GPT2Block.

    Note:
        If `encoder_hidden_states` are passed, the GPT2Block instance should be instantiated with cross-attention layers
        by setting `config.add_cross_attention=True`.

    Raises:
        ValueError: If `encoder_hidden_states` are passed, but the GPT2Block instance does not have cross-attention
            layers.

    """
    def __init__(self, config, layer_idx=None):
        """
        Initializes an instance of the GPT2Block class.

        Args:
            self: The object instance.
            config:
                An object containing the configuration parameters for the GPT2Block.
                It should have the following attributes:

                - hidden_size: An integer specifying the size of the hidden layer.
                - n_inner: An optional integer representing the number of inner layers.
                If not provided, the default value is 4 times the hidden size.
                - layer_norm_epsilon: A small float value used for layer normalization.
                It ensures numerical stability in the presence of small variances.
                - add_cross_attention: A boolean indicating whether to include cross-attention.
                - is_cross_attention: A boolean indicating whether this is a cross-attention layer.
                - layer_idx: An optional integer representing the index of the layer.
            layer_idx: An optional integer representing the index of the layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def construct(
        self,
        hidden_states: Optional[Tuple[mindspore.Tensor]],
        layer_past: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
        """
        Constructs a GPT2 block with optional cross-attention functionality.

        Args:
            self: The GPT2Block instance.
            hidden_states (Optional[Tuple[mindspore.Tensor]]): The input hidden states. Default is None.
            layer_past (Optional[Tuple[mindspore.Tensor]]):
                Past hidden states for autoregressive decoding. Default is None.
            attention_mask (Optional[mindspore.Tensor]):
                Mask to prevent attention to some positions. Default is None.
            head_mask (Optional[mindspore.Tensor]):
                Mask to nullify selected heads of the attention mechanism. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                Hidden states of the encoder for cross-attention. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention. Default is None.
            use_cache (Optional[bool]): Whether to use cache for faster decoding. Default is False.
            output_attentions (Optional[bool]): Whether to output attentions weights. Default is False.

        Returns:
            Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:

                - Tuple containing the final hidden states if `use_cache` is False.
                - Tuple containing the final hidden states and additional outputs if `use_cache` is True.

        Raises:
            ValueError: If `encoder_hidden_states` are provided but the model is not instantiated with cross-attention
                layers.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPT2Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    _no_split_modules = ["GPT2Block"]
    _keys_to_ignore_on_load_unexpected = [r'^(?:transformer\.)?h\.\d+\.attn\.bias$']

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(self.config.initializer_range),
                                                 cell.weight.shape,
                                                 cell.weight.dtype)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in cell.parameters_and_names():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.set_data(initializer(Normal((self.config.initializer_range / math.sqrt(2 * self.config.n_layer))),
                                              p.shape, p.dtype))

@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        mc_loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
            Multiple choice classification loss.
        logits (`mindspore.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`mindspore.Tensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`Tuple[Tuple[mindspore.Tensor]]`, *optional*, returned when `use_cache=True` is passed
            or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            GPT2Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    loss: Optional[mindspore.Tensor] = None
    mc_loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    mc_logits: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class GPT2Model(GPT2PreTrainedModel):

    """
    This class represents a GPT-2 model for natural language processing tasks. It includes methods for initializing
    the model, setting input embeddings, pruning model heads, and constructing the model for inference or training.
    The model consists of multiple GPT2Blocks organized in layers to process input sequences and generate output
    representations. The GPT2Model class inherits from the GPT2PreTrainedModel class, which provides additional
    functionality and pretrained weights for fine-tuning or transfer learning tasks.

    Methods:
        __init__: Initializes the GPT-2 model with configuration parameters.
        get_input_embeddings: Returns the input embeddings used by the model.
        set_input_embeddings: Sets new input embeddings for the model.
        _prune_heads: Prunes specific attention heads in the model based on the provided dictionary.
        construct: Constructs the GPT-2 model for inference or training with various input options and returns
            the model output.

    Attributes:
        embed_dim: The dimensionality of the embedding layer in the model.
        wte: Embedding layer for token embeddings.
        wpe: Embedding layer for position embeddings.
        drop: Dropout layer for regularization.
        h: List of GPT2Block layers for processing input sequences.
        ln_f: Layer normalization applied to the final hidden states.
    """
    def __init__(self, config):
        """
        Initializes an instance of the GPT2Model class.

        Args:
            self: The instance of the GPT2Model class.
            config: An object of type 'config' containing the configuration parameters for the GPT2Model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the GPT2Model.

        Args:
            self (GPT2Model): The instance of the GPT2Model class.

        Returns:
            None.

        Raises:
            None.

        This method is responsible for retrieving the input embeddings of the GPT2Model.
        It takes a single parameter, 'self', which refers to the instance of the GPT2Model class.

        The GPT2Model class is designed to handle GPT-2 models, which are based on the Transformer architecture.
        Input embeddings are representations of the input tokens in the model. They are used as the initial input to
        the model and are typically generated by applying a word embedding layer to the input tokens.

        Since this method does not return any value, the return type is 'None'. The purpose of this method is to
        retrieve the input embeddings needed for further processing within the GPT2Model.

        No exceptions are raised by this method.
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings of the GPT2Model.

        Args:
            self (GPT2Model): The instance of the GPT2Model class.
            new_embeddings: The new input embeddings to be set.
                It should be a tensor of shape (vocab_size, hidden_size) representing the word embeddings.

        Returns:
            None: This method modifies the input embeddings of the GPT2Model in-place.

        Raises:
            None.

        """
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the GPT-2 model.

        Args:
            self (:obj:`GPT2Model`): An instance of the `GPT2Model` class.
            input_ids (:obj:`Optional[mindspore.Tensor]`, `optional`): 
                Input tensor of shape :obj:`(batch_size, sequence_length)`.
            past_key_values (:obj:`Optional[Tuple[Tuple[mindspore.Tensor]]]`, `optional`): 
                Tuple of :obj:`(layer_num, batch_size, num_heads, past_sequence_length, hidden_size)` tensors 
                containing the previous hidden states (key and values of the attention blocks) if they were cached, 
                used for faster decoding. Defaults to :obj:`None`.
            attention_mask (:obj:`Optional[mindspore.Tensor]`, `optional`): 
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]`` 
                where 1 indicates the token is not masked and 0 indicates the token is masked. Defaults to :obj:`None`.
            token_type_ids (:obj:`Optional[mindspore.Tensor]`, `optional`): 
                Input tensor of shape :obj:`(batch_size, sequence_length)` indicating the token types to differentiate 
                between different sentences in the input. Defaults to :obj:`None`.
            position_ids (:obj:`Optional[mindspore.Tensor]`, `optional`): Tensor of shape :obj:
                `(batch_size, sequence_length)` containing the position indices. Defaults to :obj:`None`.
            head_mask (:obj:`Optional[mindspore.Tensor]`, `optional`): Mask to nullify selected heads of the 
                self-attention modules. Mask values selected in ``[0, 1]``, where 1 indicates the head is kept and 0
                indicates the head is nullified. Defaults to :obj:`None`.
            inputs_embeds (:obj:`Optional[mindspore.Tensor]`, `optional`): 
                Input tensor of shape :obj:`(batch_size, sequence_length, hidden_size)` containing the embedded inputs. 
                Defaults to :obj:`None`.
            encoder_hidden_states (:obj:`Optional[mindspore.Tensor]`, `optional`): 
                The encoded input sequence of shape :obj:`(batch_size, sequence_length, hidden_size)` to be used in the 
                cross-attention layer. Defaults to :obj:`None`.
            encoder_attention_mask (:obj:`Optional[mindspore.Tensor]`, `optional`): 
                Cross attention mask to avoid performing attention on padding token indices. Defaults to :obj:`None`.
            use_cache (:obj:`Optional[bool]`, `optional`): 
                Whether or not the model should return the past key values when used for inference. 
                Defaults to :obj:`None`.
            output_attentions (:obj:`Optional[bool]`, `optional`):
                Whether to also return all attention weights, including the self-attention weights of 
                each attention layer. Defaults to :obj:`None`.
            output_hidden_states (:obj:`Optional[bool]`, `optional`): 
                Whether to also return all hidden states of each layer in addition to the output tensor. 
                Defaults to :obj:`None`.
            return_dict (:obj:`Optional[bool]`, `optional`): 
                Whether to return a dictionary instead of a tuple. Defaults to :obj:`None`.

        Returns:
            :obj:`Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]`: 
                A tuple or a dictionary of outputs containing the following tensors depending 
                on the value of `return_dict`:
                
                - last_hidden_state (:obj:`mindspore.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`): 
                Sequence of hidden-states at the output of the last layer of the model.
                - past_key_values (:obj:`Tuple[Tuple[mindspore.Tensor]]`): 
                Tuple of :obj:`(layer_num, batch_size, num_heads, past_sequence_length, hidden_size)` tensors containing 
                the previous hidden states (key and values of the attention blocks) if they were cached, used for 
                faster decoding.
                - hidden_states (:obj:`Tuple[mindspore.Tensor]`): 
                Tuple of :obj:`(batch_size, sequence_length, hidden_size)` tensors containing the hidden states of all 
                layers of the model if `output_hidden_states=True`.
                - attentions (:obj:`Tuple[mindspore.Tensor]`): 
                Tuple of :obj:`(batch_size, num_heads, sequence_length, sequence_length)` tensors containing the 
                attention weights of all self-attention layers of the model if `output_attentions=True`.
                - cross_attentions (:obj:`Tuple[mindspore.Tensor]`): 
                Tuple of :obj:`(batch_size, num_heads, sequence_length, sequence_length)` tensors containing the 
                attention weights of all cross-attention layers of the model if `output_attentions=True` and 
                `config.add_cross_attention=True`.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are specified simultaneously.
            ValueError: If neither `input_ids` nor `inputs_embeds` are specified.
            ValueError: If `batch_size` is not defined or is less than or equal to 0.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]
        if position_ids is None:
            position_ids = ops.arange(past_length, input_shape[-1] + past_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
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
            attention_mask = (1.0 - attention_mask) * float(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.shape[-1],)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2LMHeadModel(GPT2PreTrainedModel):

    """
    The `GPT2LMHeadModel` class is a subclass of `GPT2PreTrainedModel` that represents a language model based on the 
    GPT-2 architecture.

    This class provides methods for initializing the model, getting and setting the output embeddings, preparing inputs 
    for generation, and constructing the model. It also includes a static method for reordering the cache when using 
    beam search or beam sampling.

    Attributes:
        transformer: A GPT2Model instance representing the GPT-2 transformer model.
        lm_head: A nn.Dense layer representing the output layer of the language model.

    Methods:
        __init__: Initializes the GPT2LMHeadModel.
        get_output_embeddings: Returns the lm_head output embeddings.
        set_output_embeddings: Sets the lm_head output embeddings.
        prepare_inputs_for_generation: 
            Prepares inputs for generation by adjusting the input_ids, token_type_ids, attention_mask, and position_ids.
        construct: Constructs the GPT2LMHeadModel and returns the model outputs.
        _reorder_cache: Reorders the past_key_values cache based on the beam_idx for beam search or beam sampling.

    Note:
        - The labels for language modeling are shifted inside the model.
        - The loss is computed only for labels in [0, ..., config.vocab_size].
        - The GPT2LMHeadModel class inherits from GPT2PreTrainedModel.

    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the GPT2LMHeadModel class.

        Args:
            self: The current object instance.
            config: An instance of the GPT2Config class representing the model configuration.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the GPT2LMHeadModel.

        Args:
            self: An instance of the GPT2LMHeadModel class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the output embeddings of the GPT2LMHeadModel. The output embeddings are the weights of
        the linear layer (lm_head) which is responsible for producing the logits for each token in the language model.
        These logits are then used to calculate the probabilities of the next token in the sequence.

        Note that the returned value is of type None, as the method doesn't explicitly return any value, but rather
        directly accesses the output embeddings of the GPT2LMHeadModel.

        Example:
            ```python
            >>> odel = GPT2LMHeadModel()
            >>> output_embeddings = model.get_output_embeddings()
            ```
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the GPT2LMHeadModel.

        Args:
            self (GPT2LMHeadModel): The instance of the GPT2LMHeadModel class.
                It represents the GPT-2 language model head model.
            new_embeddings (any): The new embeddings to be set as the output embeddings.
                These embeddings will replace the current output embeddings in the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        '''
        Prepare inputs for generation.

        Args:
            self (GPT2LMHeadModel): The GPT2LMHeadModel instance.
            input_ids (torch.Tensor): The input token ids of shape [batch_size, sequence_length].
            past_key_values (Tuple[torch.Tensor]): The past key values of the model.
            inputs_embeds (torch.Tensor): The input embeddings of shape [batch_size, sequence_length, hidden_size].

        Returns:
            None

        Raises:
            None
        '''
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
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
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[mindspore.Tensor]], beam_idx: mindspore.Tensor
    ) -> Tuple[Tuple[mindspore.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )


class GPT2DoubleHeadsModel(GPT2PreTrainedModel):

    """
    This class represents a GPT-2 model with two classification heads for multiple choice tasks.
    It is designed to be used for natural language processing tasks that require generating text and making multiple
    choice predictions. The model architecture is based on the GPT-2 model with additional heads for language modeling
    and multiple choice classification.

    The class includes methods for initializing the model, setting and getting output embeddings, preparing inputs for
    text generation, and constructing the model for inference or training. It also provides a method for reordering
    cache during beam search or beam sampling.

    Note that this class inherits from GPT2PreTrainedModel, which is a base class for all GPT-2 models in the
    transformers library. The GPT2DoubleHeadsModel extends the base functionality of the GPT-2 model to support multiple
    choice tasks.

    For detailed usage examples and descriptions of input parameters and return values, please refer to the method
    docstrings within the class code.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the GPT2DoubleHeadsModel class.

        Args:
            self: The object instance.
            config: An instance of the GPT2Config class that defines the model configuration.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        config = copy.deepcopy(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the GPT2DoubleHeadsModel.

        Args:
            self (GPT2DoubleHeadsModel): The current instance of the GPT2DoubleHeadsModel.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the GPT2DoubleHeadsModel.

        Args:
            self (GPT2DoubleHeadsModel): The instance of the GPT2DoubleHeadsModel class.
            new_embeddings (torch.nn.Embedding): The new embeddings to set as the output embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepares the inputs for generation in the GPT2DoubleHeadsModel class.

        Args:
            self (GPT2DoubleHeadsModel): The instance of the GPT2DoubleHeadsModel class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length) containing the input IDs.
            past_key_values (tuple, optional): A tuple of past key values. Defaults to None.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation, including the following keys:

                - 'input_ids' (torch.Tensor): The input tensor after removing the prefix.
                Shape: (batch_size, sequence_length)
                - 'past_key_values' (tuple): The updated past key values.
                - 'use_cache' (bool): The value of the 'use_cache' keyword argument.
                - 'position_ids' (torch.Tensor): The position IDs tensor. Shape: (batch_size, sequence_length)
                - 'attention_mask' (torch.Tensor): The attention mask tensor. Shape: (batch_size, sequence_length)
                - 'token_type_ids' (torch.Tensor): The token type IDs tensor. Shape: (batch_size, sequence_length)

        Raises:
            None.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        mc_token_ids: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        mc_labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, GPT2DoubleHeadsModelOutput]:
        r"""
        Args:
            mc_token_ids (`mindspore.Tensor` of shape `(batch_size, num_choices)`, *optional*,
                default to index of the last token of the input):
                Index of the classification token in each input sequence. Selected in the range `[0, input_ids.shape[-1] -
                1]`.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
                `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
            mc_labels (`mindspore.Tensor` of shape `(batch_size)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
                where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

        Returns:
            Union[Tuple, GPT2DoubleHeadsModelOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, GPT2DoubleHeadsModel
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = GPT2DoubleHeadsModel.from_pretrained("gpt2")
            ...
            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
            >>> # Update the model embeddings with the new vocabulary size
            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))
            ...
            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
            ...
            >>> input_ids = mindspore.Tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = mindspore.Tensor([cls_token_location])  # Batch size: 1
            ...
            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.logits
            >>> mc_logits = outputs.mc_logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            mc_loss = ops.cross_entropy(mc_logits.view(-1, mc_logits.shape[-1]), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            lm_loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[mindspore.Tensor]], beam_idx: mindspore.Tensor
    ) -> Tuple[Tuple[mindspore.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )


class GPT2ForSequenceClassification(GPT2PreTrainedModel):

    """
    GPT2ForSequenceClassification represents a GPT-2 model fine-tuned for sequence classification tasks.
    This class inherits from GPT2PreTrainedModel.

    The GPT2ForSequenceClassification class provides a method 'construct' for constructing the sequence classification
    model. The 'construct' method accepts input tensors such as input_ids, past_key_values, attention_mask,
    token_type_ids, position_ids, head_mask, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states,
    and return_dict.

    The 'construct' method returns a tuple containing the sequence classification loss, logits, past_key_values,
    hidden_states, and attentions. If the return_dict parameter is set to False, the output is a tuple of pooled_logits
    and transformer_outputs. The sequence classification loss is computed based on the given labels and the model
    configuration.

    The GPT2ForSequenceClassification class also includes an __init__ method for initializing the model with the given
    configuration, number of labels, GPT2Model transformer, and score.

    Labels for computing the sequence classification/regression loss can be provided as a mindspore.Tensor of shape
    (batch_size,) in the 'construct' method. Indices for the labels should be in the range [0,
    config.num_labels - 1]. If config.num_labels == 1, a regression loss is computed (Mean-Square loss).
    If config.num_labels > 1, a classification loss is computed (Cross-Entropy).

    The class ensures proper handling of padding tokens and provides warnings for unexpected scenarios.
    Additionally, it dynamically determines the problem type based on the configuration and label data types.

    Note:
        This docstring is generated based on the provided code and does not include signatures or any other code.
    """
    def __init__(self, config):
        """Initializes a new instance of the GPT2ForSequenceClassification class.

        Args:
            self: The object itself.
            config: An instance of the GPT2Config class containing the configuration parameters for the GPT2 model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Dense(config.n_embd, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
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

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
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


class GPT2ForTokenClassification(GPT2PreTrainedModel):

    """
    This class represents a GPT-2 model for token classification, inheriting from GPT2PreTrainedModel.
    It includes methods for initialization and construction of the model for token classification tasks.
    The model utilizes a transformer architecture with configurable dropout and classifier layers for classification
    or regression loss computation based on the number of labels specified in the configuration.
    The construct method processes input data through the transformer, applies dropout, generates logits using the
    classifier layer, and computes the loss if labels are provided. The method returns the loss and output
    based on the specified return format.
    """
    def __init__(self, config):
        """
        Initializes a GPT2ForTokenClassification instance.

        Args:
            self (GPT2ForTokenClassification): The GPT2ForTokenClassification instance.
            config (GPT2Config): The configuration object containing model hyperparameters.
                This parameter is required to properly configure the GPT2 model for token classification.
                It should include the following attributes:

                - num_labels (int): The number of distinct labels for token classification.
                - classifier_dropout (float, optional): The dropout probability for the classifier layer.
                - hidden_dropout (float, optional): The dropout probability for hidden layers.
                If both 'classifier_dropout' and 'hidden_dropout' are provided, 'classifier_dropout' takes precedence.

        Returns:
            None.

        Raises:
            ValueError: If 'config' is missing the 'num_labels' attribute.
            TypeError: If 'config' is not an instance of GPT2Config.
            TypeError: If 'classifier_dropout' or 'hidden_dropout' is not a float.
            ValueError: If both 'classifier_dropout' and 'hidden_dropout' in 'config' are not None or float.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
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

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class GPT2ForQuestionAnswering(GPT2PreTrainedModel):

    """
    This class represents a GPT2 model for question answering tasks. It is a subclass of GPT2PreTrainedModel.

    GPT2ForQuestionAnswering inherits the following attributes and methods from GPT2PreTrainedModel:

    Attributes:
        config: The configuration object for the GPT2 model.
        transformer: The GPT2Model instance for the transformer part of the model.
        qa_outputs: A neural network layer for question answering outputs.

    Methods:
        __init__: Initializes the GPT2ForQuestionAnswering instance.
        construct: Constructs the GPT2ForQuestionAnswering model and performs question answering.

    The GPT2ForQuestionAnswering class provides the following functionality:

    - Initialization:

        - The GPT2ForQuestionAnswering instance is initialized with a 'config' parameter.
        - The 'config' parameter is used to set the 'num_labels' attribute.
        - The 'transformer' attribute is set to an instance of the GPT2Model class with the 'config' parameter.
        - The 'qa_outputs' attribute is set to a neural network layer with 'config.hidden_size' input size and 2 output units.

    - Construction:

        - The 'construct' method constructs the GPT2ForQuestionAnswering model.
        - The method takes several input tensors as parameters, such as 'input_ids', 'attention_mask', 'token_type_ids', etc.
        - It also takes optional parameters like 'start_positions', 'end_positions', 'output_attentions',
        'output_hidden_states', and 'return_dict'.
        - The method returns a tuple of outputs, including 'start_logits' and 'end_logits', which represent the
        predicted start and end positions for the answer span.
        - If 'start_positions' and 'end_positions' are provided, the method calculates the loss for the question
        answering task and returns the total loss along with the outputs.

    Note:
        The method parameters and return types are defined using MindSpore framework's type hints.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the GPT2ForQuestionAnswering class.
        
        Args:
            self (GPT2ForQuestionAnswering): The instance of the GPT2ForQuestionAnswering class.
            config:
                Configuration object containing necessary settings for the model initialization.

                - Type: object
                - Purpose: Configures the model based on the provided settings.
                - Restrictions: Must be a valid configuration object.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
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

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
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
    "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
    "GPT2DoubleHeadsModel",
    "GPT2ForQuestionAnswering",
    "GPT2ForSequenceClassification",
    "GPT2ForTokenClassification",
    "GPT2LMHeadModel",
    "GPT2Model",
    "GPT2PreTrainedModel",
]
