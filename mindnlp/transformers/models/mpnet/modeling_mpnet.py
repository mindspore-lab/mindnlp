# coding=utf-8
# Copyright 2024 The HuaWei Technologies Co., Microsoft Corporation.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
# pylint: disable=C0103
# pylint: disable=W0237
# pylint: disable=R1714
# pylint: disable=R1720
"""MindSpore MPNet model."""


import math
from typing import Optional, Tuple, Union
import numpy as np

import mindspore
from mindnlp.core import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindnlp.utils import logging

from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_mpnet import MPNetConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/mpnet-base"
_CONFIG_FOR_DOC = "MPNetConfig"


MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/mpnet-base",
]


class MPNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MPNetConfig
    pretrained_model_archive_map = MPNET_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "mpnet"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class MPNetEmbeddings(nn.Module):
    """forward the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetEmbeddings class.
        
        Args:
            self: The object instance.
            config: An object of type 'config' containing the configuration parameters.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs):
        """
        Method 'forward' in the class 'MPNetEmbeddings'.
        
        Args:
            self: The instance of the class.
            input_ids (torch.Tensor, optional): The input tensor representing token indices. Defaults to None.
            position_ids (torch.Tensor, optional): The input tensor representing position indices. Defaults to None.
            inputs_embeds (torch.Tensor, optional): The input tensor representing embeddings. Defaults to None.
        
        Returns:
            embeddings: The method returns the forwarded embeddings of input data.
        
        Raises:
            ValueError: If both 'input_ids' and 'inputs_embeds' are None.
            ValueError: If shape inconsistency is detected between 'input_ids' and 'inputs_embeds'.
            IndexError: If an index is out of bounds while accessing tensors.
            TypeError: If the input types are not torch tensors.
        """
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: mindspore.Tensor

        Returns: mindspore.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ops.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64
        )
        return position_ids.unsqueeze(0).broadcast_to(input_shape)


class MPNetSelfAttention(nn.Module):
    """SelfAttention Model"""
    def __init__(self, config):
        """
        Initializes a new instance of the MPNetSelfAttention class.
        
        Args:
            self: The instance of the MPNetSelfAttention class.
            config (object):
                An object containing configuration parameters for the self-attention mechanism.

                - hidden_size (int): The size of the hidden layers.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (int): The size of the embeddings.
                - attention_probs_dropout_prob (float): The dropout probability for attention probabilities.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads or if the 'embedding_size'
                attribute is not present in the config object.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = nn.Linear(config.hidden_size, self.all_head_size)
        self.k = nn.Linear(config.hidden_size, self.all_head_size)
        self.v = nn.Linear(config.hidden_size, self.all_head_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        Transposes the input tensor `x` to prepare it for multi-head attention scoring.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        """
        This method forwards self-attention mechanism for MPNetSelfAttention.

        Args:
            self: The instance of the class.
            hidden_states: Tensor containing the input hidden states. Shape: (batch_size, sequence_length, hidden_size).
            attention_mask: Optional tensor to mask out attention scores.
                Shape: (batch_size, sequence_length, sequence_length).
            head_mask: Optional tensor to mask out attention heads.
                Shape: (num_attention_heads, sequence_length, sequence_length).
            position_bias: Optional tensor containing positional bias.
                Shape: (batch_size, num_attention_heads, sequence_length, sequence_length).
            output_attentions: Boolean indicating whether to output attention probabilities.

        Returns:
            Tuple:
                Tuple containing output tensor 'o' and attention probabilities tensor.
                    If output_attentions is False, returns tuple with only 'o'.

        Raises:
            ValueError: If the dimensions of input tensors are incompatible for matrix operations.
            TypeError: If the input parameters are not of expected types.
            RuntimeError: If any runtime error occurs during the execution of the method.
        """
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(q, k.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply relative position embedding (precomputed in MPNetEncoder) if provided.
        if position_bias is not None:
            attention_scores += position_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        c = ops.matmul(attention_probs, v)

        c = c.permute(0, 2, 1, 3)
        new_c_shape = c.shape[:-2] + (self.all_head_size,)
        c = c.view(*new_c_shape)

        o = self.o(c)

        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs


class MPNetAttention(nn.Module):
    """
    Multi-head self-attention mechanism for MPNet.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MPNetAttention class.

        Args:
            self: The instance of the MPNetAttention class.
            config: A configuration object containing the settings for the MPNetAttention.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attn = MPNetSelfAttention(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        """Prunes specified attention heads from the multi-head self-attention layer."""
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attn.num_attention_heads, self.attn.attention_head_size, self.pruned_heads
        )

        self.attn.q = prune_linear_layer(self.attn.q, index)
        self.attn.k = prune_linear_layer(self.attn.k, index)
        self.attn.v = prune_linear_layer(self.attn.v, index)
        self.attn.o = prune_linear_layer(self.attn.o, index, dim=1)

        self.attn.num_attention_heads = self.attn.num_attention_heads - len(heads)
        self.attn.all_head_size = self.attn.attention_head_size * self.attn.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        """
        Constructs the attention layer for the MPNetAttention class.

        Args:
            self (MPNetAttention): An instance of the MPNetAttention class.
            hidden_states (Tensor): The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[Tensor]): A tensor of shape (batch_size, sequence_length)
                indicating which tokens should be attended to and which ones should be ignored. Defaults to None.
            head_mask (Optional[Tensor]): A tensor of shape (num_heads,) representing the mask to be applied to the
                attention scores of each head. Defaults to None.
            position_bias (Optional[Tensor]): A tensor of shape (num_heads, sequence_length, sequence_length)
                representing the position bias to be added to the attention scores. Defaults to None.
            output_attentions (bool): Whether to output the attention scores. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Union[Tensor, Tuple[Tensor]], ...]: A tuple containing the attention output tensor of shape
                (batch_size, sequence_length, hidden_size) and any additional outputs returned by the attention layer.

        Raises:
            None.
        """
        self_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            position_bias,
            output_attentions=output_attentions,
        )
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MPNetIntermediate(nn.Module):
    """Copied from transformers.models.bert.modeling_bert.BertIntermediate"""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetIntermediate class.

        Args:
            self: The instance of the MPNetIntermediate class.
            config:
                An object containing configuration parameters for the MPNetIntermediate instance.

                - Type: Any
                - Purpose: Contains configuration settings for the MPNetIntermediate instance.
                - Restrictions: None

        Returns:
            None

        Raises:
            TypeError: If the config parameter is not provided.
            ValueError: If the hidden activation function specified in the config is not supported.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the intermediate layer of the MPNet model.

        Args:
            self (MPNetIntermediate): The instance of the MPNetIntermediate class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size)
                representing the hidden states.

        Returns:
            mindspore.Tensor: The output tensor of shape (batch_size, sequence_length, hidden_size) containing
                the processed hidden states.

        Raises:
            TypeError: If the input 'hidden_states' is not a mindspore.Tensor.
            ValueError: If the shape of 'hidden_states' is not (batch_size, sequence_length, hidden_size).
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MPNetOutput(nn.Module):
    """Copied from transformers.models.bert.modeling_bert.BertOutput"""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetOutput class.

        Args:
            self: The instance of the MPNetOutput class.
            config:
                An object containing configuration parameters.

                - Type: Any
                - Purpose: The configuration object specifying model settings.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required attributes.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the MPNetOutput.

        Args:
            self (MPNetOutput): An instance of the MPNetOutput class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
                This tensor is input to the dense layer, which performs a linear transformation on the hidden states.
                The shape of this tensor should be compatible with the dense layer's weight matrix.
            input_tensor (mindspore.Tensor): A tensor containing the input states.
                This tensor is added to the hidden states after the linear transformation and dropout.
                The shape of this tensor should be compatible with the hidden states tensor.

        Returns:
            mindspore.Tensor: A tensor representing the forwarded MPNetOutput.
                The shape of the tensor is the same as the hidden_states tensor.
                The forwarded MPNetOutput is obtained by applying the dense layer, dropout, and LayerNorm operations
                to the hidden states and adding the input tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MPNetLayer(nn.Module):
    """Single layer in the MPNet model architecture."""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetLayer class.

        Args:
            self (MPNetLayer): The instance of the MPNetLayer class.
            config (object): The configuration object used to initialize the MPNetLayer.
                This object contains the settings and parameters required for the MPNetLayer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = MPNetAttention(config)
        self.intermediate = MPNetIntermediate(config)
        self.output = MPNetOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        """
        Constructs an MPNetLayer.

        Args:
            self (object): The object instance.
            hidden_states (tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (tensor, optional): The attention mask of shape (batch_size, sequence_length). Defaults to None.
            head_mask (tensor, optional): The head mask of shape (num_heads). Defaults to None.
            position_bias (tensor, optional): The position bias of shape (num_heads, sequence_length, sequence_length).
                Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            tuple: A tuple containing layer_output of shape (batch_size, sequence_length, hidden_size) and
                additional optional outputs.

        Raises:
            ValueError: If the input dimensions are invalid or incompatible.
            TypeError: If the input types are incorrect.
            RuntimeError: If there is a runtime error during the execution of the method.
            """
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class MPNetEncoder(nn.Module):
    """Encoder module for the MPNet model."""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetEncoder class.

        Args:
            self: The current object instance.
            config (object):
                The configuration object containing the settings for the MPNetEncoder.

                - Type: object
                - Purpose: Specifies the configuration settings for the MPNetEncoder.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.layer = nn.ModuleList([MPNetLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, self.n_heads)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """
        Construct method in the MPNetEncoder class.

        Args:
            self: The instance of the MPNetEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed by the encoder.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor specifying which positions
                should be attended to. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): An optional tensor specifying which heads to mask out.
                Defaults to None.
            output_attentions (bool): A flag indicating whether to output the attention weights. Defaults to False.
            output_hidden_states (bool): A flag indicating whether to output hidden states for all layers.
                Defaults to False.
            return_dict (bool): A flag indicating whether to return the outputs as a dictionary. Defaults to False.

        Returns:
            None

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the input parameters are not within the expected ranges.
        """
        position_bias = self.compute_position_bias(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                position_bias,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        """
        Computes the position bias for relative attention in the MPNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            position_ids (torch.Tensor, optional): Tensor containing position indices. If provided,
                the position indices are used to compute relative positions; otherwise, indices
                are generated based on the input tensor's sequence length. Defaults to None.
            num_buckets (int, optional): Number of buckets for relative position encoding.
                Defaults to 32.

        Returns:
            torch.Tensor: Position bias tensor of shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        bsz, qlen, klen = x.shape[0], x.shape[1], x.shape[1]
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = ops.arange(qlen, dtype=mindspore.int64)[:, None]
            memory_position = ops.arange(klen, dtype=mindspore.int64)[None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(relative_position, num_buckets=num_buckets)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.broadcast_to((bsz, -1, qlen, klen))
        return values

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Bucketizes relative positions for relative attention in the MPNet model.

        Args:
            relative_position (torch.Tensor): Tensor containing relative positions.
            num_buckets (int, optional): Number of buckets for bucketization. Defaults to 32.
            max_distance (int, optional): Maximum distance for bucketization. Defaults to 128.

        Returns:
            torch.Tensor: Bucketized relative positions.
        """
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).to(mindspore.int64) * num_buckets
        n = ops.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            ops.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(mindspore.int64)

        val_if_large_np = np.minimum(val_if_large.asnumpy(), ops.full_like(val_if_large, num_buckets - 1).asnumpy())
        val_if_large = mindspore.Tensor(val_if_large_np)
        ret += ops.where(is_small, n, val_if_large)
        return ret


class MPNetPooler(nn.Module):
    """Copied from transformers.models.bert.modeling_bert.BertPooler"""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetPooler class.

        Args:
            self (MPNetPooler): The current instance of the MPNetPooler class.
            config (object): The configuration object containing parameters for initializing the MPNetPooler.
                The config object should have a 'hidden_size' attribute indicating the size of the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards a pooled output from the hidden states of the MPNet model.

        Args:
            self: The instance of the MPNetPooler class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states of the MPNet model.
                It is expected to have a shape of (batch_size, sequence_length, hidden_size), where batch_size is the
                batch size, sequence_length is the length of the input sequence, and hidden_size is the size of the
                hidden state.

        Returns:
            mindspore.Tensor: The pooled output tensor generated from the hidden states.
                It has a shape of (batch_size, hidden_size).

        Raises:
            None.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MPNetModel(MPNetPreTrainedModel):
    """MPNet model architecture."""
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes an instance of the MPNetModel class.

        Args:
            self: The instance of the class.
            config (dict): A dictionary containing the configuration parameters for the model.
            add_pooling_layer (bool): A flag indicating whether to include a pooling layer in the model. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = MPNetEmbeddings(config)
        self.encoder = MPNetEncoder(config)
        self.pooler = MPNetPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the MPNetModel.

        Args:
            self: An instance of the MPNetModel class.

        Returns:
            None: The method returns the input embeddings from the MPNetModel.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings in the MPNetModel class.

        Args:
            self (MPNetModel): The instance of the MPNetModel class.
            value: The input value representing the embeddings to be set for the model.
                It should be compatible with the expected format for word embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPooling]:
        """
        Constructs the MPNet model.

        Args:
            self (MPNetModel): The instance of the MPNetModel class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the indices of input sequence tokens.
            attention_mask (Optional[mindspore.Tensor]): The optional attention mask tensor specifying which tokens
                should be attended to.
            position_ids (Optional[mindspore.Tensor]): The optional input tensor containing the position indices
                of each input token.
            head_mask (Optional[mindspore.Tensor]): The optional tensor specifying which heads should be masked in
                the self-attention layers.
            inputs_embeds (Optional[mindspore.Tensor]): The optional input tensor containing the embeddings of
                each input token.
            output_attentions (Optional[bool]): Whether to return the attentions.
            output_hidden_states (Optional[bool]): Whether to return the hidden states.
            return_dict (Optional[bool]): Whether to return the output as a dictionary.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPooling]:
                The output of the MPNet model.

                - If `return_dict` is `False`, a tuple containing the following elements is returned:

                    - sequence_output (mindspore.Tensor): The output tensor of the encoder.
                    - pooled_output (mindspore.Tensor): The pooled output tensor.
                    - hidden_states (Tuple[mindspore.Tensor]): The hidden states of all layers.
                    - attentions (Tuple[mindspore.Tensor]): The attentions of all layers.

                - If `return_dict` is `True`, an instance of BaseModelOutputWithPooling is returned,
                which contains the following attributes:

                    - last_hidden_state (mindspore.Tensor): The output tensor of the encoder.
                    - pooler_output (mindspore.Tensor): The pooled output tensor.
                    - hidden_states (Tuple[mindspore.Tensor]): The hidden states of all layers.
                    - attentions (Tuple[mindspore.Tensor]): The attentions of all layers.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are provided simultaneously.
            ValueError: If neither `input_ids` nor `inputs_embeds` are provided.
            ValueError: If the dimensions of `input_ids` and `attention_mask` do not match.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MPNetForMaskedLM(MPNetPreTrainedModel):
    """MPNet model for masked language modeling."""
    _tied_weights_keys = ["lm_head.decoder"]

    def __init__(self, config):
        """
        Initializes an instance of the MPNetForMaskedLM class.

        Args:
            self: The object itself.
            config (MPNetConfig): The configuration object that defines the model architecture and hyperparameters.

        Returns:
            None

        Raises:
            None.
        """
        super().__init__(config)

        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        self.lm_head = MPNetLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Retrieve the output embeddings from the decoder of the language model head.

        Args:
            self (MPNetForMaskedLM): An instance of the MPNetForMaskedLM class.
                Represents the model for Masked Language Modeling.

        Returns:
            None: The method returns the output embeddings from the decoder of the language model head.

        Raises:
            None.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for MPNetForMaskedLM model.

        Args:
            self (MPNetForMaskedLM): The instance of the MPNetForMaskedLM class.
            new_embeddings (torch.nn.Module): The new embeddings to be set as the output embeddings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mpnet(
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
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetLMHead(nn.Module):
    """MPNet Head for masked and permuted language modeling."""
    def __init__(self, config):
        """
        This method initializes an instance of the MPNetLMHead class.

        Args:
            self: The instance of the MPNetLMHead class.
            config:
                An object containing configuration parameters for the MPNetLMHead model.

                - Type: Config object
                - Purpose: Specifies the configuration settings for the MPNetLMHead model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None

        Raises:
            ValueError: If the configuration object is invalid or missing required parameters.
            TypeError: If the configuration object is not of the expected type.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        """
        This method forwards the output by processing the input features through various layers.

        Args:
            self (MPNetLMHead): Instance of the MPNetLMHead class.
            features (tensor): Input features to be processed. Expected to be a tensor data type.

        Returns:
            None: This method returns None after processing the input features through the defined layers.

        Raises:
            None.
        """
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class MPNetForSequenceClassification(MPNetPreTrainedModel):
    """MPNet model for sequence classification tasks."""
    def __init__(self, config):
        """
        Initializes an instance of MPNetForSequenceClassification.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing settings for the model initialization.
                Must contain the attribute 'num_labels' specifying the number of labels for classification.

        Returns:
            None.

        Raises:
            ValueError: If the 'config' object does not have the 'num_labels' attribute.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        self.classifier = MPNetClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mpnet(
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
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == mindspore.int64 or labels.dtype == mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetForMultipleChoice(MPNetPreTrainedModel):
    """MPNet model for multiple choice tasks."""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetForMultipleChoice class.

        Args:
            self (MPNetForMultipleChoice): An instance of the MPNetForMultipleChoice class.
            config (object): The configuration object for the MPNetModel.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.mpnet = MPNetModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], MultipleChoiceModelOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.mpnet(
            flat_input_ids,
            position_ids=flat_position_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetForTokenClassification(MPNetPreTrainedModel):
    """MPNet model for token classification tasks."""
    def __init__(self, config):
        """
        Initializes a new instance of the MPNetForTokenClassification class.

        Args:
            self: An instance of the MPNetForTokenClassification class.
            config: An instance of the MPNetConfig class containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mpnet(
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        """
        Initializes an instance of the MPNetClassificationHead class.

        Args:
            self: The instance of the class itself.
            config:
                An object containing configuration parameters for the head, including:

                - hidden_size (int): The size of the hidden layer.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.
                - num_labels (int): The number of output labels.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not of the expected type.
            ValueError: If any of the configuration parameters are invalid or missing.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        """
        Constructs the MPNetClassificationHead by performing a series of operations on the input features.

        Args:
            self: The instance of the MPNetClassificationHead class.
            features (Tensor): The input features to be processed.
                It should be a tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            None

        Raises:
            None
        """
        x = features[:, 0, :]  # take <s> token (equiv. to BERT's [CLS] token)
        x = self.dropout(x)
        x = self.dense(x)
        x = ops.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MPNetForQuestionAnswering(MPNetPreTrainedModel):
    """MPNet model for question answering tasks."""
    def __init__(self, config):
        """
        Initialize the MPNetForQuestionAnswering class.

        Args:
            self (object): The instance of the MPNetForQuestionAnswering class.
            config (object):
                An object containing configuration settings for the model.

                - Type: Custom class object
                - Purpose: Specifies the configuration parameters for the model initialization.
                - Restrictions: Must contain the 'num_labels' attribute.
        
        Returns:
            None.
        
        Raises:
            AttributeError: If the 'config' object does not have the 'num_labels' attribute.
            TypeError: If the 'config' parameter is not of the expected type.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
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
    ) -> Union[Tuple[mindspore.Tensor], QuestionAnsweringModelOutput]:
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

        outputs = self.mpnet(
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

            start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
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


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`. :param torch.Tensor x: :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = ops.cumsum(mask, axis=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


__all__ = [
    "MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MPNetForMaskedLM",
    "MPNetForMultipleChoice",
    "MPNetForQuestionAnswering",
    "MPNetForSequenceClassification",
    "MPNetForTokenClassification",
    "MPNetLayer",
    "MPNetModel",
    "MPNetPreTrainedModel",
]
