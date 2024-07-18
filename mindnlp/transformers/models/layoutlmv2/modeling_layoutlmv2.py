# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
""" Mindnlp LayoutLMv2 model."""

import math
from typing import Optional, Tuple, Union

import mindspore
import numpy as np
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import Normal, initializer, Constant
from mindspore.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

from mindnlp.transformers.ms_utils import apply_chunking_to_forward
from mindnlp.utils import logging

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_layoutlmv2 import LayoutLMv2Config
from .visual_backbone import build_resnet_fpn_backbone

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/layoutlmv2-base-uncased"
_CONFIG_FOR_DOC = "LayoutLMv2Config"

LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv2-base-uncased",
    "microsoft/layoutlmv2-large-uncased",
]


class LayoutLMv2Embeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes the LayoutLMv2Embeddings class with the provided configuration.
        
        Args:
            self: The instance of the LayoutLMv2Embeddings class.
            config:
                An object containing configuration parameters for the embeddings.

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - pad_token_id (int): The padding token ID.
                - max_position_embeddings (int): The maximum position embeddings.
                - max_2d_position_embeddings (int): The maximum 2D position embeddings.
                - coordinate_size (int): The size of coordinate embeddings.
                - shape_size (int): The size of shape embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for LayerNorm.
                - hidden_dropout_prob (float): The dropout probability.

        Returns:
            None.

        Raises:
            None.
        """
        super(LayoutLMv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.position_ids = mindspore.Tensor(np.arange(0, config.max_position_embeddings)).broadcast_to(
                (1, -1))

    def _calc_spatial_position_embeddings(self, bbox):
        """
        This method calculates spatial position embeddings based on the provided bounding box coordinates.

        Args:
            self: An instance of the LayoutLMv2Embeddings class.
            bbox: A tensor containing bounding box coordinates in the shape (batch_size, num_boxes, 4).
                The four coordinates represent the left, upper, right, and lower positions of each bounding box.
                The values should be within the range of 0 to 1000.

        Returns:
            spatial_position_embeddings: A tensor containing the calculated spatial position embeddings.
                The embeddings include left, upper, right, and lower position embeddings,
                as well as height and width position embeddings concatenated along the last dimension.

        Raises:
            IndexError: Raised if the coordinate values in bbox are outside the valid range of 0 to 1000.
        """
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = ops.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings


class LayoutLMv2SelfAttention(nn.Cell):
    """
    LayoutLMv2SelfAttention is the self-attention layer for LayoutLMv2. It is based on the implementation of
    """
    def __init__(self, config):
        """
        Initializes the LayoutLMv2SelfAttention class.

        Args:
            self (LayoutLMv2SelfAttention): An instance of the LayoutLMv2SelfAttention class.
            config (object): The configuration object that contains the settings for the self-attention layer.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads and the configuration
                object does not have an 'embedding_size' attribute.

        This method initializes the LayoutLMv2SelfAttention class by setting the necessary attributes and layers.
        It checks if the hidden size is divisible by the number of attention heads and raises a ValueError if not.
        The method also determines if the fast_qkv (fast query, key, value) method should be used based on the configuration.
        If fast_qkv is enabled, it creates a dense layer for the query, key, and value (qkv_linear), along with biases
        (q_bias and v_bias). Otherwise, it creates separate dense layers for query, key, and value. Finally, it sets the
        dropout layer based on the configuration's attention_probs_dropout_prob value.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Dense(config.hidden_size, 3 * self.all_head_size, has_bias=False)
            self.q_bias = Parameter(initializer(Constant(0.0), [1, 1, self.all_head_size], mindspore.float32))
            self.v_bias = Parameter(initializer(Constant(0.0), [1, 1, self.all_head_size], mindspore.float32))
        else:
            self.query = nn.Dense(config.hidden_size, self.all_head_size)
            self.key = nn.Dense(config.hidden_size, self.all_head_size)
            self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        Args:
            self (LayoutLMv2SelfAttention): The instance of the LayoutLMv2SelfAttention class.
            x (tensor): The input tensor to be transposed for attention scores calculation.

        Returns:
            tensor: The transposed tensor for attention scores calculation. 
                It has the shape (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            None
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        """
        This method computes the query, key, and value tensors for LayoutLMv2 self-attention mechanism.

        Args:
            self (LayoutLMv2SelfAttention): The instance of LayoutLMv2SelfAttention class.
            hidden_states (tensor): The input tensor representing the hidden states.

        Returns:
            (tuple): A tuple containing the query (q), key (k), and value (v) tensors.

        Raises:
            ValueError: If the dimensions of the query (q) tensor and the q_bias tensor do not match.
            ValueError: If the dimensions of the value (v) tensor and the v_bias tensor do not match.
        """
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = ops.chunk(qkv, 3, axis=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None,
    ):
        """
        Constructs the self-attention mechanism for the LayoutLMv2 model.

        Args:
            self (LayoutLMv2SelfAttention): The instance of the LayoutLMv2SelfAttention class.
            hidden_states (Tensor): The input hidden states with shape (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor, optional): The attention mask with shape (batch_size, sequence_length). 
                It is a binary mask where 1's indicate the positions to attend and 0's indicate the positions to
                ignore. Defaults to None.
            head_mask (Tensor, optional): The head mask with shape (num_heads,) or (num_layers, num_heads). 
                It masks the attention weights of specific heads. Defaults to None.
            output_attentions (bool, optional): Whether to output the attention probabilities. Defaults to False.
            rel_pos (Tensor, optional): The relative position bias with shape 
                (num_heads, sequence_length, sequence_length). It contains relative position information between 
                each token pair. Defaults to None.
            rel_2d_pos (Tensor, optional): The relative 2D position bias with shape 
                (num_heads, sequence_length, sequence_length). It contains relative 2D position information 
                between each token pair. Defaults to None.

        Returns:
            tuple: A tuple containing the context layer and attention probabilities 
                if output_attentions is True, otherwise only the context layer.
                
                - context_layer (Tensor): The output context layer with shape (batch_size, sequence_length, hidden_size).
                - attention_probs (Tensor, optional): The attention probabilities with shape 
                (batch_size, num_heads, sequence_length, sequence_length) if output_attentions is True.

        Raises:
            None
        """
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        attention_scores = ops.masked_fill(
            attention_scores.astype(mindspore.float32), ops.stop_gradient(attention_mask.astype(mindspore.bool_)),
            float("-1e10")
        )
        attention_probs = ops.softmax(attention_scores, axis=-1, dtype=mindspore.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LayoutLMv2Attention(nn.Cell):
    """
    LayoutLMv2Attention is the attention layer for LayoutLMv2. It is based on the implementation of
    """
    def __init__(self, config):
        """
        Initialize the LayoutLMv2Attention class.

        Args:
            self (LayoutLMv2Attention): The instance of the LayoutLMv2Attention class.
            config: Represents the configuration settings for the LayoutLMv2Attention instance.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None,
    ):
        """
        This method 'construct' is defined in the class 'LayoutLMv2Attention' and is responsible for
        constructing the attention mechanism in the LayoutLMv2 model.

        Args:
            self (LayoutLMv2Attention): The instance of the LayoutLMv2Attention class.
            hidden_states (torch.Tensor): The input hidden states to the attention mechanism.
            attention_mask (torch.Tensor, optional): Mask to prevent attention to certain positions. Default is None.
            head_mask (torch.Tensor, optional): Mask to hide certain heads of the attention mechanism. Default is None.
            output_attentions (bool): Whether to output attentions weights. Default is False.
            rel_pos (torch.Tensor, optional): Relative position encoding. Default is None.
            rel_2d_pos (torch.Tensor, optional): 2D relative position encoding. Default is None.

        Returns:
            tuple: A tuple containing the attention output and additional outputs from the attention mechanism.

        Raises:
            None
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LayoutLMv2SelfOutput(nn.Cell):
    """
    LayoutLMv2SelfOutput is the output layer for LayoutLMv2Attention. It is based on the implementation of BertSelfOutput.
    """
    def __init__(self, config):
        """
        Initializes the LayoutLMv2SelfOutput class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration parameters.

                - hidden_size (int): The size of the hidden state.
                - layer_norm_eps (float): The epsilon value for LayerNorm.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        Constructs the self-attention output of the LayoutLMv2 transformer model.

        Args:
            self (LayoutLMv2SelfOutput): An instance of the LayoutLMv2SelfOutput class.
            hidden_states (torch.Tensor): The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
                These are the intermediate outputs of the self-attention layer.
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the input embeddings to the self-attention layer.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the constructed self-attention output.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->LayoutLMv2
class LayoutLMv2Intermediate(nn.Cell):
    """
    LayoutLMv2Intermediate is a simple feedforward network. It is based on the implementation of BertIntermediate.
    """
    def __init__(self, config):
        """
        Initialize the LayoutLMv2Intermediate class.

        Args:
            self (object): The current instance of the class.
            config (object): An object containing configuration parameters for the intermediate layer.
                It must have the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str or function): The activation function for the hidden layer.
                If a string, it should be a key in the ACT2FN dictionary.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided.
            ValueError: If the config parameter does not contain the required attributes.
            KeyError: If the hidden activation function specified in the config parameter
                is not found in the ACT2FN dictionary.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method 'construct' in the class 'LayoutLMv2Intermediate'.

        Args:
            self: LayoutLMv2Intermediate object.
                Represents the instance of the LayoutLMv2Intermediate class.
            hidden_states: mindspore.Tensor.
                Input tensor containing hidden states that need to be processed.

        Returns:
            mindspore.Tensor.
                Processed hidden states returned after passing through the dense layer
                and intermediate activation function.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutLM
class LayoutLMv2Output(nn.Cell):
    """
    LayoutLMv2Output is the output layer for LayoutLMv2Intermediate. It is based on the implementation of BertOutput.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LayoutLMv2Output class.

        Args:
            self: The instance of the LayoutLMv2Output class.
            config: An object containing configuration parameters for the LayoutLMv2Output model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameters do not meet the required constraints.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the LayoutLMv2Output for the given hidden states and input tensor.

        Args:
            self (LayoutLMv2Output): An instance of the LayoutLMv2Output class.
            hidden_states (mindspore.Tensor): A tensor representing the hidden states.
                This tensor is expected to have a shape of (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): A tensor representing the input.
                This tensor is expected to have the same shape as the hidden states.

        Returns:
            mindspore.Tensor: A tensor representing the constructed LayoutLMv2Output.
                This tensor has the same shape as the hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLMv2Layer(nn.Cell):
    """
    LayoutLMv2Layer is made up of self-attention and feedforward network. It is based on the implementation of BertLayer.
    """
    def __init__(self, config):
        """Initialize a LayoutLMv2Layer.

        Args:
            self: Instance of the LayoutLMv2Layer class.
            config:
                Configuration object containing parameters for the layer initialization.

                - Type: object
                - Purpose: To configure the layer with specific settings.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None

        Raises:
            TypeError: If the config parameter is not of the expected type.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None,
    ):
        """
        Constructs a LayoutLMv2Layer by applying the attention mechanism and feed-forward neural network to
        the input hidden states.

        Args:
            self: An instance of the LayoutLMv2Layer class.
            hidden_states (torch.Tensor): The input hidden states of shape `(batch_size, sequence_length, hidden_size)`.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape `(batch_size, sequence_length)`.
                Defaults to None.
            head_mask (torch.Tensor, optional): The tensor to mask selected heads of the multi-head attention module.
                Defaults to None.
            output_attentions (bool, optional): Whether to output the attention weights. Defaults to False.
            rel_pos (torch.Tensor, optional): The tensor of relative position encoding of shape
                `(batch_size, num_heads, sequence_length, sequence_length)`. Defaults to None.
            rel_2d_pos (torch.Tensor, optional): The tensor of 2D relative position encoding of shape
                `(batch_size, num_heads, sequence_length, sequence_length, 2)`. Defaults to None.

        Returns:
            outputs (tuple):
                A tuple of the following tensors:

                - layer_output (torch.Tensor): The output tensor of shape `(batch_size, sequence_length, hidden_size)`.
                - attention_weights (torch.Tensor, optional): The attention weights tensor of shape
                `(batch_size, num_heads, sequence_length, sequence_length)`. Only returned if `output_attentions=True`.

        Raises:
            None.
        """
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Performs a feed forward operation on the given attention output in the LayoutLMv2Layer.

        Args:
            self (LayoutLMv2Layer): An instance of the LayoutLMv2Layer class.
            attention_output: The attention output tensor to be processed.
                It should have shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method modifies the internal state of the LayoutLMv2Layer instance.

        Raises:
            None.

        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    '''Calculate the relative position bucket.

    Args:
        relative_position (mindspore.Tensor): A tensor containing the relative position.
        bidirectional (bool): A boolean flag indicating whether to use bidirectional buckets (default: True).
        num_buckets (int): An integer specifying the number of buckets to use (default: 32).
        max_distance (int): An integer representing the maximum distance to bucket (default: 128).

    Returns:
        mindspore.Tensor: A tensor containing the calculated relative position bucket.

    Raises:
        TypeError: If the input tensor 'relative_position' is not a valid tensor.
        ValueError: If the 'num_buckets' or 'max_distance' values are less than or equal to zero.
    '''
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).astype(mindspore.int64) * num_buckets
        n = ops.abs(relative_position)
    else:
        n = ops.maximum(
            -relative_position, ops.zeros_like(relative_position)
        )  # to be confirmed
    # Now n is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            ops.log(n.astype(mindspore.float32) / max_exact) / math.log(max_distance / max_exact) * (
            num_buckets - max_exact)
    ).astype(mindspore.int64)

    val_if_large = ops.minimum(
        val_if_large, ops.full_like(val_if_large, num_buckets - 1)
    )

    ret += ops.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2Encoder(nn.Cell):
    """
    LayoutLMv2Encoder is a stack of LayoutLMv2Layer. It is based on the implementation of BertEncoder.
    """
    def __init__(self, config):
        '''
        Initializes a LayoutLMv2Encoder object.

        Args:
            config (object): The configuration object containing the parameters for the LayoutLMv2Encoder.
                It is used to initialize various attributes of the LayoutLMv2Encoder.

        Returns:
            None.

        Raises:
            None.
        '''
        super().__init__()
        self.config = config
        self.layer = nn.CellList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = nn.Dense(self.rel_pos_bins, config.num_attention_heads, has_bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Dense(self.rel_2d_pos_bins, config.num_attention_heads, has_bias=False)
            self.rel_pos_y_bias = nn.Dense(self.rel_2d_pos_bins, config.num_attention_heads, has_bias=False)

        self.gradient_checkpointing = False

    def _calculate_1d_position_embeddings(self, position_ids):
        """
        This method calculates 1D position embeddings for the LayoutLMv2Encoder.

        Args:
            self (LayoutLMv2Encoder): The instance of the LayoutLMv2Encoder class.
            position_ids (torch.Tensor): A 1D tensor representing the position IDs of tokens.
                It is used to calculate the relative position embeddings.
                Expected to be a tensor of shape (batch_size,) with integer values representing the position IDs.

        Returns:
            None: This method does not return a value. It updates the internal state of the LayoutLMv2Encoder instance
                to store the calculated relative position embeddings.

        Raises:
            RuntimeError: If the input position_ids tensor is not a torch.Tensor or has an incorrect shape.
            ValueError: If the number of buckets specified for relative position bucketing (rel_pos_bins) is less than 1.
            ValueError: If the max_distance for relative position bucketing (max_rel_pos) is less than 1.
        """
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        return rel_pos

    def _calculate_2d_position_embeddings(self, bbox):
        """
        Method to calculate 2D position embeddings based on the given bounding box.

        Args:
            self (LayoutLMv2Encoder): The instance of the LayoutLMv2Encoder class.
            bbox (torch.Tensor): A 3D tensor representing the bounding box coordinates with shape
                (batch_size, num_boxes, 4). The bounding box tensor contains the x and y coordinates of the top-left
                and bottom-right corners of each box.

        Returns:
            None: This method does not return any value directly.
                It calculates and updates the relative 2D position embeddings.

        Raises:
            None.
        """
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            bbox=None,
            position_ids=None,
    ):
        """
        This method constructs the LayoutLMv2Encoder.

        Args:
            self: The instance of the class LayoutLMv2Encoder.
            hidden_states (Tensor): The input hidden states to the encoder.
            attention_mask (Tensor, optional): Mask to avoid performing attention on padding token indices.
            head_mask (List, optional): Mask for attention heads. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to False.
            return_dict (bool, optional): Whether to return the output as a dictionary. Defaults to True.
            bbox (Tensor, optional): Bounding box coordinates for spatial attention bias. Defaults to None.
            position_ids (Tensor, optional): Position IDs for relative positional embeddings. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If the input parameters are not in the expected format.
            RuntimeError: If an error occurs during the execution of the method.
            IndexError: If there is an issue with accessing elements in the head_mask list.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        rel_pos = self._calculate_1d_position_embeddings(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._calculate_2d_position_embeddings(bbox) if self.has_spatial_attention_bias else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LayoutLMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    _keys_to_ignore_on_load_unexpected = ['num_batches_tracked']
    config_class = LayoutLMv2Config
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlmv2"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(sigma=self.config.initializer_range),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(Tensor(weight, dtype=cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class LayoutLMv2VisualBackbone(nn.Cell):
    """
    LayoutLMv2VisualBackbone is a visual backbone for LayoutLMv2. It is based on the implementation of VisualBackboneBase.
    """
    def __init__(self, config):
        """
        Initializes an instance of the LayoutLMv2VisualBackbone class.

        Args:
            self: The instance of the class itself.
            config: An object that contains configuration parameters.

        Returns:
            None.

        Raises:
            ValueError: If the lengths of the pixel mean and pixel standard deviation in the configuration are not equal.
        """
        super(LayoutLMv2VisualBackbone, self).__init__()
        self.cfg = config.get_detectron2_config()
        self.backbone = build_resnet_fpn_backbone(self.cfg)

        if len(self.cfg.MODEL.PIXEL_MEAN) != len(self.cfg.MODEL.PIXEL_STD):
            raise ValueError(
                "cfg.model.pixel_mean is not equal with cfg.model.pixel_std."
            )
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)

        self.pixel_mean = Parameter(
            mindspore.Tensor(self.cfg.MODEL.PIXEL_MEAN).reshape((num_channels, 1, 1)),
            name="pixel_mean",
            requires_grad=False,
        )
        self.pixel_std = Parameter(
            mindspore.Tensor(self.cfg.MODEL.PIXEL_STD).reshape((num_channels, 1, 1)),
            name="pixel_std",
            requires_grad=False,
        )

        self.out_feature_key = "p2"
        self.pool_shape = tuple(config.image_feature_pool_shape[:2])  # (7,7)
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(
                self.backbone.output_shape()[self.out_feature_key].channels
            )

        input_shape = (224, 224)
        outsize = config.image_feature_pool_shape[0]  # (7,7)
        insize = (input_shape[0] + 4 - 1) // 4
        shape_info = self.backbone.output_shape()[self.out_feature_key]
        channels = shape_info.channels
        stride = insize // outsize
        kernel = insize - (outsize - 1) * stride

        self.weight = mindspore.Tensor(np.ones([channels, 1, kernel, kernel]), dtype=mindspore.float32) / (
                kernel * kernel)
        self.conv2d = ops.Conv2D(channels, kernel, stride=stride, group=channels)

    def pool(self, features):
        """
        To enhance performance, customize the AdaptiveAvgPool2d layer
        """
        features = self.conv2d(features, self.weight)
        return features

    def freeze(self):
        """
        Freeze parameters
        """
        for param in self.trainable_params():
            param.requires_grad = False

    def construct(self, images):
        """
        This method 'construct' is defined within the class 'LayoutLMv2VisualBackbone'
        and is responsible for processing images through the visual backbone network.

        Args:
            self:
                An instance of the 'LayoutLMv2VisualBackbone' class.

                - Type: LayoutLMv2VisualBackbone
                - Purpose: Represents the current instance of the LayoutLMv2VisualBackbone class.

            images:
                The input images to be processed by the visual backbone.

                - Type: N-dimensional array
                - Purpose: Represents the input images for processing.
                - Restrictions: Must be compatible with the model input size.

        Returns:
            features:
                The processed features of the input images after passing through the visual backbone network.

                - Type: Numpy array
                - Purpose: Represents the extracted features from the input images.

        Raises:
            None.
        """
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        for item in features:
            if item[0] == self.out_feature_key:
                features = item[1]
        features = self.pool(features)
        return features.flatten(start_dim=2).transpose(0, 2, 1)


class LayoutLMv2Pooler(nn.Cell):
    """
    LayoutLMv2Pooler is a simple feedforward network. It is based on the implementation of BertPooler.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LayoutLMv2Pooler class.

        Args:
            self (LayoutLMv2Pooler): The current instance of the LayoutLMv2Pooler class.
            config: The configuration object specifying the settings for the LayoutLMv2Pooler.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        """
        Constructs the pooled output tensor for the LayoutLMv2Pooler class.

        Args:
            self: An instance of the LayoutLMv2Pooler class.
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size)
                representing the hidden states of the input sequence.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) representing the pooled output.

        Raises:
            None.

        This method takes the hidden states of the input sequence and applies pooling to obtain a
        pooled output tensor. It first selects the first token tensor from the hidden states tensor
        using slicing, and then passes it through a dense layer. The resulting tensor is then
        activated using the specified activation function. Finally, the pooled output tensor is
        returned.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
    """
    LayoutLMv2Model is a LayoutLMv2 model with a visual backbone. It is based on the implementation of LayoutLMv2Model.
    """
    def __init__(self, config):
        """
        Initializes an instance of the LayoutLMv2Model class.

        Args:
            self: The instance of the LayoutLMv2Model class.
            config:
                A configuration object containing various settings and hyperparameters for the model.

                - Type: dict
                - Purpose: Configure the model with specific settings.
                - Restrictions: Must contain specific keys and values required by the model.

        Returns:
            None.

        Raises:
            ValueError: If the provided configuration is missing required keys or has invalid values.
            TypeError: If the configuration object is not of the expected type.
        """
        super().__init__(config)
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.use_visual_backbone = config.use_visual_backbone
        self.embeddings = LayoutLMv2Embeddings(config)
        if self.use_visual_backbone is True:
            self.visual = LayoutLMv2VisualBackbone(config)
            self.visual_proj = nn.Dense(config.image_feature_pool_shape[-1], config.hidden_size)
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        self.visual_LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.encoder = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2Pooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings of the LayoutLMv2Model.

        Args:
            self: The instance of the LayoutLMv2Model class.

        Returns:
            None: This method returns the input embeddings of the LayoutLMv2Model.
                The input embeddings are of type 'None'.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the LayoutLMv2Model.

        Args:
            self (LayoutLMv2Model): An instance of the LayoutLMv2Model class.
            value: The input embeddings to be set. It should be a tensor or any object that can be assigned to
                the word_embeddings attribute of the embeddings object.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
        """
        Calculates the text embeddings for the LayoutLMv2Model.

        Args:
            self (LayoutLMv2Model): The instance of the LayoutLMv2Model class.
            input_ids (Tensor): The input tensor of shape [batch_size, seq_length] containing the input token IDs.
            bbox (Tensor): The input tensor of shape [batch_size, seq_length, 4]
                containing the bounding box coordinates for each token.
            position_ids (Tensor): The input tensor of shape [batch_size, seq_length]
                containing the positional IDs for each token.
            token_type_ids (Tensor): The input tensor of shape [batch_size, seq_length]
                containing the token type IDs for each token.
            inputs_embeds (Tensor, optional): The optional input tensor of shape [batch_size, seq_length, hidden_size]
                containing pre-computed embeddings.

        Returns:
            Tensor: The resulting tensor of shape [batch_size, seq_length, hidden_size] containing
                the calculated text embeddings.

        Raises:
            MindSporeError: If the input_ids and inputs_embeds tensors have incompatible shapes.
            MindSporeError: If the position_ids and input_ids tensors have incompatible shapes.
            MindSporeError: If the token_type_ids and input_ids tensors have incompatible shapes.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)

        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        """
        Calculate image embeddings for the LayoutLMv2Model.

        Args:
            self (LayoutLMv2Model): The instance of the LayoutLMv2Model class.
            image (numpy.ndarray): The input image for which embeddings need to be calculated.
            bbox (numpy.ndarray): The bounding box coordinates associated with the image.
            position_ids (numpy.ndarray): The position IDs used for positional embeddings.

        Returns:
            The calculated embeddings are stored within the class instance.

        Raises:
            ValueError: If the image is None and visual backbone is required.
            TypeError: If the image data type cannot be converted to 'mindspore.float32'.
            AssertionError: If an unexpected condition occurs while calculating embeddings.
        """
        use_image_info = self.use_visual_backbone and image is not None
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(
            bbox
        )
        if use_image_info:
            visual_embeddings = self.visual_proj(self.visual(image.astype(mindspore.float32)))
            embeddings = (
                    visual_embeddings + position_embeddings + spatial_position_embeddings
            )
        else:
            embeddings = position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, visual_shape):
        '''
        Calculate the visual bounding box based on the given image features.

        Args:
            self (LayoutLMv2Model): An instance of the LayoutLMv2Model class.
            image_feature_pool_shape (tuple): The shape of the image feature pool as (y_size, x_size).
            bbox (tensor): The bounding box tensor.
            visual_shape (tuple): The desired shape of the visual bounding box.

        Returns:
            visual_bbox (tensor): The calculated visual bounding box tensor.

        Raises:
            None.
        '''
        x_size = image_feature_pool_shape[1]
        y_size = image_feature_pool_shape[0]
        visual_bbox_x = mindspore.Tensor(
            np.arange(0, 1000 * (x_size + 1), 1000) // x_size, dtype=mindspore.int64
        )
        visual_bbox_y = mindspore.Tensor(
            np.arange(0, 1000 * (y_size + 1), 1000) // y_size, dtype=mindspore.int64
        )
        expand_shape = image_feature_pool_shape[0:2]
        expand_shape = tuple(expand_shape)
        visual_bbox = ops.stack(
            [
                visual_bbox_x[:-1].broadcast_to(expand_shape),
                visual_bbox_y[:-1].broadcast_to(expand_shape[::-1]).transpose((1, 0)),
                visual_bbox_x[1:].broadcast_to(expand_shape),
                visual_bbox_y[1:].broadcast_to(expand_shape[::-1]).transpose((1, 0)),
            ],
            axis=-1,
        ).reshape((expand_shape[0] * expand_shape[1], ops.shape(bbox)[-1]))
        visual_bbox = visual_bbox.broadcast_to(
            (visual_shape[0], visual_bbox.shape[0], visual_bbox.shape[1])
        )
        return visual_bbox

    def _get_input_shape(self, input_ids=None, inputs_embeds=None):
        """
        Returns the shape of the input tensor for the LayoutLMv2Model.

        Args:
            self (LayoutLMv2Model): The instance of the LayoutLMv2Model class.
            input_ids (Optional[torch.Tensor]): The input tensor representing the tokenized input sequence.
                Default: None.
            inputs_embeds (Optional[torch.Tensor]): The input tensor representing the embedded input sequence.
                Default: None.

        Returns:
            torch.Size or Tuple[int]: The shape of the input tensor, excluding the batch size dimension.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.

        Note:
            - It is required to specify either input_ids or inputs_embeds.
            - If input_ids is specified, the shape of the input_ids tensor is returned.
            - If inputs_embeds is specified, the shape of the inputs_embeds tensor,
            excluding the last dimension, is returned.
            - The shape represents the dimensions of the input tensor, excluding the batch size dimension.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            return input_ids.shape
        elif inputs_embeds is not None:
            return inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            bbox: Optional[mindspore.Tensor] = None,
            image: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Return:
            Union[Tuple, BaseModelOutputWithPooling]

        Example:
            ```python
            >>> from transformers import AutoProcessor, LayoutLMv2Model, set_seed
            >>> from PIL import Image
            >>> import torch
            >>> from datasets import load_dataset
            ...
            >>> set_seed(88)
            ...
            >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
            >>> model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")
            ...
            ...
            >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
            >>> image_path = dataset["test"][0]["file"]
            >>> image = Image.open(image_path).convert("RGB")
            ...
            >>> encoding = processor(image, return_tensors="pt")
            ...
            >>> outputs = model(**encoding)
            >>> last_hidden_states = outputs.last_hidden_state
            ...
            >>> last_hidden_states.shape
            ops.Size([1, 342, 768])
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = self._get_input_shape(input_ids, inputs_embeds)

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        # visual_shape = ops.Size(visual_shape)
        # needs a new copy of input_shape for tracing. Otherwise wrong dimensions will occur
        final_shape = list(self._get_input_shape(input_ids, inputs_embeds))
        final_shape[1] += visual_shape[1]
        # final_shape = ops.Size(final_shape)

        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, final_shape)
        final_bbox = ops.cat([bbox, visual_bbox], axis=1)

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)

        visual_attention_mask = ops.ones(tuple(visual_shape), dtype=mindspore.float32)
        attention_mask = attention_mask.astype(visual_attention_mask.dtype)
        final_attention_mask = ops.cat([attention_mask, visual_attention_mask], axis=1)

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.broadcast_to(input_shape)

        visual_position_ids = mindspore.Tensor(np.arange(0, visual_shape[1])).broadcast_to(
            (input_shape[0], visual_shape[1])
        )
        position_ids = position_ids.astype(visual_position_ids.dtype)
        final_position_ids = ops.cat([position_ids, visual_position_ids], axis=1)

        if bbox is None:
            bbox = ops.zeros(tuple(list(input_shape) + [4]), dtype=mindspore.int64)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = ops.cat([text_layout_emb, visual_emb], axis=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * mindspore.tensor(
            np.finfo(mindspore.dtype_to_nptype(self.dtype)).min)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.broadcast_to(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask_dtype = next(iter(self.parameters_dict().items()))[1].dtype
            head_mask = head_mask.to(dtype=head_mask_dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):
    """
    LayoutLMv2ForSequenceClassification is a LayoutLMv2 model with a sequence classification head on top (a linear
    layer on top of the pooled output) It is based on the implementation of LayoutLMv2ForSequenceClassification.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LayoutLMv2ForSequenceClassification class.

        Args:
            self: The object instance.
            config:
                An instance of the LayoutLMv2Config class containing the configuration parameters for the model.

                - Type: LayoutLMv2Config
                - Purpose: Specifies the model's configuration parameters.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size * 3, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the LayoutLMv2 model for sequence classification.

        Args:
            self: LayoutLMv2ForSequenceClassification object.
                Represents the instance of the LayoutLMv2ForSequenceClassification class.

        Returns:
            None: This method returns None as it simply retrieves the input embeddings without any additional processing.

        Raises:
            None.
        """
        return self.layoutlmv2.embeddings.word_embeddings

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            bbox: Optional[mindspore.Tensor] = None,
            image: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[Tuple, SequenceClassifierOutput]

        Example:
            ```python
            >>> from transformers import AutoProcessor, LayoutLMv2ForSequenceClassification, set_seed
            >>> from PIL import Image
            >>> import torch
            >>> from datasets import load_dataset
            ...
            >>> set_seed(88)
            ...
            >>> dataset = load_dataset("rvl_cdip", split="train", streaming=True)
            >>> data = next(iter(dataset))
            >>> image = data["image"].convert("RGB")
            ...
            >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
            >>> model = LayoutLMv2ForSequenceClassification.from_pretrained(
            ...     "microsoft/layoutlmv2-base-uncased", num_labels=dataset.info.features["label"].num_classes
            ... )
            ...
            >>> encoding = processor(image, return_tensors="pt")
            >>> sequence_label = torch.tensor([data["label"]])
            ...
            >>> outputs = model(**encoding, labels=sequence_label)
            ...
            >>> loss, logits = outputs.loss, outputs.logits
            >>> predicted_idx = logits.argmax(axis=-1).item()
            >>> predicted_answer = dataset.info.features["label"].names[4]
            >>> predicted_idx, predicted_answer
            (4, 'advertisement')
            ```
        """
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

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]

        visual_bbox = self.layoutlmv2._calc_visual_bbox(
            self.config.image_feature_pool_shape, bbox, final_shape
        )

        visual_position_ids = ops.arange(0, visual_shape[1], dtype=mindspore.int64).repeat(
            input_shape[0], 1
        )

        initial_image_embeddings = self.layoutlmv2._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        sequence_output, final_image_embeddings = outputs[0][:, :seq_length], outputs[0][:, seq_length:]

        cls_final_output = sequence_output[:, 0, :]

        # average-pool the visual embeddings
        pooled_initial_image_embeddings = initial_image_embeddings.mean(axis=1)
        pooled_final_image_embeddings = final_image_embeddings.mean(axis=1)
        # concatenate with cls_final_output
        sequence_output = ops.cat(
            [cls_final_output, pooled_initial_image_embeddings, pooled_final_image_embeddings], axis=1
        )
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).astype(mindspore.int32))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
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


class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    """
    LayoutLMv2ForTokenClassification is a LayoutLMv2 model with a token classification head.
    It is based on the implementation of LayoutLMv2ForTokenClassification.
    """
    def __init__(self, config):
        """
        Initializes a LayoutLMv2ForTokenClassification instance.

        Args:
            self (LayoutLMv2ForTokenClassification): The instance of the LayoutLMv2ForTokenClassification class.
            config:
                An object containing the configuration settings for the LayoutLMv2 model.

                - Type: LayoutLMv2Config
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: Must be an instance of LayoutLMv2Config.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not an instance of LayoutLMv2Config.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings for LayoutLMv2ForTokenClassification.

        Args:
            self: An instance of the LayoutLMv2ForTokenClassification class.

        Returns:
            None: The method returns the input embeddings for the LayoutLMv2ForTokenClassification.

        Raises:
            None.
        """
        return self.layoutlmv2.embeddings.word_embeddings

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            bbox: Optional[mindspore.Tensor] = None,
            image: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:
            Union[Tuple, TokenClassifierOutput]

        Example:
            ```python
            >>> from transformers import AutoProcessor, LayoutLMv2ForTokenClassification, set_seed
            >>> from PIL import Image
            >>> from datasets import load_dataset
            ...
            >>> set_seed(88)
            ...
            >>> datasets = load_dataset("nielsr/funsd", split="test")
            >>> labels = datasets.features["ner_tags"].feature.names
            >>> id2label = {v: k for v, k in enumerate(labels)}
            ...
            >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
            >>> model = LayoutLMv2ForTokenClassification.from_pretrained(
            ...     "microsoft/layoutlmv2-base-uncased", num_labels=len(labels)
            ... )
            ...
            >>> data = datasets[0]
            >>> image = Image.open(data["image_path"]).convert("RGB")
            >>> words = data["words"]
            >>> boxes = data["bboxes"]  # make sure to normalize your bounding boxes
            >>> word_labels = data["ner_tags"]
            >>> encoding = processor(
            ...     image,
            ...     words,
            ...     boxes=boxes,
            ...     word_labels=word_labels,
            ...     padding="max_length",
            ...     truncation=True,
            ...     return_tensors="pt",
            ... )
            ...
            >>> outputs = model(**encoding)
            >>> logits, loss = outputs.logits, outputs.loss
            ...
            >>> predicted_token_class_ids = logits.argmax(-1)
            >>> predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids[0]]
            >>> predicted_tokens_classes[:5]
            ['B-ANSWER', 'B-HEADER', 'B-HEADER', 'B-HEADER', 'B-HEADER']
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).astype(mindspore.int32))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMv2ForQuestionAnswering(LayoutLMv2PreTrainedModel):
    """

    LayoutLMv2ForQuestionAnswering is a LayoutLMv2 model with a question answering head.
    It is based on the implementation of LayoutLMv2ForQuestionAnswering.
    """
    def __init__(self, config, has_visual_segment_embedding=True):
        """
        Initialize the LayoutLMv2ForQuestionAnswering class.

        Args:
            self (LayoutLMv2ForQuestionAnswering): The object instance of the LayoutLMv2ForQuestionAnswering class.
            config (LayoutLMv2Config): The configuration object for the LayoutLMv2 model.
            has_visual_segment_embedding (bool, optional): A boolean flag indicating whether visual segment embedding
                is enabled. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        config.has_visual_segment_embedding = has_visual_segment_embedding
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from LayoutLMv2 model for question answering.

        Args:
            self (LayoutLMv2ForQuestionAnswering): The instance of the LayoutLMv2ForQuestionAnswering class.
                This parameter represents the current instance of the LayoutLMv2ForQuestionAnswering class
                where the method is called. It is used to access the model's embeddings to retrieve the input embeddings.

        Returns:
            None: This method does not return any value. It simply returns the word embeddings from the LayoutLMv2 model.

        Raises:
            None
        """
        return self.layoutlmv2.embeddings.word_embeddings

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            bbox: Optional[mindspore.Tensor] = None,
            image: Optional[mindspore.Tensor] = None,
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
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.

        Returns:
            Union[Tuple, QuestionAnsweringModelOutput]

        Example:
            In this example below, we give the LayoutLMv2 model an image (of texts) and ask it a question. It will give us
            a prediction of what it thinks the answer is (the span of the answer within the texts parsed from the image).
            ```python
            >>> from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering, set_seed
            >>> import torch
            >>> from PIL import Image
            >>> from datasets import load_dataset
            ...
            >>> set_seed(88)
            >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
            >>> model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")
            ...
            >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
            >>> image_path = dataset["test"][0]["file"]
            >>> image = Image.open(image_path).convert("RGB")
            >>> question = "When is coffee break?"
            >>> encoding = processor(image, question, return_tensors="pt")
            ...
            >>> outputs = model(**encoding)
            >>> predicted_start_idx = outputs.start_logits.argmax(-1).item()
            >>> predicted_end_idx = outputs.end_logits.argmax(-1).item()
            >>> predicted_start_idx, predicted_end_idx
            (154, 287)
            >>> predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
            >>> predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)
            >>> predicted_answer  # results are not very good without further fine-tuning
            'council mem - bers conducted by trrf treasurer philip g. kuehn to get answers which the public ...
            ```

            ```python
            >>> target_start_index = torch.tensor([7])
            >>> target_end_index = torch.tensor([14])
            >>> outputs = model(**encoding, start_positions=target_start_index, end_positions=target_end_index)
            >>> predicted_answer_span_start = outputs.start_logits.argmax(-1).item()
            >>> predicted_answer_span_end = outputs.end_logits.argmax(-1).item()
            >>> predicted_answer_span_start, predicted_answer_span_end
            (154, 287)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]

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

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_positions = start_positions.astype(mindspore.int32)
            start_loss = loss_fct(start_logits, start_positions)
            end_positions = end_positions.astype(mindspore.int32)
            end_loss = loss_fct(end_logits, end_positions)
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
    "LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST",
    "LayoutLMv2ForQuestionAnswering",
    "LayoutLMv2ForSequenceClassification",
    "LayoutLMv2ForTokenClassification",
    "LayoutLMv2Layer",
    "LayoutLMv2Model",
    "LayoutLMv2PreTrainedModel",
]
