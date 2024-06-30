# coding=utf-8
# Copyright 2023-present NAVER Corp, The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
""" PyTorch Bros model."""


import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Normal

from mindnlp.modules.functional import finfo
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ....utils import (
    ModelOutput,
    logging,
)
from .configuration_bros import BrosConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "jinho8345/bros-base-uncased"
_CONFIG_FOR_DOC = "BrosConfig"


@dataclass
class BrosSpadeOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        initial_token_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores for entity initial tokens (before SoftMax).
        subsequent_token_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, sequence_length+1)`):
            Classification scores for entity sequence tokens (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    initial_token_logits: mindspore.Tensor = None
    subsequent_token_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class BrosPositionalEmbedding1D(nn.Cell):

    """
    Represents a 1D positional embedding for Bros.
    
    This class inherits from nn.Cell and is used to create a 1D positional embedding for Bros.
    
    The positional embedding is constructed based on the input configuration and positional sequence.
    The constructor initializes the dimensional parameters for the positional embedding, and the construct
    method generates the positional embedding based on the given positional sequence.

    Attributes:
        dim_bbox_sinusoid_emb_1d (int): The dimensional parameter for the sinusoidal positional embedding.
        inv_freq (Tensor): The inverse frequency used in constructing the positional embedding.

    Methods:
        construct: Constructs the positional embedding based on the given positional sequence.

    Example:
        ```python
        >>> # Create a BrosPositionalEmbedding1D instance
        >>> config = Config()
        >>> bros_positional_embedding = BrosPositionalEmbedding1D(config)
        >>> pos_sequence = Tensor([1, 2, 3])
        >>> pos_embedding = bros_positional_embedding.construct(pos_sequence)
        ```
    """
    # Reference: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15

    def __init__(self, config):
        """
        Initializes an instance of the BrosPositionalEmbedding1D class.

        Args:
            self: The current instance of the class.
            config:
                A configuration object containing parameters for the initialization.

                - Type: Any
                - Purpose: Provides configuration options for the BrosPositionalEmbedding1D instance.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super(BrosPositionalEmbedding1D, self).__init__()

        self.dim_bbox_sinusoid_emb_1d = config.dim_bbox_sinusoid_emb_1d

        inv_freq = 1 / (
            10000 ** (ops.arange(0.0, self.dim_bbox_sinusoid_emb_1d, 2.0) / self.dim_bbox_sinusoid_emb_1d)
        )
        self.inv_freq = inv_freq

    def construct(self, pos_seq: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a positional embedding for 1D bounding box position sequences.

        Args:
            self: The instance of the BrosPositionalEmbedding1D class.
            pos_seq (mindspore.Tensor): The input tensor containing position sequences.
                It should have the shape (batch_size, sequence_length, feature_dim).

        Returns:
            mindspore.Tensor: A tensor representing the positional embedding of the input position sequences.
                The shape of the returned tensor will be (batch_size, sequence_length, feature_dim).

        Raises:
            None
        """
        seq_size = pos_seq.shape
        b1, b2, b3 = seq_size
        sinusoid_inp = pos_seq.view(b1, b2, b3, 1) * self.inv_freq.view(1, 1, 1, self.dim_bbox_sinusoid_emb_1d // 2)
        pos_emb = ops.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], axis=-1)
        return pos_emb


class BrosPositionalEmbedding2D(nn.Cell):

    """
    BrosPositionalEmbedding2D represents a 2D positional embedding operation for bounding boxes.
    This class inherits from nn.Cell and is used to create positional embeddings for 2D spatial information.

    Parameters:
        config (dict): A dictionary containing configuration parameters for the positional embedding.
        dim_bbox (int): The dimension of the bounding box.
        x_pos_emb (BrosPositionalEmbedding1D): An instance of BrosPositionalEmbedding1D for the x-axis positional embedding.
        y_pos_emb (BrosPositionalEmbedding1D): An instance of BrosPositionalEmbedding1D for the y-axis positional embedding.

    Methods:
        construct: Constructs the positional embedding for the input bounding box tensor.

            Args:

            - bbox (mindspore.Tensor): Input bounding box tensor.

            Returns:

            - mindspore.Tensor: Positional embedding tensor for the input bounding box.

    Note:
        - The construct method iterates over the dimensions of the bounding box and applies positional embedding based on the dimension index.
        - The positional embeddings for x and y axes are created separately using BrosPositionalEmbedding1D instances and concatenated along the last axis.
    """
    def __init__(self, config):
        """
        Initializes an instance of BrosPositionalEmbedding2D.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing parameters for initialization.

                - Type: object
                - Purpose: Configures the positional embedding.
                - Restrictions: Must contain the 'dim_bbox' attribute.

        Returns:
            None.

        Raises:
            None.
        """
        super(BrosPositionalEmbedding2D, self).__init__()

        self.dim_bbox = config.dim_bbox
        self.x_pos_emb = BrosPositionalEmbedding1D(config)
        self.y_pos_emb = BrosPositionalEmbedding1D(config)

    def construct(self, bbox: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a positional embedding for a 2D bounding box.

        Args:
            self (BrosPositionalEmbedding2D): An instance of the BrosPositionalEmbedding2D class.
            bbox (mindspore.Tensor): A tensor representing the bounding box.
                The shape of the tensor should be (..., dim_bbox) where dim_bbox is the number of dimensions in the bounding box.
                The tensor should contain the coordinates of the bounding box, with each dimension represented by two consecutive elements in the last dimension of the tensor.
                For example, if dim_bbox is 4, the last dimension of the tensor should contain [x1, y1, x2, y2] coordinates of the bounding box.

        Returns:
            mindspore.Tensor: A tensor representing the positional embedding of the bounding box.
                The shape of the tensor will be the same as the input tensor, except for the last dimension which will be doubled.
                The tensor will contain the positional embeddings of each dimension of the bounding box, with each dimension represented by two consecutive elements in the last dimension of the tensor.
                For example, if dim_bbox is 4, the last dimension of the tensor will contain [x1_emb, y1_emb, x2_emb, y2_emb] positional embeddings of the bounding box.

        Raises:
            None.
        """
        stack = []
        for i in range(self.dim_bbox):
            if i % 2 == 0:
                stack.append(self.x_pos_emb(bbox[..., i]))
            else:
                stack.append(self.y_pos_emb(bbox[..., i]))
        bbox_pos_emb = ops.cat(stack, axis=-1)
        return bbox_pos_emb


class BrosBboxEmbeddings(nn.Cell):

    """
    The BrosBboxEmbeddings class represents a neural network cell for generating positional embeddings for bounding boxes.
    It inherits from the nn.Cell class.

    This class initializes with a configuration object and sets up the necessary components for generating positional embeddings for bounding boxes.
    It includes a positional embedding layer and a projection layer for processing the positional embeddings.

    The construct method takes a tensor representing bounding boxes as input and performs operations to generate positional embeddings for the bounding boxes.
    It first processes the input tensor and then applies the positional embedding and projection layers to obtain the positional embeddings for the bounding boxes.

    This class provides functionality for generating positional embeddings specifically tailored for bounding boxes, which can be used in various machine learning applications.
    """
    def __init__(self, config):
        """
        Initializes a BrosBboxEmbeddings instance.

        Args:
            self (BrosBboxEmbeddings): The instance of the BrosBboxEmbeddings class.
            config: The configuration object containing parameters for initializing the BrosBboxEmbeddings instance.
                This parameter is of a custom type and should include necessary fields for initialization.
                It is required for setting up the positional embeddings and projections within the BrosBboxEmbeddings instance.

        Returns:
            None.

        Raises:
            None.
        """
        super(BrosBboxEmbeddings, self).__init__()
        self.bbox_sinusoid_emb = BrosPositionalEmbedding2D(config)
        self.bbox_projection = nn.Dense(config.dim_bbox_sinusoid_emb_2d, config.dim_bbox_projection, has_bias=False)

    def construct(self, bbox: mindspore.Tensor):
        """
        Constructs the bounding box embeddings for the BrosBboxEmbeddings class.

        Args:
            self (BrosBboxEmbeddings): The instance of the BrosBboxEmbeddings class.
            bbox (mindspore.Tensor): The input tensor containing the bounding box coordinates.
                It should have a shape of (N, 4), where N is the number of bounding boxes.
                The bounding box coordinates should be in the format [xmin, ymin, xmax, ymax].

        Returns:
            mindspore.Tensor: The output tensor representing the calculated bounding box embeddings.
                It has a shape of (N, N, C), where N is the number of bounding boxes and C is the embedding dimension.

        Raises:
            None.
        """
        bbox_t = bbox.swapaxes(0, 1)
        bbox_pos = bbox_t[None, :, :, :] - bbox_t[:, None, :, :]
        bbox_pos_emb = self.bbox_sinusoid_emb(bbox_pos)
        bbox_pos_emb = self.bbox_projection(bbox_pos_emb)

        return bbox_pos_emb


class BrosTextEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes the BrosTextEmbeddings class with the given configuration.

        Args:
            self: An instance of the BrosTextEmbeddings class.
            config:
                A configuration object containing various parameters for setting up the embeddings.

                - Type: object
                - Purpose: Specifies the configuration settings for the embeddings.
                - Restrictions: Must contain specific attributes such as vocab_size, hidden_size, max_position_embeddings,
                    type_vocab_size, pad_token_id, layer_norm_eps, hidden_dropout_prob, position_embedding_type.

        Returns:
            None.

        Raises:
            AttributeError: If the config object does not contain the required attributes for setting up the embeddings.
            ValueError: If there are any issues with the configuration values provided.
        """
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).reshape(1, -1)
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> mindspore.Tensor:
        """
        This method constructs text embeddings based on the provided input data and optional parameters.

        Args:
            self (BrosTextEmbeddings): The instance of the BrosTextEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing tokenized input data. Defaults to None.
            token_type_ids (Optional[mindspore.Tensor]): The tensor containing token type information. Defaults to None.
            position_ids (Optional[mindspore.Tensor]): The tensor containing positional information. Defaults to None.
            inputs_embeds (Optional[mindspore.Tensor]): The tensor containing input embeddings. Defaults to None.
            past_key_values_length (int): The length of past key values. Defaults to 0.

        Returns:
            mindspore.Tensor: The constructed text embeddings represented as a tensor.

        Raises:
            ValueError: If the input shape is invalid or if an operation on the tensors fails.
            TypeError: If the input data types are not compatible with the expected types.
            RuntimeError: If an unexpected error occurs during the construction process.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BrosSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism for the Bros model.
    It is used to calculate attention scores and generate context layers based on input hidden states.

    The BrosSelfAttention class inherits from the nn.Cell class and implements the following methods:

    - __init__: Initializes the BrosSelfAttention instance with the given configuration.
    It checks if the hidden size is a multiple of the number of attention heads and sets up the required
    layers and embeddings.

    - swapaxes_for_scores:
    Swaps axes of the input tensor to prepare it for calculating attention scores. Returns the modified tensor.

    - construct: Constructs the self-attention mechanism. It takes hidden states, positional embeddings, attention masks,
    and other optional inputs as arguments. Returns a tuple containing the context layer and optional attention scores
    or past key-value pairs.

    Please note that this class is designed to be used as part of the Bros model and assumes specific configurations and dependencies.

    """
    def __init__(self, config):
        """
        Initializes a BrosSelfAttention instance.

        Args:
            self: The instance of the BrosSelfAttention class.
            config: An object containing configuration parameters for the BrosSelfAttention instance.
                It should have attributes:

                - hidden_size (int): The size of the hidden state.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (int, optional): The size of the embedding.
                If not provided, the hidden size should be a multiple of the number of attention heads.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size is not a multiple of the number of attention heads and the config does not have the 'embedding_size' attribute.
            AttributeError: If the config does not have the 'embedding_size' attribute when the hidden_size is not a multiple of the number of attention heads.
            AttributeError: If the position_embedding_type attribute in the config is 'relative_key' or 'relative_key_query', and the config does not have the 'max_position_embeddings' attribute.
            AttributeError: If the position_embedding_type attribute in the config is 'relative_key' or 'relative_key_query', and the config does not have the 'is_decoder' attribute.
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

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def swapaxes_for_scores(self, x: mindspore.Tensor):
        """
        Performs axis swapping and reshaping operations on a given tensor for self-attention scoring in the BrosSelfAttention class.

        Args:
            self (BrosSelfAttention): The instance of the BrosSelfAttention class.
            x (mindspore.Tensor): The input tensor to be processed. It should have shape (batch_size, seq_length, hidden_size).

        Returns:
            None: The method modifies the input tensor in-place.

        Raises:
            None.

        Description:
            This method reshapes the input tensor 'x' by swapping and permuting its axes to prepare it for self-attention scoring.
            The input tensor 'x' is expected to have shape (batch_size, seq_length, hidden_size),
            where 'batch_size' represents the number of samples in a batch, 'seq_length' represents the length of the sequence,
            and 'hidden_size' represents the size of the hidden dimension.

            The method first calculates the new shape for the tensor after axis swapping and reshaping.
            The new shape is obtained by taking all dimensions of 'x' except the last one, and appending two additional
            dimensions: 'self.num_attention_heads' representing the number of attention heads and 'self.attention_head_size'
            representing the size of each attention head.

            Then, the method reshapes the tensor 'x' using the calculated new shape.
            This results in a tensor with shape (batch_size, seq_length, self.num_attention_heads, self.attention_head_size).

            Finally, the method permutes the axes of the tensor 'x' using the permutation order (0, 2, 1, 3).
            The permutation reorders the axes of the tensor to (batch_size, self.num_attention_heads, seq_length,
            self.attention_head_size).

            This method modifies the input tensor 'x' in-place and does not return any value.
        """
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        bbox_pos_emb: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[mindspore.Tensor] = False,
    ) -> Tuple[mindspore.Tensor]:
        '''
        This method constructs self-attention mechanism for the BrosSelfAttention class.

        Args:
            self (BrosSelfAttention): The instance of the BrosSelfAttention class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            bbox_pos_emb (mindspore.Tensor): The input tensor representing the positional embeddings for bounding boxes.
            attention_mask (Optional[mindspore.Tensor]): An optional input tensor representing the attention mask. Default is None.
            head_mask (Optional[mindspore.Tensor]): An optional input tensor representing the head mask. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional input tensor representing the encoder hidden states. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional input tensor representing the encoder attention mask. Default is None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional input representing the past key value. Default is None.
            output_attentions (Optional[mindspore.Tensor]): An optional input representing the output attentions. Default is False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the context layer and attention probabilities if output_attentions is True, else only the context layer.

        Raises:
            TypeError: If the input types are incorrect.
            ValueError: If the input values are invalid.
        '''
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.swapaxes_for_scores(self.key(encoder_hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))
            key_layer = ops.cat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.cat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            seq_length = hidden_states.shape[1]
            position_ids_l = ops.arange(seq_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(seq_length, dtype=mindspore.int64).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = ops.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)

                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # bbox positional encoding
        batch_size, n_head, seq_length, d_head = query_layer.shape
        bbox_pos_emb = bbox_pos_emb.view(seq_length, seq_length, batch_size, d_head)
        bbox_pos_emb = bbox_pos_emb.permute([2, 0, 1, 3])
        bbox_pos_scores = ops.einsum("bnid,bijd->bnij", (query_layer, bbox_pos_emb))

        attention_scores = attention_scores + bbox_pos_scores

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BrosModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

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

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Bros
class BrosSelfOutput(nn.Cell):

    """
    This class represents a self-output layer for a neural network, providing methods for processing and transforming input tensors.
    It inherits from the nn.Cell class.

    The BrosSelfOutput class initializes with the given configuration and contains methods for constructing the self-output layer.
    The construct method takes hidden_states and input_tensor as input tensors and processes them using dense, dropout,
    and LayerNorm operations to produce the final hidden_states output tensor.

    Detailed descriptions of the methods:

    - __init__: Initializes the BrosSelfOutput with the provided configuration, setting up the dense, LayerNorm, and dropout layers.
    - construct: Processes the input hidden_states tensor using the dense, dropout, and LayerNorm operations, and
    returns the resulting hidden_states tensor.

    Note:
        This docstring provides an overview of the class and its methods.
        For detailed information on the configuration and operations,
        refer to the class implementation and the corresponding neural network framework documentation.
    """
    def __init__(self, config):
        """
        Initializes a BrosSelfOutput instance.

        Args:
            self: The instance of the BrosSelfOutput class.
            config: A configuration object containing the settings for the BrosSelfOutput.
                It includes the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            ValueError: If the hidden_size or layer_norm_eps attributes in the config object are not valid.
            RuntimeError: If there is an issue with initializing the dense, LayerNorm, or dropout layers.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the self-attention output for the BrosSelfOutput class.

        Args:
            self: The instance of the BrosSelfOutput class.
            hidden_states (mindspore.Tensor): The tensor representing the hidden states.
            input_tensor (mindspore.Tensor): The input tensor to be added to the hidden states.

        Returns:
            mindspore.Tensor: The constructed tensor representing the self-attention output.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BrosAttention(nn.Cell):

    """
    This class represents the BrosAttention module, which is a part of the Bros model architecture.
    BrosAttention is responsible for performing self-attention and output computation.

    The BrosAttention class inherits from the nn.Cell class.

    Attributes:
        self (BrosSelfAttention): The self-attention layer responsible for computing self-attention scores.
        output (BrosSelfOutput): The output layer responsible for computing the final attention output.
        pruned_heads (set): A set containing the indices of the pruned attention heads.

    Methods:
        __init__(self, config):
            Initializes a BrosAttention instance.

            Args:

            - config: A configuration object containing the BrosAttention parameters.

        prune_heads(self, heads):
            Prunes the specified attention heads from the model.

           Args:

            - heads: A list of integers representing the indices of the attention heads to be pruned.

        construct(self, hidden_states, bbox_pos_emb, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                  encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            Constructs the BrosAttention module.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states.
            - bbox_pos_emb (mindspore.Tensor): The positional embeddings for bounding boxes.
            - attention_mask (Optional[mindspore.Tensor]): An optional tensor containing attention masks.
            - head_mask (Optional[mindspore.Tensor]): An optional tensor containing head masks.
            - encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor containing encoder hidden states.
            - encoder_attention_mask (Optional[mindspore.Tensor]): An optional tensor containing encoder attention masks.
            - past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple containing past key-value states.
            - output_attentions (Optional[bool]): Whether to output attention scores.

            Returns:

            - Tuple[mindspore.Tensor]: A tuple containing the attention output and other optional outputs.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BrosAttention class.

        Args:
            self (BrosAttention): The instance of the BrosAttention class.
            config: The configuration object containing the necessary parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.self = BrosSelfAttention(config)
        self.output = BrosSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' belongs to the class 'BrosAttention' and is used to perform pruning on the attention heads in a neural network model.

        Args:
            self (object): The instance of the 'BrosAttention' class.
            heads (list): A list of integers representing the attention heads to be pruned from the model. It is used to identify the attention heads to be pruned.

        Returns:
            None:
                This method does not return any value. It operates in place by updating the internal state of the 'BrosAttention' instance.

        Raises:
            ValueError: If the length of the 'heads' list is zero, indicating that no attention heads were specified for pruning.
            TypeError: If the provided 'heads' parameter is not a list.

        Note:
            The method internally modifies the 'query', 'key', 'value', and 'output.dense' attributes of the 'BrosAttention'
            instance by pruning the specified attention heads.
            It also updates the 'num_attention_heads', 'all_head_size', and 'pruned_heads' attributes accordingly.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        bbox_pos_emb: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Constructs the attention mechanism for the BrosAttention class.

        Args:
            self (BrosAttention): The instance of the BrosAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of the attention mechanism.
            bbox_pos_emb (mindspore.Tensor): The positional embeddings for the bounding box positions.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor, if provided. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor, if provided. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder, if provided. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder, if provided. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key-value tensors, if provided. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor and other optional outputs.

        Raises:
            None.
        """
        self_outputs = self.self(
            hidden_states=hidden_states,
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Bros
class BrosIntermediate(nn.Cell):

    """
    This class represents an intermediate layer of a neural network model called BrosIntermediate.
    It inherits from the nn.Cell class.

    Attributes:
        dense (nn.Dense): A fully connected layer that maps the input tensor to the hidden size specified in the configuration.
        intermediate_act_fn (function): An activation function applied to the intermediate hidden states.

    Methods:
        __init__: Initializes the BrosIntermediate instance with the given configuration.
        construct: Constructs the intermediate hidden states of the BrosIntermediate model.

    """
    def __init__(self, config):
        """
        Initializes a BrosIntermediate object with the provided configuration.

        Args:
            self (BrosIntermediate): The instance of the BrosIntermediate class.
            config (object): An object containing configuration parameters for the BrosIntermediate instance.
                It should have attributes including:

                - hidden_size (int): The size of the hidden layers.
                - intermediate_size (int): The size of the intermediate layers.
                - hidden_act (str or function): The activation function for the hidden layers.
                Can be either a string representing a predefined activation function or a custom activation function.

        Returns:
            None.

        Raises:
            TypeError: If the config object does not have the required attributes.
            ValueError: If the config object attributes are invalid or out of range.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method to construct hidden states in the BrosIntermediate class.

        Args:
            self: The instance of the BrosIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states tensor to be processed.
                This tensor represents the hidden states that will undergo transformation.

        Returns:
            mindspore.Tensor: The processed hidden states tensor after passing through the dense layer
                and intermediate activation function. This tensor represents the updated hidden states.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BrosOutput(nn.Cell):

    """
    This class represents an output layer for the Bros Model. It is used to apply a series of transformations
    to the input tensor and produce the final output.

    The BrosOutput class inherits from the nn.Cell class, which is a base class for neural network modules in MindSpore.

    Attributes:
        dense (nn.Dense): A fully connected layer that applies a linear transformation to the input tensor.
        LayerNorm (nn.LayerNorm): A layer normalization operation that normalizes the hidden states.
        dropout (nn.Dropout): A dropout operation that randomly sets elements to zero to prevent overfitting.

    Methods:
        __init__:
            Initializes the BrosOutput instance.

            Args:

            - config (object): An object containing the configuration parameters for the BrosOutput instance.

        construct:
            Applies a series of transformations to the input tensor and returns the final output.

            Args:

            - hidden_states (mindspore.Tensor): The hidden states tensor.
            - input_tensor (mindspore.Tensor): The input tensor.

            Returns:

            - mindspore.Tensor: The output tensor after applying the transformations.

    Note:
        - The dense layer applies a linear transformation to the hidden states tensor.
        - The dropout operation helps prevent overfitting by randomly dropping elements.
        - The LayerNorm operation normalizes the hidden states tensor by calculating the mean and variance.

    Example:
        ```python
        >>> config = BrosConfig()
        >>> output_layer = BrosOutput(config)
        >>> hidden_states = mindspore.Tensor(np.random.randn(32, 64), dtype=mindspore.float32)
        >>> input_tensor = mindspore.Tensor(np.random.randn(32, 64), dtype=mindspore.float32)
        >>> output = output_layer.construct(hidden_states, input_tensor)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the BrosOutput class.

        Args:
            self (BrosOutput): The instance of the BrosOutput class.
            config:
                The configuration object containing parameters for the model.

                - Type: Any
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output tensor by applying a series of transformations to the hidden states and input tensor.

        Args:
            self (BrosOutput): The instance of the BrosOutput class.
            hidden_states (mindspore.Tensor): The tensor representing the hidden states.
                This tensor is processed through dense layers and normalization.
            input_tensor (mindspore.Tensor): The input tensor that is added to the processed hidden_states.
                This tensor is used to adjust the hidden states.

        Returns:
            mindspore.Tensor: The output tensor resulting from the transformation operations.
                This tensor represents the final processed hidden states with the input tensor incorporated.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BrosLayer(nn.Cell):

    """
    This class represents a custom layer implementation, BrosLayer, for a neural network model.
    It includes functionality for self-attention and cross-attention mechanisms, feed-forward processing, and chunking operations.

    Attributes:
        chunk_size_feed_forward (int): The size of chunks to be used in feed-forward processing.
        seq_len_dim (int): The dimension of sequence length.
        attention (BrosAttention): An instance of BrosAttention class for self-attention mechanism.
        is_decoder (bool): A flag indicating whether the layer is used as a decoder model.
        add_cross_attention (bool): A flag indicating whether cross-attention mechanism is added.
        crossattention (BrosAttention): An instance of BrosAttention class for cross-attention mechanism.
        intermediate (BrosIntermediate): An instance of BrosIntermediate class for intermediate processing.
        output (BrosOutput): An instance of BrosOutput class for final output processing.

    Methods:
        construct:
            Constructs the layer by processing the input hidden states with attention mechanisms and feed-forward processing.

        feed_forward_chunk:
            Processes the attention output through intermediate and output layers for feed-forward processing.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BrosLayer class.

        Args:
            self: The object instance.
            config:
                An object of type 'Config' containing the configuration parameters for the BrosLayer.

                - chunk_size_feed_forward (int): The size of chunks used for feed-forward operations.
                - is_decoder (bool): Indicates whether the BrosLayer is used as a decoder model.
                - add_cross_attention (bool): Indicates whether cross attention is added.

        Returns:
            None

        Raises:
            Exception: Raised if the BrosLayer is not used as a decoder model but cross attention is added.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BrosAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise Exception(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BrosAttention(config)
        self.intermediate = BrosIntermediate(config)
        self.output = BrosOutput(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        bbox_pos_emb: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        This method constructs a BrosLayer by processing the input hidden states and additional parameters.

        Args:
            self: The instance of the BrosLayer class.
            hidden_states (mindspore.Tensor): The hidden states input tensor.
            bbox_pos_emb (mindspore.Tensor): The positional embedding tensor for bounding box positions.
            attention_mask (Optional[mindspore.Tensor]): Optional tensor for attention masking.
            head_mask (Optional[mindspore.Tensor]): Optional tensor for head masking.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional tensor for encoder hidden states.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional tensor for encoder attention masking.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional tuple for storing past key and value tensors.
            output_attentions (Optional[bool]): Optional boolean flag to indicate whether to output attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the output tensors after processing the input hidden states.

        Raises:
            Exception:
                If `encoder_hidden_states` are passed while cross-attention layers are not instantiated by setting
                `config.add_cross_attention=True`.
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if hasattr(self, "crossattention"):
                raise Exception(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Method to perform feed forward computation on a chunk of data in the BrosLayer class.

        Args:
            self: Instance of the BrosLayer class. It represents the current object instance.
            attention_output: Input data representing the attention output.
            This parameter is expected to be of a specific format for processing.

        Returns:
            None: This method does not return any value as it modifies the provided data in-place.

        Raises:
            No specific exceptions are raised within this method under normal circumstances.
            However, potential exceptions that could be raised by the underlying operations should be handled accordingly.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BrosEncoder(nn.Cell):

    """
    The BrosEncoder class represents a custom encoder module for processing input data in a neural network.
    It inherits from nn.Cell and contains methods for initializing the encoder and performing the encoding process.

    Attributes:
        config (object): Configuration object containing settings for the encoder.
        layer (nn.CellList): List of BrosLayer instances representing the encoder layers.

    Methods:
        __init__: Initializes the BrosEncoder with the provided configuration.
        construct:
            Constructs the encoder using the specified input tensors and parameters, returning the output hidden states
            and optional additional information based on the provided flags.

    Returns:
        Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        A tuple containing the final hidden state, past key values, hidden states for all layers, self-attentions,
        and cross-attentions if enabled, based on the specified return format.
    """
    def __init__(self, config):
        """
        Initializes a BrosEncoder object with the provided configuration.

        Args:
            self (BrosEncoder): The instance of the BrosEncoder class.
            config (dict): A dictionary containing configuration parameters for the encoder.
                Must include the following key:

                - num_hidden_layers (int): Number of hidden layers in the encoder.

        Returns:
            None.

        Raises:
            AttributeError: If the 'config' parameter is missing the 'num_hidden_layers' key.
            TypeError: If the 'config' parameter is not of type dict.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([BrosLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        bbox_pos_emb: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the BrosEncoder.

        Args:
            self: The BrosEncoder object.
            hidden_states (mindspore.Tensor): The hidden states of the input sequence. Shape: (batch_size, sequence_length, hidden_size).
            bbox_pos_emb (mindspore.Tensor): The positional embeddings for bounding box inputs. Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor indicating which tokens should be attended to. Shape: (batch_size, sequence_length, sequence_length).
            head_mask (Optional[mindspore.Tensor]): The head mask tensor indicating which heads should be masked. Shape: (num_layers, num_attention_heads).
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder outputs. Shape: (batch_size, encoder_sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for the encoder outputs. Shape: (batch_size, sequence_length, encoder_sequence_length).
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key-value tensors for previous time steps. Shape: (num_layers, 2, batch_size, num_attention_heads, sequence_length, hidden_size).
            use_cache (Optional[bool]): Whether to use cache for faster decoding. Default is None.
            output_attentions (Optional[bool]): Whether to output attention tensors. Default is False.
            output_hidden_states (Optional[bool]): Whether to output hidden state tensors. Default is False.
            return_dict (Optional[bool]): Whether to return a dictionary as output. Default is True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]: The output of the BrosEncoder.
                If return_dict is True, a dictionary with the following keys will be returned:

                - last_hidden_state (mindspore.Tensor): The last layer hidden state of the BrosEncoder.
                Shape: (batch_size, sequence_length, hidden_size).
                - past_key_values (Tuple[Tuple[mindspore.Tensor]]): The past key-value tensors for next time steps.
                Shape: (num_layers, 2, batch_size, num_attention_heads, sequence_length, hidden_size).
                - hidden_states (Tuple[mindspore.Tensor]):
                All the hidden states of the BrosEncoder if output_hidden_states is True.
                Shape: (num_layers, batch_size, sequence_length, hidden_size).
                - attentions (Tuple[mindspore.Tensor]):
                All the attention tensors of the BrosEncoder if output_attentions is True.
                Shape: (num_layers, batch_size, num_attention_heads, sequence_length, sequence_length).
                - cross_attentions (Tuple[mindspore.Tensor]):
                All the cross-attention tensors of the BrosEncoder if output_attentions and add_cross_attention are True.
                Shape: (num_layers, batch_size, num_attention_heads, sequence_length, encoder_sequence_length).

            If return_dict is False, a tuple of the above values will be returned.

        Raises:
            None.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    bbox_pos_emb,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    bbox_pos_emb=bbox_pos_emb,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Bros
class BrosPooler(nn.Cell):

    """
    Represents a custom pooling layer named BrosPooler that performs pooling on the input tensor.

    This class inherits from nn.Cell and includes methods for initialization and constructing the pooling layer.

    Attributes:
        dense (nn.Dense): A fully connected layer for the pooling operation.
        activation (nn.Tanh): Activation function applied to the pooled output.

    Methods:
        __init__: Initializes the BrosPooler class with the specified configuration.
        construct: Constructs the pooling layer on the input tensor.

    """
    def __init__(self, config):
        """
        Initializes an instance of BrosPooler.

        Args:
            self: Instance of the BrosPooler class.
            config:
                Configuration object containing parameters for initialization.

                - Type: Custom class
                - Purpose: Specifies the configuration settings for the BrosPooler instance.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None

        Raises:
            TypeError: If the config parameter is not of the correct type.
            ValueError: If the config parameter is missing required attributes.
            RuntimeError: If an issue occurs during initialization.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a pooled output tensor based on the given hidden states.

        Args:
            self (BrosPooler): An instance of the BrosPooler class.
            hidden_states (mindspore.Tensor):
                A tensor containing the hidden states. Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            mindspore.Tensor: A tensor representing the pooled output. Shape: (batch_size, hidden_size)

        Raises:
            None: This method does not raise any exceptions.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BrosRelationExtractor(nn.Cell):

    """
    The BrosRelationExtractor class represents a relation extractor module for processing structured data.
    This class inherits from nn.Cell and implements methods for constructing relation scores based on query and key layers.

    Attributes:
        n_relations (int): Number of relations to consider.
        backbone_hidden_size (int): Size of the hidden layer in the backbone network.
        head_hidden_size (int): Size of the hidden layer in the head network.
        classifier_dropout_prob (float): Dropout probability for the classifier layer.
        drop (nn.Dropout): Dropout layer with specified probability.
        query (nn.Dense): Dense layer for processing query data.
        key (nn.Dense): Dense layer for processing key data.
        dummy_node (Parameter): Parameter representing a dummy node in the network.

    Methods:
        __init__: Constructor method for initializing class attributes.
        construct: Method for constructing relation scores based on query and key layers.

    Example Usage:
        ```python
        >>> config = Configuration(n_relations=5, hidden_size=64, classifier_dropout_prob=0.5)
        >>> relation_extractor = BrosRelationExtractor(config)
        >>> query_layer = mindspore.Tensor(...)
        >>> key_layer = mindspore.Tensor(...)
        >>> relation_scores = relation_extractor.construct(query_layer, key_layer)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the BrosRelationExtractor class.

        Args:
            self (BrosRelationExtractor): The instance of the BrosRelationExtractor class.
            config: 
                A configuration object containing the following attributes:
                
                - n_relations (int): The number of relations.
                - hidden_size (int): The size of the hidden layers in the backbone and head.
                - classifier_dropout_prob (float): The dropout probability for the classifier.

        Returns:
            None.

        Raises:
            ValueError: If n_relations, hidden_size, or classifier_dropout_prob are not valid or if any of the parameters are missing.
            TypeError: If the config parameter is not of the expected type.
            RuntimeError: If there is an issue with initializing the neural network layers.
        """
        super().__init__()
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size
        self.head_hidden_size = config.hidden_size
        self.classifier_dropout_prob = config.classifier_dropout_prob

        self.drop = nn.Dropout(p=self.classifier_dropout_prob)
        self.query = nn.Dense(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        self.key = nn.Dense(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        self.dummy_node = Parameter(ops.zeros(1, self.backbone_hidden_size))

    def construct(self, query_layer: mindspore.Tensor, key_layer: mindspore.Tensor):
        """
        Method to construct a relation score matrix based on query and key layers.

        Args:
            self (BrosRelationExtractor): An instance of BrosRelationExtractor class.
            query_layer (mindspore.Tensor): The input query layer tensor of shape (batch_size, sequence_length, hidden_size).
            key_layer (mindspore.Tensor): The input key layer tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            relation_score (mindspore.Tensor): 
                A tensor representing the relation score matrix of shape (n_relations, batch_size, sequence_length, sequence_length).

        Raises:
            ValueError: If the shapes of query_layer and key_layer are incompatible for matrix multiplication.
            AssertionError: If the dimensions of query_layer and key_layer do not match the expected values.
        """
        query_layer = self.query(self.drop(query_layer))

        dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, key_layer.shape[1], 1)
        key_layer = ops.cat([key_layer, dummy_vec], axis=0)
        key_layer = self.key(self.drop(key_layer))

        query_layer = query_layer.view(
            query_layer.shape[0], query_layer.shape[1], self.n_relations, self.head_hidden_size
        )
        key_layer = key_layer.view(key_layer.shape[0], key_layer.shape[1], self.n_relations, self.head_hidden_size)

        relation_score = ops.matmul(
            query_layer.permute(2, 1, 0, 3), key_layer.permute(2, 1, 3, 0)
        )  # equivalent to ops.einsum("ibnd,jbnd->nbij", (query_layer, key_layer))

        return relation_score


class BrosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BrosConfig
    base_model_prefix = "bros"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.initialize(Normal(self.config.initializer_range))
            if cell.bias is not None:
                cell.bias.initialize('zeros')
        elif isinstance(cell, nn.Embedding):
            cell.weight.initialize(Normal(self.config.initializer_range))
            if cell.padding_idx is not None:
                cell.weight[cell.padding_idx] = 0
        elif isinstance(cell, nn.LayerNorm):
            cell.bias.initialize('zeros')
            cell.weight.initialize('ones')


class BrosModel(BrosPreTrainedModel):

    """
    A BrosModel represents a Bros language model that is used for various natural language processing tasks. 
    It is designed to handle inputs with both text and bounding box information and provides a comprehensive set of 
    functionalities for processing and encoding text data.

    Attributes:
        config: The configuration object that stores the model's hyperparameters and settings.
        embeddings: An instance of BrosTextEmbeddings that handles the word embeddings for the input text.
        bbox_embeddings: An instance of BrosBboxEmbeddings that handles the encoding of bounding box information.
        encoder: An instance of BrosEncoder that performs the main encoding operations on the input.
        pooler: An optional instance of BrosPooler that performs pooling operations on the encoded sequence.

    Methods:
        __init__: Initializes a BrosModel instance with the given configuration.
        get_input_embeddings: Returns the word embeddings used for input text.
        set_input_embeddings: Sets the word embeddings used for input text to the given value.
        _prune_heads: Prunes specific attention heads in the model.
        construct: Constructs the model with the given input and returns the encoded sequence and other optional outputs.

    Example:
        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosModel
        ... 
        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
        ... 
        >>> model = BrosModel.from_pretrained("jinho8345/bros-base-uncased")
        ... 
        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox
        ... 
        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes the BrosModel.

        Args:
            self (BrosModel): The instance of the BrosModel class.
            config (object): The configuration object containing model parameters and settings.
            add_pooling_layer (bool): A flag indicating whether to include a pooling layer in the model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BrosTextEmbeddings(config)
        self.bbox_embeddings = BrosBboxEmbeddings(config)
        self.encoder = BrosEncoder(config)

        self.pooler = BrosPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the BrosModel class.

        Args:
            self: BrosModel instance. The self parameter is a reference to the current instance of the class. 
                It is used to access the attributes and methods of the class within the method.

        Returns:
            None: This method does not return any value explicitly, 
                as it directly returns the input embeddings from the BrosModel class.

        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the BrosModel.

        Args:
            self (BrosModel): The instance of the BrosModel class.
            value (object): The input embeddings value to be set for the BrosModel. 
                It should be of the appropriate type and format compatible with the word_embeddings attribute of the embeddings object.

        Returns:
            None.

        Raises:
            None
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]

        Example:
            ```python
            >>> import torch
            >>> from transformers import BrosProcessor, BrosModel
            ...
            >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> model = BrosModel.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
            >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
            >>> encoding["bbox"] = bbox
            ...
            >>> outputs = model(**encoding)
            >>> last_hidden_states = outputs.last_hidden_state
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if bbox is None:
            raise ValueError("You have to specify bbox")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # if bbox has 2 points (4 float tensors) per token, convert it to 4 points (8 float tensors) per token
        if bbox.shape[-1] == 4:
            bbox = bbox[:, :, [0, 1, 2, 1, 2, 3, 0, 3]]
        scaled_bbox = bbox * self.config.bbox_scale
        bbox_position_embeddings = self.bbox_embeddings(scaled_bbox)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox_pos_emb=bbox_position_embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BrosForTokenClassification(BrosPreTrainedModel):

    """
    BrosForTokenClassification is a class for token classification tasks using the Bros model.
    It inherits from BrosPreTrainedModel and is designed to be used for token classification tasks such as named
    entity recognition or part-of-speech tagging.

    Returns:
        TokenClassifierOutput: A data class that holds the outputs of the BrosForTokenClassification model.

    Example:
        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosForTokenClassification
        ...
        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
        ...
        >>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")
        ...
        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox
        ...
        >>> outputs = model(**encoding)
        ```
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes an instance of the BrosForTokenClassification class.

        Args:
            self (BrosForTokenClassification): The object itself.
            config (BrosConfig): The configuration object containing various settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bros = BrosModel(config)
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        self.init_weights()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        bbox_first_token_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""

        Returns:
            `Union[Tuple[mindspore.Tensor], TokenClassifierOutput]`

        Example:
            ```python
            >>> import torch
            >>> from transformers import BrosProcessor, BrosForTokenClassification
            ...
            >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
            >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
            >>> encoding["bbox"] = bbox
            ...
            >>> outputs = model(**encoding)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bros(
            input_ids,
            bbox=bbox,
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                loss = ops.cross_entropy(
                    logits.view(-1, self.num_labels)[bbox_first_token_mask], labels.view(-1)[bbox_first_token_mask]
                )
            else:
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


class BrosSpadeEEForTokenClassification(BrosPreTrainedModel):

    """
    This class represents a BrosSpadeEEForTokenClassification model for token classification tasks.
    It is a subclass of BrosPreTrainedModel.

    The BrosSpadeEEForTokenClassification model consists of a BrosModel backbone and two token classifiers:
    initial_token_classifier and subsequent_token_classifier. The initial_token_classifier is used to
    classify the initial tokens in the input sequence, while the subsequent_token_classifier is used to classify the subsequent tokens.

    The class provides a 'construct' method that takes various input tensors such as input_ids, bbox, attention_mask,
    token_type_ids, etc. It returns the predicted initial token logits and subsequent token
    logits. Optionally, it can also return hidden states and attentions if specified.

    Example:
        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosSpadeEEForTokenClassification
        ...
        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
        >>> model = BrosSpadeEEForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")
        ...
        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox
        ...
        >>> outputs = model(**encoding)
        ```

    Please note that the docstring above is a summary of the class functionality and does not include method signatures or additional details.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes a BrosSpadeEEForTokenClassification instance.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing the model configuration parameters.

                - Type: object
                - Purpose: The configuration for initializing the model.
                - Restrictions: Must contain the following attributes: 'num_labels', 'n_relations', 'hidden_size', 'classifier_dropout', 'hidden_dropout_prob'.

        Returns:
            None.

        Raises:
            AttributeError: If the 'config' object does not contain the required attributes.
            ValueError: If the 'config' attributes have invalid values or types.
            TypeError: If the 'config' parameter is not of type object.
        """
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size

        self.bros = BrosModel(config)
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )

        # Initial token classification for Entity Extraction (NER)
        self.initial_token_classifier = nn.SequentialCell(
            nn.Dropout(p=classifier_dropout),
            nn.Dense(config.hidden_size, config.hidden_size),
            nn.Dropout(p=classifier_dropout),
            nn.Dense(config.hidden_size, config.num_labels),
        )

        # Subsequent token classification for Entity Extraction (NER)
        self.subsequent_token_classifier = BrosRelationExtractor(config)

        self.init_weights()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        bbox_first_token_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        initial_token_labels: Optional[mindspore.Tensor] = None,
        subsequent_token_labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BrosSpadeOutput]:
        r"""

        Returns:
            Union[Tuple[mindspore.Tensor], BrosSpadeOutput]

        Example:
            ```python
            >>> import torch
            >>> from transformers import BrosProcessor, BrosSpadeEEForTokenClassification
            ...
            >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> model = BrosSpadeEEForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
            >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
            >>> encoding["bbox"] = bbox
            ...
            >>> outputs = model(**encoding)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bros(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.swapaxes(0, 1)
        initial_token_logits = self.initial_token_classifier(last_hidden_states).swapaxes(0, 1)
        subsequent_token_logits = self.subsequent_token_classifier(last_hidden_states, last_hidden_states).squeeze(0)

        # make subsequent token (sequence token classification) mask
        inv_attention_mask = 1 - attention_mask
        batch_size, max_seq_length = inv_attention_mask.shape
        invalid_token_mask = ops.cat([inv_attention_mask, ops.zeros((batch_size, 1)).astype(inv_attention_mask.dtype)], axis=1).bool()
        subsequent_token_logits = subsequent_token_logits.masked_fill(
            invalid_token_mask[:, None, :], finfo(subsequent_token_logits.dtype, 'min')
        )
        self_token_mask = ops.eye(max_seq_length, max_seq_length + 1).bool()
        subsequent_token_logits = subsequent_token_logits.masked_fill(
            self_token_mask[None, :, :], finfo(subsequent_token_logits.dtype, 'min')
        )
        subsequent_token_mask = attention_mask.view(-1).bool()

        loss = None
        if initial_token_labels is not None and subsequent_token_labels is not None:
            # get initial token loss
            initial_token_labels = initial_token_labels.view(-1)
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                initial_token_loss = ops.cross_entropy(
                    initial_token_logits.view(-1, self.num_labels)[bbox_first_token_mask],
                    initial_token_labels[bbox_first_token_mask],
                )
            else:
                initial_token_loss = ops.cross_entropy(initial_token_logits.view(-1, self.num_labels), initial_token_labels)

            subsequent_token_labels = subsequent_token_labels.view(-1)
            subsequent_token_loss = ops.cross_entropy(
                subsequent_token_logits.view(-1, max_seq_length + 1)[subsequent_token_mask],
                subsequent_token_labels[subsequent_token_mask],
            )

            loss = initial_token_loss + subsequent_token_loss

        if not return_dict:
            output = (initial_token_logits, subsequent_token_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return BrosSpadeOutput(
            loss=loss,
            initial_token_logits=initial_token_logits,
            subsequent_token_logits=subsequent_token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BrosSpadeELForTokenClassification(BrosPreTrainedModel):

    """
    This class represents a Bros Spade Entity Linking model for token classification.

    The BrosSpadeELForTokenClassification class is a subclass of the BrosPreTrainedModel class and is used for token classification tasks. It inherits the __init__ and construct methods from the
    BrosPreTrainedModel class.

    Attributes:
        config: The configuration object used to initialize the model.
        num_labels: The number of labels for token classification.
        n_relations: The number of relations used in the model.
        backbone_hidden_size: The hidden size of the model's backbone.
        bros: An instance of the BrosModel class.
        entity_linker: An instance of the BrosRelationExtractor class.

    Methods:
        __init__(self, config): Initializes the BrosSpadeELForTokenClassification object with the given config.
        construct: Constructs the model and performs token classification.

    Returns:
        Conditional returns:

            - If return_dict is False:

                - A tuple containing the logits and other model outputs.

            - If return_dict is True:

                - An instance of the TokenClassifierOutput class containing the loss, logits, hidden states, and attentions.

    Example:
        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosSpadeELForTokenClassification
        ...
        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
        ...
        >>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")
        ...
        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox
        ...
        >>> outputs = model(**encoding)
        ```
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes an instance of the BrosSpadeELForTokenClassification class.

        Args:
            self: The object instance.
            config: An instance of the BrosConfig class containing the configuration parameters.
                It should have the following attributes:

                - num_labels (int): The number of possible labels for token classification.
                - n_relations (int): The number of possible relations.
                - hidden_size (int): The hidden size of the model.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method initializes the BrosSpadeELForTokenClassification instance by setting the provided configuration parameters.
            It also initializes the bros model and the entity linker for relation extraction.
            The method init_weights() is called to initialize the weights of the model.
        """
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size

        self.bros = BrosModel(config)
        # (config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob)

        self.entity_linker = BrosRelationExtractor(config)

        self.init_weights()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        bbox_first_token_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Returns:
            Union[Tuple[mindspore.Tensor], TokenClassifierOutput]

        Example:
            ```python
            >>> import torch
            >>> from transformers import BrosProcessor, BrosSpadeELForTokenClassification
            ...
            >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")
            ...
            >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
            >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
            >>> encoding["bbox"] = bbox
            ...
            >>> outputs = model(**encoding)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bros(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.swapaxes(0, 1)

        logits = self.entity_linker(last_hidden_states, last_hidden_states).squeeze(0)

        loss = None
        if labels is not None:
            batch_size, max_seq_length = attention_mask.shape

            self_token_mask = ops.eye(max_seq_length, max_seq_length + 1).bool()

            mask = bbox_first_token_mask.view(-1)
            bbox_first_token_mask = ops.cat(
                [
                    ~bbox_first_token_mask,
                    ops.zeros((batch_size, 1), dtype=mindspore.bool_),
                ],
                axis=1,
            )
            logits = logits.masked_fill(bbox_first_token_mask[:, None, :], finfo(logits.dtype, 'min'))
            logits = logits.masked_fill(self_token_mask[None, :, :], finfo(logits.dtype, 'min'))

            loss = ops.cross_entropy(logits.view(-1, max_seq_length + 1)[mask], labels.view(-1)[mask])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "BrosPreTrainedModel",
    "BrosModel",
    "BrosForTokenClassification",
    "BrosSpadeEEForTokenClassification",
    "BrosSpadeELForTokenClassification",
]
