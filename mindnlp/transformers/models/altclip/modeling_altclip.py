# coding=utf-8
# Copyright 2022 The BAAI Teams Authors and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore AltCLIP model."""
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import functional as F
from mindnlp.utils import ModelOutput, logging
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndProjection,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_altclip import AltCLIPConfig, AltCLIPTextConfig, AltCLIPVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "BAAI/AltCLIP"
_CONFIG_FOR_DOC = "AltCLIPConfig"

ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "BAAI/AltCLIP",
    # See all AltCLIP models at https://hf-mirror.com/models?filter=altclip
]

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: mindspore.Tensor) -> mindspore.Tensor:
    """
    This function calculates the contrastive loss given a tensor of logits.
    
    Args:
        logits (mindspore.Tensor): A tensor containing the logits.
    
    Returns:
        mindspore.Tensor: A tensor representing the contrastive loss.
    
    Raises:
        None.
    """
    return F.cross_entropy(logits, ops.arange(len(logits)))


def clip_loss(similarity: mindspore.Tensor) -> mindspore.Tensor:
    """
    This function calculates the average of caption loss and image loss based on the input similarity tensor.
    
    Args:
        similarity (mindspore.Tensor): A tensor representing the similarity between captions and images.
    
    Returns:
        mindspore.Tensor: A tensor containing the average of caption loss and image loss.
    
    Raises:
        None
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->AltCLIP
class AltCLIPOutput(ModelOutput):
    """
    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`mindspore.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`mindspore.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`AltCLIPTextModel`].
        image_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`AltCLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`AltCLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`AltCLIPVisionModel`].
    """
    loss: Optional[mindspore.Tensor] = None
    logits_per_image: mindspore.Tensor = None
    logits_per_text: mindspore.Tensor = None
    text_embeds: mindspore.Tensor = None
    image_embeds: mindspore.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        """
        Converts the AltCLIPOutput object to a tuple.
        
        Args:
            self (AltCLIPOutput): The AltCLIPOutput object to convert to a tuple.
        
        Returns:
            Tuple[Any]: A tuple representation of the AltCLIPOutput object. 
                The tuple contains the values of all attributes in the object, except for 'text_model_output' and
                'vision_model_output'. If any of these attributes are present, their values will be recursively
                converted to tuples as well.

        Raises:
            None.

        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->AltRoberta
class AltRobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        """
        Args:
            self (object): The instance of the class.
            config (object): 
                An object containing configuration parameters for the embeddings. It should have the following attributes:
                
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The dimension of the hidden layers.
                - max_position_embeddings (int): The maximum position for position embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layers.
                - position_embedding_type (str, optional): The type of position embedding, defaults to 'absolute'.
                - pad_token_id (int): The ID of the padding token.

        Returns:
            None.

        Raises:
            AttributeError: If the config object does not have the required attributes.
            ValueError: If the config attributes have invalid values or types.
            RuntimeError: If there is an issue with the initialization of embeddings or other components.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer('position_ids', ops.broadcast_to(ops.arange(config.max_position_embeddings), (1, -1)))
        self.register_buffer('token_type_ids', ops.zeros(*self.position_ids.shape, dtype=mindspore.int64))

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        """
        Constructs the embeddings for the AltRoberta model.

        Args:
            self (AltRobertaEmbeddings): An instance of the AltRobertaEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the token indices.
                Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The input tensor containing the token type indices.
                Default is None.
            position_ids (Optional[mindspore.Tensor]): The input tensor containing the position indices.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The input tensor containing the embedded inputs.
                Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            mindspore.Tensor: The forwarded embeddings tensor.

        Raises:
            None.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in forwardor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = ops.broadcast_to(buffered_token_type_ids, (input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(*input_shape, dtype=mindspore.int64)

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

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: mindspore.Tensor

        Returns:
            mindspore.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ops.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64
        )
        return ops.broadcast_to(position_ids.unsqueeze(0), (input_shape))


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->AltRoberta
class AltRobertaSelfAttention(nn.Module):

    """
    Represents a self-attention mechanism for the ALBERT model.
    This class implements the self-attention mechanism used in the ALBERT model for natural language processing tasks.
    The attention mechanism calculates attention scores between different parts of the input sequence and generates context-aware representations.
    The class includes methods for initializing the self-attention layer, processing input tensors, and generating output representations.

    Attributes:
        num_attention_heads (int): Number of attention heads used in the self-attention mechanism.
        attention_head_size (int): Size of each attention head.
        all_head_size (int): Total size of all attention heads.
        query (nn.Linear): Dense layer for query transformation.
        key (nn.Linear): Dense layer for key transformation.
        value (nn.Linear): Dense layer for value transformation.
        dropout (nn.Dropout): Dropout layer for attention probabilities.
        position_embedding_type (str): Type of position embeddings used in the attention mechanism.
        max_position_embeddings (int): Maximum number of position embeddings.
        distance_embedding (nn.Embedding): Embedding layer for distance-based positional encodings.
        is_decoder (bool): Indicates if the self-attention layer is used in a decoder context.

    Methods:
        swapaxes_for_scores:
            Reshapes the input tensor 'x' to prepare it for calculating attention scores.

        forward:
            Constructs the self-attention mechanism using the given input tensors and masks.
            Calculates attention scores, applies position embeddings, and produces the final context layer.
            Supports optional parameters for handling cross-attention, caching key-value pairs, and outputting attention scores.
            Returns a tuple containing the context layer and optionally attention probabilities and cached key-value pairs.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the AltRobertaSelfAttention class.

        Args:
            self: The instance of the class.
            config (object): An object containing configuration parameters for the self-attention mechanism.
            position_embedding_type (str, optional): The type of position embedding to be used. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size in the config is not a multiple of the number of attention heads.
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

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def swapaxes_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method 'swapaxes_for_scores' is defined in the class 'AltRobertaSelfAttention' and is used to perform a specific transformation on the input tensor 'x'.

        Args:
            self (AltRobertaSelfAttention): The instance of the AltRobertaSelfAttention class.
            x (mindspore.Tensor): The input tensor on which the swapaxes operation will be performed. It should be of type mindspore.Tensor.

        Returns:
            mindspore.Tensor: Returns a new tensor after performing the swapaxes operation.

        Raises:
            None.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        '''
        This method forwards the self-attention mechanism for the AltRoberta model.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states. 
                It is a tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional tensor for masking the attention scores. 
                It has the same shape as hidden_states and contains 0s for positions that should be masked and
                -10000s for positions that should be kept.
            head_mask (Optional[mindspore.Tensor]): An optional tensor for masking the attention scores per head. 
                It has the shape (num_heads,) and is a tensor of 0s and 1s.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor representing hidden states from the encoder. 
                It has the same shape as hidden_states.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional tensor for masking the attention scores in encoder_hidden_states. 
                It has the same shape as encoder_hidden_states.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of past key and value tensors.
                Each tensor has the shape (batch_size, num_heads, past_sequence_length, head_size).
            output_attentions (Optional[bool]): An optional flag to indicate whether to output attention scores.

        Returns:
            Tuple[mindspore.Tensor]: 
                The output of the self-attention mechanism, which is a tuple containing context_layer and attention_probs.

        Raises:
            None
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
            key_layer = ops.cat([past_key_value[0], key_layer], dim=2)
            value_layer = ops.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
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
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = mindspore.tensor(key_length - 1, dtype=mindspore.int64).view(
                    -1, 1
                )
            else:
                position_ids_l = ops.arange(query_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(key_length, dtype=mindspore.int64).view(1, -1)
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

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in AltRobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput
class AltRobertaSelfOutput(nn.Module):
    """A class representing the self-attention output module of the alternative implementation of the RoBERTa model.

    This class inherits from the `nn.Module` class and implements the functionality of the self-attention output module in the RoBERTa model. 
    It applies a dense layer, followed by a dropout layer, and then applies layer normalization to the output. 
    The output is the sum of the layer-normalized hidden states and the input tensor.

    Attributes:
        dense (nn.Linear): The dense layer used to transform the hidden states.
        LayerNorm (nn.LayerNorm): The layer normalization module.
        dropout (nn.Dropout): The dropout module.

    Methods:
        forward:
            Applies the self-attention output module to the given hidden states and input tensor.

    Example:
        ```python
        >>> config = RobertaConfig(hidden_size=768, layer_norm_eps=1e-12, hidden_dropout_prob=0.1)
        >>> self_output = AltRobertaSelfOutput(config)
        >>> hidden_states = mindspore.Tensor(np.random.rand(2, 3, 768), mindspore.float32)
        >>> input_tensor = mindspore.Tensor(np.random.rand(2, 3, 768), mindspore.float32)
        >>> output = self_output.forward(hidden_states, input_tensor)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the AltRobertaSelfOutput class.

        Args:
            self: The instance of the AltRobertaSelfOutput class.
            config: An object containing configuration parameters for the self output layer, including the hidden size
                and dropout probability. It is of type Config.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters are invalid or missing.
            TypeError: If the input parameters are of incorrect types.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the output of the self-attention mechanism in the AltRoberta model.

        Args:
            self (AltRobertaSelfOutput): The instance of the AltRobertaSelfOutput class.
            hidden_states (mindspore.Tensor): The tensor representing the hidden states from the previous layer.
            input_tensor (mindspore.Tensor): The tensor representing the input to the current layer.

        Returns:
            mindspore.Tensor: Returns a tensor representing the forwarded hidden states after self-attention mechanism.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaAttention with Roberta->AltRoberta
class AltRobertaAttention(nn.Module):
    '''
    The AltRobertaAttention class represents the attention mechanism used in the AltRoberta model.
    This class inherits from nn.Module and includes methods for initializing the attention mechanism, pruning
    attention heads, and forwarding the attention output.

    Attributes:
        config: The configuration parameters for the attention mechanism.
        position_embedding_type: The type of position embedding to be used.

    Methods:
        __init__: Initializes the AltRobertaAttention class.
        prune_heads: Prunes the specified attention heads from the attention mechanism.
        forward: Constructs the attention output using the given inputs.

    Note:
        The class inherits from nn.Module and is designed for use in the AltRoberta model.
    '''
    def __init__(self, config, position_embedding_type=None):
        """
        __init__

        Initializes an instance of the AltRobertaAttention class.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing the settings for the attention mechanism.
            position_embedding_type (str, optional): The type of position embedding to be used. Default is None.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.self = AltRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = AltRobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'AltRobertaAttention' and is used to prune the attention heads based on the specified 'heads'.

        Args:
            self: The instance of the 'AltRobertaAttention' class.
            heads: A list of integers representing the attention heads to be pruned.

        Returns:
            None: The method performs in-place modifications on the instance variables of the 'AltRobertaAttention' class.

        Raises:
            This method does not raise any exceptions explicitly.
                However, it assumes that the input parameters are of the correct type and format.
                Any exceptions raised within the called functions (e.g.,find_pruneable_heads_and_indices, prune_linear_layer)
                will propagate to the caller.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        This method forwards the attention mechanism for the AltRoberta model.

        Args:
            self (AltRobertaAttention): The current instance of the AltRobertaAttention class.
            hidden_states (mindspore.Tensor): The input hidden states. Shape: [batch_size, sequence_length, hidden_size].
            attention_mask (Optional[mindspore.Tensor]): Mask to avoid performing attention on padding tokens. Shape: [batch_size, 1, sequence_length].
            head_mask (Optional[mindspore.Tensor]): Mask to exclude certain heads. Shape: [num_heads].
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states of the encoder. Shape: [batch_size, sequence_length, hidden_size].
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention. Shape: [batch_size, 1, sequence_length].
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Tuple containing the previous key and value tensors for faster decoding.
            output_attentions (Optional[bool]): Flag to output attentions weights. Default is False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the attention output tensor. Shape: [batch_size, sequence_length, hidden_size].

        Raises:
            None
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaIntermediate with Roberta->AltRoberta
class AltRobertaIntermediate(nn.Module):

    """
    Represents the intermediate layer for an alternative implementation of the Roberta model.

    This class inherits from nn.Module and contains methods to initialize and forward the intermediate layer.

    Attributes:
        config (obj): Configuration object for the intermediate layer.

    Methods:
        __init__: Initializes the AltRobertaIntermediate class with the provided configuration.
        forward: Constructs the intermediate layer using the given hidden states tensor.

    """
    def __init__(self, config):
        """
        Initializes an instance of the AltRobertaIntermediate class.

        Args:
            self: The instance of the class.
            config:
                An object of type 'config' containing the configuration parameters for the model.

                - Type: 'config'
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method to forward hidden states in the AltRobertaIntermediate class.

        Args:
            self: Instance of the AltRobertaIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states tensor to be processed.
                It represents the intermediate hidden states that are to be transformed.

        Returns:
            mindspore.Tensor: The transformed hidden states tensor after passing through the dense layer
                and applying the intermediate activation function.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput
class AltRobertaOutput(nn.Module):

    """
    Represents the output of an alternative Roberta model.

    This class inherits from nn.Module and includes methods for initializing the class and forwarding the output tensor
    based on the input hidden states and input tensor.
    The output tensor is obtained by applying dense layers, dropout, and layer normalization to the input hidden states
    and input tensor.

    Attributes:
        dense (nn.Linear): A dense layer with the specified intermediate and hidden sizes.
        LayerNorm (nn.LayerNorm): A layer normalization module with the specified hidden size and epsilon value.
        dropout (nn.Dropout): A dropout module with the specified dropout probability.

    Methods:
        __init__: Initializes the AltRobertaOutput class with the given configuration.
        forward: Constructs the output tensor by applying dense layers, dropout, and layer normalization to the input hidden states and input tensor.

    """
    def __init__(self, config):
        """
        Initializes an instance of the AltRobertaOutput class.

        Args:
            self (AltRobertaOutput): The instance of the class.
            config: A configuration object containing the parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the output of the alternative Roberta model.

        Args:
            self (AltRobertaOutput): The instance of the AltRobertaOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor to be processed.
            input_tensor (mindspore.Tensor): The input tensor to be added to the processed hidden states.

        Returns:
            mindspore.Tensor: The processed hidden states tensor.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaLayer with Roberta->AltRoberta
class AltRobertaLayer(nn.Module):

    """
    This class represents a layer in the AltRoberta model. It inherits from the nn.Module class.

    Attributes:
        chunk_size_feed_forward (int): The chunk size for feed-forward operations.
        seq_len_dim (int): The dimension of the sequence length.
        attention (AltRobertaAttention): The attention module used in the layer.
        is_decoder (bool): Indicates if the layer is a decoder.
        add_cross_attention (bool): Indicates if cross-attention is added.
        crossattention (AltRobertaAttention, optional): The cross-attention module used in the layer, if cross-attention is added.
        intermediate (AltRobertaIntermediate): The intermediate module used in the layer.
        output (AltRobertaOutput): The output module used in the layer.

    Methods:
        forward:
            Constructs the layer by applying attention, intermediate, and output operations.
        feed_forward_chunk:
            Applies the feed-forward operations on the attention output.

    """
    def __init__(self, config):
        """
        Initializes an instance of the AltRobertaLayer class.

        Args:
            self: The instance of the class.
            config:
                An object of the configuration class that contains various settings and hyperparameters for the model.

                - Type: Custom Configuration object
                - Purpose: Provides necessary configuration parameters for initializing the AltRobertaLayer instance.
                - Restrictions: None

        Returns:
            None.

        Raises:
            ValueError:
                If `self` is not used as a decoder model and `add_cross_attention` is `True`.

                - Raised when the `add_cross_attention` parameter is set to `True` but `self` is not used as a decoder model.
                - Purpose: Ensures that cross attention is only added when the layer is used as a decoder model.
                - Restrictions: None
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AltRobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = AltRobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = AltRobertaIntermediate(config)
        self.output = AltRobertaOutput(config)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Constructs the layer of the alternative Roberta model.

        Args:
            self (AltRobertaLayer): The instance of the AltRobertaLayer class.
            hidden_states (mindspore.Tensor): The input hidden states for the layer.
            attention_mask (Optional[mindspore.Tensor]): The attention mask for the input hidden states. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask for the attention mechanism. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder if the layer is a decoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder_hidden_states if the layer is a decoder. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key and value tensors for fast autoregressive decoding. Defaults to None.
            output_attentions (Optional[bool]): Whether to return attentions as part of the output. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the layer output and additional values if the layer is a decoder.

        Raises:
            ValueError:
                If `encoder_hidden_states` are passed, and the cross-attention layers are not instantiated with `config.add_cross_attention=True`.
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
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
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
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
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        This method applies the feed-forward chunk operation to the given attention output.

        Args:
            self (AltRobertaLayer): The instance of the AltRobertaLayer class invoking this method.
            attention_output (tensor): The input tensor representing the attention output.

        Returns:
            tensor: The output tensor obtained after applying the feed-forward chunk operation.

        Raises:
            ValueError: If the attention_output is not a valid tensor.
            TypeError: If the input parameters are not of the expected types.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaEncoder with Roberta->AltRoberta
class AltRobertaEncoder(nn.Module):

    """
    The 'AltRobertaEncoder' class is responsible for encoding input data using a variation of the Roberta model.
    This class inherits from the 'nn.Module' class and provides methods to forward the encoder and generate the output based on the input.

    The class consists of an initialization method that takes a configuration object as input and sets up the encoder layers.
    The 'forward' method takes various input tensors and optional parameters to forward the encoder output based on the Roberta model's architecture.

    The 'forward' method supports options such as attention masks, head masks, encoder hidden states, past key values, cache usage, and output configurations.
    It also handles gradient checkpointing during training if enabled.
    The method returns the encoder output in the form of a tuple or a custom 'BaseModelOutputWithPastAndCrossAttentions' object based on the specified return dict option.

    Overall, the 'AltRobertaEncoder' class encapsulates the functionality to encode input data using the specified Roberta model architecture
    and provides flexibility in configuring the output based on the input parameters.
    """
    def __init__(self, config):
        """
        Initializes an instance of the AltRobertaEncoder class.

        Args:
            self: The instance of the AltRobertaEncoder class.
            config:
                A dictionary containing configuration settings for the encoder.

                - Type: dict
                - Purpose: Contains various parameters to configure the encoder.
                - Restrictions: Must be a valid dictionary object.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AltRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: mindspore.Tensor,
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
        This method forwards the AltRobertaEncoder by processing the input hidden states and optional parameters to generate the output.

        Args:
            self: The instance of the AltRobertaEncoder class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask. Default is None.
            head_mask (Optional[mindspore.Tensor]): An optional tensor representing the head mask. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor representing the encoder hidden states. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the encoder attention mask. Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of past key values. Default is None.
            use_cache (Optional[bool]): An optional boolean flag indicating whether to use cache. Default is None.
            output_attentions (Optional[bool]): An optional boolean flag indicating whether to output attentions. Default is False.
            output_hidden_states (Optional[bool]): An optional boolean flag indicating whether to output hidden states. Default is False.
            return_dict (Optional[bool]): An optional boolean flag indicating whether to return a dictionary. Default is True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                Depending on the return_dict flag, returns either a tuple of relevant tensors or a BaseModelOutputWithPastAndCrossAttentions object.

        Raises:
            None

        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
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


# Copied from transformers.models.roberta.modeling_roberta.RobertaPooler
class AltRobertaPooler(nn.Module):

    """
    Represents a custom pooler layer for an alternative implementation of the RoBERTa model.

    This class inherits from the nn.Module module and provides a custom pooler layer for an alternative implementation of the RoBERTa model.

    The AltRobertaPooler class initializes with a configuration object and includes methods to forward the pooler layer.
    The forwardor initializes the dense layer and activation function.
    The forward method takes hidden_states as input, extracts the first token tensor, applies the dense layer,
    applies the activation function, and returns the pooled output.

    This class is designed to be used as part of a custom RoBERTa model implementation and provides
    an alternative approach to pooling hidden states for downstream tasks.
    """
    def __init__(self, config):
        """
        Initializes an instance of the AltRobertaPooler class.

        Args:
            self (object): The instance of the AltRobertaPooler class.
            config (object):
                An object containing configuration settings.

                - Type: Any
                - Purpose: Specifies the configuration settings for the pooler.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a pooled output tensor from the given hidden states tensor.

        Args:
            self (AltRobertaPooler): The instance of the AltRobertaPooler class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size) containing hidden states.

        Returns:
            mindspore.Tensor: The output tensor of shape (batch_size, hidden_size) representing the pooled output.

        Raises:
            None.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->AltCLIP
class AltCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config):
        """
        Initializes an instance of the AltCLIPAttention class.

        Args:
            self: The instance of the class AltCLIPAttention.
            config:
                An object containing configuration parameters for the attention mechanism.

                - Type: object
                - Purpose: To configure the attention mechanism.
                - Restrictions: None

        Returns:
            None

        Raises:
            ValueError:
                If embed_dim is not divisible by num_heads.

                - Purpose: To indicate that the configuration is invalid.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method '_shape' in the class 'AltCLIPAttention' reshapes the input tensor based on the provided sequence length and batch size.

        Args:
            self (AltCLIPAttention): The instance of the AltCLIPAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The size of the batch.

        Returns:
            None: This method does not return any value, as it directly reshapes the input tensor.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        causal_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->AltCLIP
class AltCLIPMLP(nn.Module):

    """
    This class represents an alternative implementation of the Multilayer Perceptron (MLP) used in the Contrastive Language-Image Pretraining (CLIP) model.
    It inherits from the `nn.Module` class.

    Attributes:
        config (object): The configuration object containing the hyperparameters for the MLP.
        activation_fn (function): The activation function used for the hidden layers.
        fc1 (mindspore.nn.Linear): The first fully connected layer of the MLP.
        fc2 (mindspore.nn.Linear): The second fully connected layer of the MLP.

    Methods:
        __init__: Initializes a new instance of the AltCLIPMLP class.
        forward: Constructs the forward pass of the MLP.

    """
    def __init__(self, config):
        """
        Initializes an instance of the AltCLIPMLP class.

        Args:
            self (AltCLIPMLP): The instance of the AltCLIPMLP class.
            config:
                The configuration object containing the settings for the AltCLIPMLP model.

                - Type: Any
                - Purpose: Specifies the configuration settings for the AltCLIPMLP model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Construct the AltCLIPMLP model.

        This method takes in a tensor of hidden states and applies linear transformations and activation functions to forward the AltCLIPMLP model.

        Args:
            self (AltCLIPMLP): An instance of the AltCLIPMLP class.
            hidden_states (mindspore.Tensor): A tensor of hidden states to be processed by the model.

        Returns:
            mindspore.Tensor: A tensor representing the output of the AltCLIPMLP model.

        Raises:
            None.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->AltCLIP
class AltCLIPEncoderLayer(nn.Module):

    """
    This class represents a single layer of the AltCLIPEncoder.
    It applies multi-head self-attention and feed-forward neural network operations to the input hidden states.

    The AltCLIPEncoderLayer class inherits from the nn.Module class.

    Attributes:
        embed_dim (int): The dimensionality of the input hidden states.
        self_attn (AltCLIPAttention): An instance of the AltCLIPAttention class that performs the self-attention operation.
        layer_norm1 (nn.LayerNorm): A layer normalization operation applied after the self-attention.
        mlp (AltCLIPMLP): An instance of the AltCLIPMLP class that performs the feed-forward neural network operation.
        layer_norm2 (nn.LayerNorm): A layer normalization operation applied after the feed-forward neural network operation.

    Methods:
        forward(hidden_states, attention_mask, causal_attention_mask, output_attentions=False):
            Applies the AltCLIPEncoderLayer operations to the input hidden states.

            Args:

                - hidden_states (mindspore.Tensor): The input hidden states of shape (batch, seq_len, embed_dim).
                - attention_mask (mindspore.Tensor): The attention mask of size (batch, 1, tgt_len, src_len),
                where padding elements are indicated by very large negative values.
                - causal_attention_mask (mindspore.Tensor): The causal attention mask of size (batch, 1, tgt_len, src_len),
                which is used to mask future positions during self-attention.
                - output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers. Defaults to False.

            Returns:
                Tuple[mindspore.Tensor]: A tuple containing the output hidden states.
                    If output_attentions is True, the tuple also contains the attention weights tensor.

    Note:
        The forward method performs the following operations in order:

        1. Applies layer normalization to the input hidden states.
        2. Performs the self-attention operation using the self_attn instance.
        3. Adds the residual connection from step 1 to the output of step 2.
        4. Applies layer normalization to the output of step 3.
        5. Applies the feed-forward neural network operation using the mlp instance.
        6. Adds the residual connection from step 3 to the output of step 5.
        7. Returns the output hidden states.

        If output_attentions is True, the attention weights tensor is also returned.

    """
    def __init__(self, config: AltCLIPConfig):
        """
        Initializes an instance of the AltCLIPEncoderLayer class.

        Args:
            self: The instance of the AltCLIPEncoderLayer class.
            config (AltCLIPConfig): An object of type AltCLIPConfig containing the configuration parameters for the encoder layer.
                It specifies the hidden size and layer normalization epsilon for the encoder layer.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = AltCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = AltCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        causal_attention_mask: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->AltCLIP
class AltCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`AltCLIPEncoderLayer`].

    Args:
        config: AltCLIPConfig
    """
    def __init__(self, config: AltCLIPConfig):
        """
        __init__

        Initializes an instance of the AltCLIPEncoder class.

        Args:
            self: AltCLIPEncoder - The instance of the AltCLIPEncoder class.
            config (AltCLIPConfig): The configuration object for the AltCLIPEncoder instance.
                This parameter specifies the configuration settings for the AltCLIPEncoder.
                It should be an instance of the AltCLIPConfig class.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AltCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[mindspore.Tensor] = None,
        causal_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings with CLIP->AltCLIP
class AltCLIPVisionEmbeddings(nn.Module):

    """
    A class representing the alternate vision embeddings for CLIP (Contrastive Language-Image Pre-training) models in MindSpore.

    This class extends the nn.Module module and is used to generate embeddings for images in the CLIP model architecture.
    The embeddings are forwarded based on the input pixel values using a combination of class embedding, patch embedding, and position embedding.

    Attributes:
        config (AltCLIPVisionConfig): The configuration object containing parameters for the embeddings.
        embed_dim (int): The dimensionality of the embeddings.
        image_size (int): The size of the input image.
        patch_size (int): The size of the patch used for patch embedding.
        class_embedding (Parameter): The learnable parameter representing the class embedding.
        patch_embedding (nn.Conv2d): The convolutional layer for patch embedding.
        num_patches (int): The total number of patches in the image.
        num_positions (int): The total number of positions including the patches and the class embedding.
        position_embedding (nn.Embedding): The embedding layer for positional encoding.
        position_ids (mindspore.Tensor): The tensor containing position indices.

    Methods:
        forward:
            Constructs the embeddings for the input pixel values by combining class embedding, patch embedding, and position embedding.

    Returns:
        mindspore.Tensor: The generated embeddings for the input pixel values.

    """
    def __init__(self, config: AltCLIPVisionConfig):
        """
        Initialize the AltCLIPVisionEmbeddings class.

        Args:
            self: The instance of the AltCLIPVisionEmbeddings class.
            config (AltCLIPVisionConfig): An instance of the AltCLIPVisionConfig class containing configuration parameters.
                The config parameter is used to initialize various attributes within the AltCLIPVisionEmbeddings instance.
                It must be of type AltCLIPVisionConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = Parameter(ops.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer('position_ids', ops.broadcast_to(ops.arange(self.num_positions), ((1, -1))))

    def forward(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the embeddings for the AltCLIPVisionEmbeddings model.

        Args:
            self (AltCLIPVisionEmbeddings): An instance of the AltCLIPVisionEmbeddings class.
            pixel_values (mindspore.Tensor): A tensor containing pixel values of images.
                The shape of the tensor is (batch_size, channels, height, width).

        Returns:
            mindspore.Tensor: A tensor containing the forwarded embeddings.
                The shape of the tensor is (batch_size, num_patches + 1, embed_dim),
                where num_patches is the number of patches extracted from the images and
                embed_dim is the dimension of the embedding.

        Raises:
            None.
        """
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = ops.broadcast_to(self.class_embedding, (batch_size, 1, -1))
        embeddings = ops.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class AltCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AltCLIPConfig
    base_model_prefix = "altclip"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(cell, AltCLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            cell.class_embedding.assign_value(initializer(Normal(cell.embed_dim**-0.5 * factor),
                                                    cell.class_embedding.shape, cell.class_embedding.dtype))
            cell.patch_embedding.weight.assign_value(initializer(Normal(cell.config.initializer_range * factor),
                                                    cell.patch_embedding.weight.shape, cell.patch_embedding.weight.dtype))
            cell.position_embedding.weight.assign_value(initializer(Normal(cell.config.initializer_range * factor),
                                                    cell.position_embedding.weight.shape, cell.position_embedding.weight.dtype))
        elif isinstance(cell, AltCLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (cell.embed_dim**-0.5) * ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (cell.embed_dim**-0.5) * factor
            cell.q_proj.weight.assign_value(initializer(Normal(in_proj_std),
                                                    cell.q_proj.weight.shape, cell.q_proj.weight.dtype))
            cell.k_proj.weight.assign_value(initializer(Normal(in_proj_std),
                                                    cell.k_proj.weight.shape, cell.k_proj.weight.dtype))
            cell.v_proj.weight.assign_value(initializer(Normal(in_proj_std),
                                                    cell.v_proj.weight.shape, cell.v_proj.weight.dtype))
            cell.out_proj.weight.assign_value(initializer(Normal(out_proj_std),
                                                    cell.out_proj.weight.shape, cell.out_proj.weight.dtype))

        elif isinstance(cell, AltCLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (cell.config.hidden_size**-0.5) * ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * cell.config.hidden_size) ** -0.5 * factor

            cell.fc1.weight.assign_value(initializer(Normal(fc_std),
                                                cell.fc1.weight.shape, cell.fc1.weight.dtype))
            cell.fc2.weight.assign_value(initializer(Normal(in_proj_std),
                                                cell.fc2.weight.shape, cell.fc2.weight.dtype))

        elif isinstance(cell, AltCLIPModel):
            cell.text_projection.weight.assign_value(initializer(Normal(cell.text_embed_dim**-0.5 * self.config.initializer_factor),
                                                cell.text_projection.weight.shape, cell.text_projection.weight.dtype))
            cell.text_projection._is_initialized = True
            cell.visual_projection.weight.assign_value(initializer(Normal(cell.vision_embed_dim**-0.5 * self.config.initializer_factor),
                                                cell.visual_projection.weight.shape, cell.visual_projection.weight.dtype))
            cell.visual_projection._is_initialized = True

        elif isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.assign_value(initializer(Normal(self.config.initializer_factor),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_factor, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.assign_value(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.assign_value(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class AltCLIPVisionTransformer(nn.Module):

    """
    This class represents a vision transformer model for the Alternative Contrastive Learning for Image and Text (AltCLIP) framework.
    It encapsulates the functionality to process visual inputs using the AltCLIP vision transformer architecture.

    The AltCLIPVisionTransformer class inherits from the nn.Module class and consists of methods for initialization and forwardion.

    The __init__ method initializes the AltCLIPVisionTransformer with the provided AltCLIPVisionConfig.
    It sets up the embeddings, encoder, and layer normalization components required for processing visual inputs.

    The forward method processes the input pixel values using the initialized components, performs encoding, and returns the last hidden state and pooled output.
    It also handles the optional arguments for controlling the output format.

    For more details on the AltCLIPVisionTransformer model and its usage, refer to the AltCLIPVisionTransformer documentation and examples.
    """
    def __init__(self, config: AltCLIPVisionConfig):
        """
        Initializes an instance of the AltCLIPVisionTransformer class.

        Args:
            self: The instance of the class.
            config (AltCLIPVisionConfig): An instance of AltCLIPVisionConfig containing the configuration parameters for the transformer.
                This parameter is required to initialize the transformer and should be an instance of AltCLIPVisionConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = AltCLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = AltCLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            Union[Tuple, BaseModelOutputWithPooling]

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AltCLIPVisionModel(AltCLIPPreTrainedModel):

    """
    The 'AltCLIPVisionModel' class represents a vision model for the AltCLIP framework.
    It inherits from the 'AltCLIPPreTrainedModel' class and contains methods for initializing the model, obtaining input
    embeddings, and forwarding the model output.
    The 'AltCLIPVisionModel' class is designed to work with image inputs and provides flexibility in handling output attentions, hidden states, and return
    dictionaries.
    It supports the use of pre-trained models and enables easy integration with image processing pipelines.

    The 'AltCLIPVisionModel' class can be instantiated and used to process image data, extract features,
    and perform inference in the context of the AltCLIP framework.
    It provides a convenient interface for leveraging vision transformers and accessing model outputs,
    such as hidden states and pooled representations of images.

    This class encapsulates the functionality required to utilize vision models within the AltCLIP framework,
    allowing for seamless integration with image processing workflows and enabling efficient
    utilization of pre-trained models for various vision-related tasks.
    """
    config_class = AltCLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: AltCLIPVisionConfig):
        """
        Initializes an instance of the AltCLIPVisionModel class.

        Args:
            self: The instance of the AltCLIPVisionModel class.
            config (AltCLIPVisionConfig):
                An instance of AltCLIPVisionConfig representing the configuration parameters for the vision model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.vision_model = AltCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the input embeddings of the AltCLIPVisionModel.

        Args:
            self (AltCLIPVisionModel): An instance of the AltCLIPVisionModel class.

        Returns:
            nn.Module: The input embeddings of the AltCLIPVisionModel.

        Raises:
            None.
        """
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            Union[Tuple, BaseModelOutputWithPooling]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, AltCLIPVisionModel
            ...
            >>> model = AltCLIPVisionModel.from_pretrained("BAAI/AltCLIP")
            >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(images=image, return_tensors="ms")
            ...
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output  # pooled CLS states
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class AltRobertaModel(AltCLIPPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """
    config_class = AltCLIPTextConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->AltRoberta
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes an instance of the AltRobertaModel class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing the model's settings.
            add_pooling_layer (bool): Flag indicating whether to add a pooling layer. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config

        self.embeddings = AltRobertaEmbeddings(config)
        self.encoder = AltRobertaEncoder(config)

        self.pooler = AltRobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the AltRobertaModel.

        Args:
            self (AltRobertaModel): The instance of the AltRobertaModel class.

        Returns:
            None: This method returns the input embeddings as a word_embeddings object.

        Raises:
            None
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the AltRobertaModel.

        Args:
            self (AltRobertaModel): The instance of the AltRobertaModel.
            value (object): The input embeddings value to be set. It can be of any valid type.

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

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
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
        Args:
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input.
                This mask is used in the cross-attention if the model is configured as a decoder.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
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
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ops.ones(batch_size, seq_length + past_key_values_length)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = ops.broadcast_to(buffered_token_type_ids, (batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(*input_shape, dtype=mindspore.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(*encoder_hidden_shape)
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
        encoder_outputs = self.encoder(
            embedding_output,
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


class AltCLIPTextModel(AltCLIPPreTrainedModel):

    """
    Represents an alternative implementation of the CLIP (Contrastive Language-Image Pretraining) model specifically
    tailored for text. This class extends the AltCLIPPreTrainedModel class and includes methods for initializing the
    model, getting and setting input embeddings, resizing token embeddings, and forwarding the model for inference.
    The 'forward' method takes various input tensors and optional parameters and returns the model's output,
    including the last hidden state and the pooled CLS states. Additionally, usage examples are provided for reference.

    Example:
        ```python
        >>> from transformers import AutoProcessor, AltCLIPTextModel
        >>> model = AltCLIPTextModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
        >>> texts = ["it's a cat", "it's a dog"]
        >>> inputs = processor(text=texts, padding=True, return_tensors="ms")
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```
    """
    config_class = AltCLIPTextConfig

    def __init__(self, config):
        """
        Initializes an instance of the AltCLIPTextModel class.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing parameters for the model initialization.

                - Type: dict
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid configuration dictionary.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not of type dict.
            ValueError: If the config dictionary is missing required keys or contains invalid values.
            RuntimeError: If there is an issue during the initialization process.
        """
        super().__init__(config)
        self.roberta = AltRobertaModel(config, add_pooling_layer=False)
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        This method returns the input embeddings for the AltCLIPTextModel.

        Args:
            self: AltCLIPTextModel
                The instance of the AltCLIPTextModel class.

        Returns:
            nn.Module
                The input embeddings for the AltCLIPTextModel, represented as an instance of nn.Module.

        Raises:
            None
                This method does not raise any exceptions.
        """
        return self.roberta.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """
        Method to set the input embeddings for the AltCLIPTextModel.

        Args:
            self (AltCLIPTextModel): An instance of the AltCLIPTextModel class.
                This parameter refers to the object itself.
            value (nn.Embedding): The new embedding to be set as the input embedding.
                It should be an instance of nn.Embedding representing the input embeddings.
                The value parameter will replace the existing word embeddings in the model.

        Returns:
            None: This method does not return any value. It updates the input embeddings of the model in place.

        Raises:
            None
        """
        self.roberta.embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        This method resizes the token embeddings of the AltCLIPTextModel.

        Args:
            self (AltCLIPTextModel): The instance of the AltCLIPTextModel class.

            new_num_tokens (Optional[int]): The new number of tokens for the resized embeddings.
            If None, the original number of tokens will be used. Default is None.

        Returns:
            nn.Embedding: The resized token embeddings as an instance of nn.Embedding.

        Raises:
            None: This method does not explicitly raise any exceptions.
        """
        return super().resize_token_embeddings(new_num_tokens)

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndProjection]:
        r"""
        Returns:
            Union[Tuple, BaseModelOutputWithPoolingAndProjection]

        Example:
            ```python
            >>> from transformers import AutoProcessor, AltCLIPTextModel
            ...
            >>> model = AltCLIPTextModel.from_pretrained("BAAI/AltCLIP")
            >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
            ...
            >>> texts = ["it's a cat", "it's a dog"]
            ...
            >>> inputs = processor(text=texts, padding=True, return_tensors="ms")
            ...
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output  # pooled CLS states
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last module outputs
        sequence_output = outputs[0]

        # project every module
        sequence_output = self.pre_LN(sequence_output)

        # pooler
        projection_state = self.transformation(sequence_output)
        pooler_output = projection_state[:, 0]

        if not return_dict:
            return (projection_state, pooler_output) + outputs[2:4]

        return BaseModelOutputWithPoolingAndProjection(
            last_hidden_state=projection_state,
            pooler_output=pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AltCLIPModel(AltCLIPPreTrainedModel):

    """
    AltCLIPModel
    Represents an alternative implementation of the Contrastive Language-Image Pretraining (CLIP) model.

    This class inherits from the `AltCLIPPreTrainedModel` class and includes methods to obtain text and image features, as well as to forward the final output.

    The `AltCLIPModel` class includes the following methods:

    - get_text_features: Returns the text embeddings obtained by applying the projection layer to the pooled output of `AltCLIPTextModel`.
    - get_image_features: Returns the image embeddings obtained by applying the projection layer to the pooled output of `AltCLIPVisionModel`.
    - forward: Constructs the final output, including image-text similarity scores and label probabilities.

    Example:
        ```python
        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="ms")
        >>> image_features = model.get_image_features(**inputs)
        ...
        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="ms", padding=True
        ... )
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```
    """
    config_class = AltCLIPConfig

    def __init__(self, config: AltCLIPConfig):
        """Initialize the AltCLIPModel with the provided configuration.

        Args:
            self: The instance of the AltCLIPModel class.
            config (AltCLIPConfig): The configuration object containing the settings for the AltCLIPModel.

        Returns:
            None.

        Raises:
            ValueError: If the 'config.vision_config' is not an instance of AltCLIPVisionConfig.
            ValueError: If the 'config.text_config' is not an instance of AltCLIPTextConfig.
        """
        super().__init__(config)

        if not isinstance(config.vision_config, AltCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type AltCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )
        if not isinstance(config.text_config, AltCLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type AltCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.project_dim
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = AltCLIPTextModel(text_config)
        self.vision_model = AltCLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = Parameter(mindspore.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        token_type_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""

        Returns:
            text_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`AltCLIPTextModel`].

        Example:
            ```python
            >>> from transformers import AutoProcessor, AltCLIPModel
            ...
            >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
            >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
            >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="ms")
            >>> text_features = model.get_text_features(**inputs)
            ```
        """
        # Use AltCLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""

        Returns:
            image_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`AltCLIPVisionModel`].

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, AltCLIPModel
            ...
            >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
            >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> inputs = processor(images=image, return_tensors="ms")
            >>> image_features = model.get_image_features(**inputs)
            ```
        """
        # Use AltCLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AltCLIPOutput]:
        r"""

        Returns:
            `Union[Tuple, AltCLIPOutput]`

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, AltCLIPModel
            ...
            >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
            >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> inputs = processor(
            ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="ms", padding=True
            ... )
            >>> outputs = model(**inputs)
            >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            ```
        """
        # Use AltCLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = ops.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return AltCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: mindspore.Tensor x:

    Returns:
        mindspore.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (ops.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

__all__ = [
    "ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
    "AltCLIPPreTrainedModel",
    "AltCLIPModel",
    "AltCLIPTextModel",
    "AltCLIPVisionModel",
]
