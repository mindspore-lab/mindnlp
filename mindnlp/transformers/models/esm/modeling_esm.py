# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore ESM model."""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_esm import EsmConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/esm2_t6_8M_UR50D"
_CONFIG_FOR_DOC = "EsmConfig"

ESM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    # This is not a complete list of all ESM models!
    # See all ESM models at https://hf-mirror.com/models?filter=esm
]


def rotate_half(x):
    """
    Rotate the input array by half of its length.
    
    Args:
        x (ndarray): The input array to be rotated. It should have a shape compatible with the chunk operation.
        
    Returns:
        None.
    
    Raises:
        None
    """
    x1, x2 = x.chunk(2, axis=-1)
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary positional embeddings to the input tensor.
    
    Args:
        x (Tensor): Input tensor to which the positional embeddings will be applied.
        cos (Tensor): Cosine values for rotary positional embeddings.
        sin (Tensor): Sine values for rotary positional embeddings.
        
    Returns:
        None: The function modifies the input tensor x in-place by applying rotary positional embeddings.
    
    Raises:
        None.
    """
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + ops.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.swapaxes(-1, -2)


def average_product_correct(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keep_dims=True)
    a2 = x.sum(-2, keep_dims=True)
    a12 = x.sum((-1, -2), keep_dims=True)

    avg = a1 * a2
    avg = avg.div(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


class RotaryEmbedding(nn.Cell):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://hf-mirror.com/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """
    def __init__(self, dim: int):
        """
        Initializes a new instance of the RotaryEmbedding class.
        
        Args:
            self (RotaryEmbedding): The current instance of the RotaryEmbedding class.
            dim (int): The dimension of the rotary embedding. It determines the size of the inv_freq tensor.
                Must be a positive integer value.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (ops.arange(0, dim, 2, dtype=mindspore.float32) / dim))
        self.inv_freq = inv_freq

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        """
        Method to update cosine and sine tables for rotary embedding.
        
        Args:
            self (RotaryEmbedding): The instance of the RotaryEmbedding class.
            x (Tensor): The input tensor for which cosine and sine tables are being updated.
            seq_dimension (int, optional): The dimension along which to calculate the sequence length. Default is 2.
        
        Returns:
            None. This method updates the internal cosine and sine tables of the RotaryEmbedding instance.
        
        Raises:
            ValueError: If the sequence length of the input tensor does not match the cached sequence length.
            ValueError: If an error occurs during arithmetic operations with the input tensor.
        """
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = ops.arange(x.shape[seq_dimension]).astype(self.inv_freq.dtype)
            freqs = ops.outer(t, self.inv_freq)
            emb = ops.cat((freqs, freqs), axis=-1)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def construct(self, q: mindspore.Tensor, k: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Constructs the rotary embedding for the given input tensors q and k.
        
        Args:
            self (RotaryEmbedding): An instance of the RotaryEmbedding class.
            q (mindspore.Tensor): The input tensor q for rotary embedding construction.
                It should have a shape compatible with the rotary embedding dimensions.
            k (mindspore.Tensor): The input tensor k for rotary embedding construction.
                It should have a shape compatible with the rotary embedding dimensions.
        
        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor]: A tuple containing two tensors.
            The first tensor is the result of applying rotary positional embedding to the input tensor q.
            The second tensor is the result of applying rotary positional embedding to the input tensor k.
        
        Raises:
            None.
        
        Note:
            - The rotary embedding is constructed using the provided q and k tensors.
            - The rotary embedding dimensions are determined by the dimensions of the input tensors.
            - The rotary embedding is calculated using the cosine and sine tables obtained from
            the _update_cos_sin_tables method.
            - The cosine and sine tables are cached within the RotaryEmbedding instance.
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class EsmContactPredictionHead(nn.Cell):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""
    def __init__(
        self,
        in_features: int,
        bias=True,
        eos_idx: int = 2,
    ):
        """
        Initializes an instance of the EsmContactPredictionHead class.
        
        Args:
            self: An instance of the EsmContactPredictionHead class.
            in_features (int): The number of input features for the head.
            bias (bool, optional): Whether to include bias terms. Defaults to True.
            eos_idx (int, optional): The index of the end-of-sentence token. Defaults to 2.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.in_features = in_features
        self.eos_idx = eos_idx
        self.regression = nn.Dense(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def construct(self, tokens, attentions):
        """
        This method constructs attentions for contact prediction in the ESM model.
        
        Args:
            self (EsmContactPredictionHead): An instance of the EsmContactPredictionHead class.
            tokens (torch.Tensor): A tensor containing tokens for the input sequences.
                Shape: (batch_size, seqlen)
                Restrictions: Should not contain the end-of-sequence index (eos_idx).
            attentions (torch.Tensor): A tensor containing attention weights for the input sequences.
                Shape: (batch_size, layers, heads, seqlen, seqlen)
        
        Returns:
            None: This method does not return any value but modifies the attentions tensor in-place.
        
        Raises:
            None
        """
        # remove eos token attentions
        eos_mask = tokens.ne(self.eos_idx).to(attentions.dtype)
        eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.shape
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # attentions always float32, may need to convert to float16
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))


class EsmEmbeddings(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        """
        __init__
        
        Initializes the EsmEmbeddings class.
        
        Args:
            self: EsmEmbeddings instance. A reference to the class instance.
            config: object.
                A configuration object containing parameters for the embedding layer.

                - vocab_size: int. The size of the vocabulary.
                - hidden_size: int. The dimension of the hidden layer.
                - pad_token_id: int. The index of the padding token in the vocabulary.
                - emb_layer_norm_before: bool. Specifies whether layer normalization is applied before the embedding layer.
                - layer_norm_eps: float. The epsilon value for layer normalization.
                - hidden_dropout_prob: float. The dropout probability for the hidden layer.
                - position_embedding_type: str. The type of position embedding ('absolute' by default).
                - max_position_embeddings: int. The maximum number of positions for position embeddings.
                - token_dropout: float. The token dropout probability.
                - mask_token_id: int. The index of the mask token in the vocabulary.

        Returns:
            None.

        Raises:
            AttributeError: If the config object does not contain the required attributes.
            ValueError: If the config parameters are invalid or out of range.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).expand((1, -1))

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def construct(
        self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        """
        This method constructs embeddings for the EsmEmbeddings class.

        Args:
            self (EsmEmbeddings): The instance of the EsmEmbeddings class.
            input_ids (torch.Tensor, optional): The input tensor of token indices. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (torch.Tensor, optional): The position indices tensor. Defaults to None.
            inputs_embeds (torch.Tensor, optional): The input embeddings tensor. Defaults to None.
            past_key_values_length (int): The length of past key values. Defaults to 0.

        Returns:
            embeddings (torch.Tensor): The constructed embeddings tensor.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None, or if position_ids is None and input_ids is None.
            TypeError: If input_ids, attention_mask, position_ids, or inputs_embeds are not of type torch.Tensor.
            RuntimeError: If there is a runtime error during the construction process.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Note that if we want to support ESM-1 (not 1b!) in future then we need to support an
        # embedding_scale factor here.
        embeddings = inputs_embeds

        # Matt: ESM has the option to handle masking in MLM in a slightly unusual way. If the token_dropout
        # flag is False then it is handled in the same was as BERT/RoBERTa. If it is set to True, however,
        # masked tokens are treated as if they were selected for input dropout and zeroed out.
        # This "mask-dropout" is compensated for when masked tokens are not present, by scaling embeddings by
        # a factor of (fraction of unmasked tokens during training) / (fraction of unmasked tokens in sample).
        # This is analogous to the way that dropout layers scale down outputs during evaluation when not
        # actually dropping out values (or, equivalently, scale up their un-dropped outputs in training).
        if self.token_dropout:
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
                embeddings.dtype
            )

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        # Matt: I think this line was copied incorrectly from BERT, disabling it for now.
        # embeddings = self.dropout(embeddings)
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
        return position_ids.unsqueeze(0).expand(input_shape)


class EsmSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism for the ESM (Evolving Scalable Models) architecture.
    It calculates attention scores and produces context layers based on input hidden states.
    The class provides functionalities for processing queries, keys, and values, handling position embeddings,
    and implementing attention mechanisms for transformers.
    The class inherits from nn.Cell and includes methods for initializing the self-attention mechanism,
    swapping axes for attention scores, and constructing the attention mechanism.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes a new instance of the EsmSelfAttention class.

        Args:
            self: The object instance.
            config:
                An object that contains configuration parameters for the self-attention layer.

                - Type: object
                - Purpose: Specifies the configuration settings for the self-attention layer.
            position_embedding_type:
                The type of position embedding to use.

                - Type: str
                - Purpose: Specifies the type of position embedding to be used in the self-attention layer.
                - Restrictions: If None, the default value is 'absolute'.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads and the config object
                does not have an 'embedding_size' attribute.

                - Purpose: Raises an exception when the hidden size is not divisible by the number of attention heads,
                indicating an invalid configuration.
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
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

        self.is_decoder = config.is_decoder

    def swapaxes_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method swaps and reshapes the input tensor to prepare it for self-attention calculation.

        Args:
            self (EsmSelfAttention): The instance of the EsmSelfAttention class.
            x (mindspore.Tensor): The input tensor representing the scores for self-attention calculation.
                It should have a shape that can be reshaped to include the specified number of attention heads and
                attention head size.

        Returns:
            mindspore.Tensor: The tensor after swapping and reshaping to prepare for self-attention calculation.

        Raises:
            ValueError: If the input tensor shape is not compatible with the specified number of attention heads
                and attention head size.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
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
        Constructs the self-attention mechanism for the EsmSelfAttention class.

        Args:
            self (EsmSelfAttention): The instance of the EsmSelfAttention class.
            hidden_states (mindspore.Tensor):
                The input tensor of shape (batch_size, sequence_length, hidden_size) containing the hidden states.
            attention_mask (Optional[mindspore.Tensor]):
                An optional input tensor of shape (batch_size, sequence_length) containing attention mask values.
                Default is None.
            head_mask (Optional[mindspore.Tensor]):
                An optional input tensor of shape (num_attention_heads,) containing the head mask values. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                An optional input tensor of shape (batch_size, sequence_length, hidden_size) containing hidden states
                from the encoder. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                An optional input tensor of shape (batch_size, sequence_length) containing attention mask values for
                the encoder. Default is None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                An optional tuple of past key and value tensors. Default is None.
            output_attentions (Optional[bool]):
                An optional boolean value indicating whether to output attention probabilities. Default is False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the output context tensor of shape (batch_size, sequence_length, hidden_size)
                and optionally the attention probabilities tensor of shape
                (batch_size, num_attention_heads, sequence_length, sequence_length) if output_attentions is True.

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
            key_layer = ops.cat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.cat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        # Matt: Our BERT model (which this code was derived from) scales attention logits down by sqrt(head_dim).
        # ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent,
        # but not when rotary embeddings get involved. Therefore, we scale the query here to match the original
        # ESM code and fix rotary embeddings.
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

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

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in EsmModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs.to(value_layer.dtype), value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class EsmSelfOutput(nn.Cell):

    """
    The EsmSelfOutput class represents a self-output module for the ESM model.
    This class inherits from nn.Cell and contains methods for initializing the module and constructing
    the output based on input hidden states and tensors.

    Attributes:
        dense (nn.Dense): A dense layer with the specified hidden size for the ESM self-output module.
        dropout (nn.Dropout): A dropout layer with the specified dropout probability for the ESM self-output module.

    Methods:
        __init__: Initializes the EsmSelfOutput module with the provided configuration.
        construct: Constructs the output by applying the dense layer, dropout layer,
            and adding the input tensor to the hidden states.

    Example:
        ```python
        >>> config = Config(hidden_size=768, hidden_dropout_prob=0.1)
        >>> esm_self_output = EsmSelfOutput(config)
        >>> output = esm_self_output.construct(hidden_states, input_tensor)
        ```
    """
    def __init__(self, config):
        """Initializes a new instance of the EsmSelfOutput class.

        Args:
            self (EsmSelfOutput): The current instance of the class.
            config:
                A configuration object used to customize the initialization.

                - hidden_size (int): The size of the hidden layer.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        The 'construct' method in the 'EsmSelfOutput' class processes the hidden states and input tensor to construct the output.

        Args:
            hidden_states (tensor): The hidden states to be processed.
                It should be a tensor representing the hidden states of the model.
            input_tensor (tensor): The input tensor to be combined with the processed hidden states.
                It should be a tensor representing the input data.

        Returns:
            tensor: The constructed output tensor, which is the result of processing the hidden states
                and combining them with the input tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class EsmAttention(nn.Cell):

    """
    EsmAttention

    This class represents an attention mechanism for the ESM model. It inherits from nn.Cell and contains methods for
    initializing the attention mechanism, pruning attention heads, and constructing the attention output.

    Methods:
        __init__: Initializes the EsmAttention instance with the provided configuration.
        prune_heads: Prunes the specified attention heads from the EsmAttention instance.
        construct: Constructs the attention output based on the provided input states and masks.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the EsmAttention class.

        Args:
            self (EsmAttention): The instance of the EsmAttention class.
            config: The configuration object containing the settings for the attention mechanism.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def prune_heads(self, heads):
        """
        Method to prune attention heads in the EsmAttention class.

        Args:
            self (EsmAttention): The instance of the EsmAttention class.
            heads (list): A list of integers representing the attention heads to be pruned.

        Returns:
            None.

        Raises:
            None.
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
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        Args:
            self (EsmAttention): The instance of the EsmAttention class.
            hidden_states (tensor): The input hidden states for the attention mechanism.
            attention_mask (tensor, optional): Mask for the attention scores. Defaults to None.
            head_mask (tensor, optional): Mask for the individual attention heads. Defaults to None.
            encoder_hidden_states (tensor, optional): Hidden states from the encoder. Defaults to None.
            encoder_attention_mask (tensor, optional): Mask for the encoder attention scores. Defaults to None.
            past_key_value (tuple, optional): Tuple containing the past key and value tensors. Defaults to None.
            output_attentions (bool): Flag to indicate if the attention outputs should be returned. Defaults to False.

        Returns:
            tuple: A tuple containing the attention output and additional outputs from the attention mechanism.

        Raises:
            ValueError: If the dimensions of input tensors are incompatible.
            TypeError: If the input types are invalid or incompatible.
            RuntimeError: If the method encounters a runtime issue during computation.
        """
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
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


class EsmIntermediate(nn.Cell):

    """
    The 'EsmIntermediate' class represents a neural network module that performs intermediate computations
    on input hidden states. This class inherits from the 'nn.Cell' class.

    Attributes:
        dense (nn.Dense):
            A fully connected layer that maps the input hidden states to an intermediate size.

    Methods:
        __init__: Initializes the 'EsmIntermediate' instance with the given configuration.

            Parameters:

                - config: An object containing the configuration settings for the 'EsmIntermediate' instance.
            Returns:

                - None

        construct: Performs the intermediate computations on the input hidden states.

            Parameters:

                - hidden_states (mindspore.Tensor): The input hidden states to be processed.
            Returns:

                - mindspore.Tensor: The processed hidden states after going through the intermediate computations.
    """
    def __init__(self, config):
        """
        Initializes an EsmIntermediate object.

        Args:
            self: The instance of the EsmIntermediate class.
            config: An object containing configuration parameters for the EsmIntermediate,
                including hidden_size and intermediate_size.

                - Type: object
                - Purpose: Specifies the configuration parameters for the EsmIntermediate.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the intermediate layer of the Esm model.

        Args:
            self (EsmIntermediate): The instance of the EsmIntermediate class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.

        Returns:
            mindspore.Tensor:
                The tensor representing the intermediate hidden states after applying the dense layer and the
                gelu activation function.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class EsmOutput(nn.Cell):

    """
    EsmOutput is a class representing the output layer for the ESM (Embedding-based Language Model) model.

    This class inherits from nn.Cell and contains methods for initializing the class and constructing the output layer.

    The __init__ method initializes the EsmOutput instance with the provided configuration. It sets up a dense layer
    with the specified intermediate and hidden sizes, and a dropout layer with the given dropout probability.

    The construct method takes hidden_states and input_tensor as input and processes the hidden_states through
    the dense layer, applies dropout, and adds the input_tensor to the result. The processed hidden_states are then
    returned.

    Note:
        This docstring is based on the provided information and does not include code signatures or any other code.
    """
    def __init__(self, config):
        """
        Initializes an instance of the EsmOutput class.

        Args:
            self (EsmOutput): The current instance of the EsmOutput class.
            config (object):
                An object containing configuration parameters.

                - intermediate_size (int): Size of the intermediate layer.
                - hidden_size (int): Size of the hidden layer.
                - hidden_dropout_prob (float): Dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required attributes.
            RuntimeError: If there are issues during initialization of the dense or dropout layers.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        Constructs the output of the EsmOutput class.

        Args:
            self: An instance of the EsmOutput class.
            hidden_states (tensor): The hidden states tensor.
                This tensor represents the intermediate hidden states of the model.
                It should have a shape of [batch_size, sequence_length, hidden_size].
            input_tensor (tensor): The input tensor.
                This tensor represents the input to be added to the hidden states.
                It should have the same shape as the hidden states tensor.

        Returns:
            None

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class EsmLayer(nn.Cell):

    """
    The EsmLayer class represents a layer for the ESM (Evolved Transformer with Split Mixture of Experts) model.
    It is used for processing input data and performing self-attention and feed-forward operations.
    This class inherits from the nn.Cell class.

    Attributes:
        chunk_size_feed_forward (int): The chunk size for feed-forward operations.
        seq_len_dim (int): The dimension for sequence length.
        attention (EsmAttention): An instance of the EsmAttention class for self-attention.
        is_decoder (bool): Indicates whether the layer is used as a decoder model.
        add_cross_attention (bool): Indicates whether cross-attention is added to the model.
        crossattention (EsmAttention): An instance of the EsmAttention class for cross-attention (if added).
        intermediate (EsmIntermediate): An instance of the EsmIntermediate class for intermediate processing.
        output (EsmOutput): An instance of the EsmOutput class for producing the final output.
        LayerNorm (nn.LayerNorm): An instance of the nn.LayerNorm class for layer normalization.

    Methods:
        construct: Processes the input hidden states and performs self-attention and cross-attention (if applicable),
            and returns the outputs.
        feed_forward_chunk:
            Performs layer normalization, intermediate processing, and produces the final layer output based on the
            attention output.

    Raises:
        RuntimeError: If the class is not used as a decoder model while cross-attention is added.
        AttributeError: If `encoder_hidden_states` are passed without instantiating cross-attention layers
            by setting `config.add_cross_attention=True`.
    """
    def __init__(self, config):
        """Initializes an instance of the EsmLayer class.

        Args:
            self: The EsmLayer instance.
            config: The configuration options for the EsmLayer.
                This parameter expects an object of a specific class.

                - chunk_size_feed_forward (int): The chunk size for feed-forward operations.
                - is_decoder (bool): Indicates whether the EsmLayer is used as a decoder model.
                - add_cross_attention (bool): Indicates whether cross attention is added.
                - hidden_size (int): The size of the hidden state.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            RuntimeError: If cross attention is added and the EsmLayer is not used as a decoder model.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = EsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        Construct the EsmLayer.

        This method constructs the EsmLayer by performing self-attention and cross-attention operations on the
        input hidden states.

        Args:
            self (EsmLayer): An instance of the EsmLayer class.
            hidden_states (tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (tensor, optional):
                The attention mask of shape (batch_size, sequence_length) indicating which tokens should be attended to.
                Defaults to None.
            head_mask (tensor, optional):
                The head mask of shape (num_heads,) or (num_layers, num_heads) indicating which heads should be
                masked out. Defaults to None.
            encoder_hidden_states (tensor, optional):
                The hidden states of the encoder, if the EsmLayer is used as a decoder. Defaults to None.
            encoder_attention_mask (tensor, optional):
                The attention mask for the encoder, if the EsmLayer is used as a decoder. Defaults to None.
            past_key_value (tuple, optional):
                The tuple containing the past key and value tensors of shape
                (2, batch_size, num_heads, sequence_length, head_size), if available. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            tuple:
                A tuple containing the following elements:

                - attention_output (tensor): The output of the self-attention operation of shape
                (batch_size, sequence_length, hidden_size).
                - outputs (tuple): A tuple containing the intermediate outputs of the self-attention and cross-attention
                operations.
                - present_key_value (tensor, optional): The present key and value tensors of shape
                (2, batch_size, num_heads, sequence_length, head_size), if the EsmLayer is used as a decoder.

        Raises:
            AttributeError: If `encoder_hidden_states` are passed and the EsmLayer is not instantiated with
                cross-attention layers.
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
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
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

        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Method: feed_forward_chunk

        Description:
            This method performs the feed-forward chunk operation within the EsmLayer class.

        Args:
            self (EsmLayer):
                The instance of the EsmLayer class.

                - Type: EsmLayer object
                - Purpose: Represents the current instance of the EsmLayer class.
                - Restrictions: Must be an instance of the EsmLayer class.

            attention_output (tensor):
                The input tensor representing the attention output.

                - Type: Tensor
                - Purpose: Represents the attention output to be processed.
                - Restrictions: Should be a valid input tensor.

        Returns:
            None.

        Raises:
            None.
        """
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EsmEncoder(nn.Cell):

    """
    The `EsmEncoder` class represents a Python class that serves as an encoder in the ESM
    (Encoder-Decoder Semantic Mapping) model. This class inherits from the `nn.Cell` class.

    The `EsmEncoder` class has the following attributes:

    Attributes:
        `config`: An object that encapsulates the configuration parameters for the encoder.
        `layer`: A list of `EsmLayer` instances, representing the individual layers of the encoder.
        `emb_layer_norm_after`: An instance of the `nn.LayerNorm` class, used for layer normalization of the
            hidden states.
        `gradient_checkpointing`: A boolean flag indicating whether gradient checkpointing is enabled.

    The `EsmEncoder` class provides the following methods:

    Methods:
        `__init__`: Initializes an instance of the `EsmEncoder` class.
        `construct`: Constructs the encoder layers and returns the final hidden states.

    Please note that this class assumes the existence of the `EsmLayer` class, which represents the individual layers
    of the encoder. The `EsmLayer` class is not defined within this docstring.

    For more details on the purpose and functionality of the `EsmEncoder` class, please refer to the code implementation
    and associated documentation.
    """
    def __init__(self, config):
        """
        Initializes an instance of the EsmEncoder class.

        Args:
            self: The instance of the EsmEncoder class.
            config: A dictionary containing configuration parameters for the EsmEncoder.
                It is expected to have the following keys:

                - num_hidden_layers: An integer specifying the number of hidden layers.
                - hidden_size: An integer specifying the size of the hidden layers.
                - layer_norm_eps: A float specifying the epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        This method constructs the EsmEncoder model with the given input parameters.

        Args:
            self: The instance of the EsmEncoder class.
            hidden_states (Tensor): The input hidden states for the encoder.
            attention_mask (Tensor, optional): Mask to prevent attention to certain positions.
            head_mask (List[Tensor], optional): Mask to prevent attention to certain heads.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder.
            encoder_attention_mask (Tensor, optional): Mask to prevent attention to encoder positions.
            past_key_values (Tuple[Tensor], optional): Cached key values from previous decoding steps.
            use_cache (bool, optional): Whether to use caching for decoding.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return outputs as a dict.

        Returns:
            None

        Raises:
            Warning: If `use_cache=True` is incompatible with `config.gradient_checkpointing=True`.
        """
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

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
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

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


# Copied from transformers.models.bert.modeling_bert.BertPooler
class EsmPooler(nn.Cell):

    """
    This class represents an EsmPooler, which is a pooler layer used in a neural network model.

    The EsmPooler class inherits from the nn.Cell class, which is a base class for all neural network cells in MindSpore.

    Attributes:
        dense (nn.Dense):
            A fully connected layer that takes the hidden states as input and output the pooled representation.
        activation (nn.Tanh): An activation function that is applied to the output of the dense layer.

    Methods:
        construct:
            This method takes the hidden states tensor as input and returns the pooled representation.

    Usage:
        To use the EsmPooler, instantiate an object of this class and call the construct() method passing the
        hidden states tensor.

    Example:
        ```python
        >>> config = {
        >>>     'hidden_size': 768
        >>> }
        ...
        >>> pooler = EsmPooler(config)
        >>> hidden_states = mindspore.Tensor(np.random.randn(5, 10, 768), dtype=mindspore.float32)
        >>> pooled_output = pooler.construct(hidden_states)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the EsmPooler class.

        Args:
            self (EsmPooler): The instance of the EsmPooler class.
            config (object):
                An object containing configuration settings.

                - hidden_size (int): The size of the hidden layer.
                It specifies the dimensions for the Dense layer and activation function.

                Note:
                    The config object must have the required attributes for successful initialization.

        Returns:
            None.

        Raises:
            AttributeError: If the necessary attributes are missing in the config object.
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the hidden_size value is invalid or out of range.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a pooled output from the hidden states of the ESM model.

        Args:
            self (EsmPooler): The instance of the EsmPooler class.
            hidden_states (mindspore.Tensor): The hidden states of the ESM model, represented as a tensor.
                It contains the contextual embeddings of the input tokens.
                Expected shape: (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The pooled output tensor obtained from the hidden states.
                It represents the aggregated representation of the input sequence.
                Expected shape: (batch_size, hidden_size).

        Raises:
            None
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = EsmConfig
    base_model_prefix = "esm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EsmLayer", "EsmFoldTriangularSelfAttentionBlock", "EsmEmbeddings"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
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
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class EsmModel(EsmPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes an instance of the EsmModel class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing various settings for the model.
            add_pooling_layer (bool, optional): A flag indicating whether to include a pooling layer in the model.
                Default is True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = EsmEncoder(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the ESMM model.

        Args:
            self: An instance of the EsmModel class.

        Returns:
            word_embeddings: This method returns the word embeddings for input data, represented as a tensor.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the EsmModel.

        Args:
            self (EsmModel): The instance of the EsmModel class.
            value: The input embeddings to be set. This should be of type `torch.Tensor`.

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

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
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
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors
                of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
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
            attention_mask = ops.ones(((batch_size, seq_length + past_key_values_length)))

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
            attention_mask=attention_mask,
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

    def predict_contacts(self, tokens, attention_mask):
        """
        Predicts contacts using the EsmModel.

        Args:
            self (EsmModel): An instance of the EsmModel class.
            tokens (Tensor): The input tokens for prediction.
            attention_mask (Tensor): The attention mask for the input tokens.

        Returns:
            None.

        Raises:
            None.
        """
        attns = self(tokens, attention_mask=attention_mask, return_dict=True, output_attentions=True).attentions
        attns = ops.stack(attns, axis=1)  # Matches the original model layout
        # In the original model, attentions for padding tokens are completely zeroed out.
        # This makes no difference most of the time because the other tokens won't attend to them,
        # but it does for the contact prediction task, which takes attentions as input,
        # so we have to mimic that here.
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(tokens, attns)


class EsmForMaskedLM(EsmPreTrainedModel):

    """
    Represents an ESM (Evolutionary Scale Modeling) model for masked language modeling (MLM), inheriting from EsmPreTrainedModel.
    This class provides the functionality to perform masked language modeling using the ESM model.

    The EsmForMaskedLM class contains methods for initializing the model, getting and setting output embeddings,
    constructing the model, and predicting contacts.
    The model architecture includes an ESM model and a language modeling head (lm_head).
    The construct method takes input_ids, attention_mask, position_ids, head_mask, inputs_embeds, encoder_hidden_states,
    encoder_attention_mask, labels, output_attentions, output_hidden_states, and return_dict as input arguments and
    returns the masked language modeling loss and other outputs.
    The predict_contacts method takes tokens and attention_mask as input and returns the predicted contacts using the
    ESM model.

    Note:
        - If using `EsmForMaskedLM`, ensure `config.is_decoder=False` for bi-directional self-attention.
        - Labels for computing the masked language modeling loss should be indices in `[-100, 0, ..., config.vocab_size]`.
        Tokens with indices set to `-100` are ignored (masked), and the loss is only computed for the tokens with labels
        in `[0, ..., config.vocab_size]`.

    """
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config):
        """
        Initializes an instance of EsmForMaskedLM.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing model hyperparameters.
                It must have attributes like 'is_decoder', 'add_pooling_layer', and 'init_weights'.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        """
        This method returns the output embeddings for the language model head.

        Args:
            self: An instance of the EsmForMaskedLM class.

        Returns:
            decoder: The method returns the output embeddings for the language model head.

        Raises:
            None.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the ESM model.

        Args:
            self (EsmForMaskedLM): The instance of the EsmForMaskedLM class.
            new_embeddings (torch.nn.Module): The new embeddings to be set as output embeddings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the provided new_embeddings is not of type torch.nn.Module.
            AttributeError: If the lm_head.decoder attribute is not present in the EsmForMaskedLM instance.
        """
        self.lm_head.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict_contacts(self, tokens, attention_mask):
        """
        This method predicts contacts using the ESM (Evolutionary Scale Modeling) for Masked Language Modeling.

        Args:
            self (EsmForMaskedLM): The instance of the EsmForMaskedLM class.
            tokens (Tensor): The input tokens for prediction.
            attention_mask (Tensor): The attention mask for the input tokens.
                It masks the tokens that should not be attended to, specifying which tokens should be attended to
                and which should not.

        Returns:
            None.

        Raises:
            None.
        """
        return self.esm.predict_contacts(tokens, attention_mask=attention_mask)


class EsmLMHead(nn.Cell):
    """ESM Head for masked language modeling."""
    def __init__(self, config):
        """
        Initializes an instance of the EsmLMHead class.

        Args:
            self (EsmLMHead): The instance of the class.
            config: An object that contains the configuration parameters for the EsmLMHead.
                It should have the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - vocab_size (int): The size of the vocabulary.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.bias = Parameter(ops.zeros(config.vocab_size))

    def construct(self, features, **kwargs):
        """
        Constructs the output of the EsmLMHead class.

        Args:
            self (EsmLMHead): The object instance of the EsmLMHead class.
            features (tensor): Input features for constructing the output.

        Returns:
            None: The constructed output is returned as a tensor.

        Raises:
            None.
        """
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x


class EsmForSequenceClassification(EsmPreTrainedModel):

    """
    This class represents an ESM (Evoformer) model for sequence classification tasks.
    It is a subclass of EsmPreTrainedModel, which provides the underlying architecture and functionality.

    Attributes:
        num_labels (int): The number of labels for the classification task.
        config (EsmConfig): The configuration object for the ESM model.
        esm (EsmModel): The ESM model instance.
        classifier (EsmClassificationHead): The classification head for the ESM model.

    Methods:
        __init__: Initializes the EsmForSequenceClassification instance.
        construct: Constructs the ESM model for sequence classification.

    """
    def __init__(self, config):
        """
        Initializes an instance of EsmForSequenceClassification.

        Args:
            self: The instance of the class.
            config:
                An object containing the configuration parameters for the model.

                - Type: object
                - Purpose: To configure the model and its components.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config)

        self.init_weights()

    def construct(
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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
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
                elif self.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EsmForTokenClassification(EsmPreTrainedModel):

    """
    EsmForTokenClassification is a class that represents a token classification model based on the ESM
    (Evoformer Sequence Model) architecture. This class extends EsmPreTrainedModel to leverage pre-trained
    weights and configurations for efficient token classification tasks. It includes methods for initializing the model,
    constructing the forward pass, and computing the token classification loss.

    The __init__ method initializes the EsmForTokenClassification model with configurable parameters such as the number
    of labels, dropout probability, and hidden layer sizes. It also sets up the ESM model, dropout layer, and the
    classifier for token classification.

    The construct method defines the forward pass of the model, taking input tensors such as input_ids, attention_mask,
    position_ids, etc., and returning the token classification output.
    It computes the logits for token classification based on the sequence_output from the ESM model and calculates the
    cross-entropy loss if labels are provided. The method allows for returning additional outputs like hidden states and
    attentions based on the return_dict parameter.

    Note:
        This docstring is a high-level summary and does not include method signatures or implementation details.
    """
    def __init__(self, config):
        """
        Initializes an instance of the EsmForTokenClassification class.

        Args:
            self: The instance of the EsmForTokenClassification class.
            config:
                An instance of the configuration class containing the model configuration parameters.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid instance of the configuration class.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the correct type.
            ValueError: If the config.num_labels is not provided or is invalid.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        self.init_weights()

    def construct(
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
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


class EsmClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        """
        Initializes an instance of the EsmClassificationHead class.
        
        Args:
            self (EsmClassificationHead): The instance of the class.
            config:
                An object containing configuration parameters for the head.

                - Type: object
                - Purpose: The configuration for the classification head.
                - Restrictions: Must be a valid configuration object.
        
        Returns:
            None.
        
        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.out_proj = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, features, **kwargs):
        """
        Constructs the classification head for ESM (Evoformer) model.
        
        Args:
            self (EsmClassificationHead): The instance of the EsmClassificationHead class.
            features (tensor): The input features tensor of shape (batch_size, sequence_length, num_features). 
                It represents the input features for classification.
        
        Returns:
            None: The method modifies the EsmClassificationHead instance in-place.
        
        Raises:
            ValueError: If the features tensor is not of the expected shape.
            TypeError: If the features tensor is not a valid tensor object.
        """
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ops.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: mindspore.Tensor x:

    Returns: mindspore.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (ops.cumsum(mask, axis=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

__all__ = [
    "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
    "EsmForMaskedLM",
    "EsmForSequenceClassification",
    "EsmForTokenClassification",
    "EsmModel",
    "EsmPreTrainedModel",
]
