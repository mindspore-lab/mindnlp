# coding=utf-8
# Copyright Studio Ousia and The HuggingFace Inc. team.
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
"""MindSpore LUKE model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mindspore
from mindnlp.core import nn, ops
from mindnlp.core.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, gelu
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward
from ....utils import (
    ModelOutput,
    logging,
)
from .configuration_luke import LukeConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LukeConfig"
_CHECKPOINT_FOR_DOC = "studio-ousia/luke-base"


@dataclass
class BaseLukeModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Base class for outputs of the LUKE model.

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        entity_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, entity_length, hidden_size)`):
            Sequence of entity hidden-states at the output of the last layer of the model.
        pooler_output (`mindspore.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length +
            entity_length, sequence_length + entity_length)`. Attentions weights after the attention softmax, used to
            compute the weighted average in the self-attention heads.
    """

    entity_last_hidden_state: mindspore.Tensor = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class BaseLukeModelOutput(BaseModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        entity_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, entity_length, hidden_size)`):
            Sequence of entity hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    entity_last_hidden_state: mindspore.Tensor = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class LukeMaskedLMOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            The sum of masked language modeling (MLM) loss and entity prediction loss.
        mlm_loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        mep_loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked entity prediction (MEP) loss.
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        entity_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the entity prediction head (scores for each entity vocabulary token before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[mindspore.Tensor] = None
    mlm_loss: Optional[mindspore.Tensor] = None
    mep_loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    entity_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class EntityClassificationOutput(ModelOutput):
    """
    Outputs of entity classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class EntityPairClassificationOutput(ModelOutput):
    """
    Outputs of entity pair classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class EntitySpanClassificationOutput(ModelOutput):
    """
    Outputs of entity span classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`mindspore.Tensor` of shape `(batch_size, entity_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class LukeSequenceClassifierOutput(ModelOutput):
    """
    Outputs of sentence classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class LukeTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class LukeQuestionAnsweringModelOutput(ModelOutput):
    """
    Outputs of question answering models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[mindspore.Tensor] = None
    start_logits: mindspore.Tensor = None
    end_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class LukeMultipleChoiceModelOutput(ModelOutput):
    """
    Outputs of multiple choice models.

    Args:
        loss (`mindspore.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`mindspore.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        entity_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    entity_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


class LukeEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
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


class LukeEntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super().__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, entity_ids: mindspore.Tensor, position_ids: mindspore.Tensor, token_type_ids: mindspore.Tensor = None
    ):
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = ops.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(axis=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LukeSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_entity_aware_attention = config.use_entity_aware_attention

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.use_entity_aware_attention:
            self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_hidden_states.shape[1]

        if entity_hidden_states is None:
            concat_hidden_states = word_hidden_states
        else:
            concat_hidden_states = ops.cat([word_hidden_states, entity_hidden_states], dim=1)

        key_layer = self.transpose_for_scores(self.key(concat_hidden_states))
        value_layer = self.transpose_for_scores(self.value(concat_hidden_states))

        if self.use_entity_aware_attention and entity_hidden_states is not None:
            # compute query vectors using word-word (w2w), word-entity (w2e), entity-word (e2w), entity-entity (e2e)
            # query layers
            w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

            # compute w2w, w2e, e2w, and e2e key vectors used with the query vectors computed above
            w2w_key_layer = key_layer[:, :, :word_size, :]
            e2w_key_layer = key_layer[:, :, :word_size, :]
            w2e_key_layer = key_layer[:, :, word_size:, :]
            e2e_key_layer = key_layer[:, :, word_size:, :]

            # compute attention scores based on the dot product between the query and key vectors
            w2w_attention_scores = ops.matmul(w2w_query_layer, w2w_key_layer.swapaxes(-1, -2))
            w2e_attention_scores = ops.matmul(w2e_query_layer, w2e_key_layer.swapaxes(-1, -2))
            e2w_attention_scores = ops.matmul(e2w_query_layer, e2w_key_layer.swapaxes(-1, -2))
            e2e_attention_scores = ops.matmul(e2e_query_layer, e2e_key_layer.swapaxes(-1, -2))

            # combine attention scores to create the final attention score matrix
            word_attention_scores = ops.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
            entity_attention_scores = ops.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
            attention_scores = ops.cat([word_attention_scores, entity_attention_scores], dim=2)

        else:
            query_layer = self.transpose_for_scores(self.query(concat_hidden_states))
            attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in LukeModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

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

        output_word_hidden_states = context_layer[:, :word_size, :]
        if entity_hidden_states is None:
            output_entity_hidden_states = None
        else:
            output_entity_hidden_states = context_layer[:, word_size:, :]

        if output_attentions:
            outputs = (output_word_hidden_states, output_entity_hidden_states, attention_probs)
        else:
            outputs = (output_word_hidden_states, output_entity_hidden_states)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LukeSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LukeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LukeSelfAttention(config)
        self.output = LukeSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError("LUKE does not support the pruning of attention heads")

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_hidden_states.shape[1]
        self_outputs = self.self(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        if entity_hidden_states is None:
            concat_self_outputs = self_outputs[0]
            concat_hidden_states = word_hidden_states
        else:
            concat_self_outputs = ops.cat(self_outputs[:2], dim=1)
            concat_hidden_states = ops.cat([word_hidden_states, entity_hidden_states], dim=1)

        attention_output = self.output(concat_self_outputs, concat_hidden_states)

        word_attention_output = attention_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_attention_output = None
        else:
            entity_attention_output = attention_output[:, word_size:, :]

        # add attentions if we output them
        outputs = (word_attention_output, entity_attention_output) + self_outputs[2:]

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class LukeIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LukeOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LukeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LukeAttention(config)
        self.intermediate = LukeIntermediate(config)
        self.output = LukeOutput(config)

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_hidden_states.shape[1]

        self_attention_outputs = self.attention(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        if entity_hidden_states is None:
            concat_attention_output = self_attention_outputs[0]
        else:
            concat_attention_output = ops.cat(self_attention_outputs[:2], dim=1)

        outputs = self_attention_outputs[2:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, concat_attention_output
        )
        word_layer_output = layer_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_layer_output = None
        else:
            entity_layer_output = layer_output[:, word_size:, :]

        outputs = (word_layer_output, entity_layer_output) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LukeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_word_hidden_states = () if output_hidden_states else None
        all_entity_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            word_hidden_states = layer_outputs[0]

            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    word_hidden_states,
                    all_word_hidden_states,
                    all_self_attentions,
                    entity_hidden_states,
                    all_entity_hidden_states,
                ]
                if v is not None
            )
        return BaseLukeModelOutput(
            last_hidden_state=word_hidden_states,
            hidden_states=all_word_hidden_states,
            attentions=all_self_attentions,
            entity_last_hidden_state=entity_hidden_states,
            entity_hidden_states=all_entity_hidden_states,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class LukePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EntityPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class EntityPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        self.bias = nn.Parameter(ops.zeros(config.entity_vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class LukePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LukeConfig
    base_model_prefix = "luke"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LukeAttention", "LukeEntityEmbeddings"]

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                nn.init.zeros_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)


class LukeModel(LukePreTrainedModel):
    def __init__(self, config: LukeConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = LukeEmbeddings(config)
        self.entity_embeddings = LukeEntityEmbeddings(config)
        self.encoder = LukeEncoder(config)

        self.pooler = LukePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_entity_embeddings(self):
        return self.entity_embeddings.entity_embeddings

    def set_entity_embeddings(self, value):
        self.entity_embeddings.entity_embeddings = value

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError("LUKE does not support the pruning of attention heads")

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseLukeModelOutputWithPooling]:
        r"""

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LukeModel

        >>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
        >>> model = LukeModel.from_pretrained("studio-ousia/luke-base")
        # Compute the contextualized entity representation corresponding to the entity mention "Beyoncé"

        >>> text = "Beyoncé lives in Los Angeles."
        >>> entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"

        >>> encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="ms")
        >>> outputs = model(**encoding)
        >>> word_last_hidden_state = outputs.last_hidden_state
        >>> entity_last_hidden_state = outputs.entity_last_hidden_state
        # Input Wikipedia entities to obtain enriched contextualized representations of word tokens

        >>> text = "Beyoncé lives in Los Angeles."
        >>> entities = [
        ...     "Beyoncé",
        ...     "Los Angeles",
        ... ]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "Los Angeles"
        >>> entity_spans = [
        ...     (0, 7),
        ...     (17, 28),
        ... ]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"

        >>> encoding = tokenizer(
        ...     text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="ms"
        ... )
        >>> outputs = model(**encoding)
        >>> word_last_hidden_state = outputs.last_hidden_state
        >>> entity_last_hidden_state = outputs.entity_last_hidden_state
        ```"""
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

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, seq_length))
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)
        if entity_ids is not None:
            entity_seq_length = entity_ids.shape[1]
            if entity_attention_mask is None:
                entity_attention_mask = ops.ones((batch_size, entity_seq_length))
            if entity_token_type_ids is None:
                entity_token_type_ids = ops.zeros((batch_size, entity_seq_length), dtype=mindspore.int64)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # First, compute word embeddings
        word_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Second, compute extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, entity_attention_mask)

        # Third, compute entity embeddings and concatenate with word embeddings
        if entity_ids is None:
            entity_embedding_output = None
        else:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_token_type_ids)

        # Fourth, send embeddings through the model
        encoder_outputs = self.encoder(
            word_embedding_output,
            entity_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Fifth, get the output. LukeModel outputs the same as BertModel, namely sequence_output of shape (batch_size, seq_len, hidden_size)
        sequence_output = encoder_outputs[0]

        # Sixth, we compute the pooled_output, word_sequence_output and entity_sequence_output based on the sequence_output
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseLukeModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            entity_last_hidden_state=encoder_outputs.entity_last_hidden_state,
            entity_hidden_states=encoder_outputs.entity_hidden_states,
        )

    def get_extended_attention_mask(
        self, word_attention_mask: mindspore.Tensor, entity_attention_mask: Optional[mindspore.Tensor]
    ):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            word_attention_mask (`mindspore.Tensor`):
                Attention mask for word tokens with ones indicating tokens to attend to, zeros for tokens to ignore.
            entity_attention_mask (`mindspore.Tensor`, *optional*):
                Attention mask for entity tokens with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            `mindspore.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = ops.cat([attention_mask, entity_attention_mask], dim=-1)

        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * float(ops.finfo(self.dtype).min)
        return extended_attention_mask


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: mindspore.Tensor x:

    Returns: mindspore.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (ops.cumsum(mask, dim=1).type_as(mask)) * mask
    return incremental_indices.long() + padding_idx


# Copied from transformers.models.roberta.modeling_roberta.RobertaLMHead
class LukeLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(ops.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        self.bias = self.decoder.bias


class LukeForMaskedLM(LukePreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias", "entity_predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)

        self.lm_head = LukeLMHead(config)
        self.entity_predictions = EntityPredictionHead(config)

        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        super().tie_weights()
        self._tie_or_clone_weights(self.entity_predictions.decoder, self.luke.entity_embeddings.entity_embeddings)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        entity_labels: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LukeMaskedLMOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        entity_labels (`mindspore.Tensor` of shape `(batch_size, entity_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        loss = None

        mlm_loss = None
        logits = self.lm_head(outputs.last_hidden_state)
        if labels is not None:
            mlm_loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            if loss is None:
                loss = mlm_loss

        mep_loss = None
        entity_logits = None
        if outputs.entity_last_hidden_state is not None:
            entity_logits = self.entity_predictions(outputs.entity_last_hidden_state)
            if entity_labels is not None:
                mep_loss = self.loss_fn(entity_logits.view(-1, self.config.entity_vocab_size), entity_labels.view(-1))
                if loss is None:
                    loss = mep_loss
                else:
                    loss = loss + mep_loss

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    mlm_loss,
                    mep_loss,
                    logits,
                    entity_logits,
                    outputs.hidden_states,
                    outputs.entity_hidden_states,
                    outputs.attentions,
                ]
                if v is not None
            )

        return LukeMaskedLMOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            mep_loss=mep_loss,
            logits=logits,
            entity_logits=entity_logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )


class LukeForEntityClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EntityClassificationOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)` or `(batch_size, num_labels)`, *optional*):
            Labels for computing the classification loss. If the shape is `(batch_size,)`, the cross entropy loss is
            used for the single-label classification. In this case, labels should contain the indices that should be in
            `[0, ..., config.num_labels - 1]`. If the shape is `(batch_size, num_labels)`, the binary cross entropy
            loss is used for the multi-label classification. In this case, labels should only contain `[0, 1]`, where 0
            and 1 indicate false and true, respectively.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LukeForEntityClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
        >>> model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

        >>> text = "Beyoncé lives in Los Angeles."
        >>> entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
        >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="ms")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: person
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        feature_vector = outputs.entity_last_hidden_state[:, 0, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 1, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 1:
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

        if not return_dict:
            return tuple(
                v
                for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions]
                if v is not None
            )

        return EntityClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )


class LukeForEntityPairClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels, False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EntityPairClassificationOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)` or `(batch_size, num_labels)`, *optional*):
            Labels for computing the classification loss. If the shape is `(batch_size,)`, the cross entropy loss is
            used for the single-label classification. In this case, labels should contain the indices that should be in
            `[0, ..., config.num_labels - 1]`. If the shape is `(batch_size, num_labels)`, the binary cross entropy
            loss is used for the multi-label classification. In this case, labels should only contain `[0, 1]`, where 0
            and 1 indicate false and true, respectively.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LukeForEntityPairClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
        >>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

        >>> text = "Beyoncé lives in Los Angeles."
        >>> entity_spans = [
        ...     (0, 7),
        ...     (17, 28),
        ... ]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
        >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="ms")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: per:cities_of_residence
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        feature_vector = ops.cat(
            [outputs.entity_last_hidden_state[:, 0, :], outputs.entity_last_hidden_state[:, 1, :]], dim=1
        )
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 1, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 1:
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

        if not return_dict:
            return tuple(
                v
                for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions]
                if v is not None
            )

        return EntityPairClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )


class LukeForEntitySpanClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        entity_start_positions: Optional[mindspore.Tensor] = None,
        entity_end_positions: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EntitySpanClassificationOutput]:
        r"""
        entity_start_positions (`mindspore.Tensor`):
            The start positions of entities in the word token sequence.

        entity_end_positions (`mindspore.Tensor`):
            The end positions of entities in the word token sequence.

        labels (`mindspore.Tensor` of shape `(batch_size, entity_length)` or `(batch_size, entity_length, num_labels)`, *optional*):
            Labels for computing the classification loss. If the shape is `(batch_size, entity_length)`, the cross
            entropy loss is used for the single-label classification. In this case, labels should contain the indices
            that should be in `[0, ..., config.num_labels - 1]`. If the shape is `(batch_size, entity_length,
            num_labels)`, the binary cross entropy loss is used for the multi-label classification. In this case,
            labels should only contain `[0, 1]`, where 0 and 1 indicate false and true, respectively.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LukeForEntitySpanClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
        >>> model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

        >>> text = "Beyoncé lives in Los Angeles"
        # List all possible entity spans in the text

        >>> word_start_positions = [0, 8, 14, 17, 21]  # character-based start positions of word tokens
        >>> word_end_positions = [7, 13, 16, 20, 28]  # character-based end positions of word tokens
        >>> entity_spans = []
        >>> for i, start_pos in enumerate(word_start_positions):
        ...     for end_pos in word_end_positions[i:]:
        ...         entity_spans.append((start_pos, end_pos))

        >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="ms")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> predicted_class_indices = logits.argmax(-1).squeeze().tolist()
        >>> for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
        ...     if predicted_class_idx != 0:
        ...         print(text[span[0] : span[1]], model.config.id2label[predicted_class_idx])
        Beyoncé PER
        Los Angeles LOC
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_size = outputs.last_hidden_state.shape[-1]

        entity_start_positions = entity_start_positions.unsqueeze(-1).broadcast_to((-1, -1, hidden_size))
        start_states = ops.gather(outputs.last_hidden_state, -2, entity_start_positions)

        entity_end_positions = entity_end_positions.unsqueeze(-1).broadcast_to((-1, -1, hidden_size))
        end_states = ops.gather(outputs.last_hidden_state, -2, entity_end_positions)

        feature_vector = ops.cat([start_states, end_states, outputs.entity_last_hidden_state], dim=2)

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 2, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 2:
                loss = nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

        if not return_dict:
            return tuple(
                v
                for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions]
                if v is not None
            )

        return EntitySpanClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )


class LukeForSequenceClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.luke = LukeModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LukeSequenceClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            return tuple(
                v
                for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions]
                if v is not None
            )

        return LukeSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )


class LukeForTokenClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.luke = LukeModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LukeTokenClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            return tuple(
                v
                for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions]
                if v is not None
            )

        return LukeTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )


class LukeForQuestionAnswering(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.luke = LukeModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LukeQuestionAnsweringModelOutput]:
        r"""
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

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

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
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            return tuple(
                v
                for v in [
                    total_loss,
                    start_logits,
                    end_logits,
                    outputs.hidden_states,
                    outputs.entity_hidden_states,
                    outputs.attentions,
                ]
                if v is not None
            )

        return LukeQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )


class LukeForMultipleChoice(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        entity_ids: Optional[mindspore.Tensor] = None,
        entity_attention_mask: Optional[mindspore.Tensor] = None,
        entity_token_type_ids: Optional[mindspore.Tensor] = None,
        entity_position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LukeMultipleChoiceModelOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        entity_ids = entity_ids.view(-1, entity_ids.shape[-1]) if entity_ids is not None else None
        entity_attention_mask = (
            entity_attention_mask.view(-1, entity_attention_mask.shape[-1])
            if entity_attention_mask is not None
            else None
        )
        entity_token_type_ids = (
            entity_token_type_ids.view(-1, entity_token_type_ids.shape[-1])
            if entity_token_type_ids is not None
            else None
        )
        entity_position_ids = (
            entity_position_ids.view(-1, entity_position_ids.shape[-2], entity_position_ids.shape[-1])
            if entity_position_ids is not None
            else None
        )

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    reshaped_logits,
                    outputs.hidden_states,
                    outputs.entity_hidden_states,
                    outputs.attentions,
                ]
                if v is not None
            )

        return LukeMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
        "LukeForEntityClassification",
        "LukeForEntityPairClassification",
        "LukeForEntitySpanClassification",
        "LukeForMultipleChoice",
        "LukeForQuestionAnswering",
        "LukeForSequenceClassification",
        "LukeForTokenClassification",
        "LukeForMaskedLM",
        "LukeModel",
        "LukePreTrainedModel",
    ]
