# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Ernie Models
"""

from typing import Optional, Tuple

import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer
from .ernie_config import ErnieConfig
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel


__all__ = [
    "ErnieModel",
    "ErniePretrainedModel",
    "ErnieForSequenceClassification",
    "ErnieForTokenClassification",
    "ErnieForQuestionAnswering",
    "ErnieForPretraining",
    "ErniePretrainingCriterion",
    "ErnieForMaskedLM",
    "ErnieForMultipleChoice",
    "UIE"
]

class ErnieEmbeddings(nn.Cell):
    """
    Ernie Embeddings for word, position and token_type embeddings.
    """

    def __init__(self, config: ErnieConfig):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
        self.use_task_id = config.use_task_id
        self.task_id = config.task_id
        if self.use_task_id:
            self.task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size])
        self.dropout = nn.Dropout(config.hidden_dropout_prob, p=0.5)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        task_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values_length: int = 0,
    ):
        r"""
            ErnieEmbedding Construct
        """
        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        input_shape = ops.shape(inputs_embeds)[:-1]

        if position_ids is None:
            ones = ops.ones(input_shape, mindspore.int64)
            seq_length = ops.cumsum(ones, axis=1)
            position_ids = seq_length - ones

            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length

            ops.stop_gradient(position_ids)
            # position_ids.stop_gradient = True

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = ops.zeros(input_shape, mindspore.int64)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = ops.ones(
                    input_shape, mindspore.int64) * self.task_id
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings = embeddings + task_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErniePretrainedModel(PreTrainedModel):
    """
    Ernie Pretrained Model.
    """

    config_class = ErnieConfig


    # TODO
    def get_input_embeddings(self):
        pass

    #TODO
    def get_position_embeddings(self):
        pass

    #TODO
    def resize_position_embeddings(self):
        pass

    #TODO
    def set_input_embeddings(self):
        pass

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.Dense):
            if isinstance(layer.weight, mindspore.Tensor):
                layer.weight.set_data(
                    ops.normal(
                        mean=0.0,
                        stddev=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )
        if isinstance(layer, nn.Embedding):
            if isinstance(layer.embedding_table, mindspore.Tensor):
                layer.embedding_table.set_data(
                    ops.normal(
                        mean=0.0,
                        stddev=self.config.initializer_range,
                        shape=layer.embedding_table.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.epsilon = 1e-12


class ErniePooler(nn.Cell):
    """
    Ernie Pooler.
    """
    def __init__(self, config: ErnieConfig):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        r"""
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieModel(ErniePretrainedModel):
    """
    Ernie model.
    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.nheads = config.num_attention_heads
        self.embeddings = ErnieEmbeddings(config=config)
        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.pooler = ErniePooler(config)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        task_type_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Ernir Model
        """
        batch_size, seq_length = input_ids.shape

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        # use_cache = use_cache if use_cache is not None else False
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask = ((input_ids == self.pad_token_id).astype(
                self.pooler.dense.weight.dtype) * -1e4).unsqueeze(1).unsqueeze(2)

            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = ops.zeros(
                    [batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = ops.concat(
                    [past_mask, attention_mask], axis=-1)

            attention_mask = ops.tile(
                attention_mask, (1, self.nheads, seq_length, 1)).reshape(-1, seq_length, seq_length)
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_mask = ops.tile(
                attention_mask, (1, self.nheads, seq_length, 1)).reshape(-1, seq_length, seq_length)

        ops.stop_gradient(attention_mask)
        # attention_mask.stop_gradient = True

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
        )
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            pooled_output = self.pooler(sequence_output)
            return (sequence_output, pooled_output)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return (sequence_output, pooled_output) + encoder_outputs[1:]

class ErnieForSequenceClassification(ErniePretrainedModel):
    r"""
    Ernie Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct ErnieForSequenceClassification.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)
        self.apply(self.init_weights)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
            inputs_embeds(Tensor, optional):
                See :class:`ErnieModel`.
            labels (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ..., num_labels - 1]`. If `num_labels == 1`
                a regression loss is computed (Mean-Square loss), If `num_labels > 1`
                a classification loss is computed (Cross-Entropy).
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments)
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss('none')
                loss = loss_fct(logits, labels)
            elif labels.dtype in ( mindspore.int64, mindspore.int32 ):
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape(
                    (-1, self.num_labels)), labels.reshape((-1,)))
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

class ErnieForQuestionAnswering(ErniePretrainedModel):
    """
    Ernie Model with a linear layer on top of the hidden-states
    output to compute `span_start_logits` and `span_end_logits`,
    designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct ErnieForQuestionAnswering.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        self.classifier = nn.Dense(config.hidden_size, 2)
        self.apply(self.init_weights)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
            inputs_embeds(Tensor, optional):
                See :class:`ErnieModel`.
            start_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the start of the labelled span
                for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`).
                Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the end of the labelled span
                for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`).
                Position outside of the sequence are not taken into account for computing the loss.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            it returns a tuple of tensors corresponding to ordered
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        logits = ops.transpose(input=logits, input_perm=(2, 0, 1))
        start_logits, end_logits = ops.unstack(input_x=logits)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = ops.shape(start_logits)[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

class ErnieForTokenClassification(ErniePretrainedModel):
    r"""
    ERNIE Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfigused to construct ErnieForTokenClassification.
    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)
        self.apply(self.init_weights)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
            inputs_embeds(Tensor, optional):
                See :class:`ErnieModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the token classification loss. Indices should be in `[0, ..., num_labels - 1]`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            it returns a tuple of tensors corresponding to ordered and
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(
                (-1, self.num_labels)), labels.reshape((-1,)))
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

class ErnieLMPredictionHead(nn.Cell):
    r"""
    Ernie Model with a `language modeling` head on top.
    """

    def __init__(
        self,
        config: ErnieConfig,
    ):
        super().__init__()
        self.transform = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm([config.hidden_size])
        # self.decoder_weight = (
        #     Parameter(
        #         initializer(XavierNormal(),
        #                  [config.vocab_size, config.hidden_size],
        #                  self.transform.weight.dtype)
        #     )
        #     if embedding_weights is None
        #     else embedding_weights
        # )
        self.decoder = nn.Dense(config.vocab_size, config.hidden_size)
        self.decoder_bias =Parameter(
            initializer('zeros',
                        [config.vocab_size],
                        dtype=mindspore.float32))

    def construct(self, hidden_states = None, masked_positions = None):
        r"""
        ErniePredictionHead
        """
        if masked_positions is not None:
            hidden_states = ops.reshape(
                hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = ops.GatherD(
                x=hidden_states, index=masked_positions,dim=None)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = ops.matmul(hidden_states, self.decoder.weight) + self.decoder_bias
        return hidden_states

class ErniePretrainingHeads(nn.Cell):
    r""""
    ErinePretrainingHeads
    """
    def __init__(
        self,
        config: ErnieConfig,
    ):
        super().__init__()
        self.predictions = ErnieLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output, masked_positions=None):
        r"""
        ErniePretrainingHeads
        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class ErnieForPretraining(ErniePretrainedModel):
    r"""
    Ernie Model with a `masked language modeling` head and a `sentence order prediction` head
    on top.

    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        self.cls = ErniePretrainingHeads(config=config)

        self.apply(self.init_weights)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        masked_positions: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        next_sentence_label: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
            inputs_embeds(Tensor, optional):
                See :class:`ErnieModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., vocab_size]`.
            next_sentence_label (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.bert.ErnieForPreTrainingOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            it returns a tuple of tensors corresponding to ordered


        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_positions)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(
                    (-1, ops.shape(prediction_scores)[-1])), labels.reshape((-1,))
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.reshape(
                    (-1, 2)), next_sentence_label.reshape((-1,))
            )
            total_loss = masked_lm_loss + next_sentence_loss
        output = (prediction_scores, seq_relationship_score) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

class ErniePretrainingCriterion(nn.Cell):
    r"""
    The loss output of Ernie Model during the pretraining:
    a `masked language modeling` head and a `next sentence prediction (classification)` head.

    """

    def __init__(self, with_nsp_loss=True):
        super().__init__()
        self.with_nsp_loss = with_nsp_loss
        # self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)

    def construct(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels=None):
        """
        Args:
            prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size]
            seq_relationship_score(Tensor):
                The scores of next sentence prediction. Its data type should be float32 and
                its shape is [batch_size, 2]
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, its dimensionality is equal to `prediction_scores`.
                Its data type should be int64. If `masked_positions` is None,
                its shape is [batch_size, sequence_length, 1].
                Otherwise, its shape is [batch_size, mask_token_num, 1]
            next_sentence_labels(Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and
                its shape is [batch_size, 1]

        Returns:
            Tensor: The pretraining loss, equals to the sum of `masked_lm_loss` plus the mean of `next_sentence_loss`.
            Its data type should be float32 and its shape is [1].

        """

        masked_lm_loss = ops.cross_entropy(prediction_scores, masked_lm_labels, reduction="none")

        if not self.with_nsp_loss:
            return ops.mean(masked_lm_loss)

        next_sentence_loss = ops.cross_entropy(seq_relationship_score, next_sentence_labels, reduction="none")
        return ops.mean(masked_lm_loss), ops.mean(next_sentence_loss)

class ErnieOnlyMLMHead(nn.Cell):
    r"""
    ErnieOnlyMLMHead
    """
    def __init__(self, config: ErnieConfig):
        super().__init__()
        self.predictions = ErnieLMPredictionHead(config=config)

    def construct(self, sequence_output, masked_positions=None):
        r"""
        ErnieOnlyMLMHead
        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores

class ErnieForMaskedLM(ErniePretrainedModel):
    """
    Ernie Model with a `masked language modeling` head on top.

    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct ErnieForMaskedLM.

    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        self.cls = ErnieOnlyMLMHead(config=config)

        self.apply(self.init_weights)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        masked_positions: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
            masked_positions:
                masked positions of output.
            inputs_embeds(Tensor, optional):
                See :class:`ErnieModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., vocab_size]`
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(
            sequence_output, masked_positions=masked_positions)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(
                    (-1, ops.shape(prediction_scores)[-1])), labels.reshape((-1,))
            )
        output = (prediction_scores,) + outputs[2:]
        return (
            ((masked_lm_loss,) + output)
            if masked_lm_loss is not None
            else (output[0] if len(output) == 1 else output)
        )

class ErnieForMultipleChoice(ErniePretrainedModel):
    """
    Ernie Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct ErnieForMultipleChoice
    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        #self.num_choices = config.num_choices if config.num_choices is not None else 2
        self.num_choices = 2
        #self.dropout = nn.Dropout(
        #    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        #)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)
        self.apply(self.init_weights)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The ErnieForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ErnieModel` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids(Tensor, optional):
                See :class:`ErnieModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`ErnieModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`ErnieModel` and shape as [batch_size, num_choice, sequence_length].
            inputs_embeds(Tensor, optional):
                See :class:`ErnieModel` and shape as [batch_size, num_choice, sequence_length, hidden_size].
            labels (Tensor of shape `(batch_size, )`, optional):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            Otherwise it returns a tuple of tensors corresponding to ordered
            not None (depending on the input arguments)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # input_ids: [bs, num_choice, seq_l]
        if input_ids is not None:
            # flat_input_ids: [bs*num_choice,seq_l]
            input_ids = input_ids.reshape((-1, input_ids.shape[-1]))

        if position_ids is not None:
            position_ids = position_ids.reshape(
                (-1, position_ids.shape[-1]))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(
                (-1, token_type_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(
                (-1, attention_mask.shape[-1]))

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(
                (-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]))

        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(
            (-1, self.num_choices))  # logits: (bs, num_choice)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        output = (reshaped_logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)



class UIE(ErniePretrainedModel):
    """
    UIE model based on Ernie.
    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        self.linear_start = nn.Dense(config.hidden_size, 1)
        self.linear_end = nn.Dense(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        return_dict: Optional[Tensor] = None,
    ):
        r"""
        UIE
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        sequence_output, _ = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )
        start_logits = self.linear_start(sequence_output)
        start_logits = ops.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = ops.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob
