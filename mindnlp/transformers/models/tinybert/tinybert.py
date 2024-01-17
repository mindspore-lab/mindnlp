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
# pylint: disable=C0103
# pylint: disable=C0415

"""
TinyBert Models
"""
import math
import os
from typing import Union
import mindspore
from mindspore import nn, ops
from .tinybert_config import TinyBertConfig
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel


class TinyBertEmbeddings(nn.Cell):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        """
        init BertEmbeddings
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids=None):
        """
        Construct the embeddings from word, position and token_type embeddings.
        """
        seq_length = input_ids.shape[1]
        position_ids = ops.arange(seq_length, dtype=mindspore.int64)
        position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TinyBertSelfAttention(nn.Cell):
    r"""
    TinyBertSelfAttention
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the " +
                f"number of attention heads {config.num_attention_heads}")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        transpose_for_scores
        """
        new_x_shape = x.shape[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(
            query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class TinyBertAttention(nn.Cell):
    """
    TinyBertAttention
    """

    def __init__(self, config):
        super().__init__()

        self.self_ = TinyBertSelfAttention(config)
        self.output = TinyBertSelfOutput(config)

    def construct(self, input_tensor, attention_mask):
        self_output, layer_att = self.self_(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att


class TinyBertSelfOutput(nn.Cell):
    """
    TinyBertSelfOutput
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TinyBertIntermediate(nn.Cell):
    """
    TinyBertIntermediate
    """

    def __init__(self, config, intermediate_size=-1):
        super().__init__()
        if intermediate_size < 0:
            self.dense = nn.Dense(
                config.hidden_size, config.intermediate_size)
        else:
            self.dense = nn.Dense(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TinyBertOutput(nn.Cell):
    """
    TinyBertOutput
    """

    def __init__(self, config, intermediate_size=-1):
        super().__init__()
        if intermediate_size < 0:
            self.dense = nn.Dense(
                config.intermediate_size, config.hidden_size)
        else:
            self.dense = nn.Dense(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TinyBertLayer(nn.Cell):
    """
    TinyBertLayer
    """

    def __init__(self, config):
        super().__init__()
        self.attention = TinyBertAttention(config)
        self.intermediate = TinyBertIntermediate(config)
        self.output = TinyBertOutput(config)

    def construct(self, hidden_states, attention_mask):
        attention_output, layer_att = self.attention(
            hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, layer_att


class TinyBertEncoder(nn.Cell):
    """
    TinyBertEncoder
    """

    def __init__(self, config):
        super().__init__()
        self.layer = nn.CellList([TinyBertLayer(config)
                                  for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_atts = []
        for _, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att = layer_module(
                hidden_states, attention_mask)
            all_encoder_atts.append(layer_att)

        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_atts


class TinyBertPooler(nn.Cell):
    """
    TinyBertPooler
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. "-1" refers to last layer
        pooled_output = hidden_states[-1][:, 0]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class TinyBertPredictionHeadTransform(nn.Cell):
    """
    TinyBertPredictionHeadTransform
    """

    def __init__(self, config):
        super().__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=1e-12)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TinyBertLMPredictionHead(nn.Cell):
    """
    TinyBertLMPredictionHead
    """

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = TinyBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(bert_model_embedding_weights.shape[1],
                                bert_model_embedding_weights.shape[0],
                                has_bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = mindspore.Parameter(ops.zeros(
            bert_model_embedding_weights.shape[0]))

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class TinyBertOnlyMLMHead(nn.Cell):
    """
    TinyBertOnlyMLMHead
    """

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = TinyBertLMPredictionHead(
            config, bert_model_embedding_weights)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TinyBertOnlyNSPHead(nn.Cell):
    """
    TinyBertOnlyNSPHead
    """

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class TinyBertPreTrainingHeads(nn.Cell):
    """
    TinyBertPreTrainingHeads
    """

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = TinyBertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class TinyBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """
    config_class = TinyBertConfig

    base_model_prefix = 'bert'

    def __init__(self, config):
        super().__init__(config)
        if not isinstance(config, TinyBertConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of " +
                "class `BertConfig`. To create a model from a Google pretrained model use " +
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def init_model_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight = ops.normal(
                shape=module.weight.shape,
                mean=0.0,
                stddev=self.config.initializer_range
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias = ops.fill(
                type=module.bias.dtype, shape=module.bias.shape, value=0)
            module.weight = ops.fill(
                type=module.weight.dtype, shape=module.weight.shape, value=1.0)
        if isinstance(module, nn.Dense):
            module.weight = ops.normal(
                shape=module.weight.shape, mean=0.0, stddev=self.config.initializer_range)
            if module.bias is not None:
                module.bias = ops.fill(
                    type=module.bias.dtype, shape=module.bias.shape, value=0)

    def get_input_embeddings(self) -> "nn.Cell":
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Cell`: A mindspore cell mapping vocabulary to hidden states.
        """

    def set_input_embeddings(self, new_embeddings: "nn.Cell"):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Cell`): A mindspore cell mapping vocabulary to hidden states.
        """

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        resize the model position embeddings if necessary
        """

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """

    def save(self, save_dir: Union[str, os.PathLike]):
        "save pretrain model"

    #TODO
    def post_init(self):
        """post init."""


class TinyBertModel(TinyBertPreTrainedModel):
    """
    TinyBERT model
    """

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = TinyBertEmbeddings(config)
        self.encoder = TinyBertEncoder(config)
        self.pooler = TinyBertPooler(config)
        self.apply(self.init_model_weights)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None,
                  output_all_encoded_layers=True, output_att=True):
        """construct."""
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # extended_attention_mask = extended_attention_mask.to(
        #     dtype=next(self.parameters()).dtype)  # fp16 compatibility

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, layer_atts = self.encoder(embedding_output,
                                                  extended_attention_mask)

        pooled_output = self.pooler(encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if not output_att:
            return encoded_layers, pooled_output

        return encoded_layers, layer_atts, pooled_output


class TinyBertForPreTraining(TinyBertPreTrainedModel):
    """
    BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_model_weights)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None,
                  masked_lm_labels=None, next_sentence_label=None):
        """construct."""
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, output_att=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss
            return total_loss
        return prediction_scores, seq_relationship_score


class TinyBertFitForPreTraining(TinyBertPreTrainedModel):
    """
    TinyBertForPreTraining with fit dense
    """

    def __init__(self, config, fit_size=768):
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.fit_dense = nn.Dense(config.hidden_size, fit_size)
        self.apply(self.init_model_weights)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None):
        """construct."""
        sequence_output, att_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask)
        tmp = []
        for _, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output


class TinyBertForMaskedLM(TinyBertPreTrainedModel):
    """
    BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_model_weights)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                  output_att=False):
        """construct."""
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=True, output_att=output_att)

        if output_att:
            sequence_output, att_output = sequence_output
        prediction_scores = self.cls(sequence_output[-1])

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            if not output_att:
                return masked_lm_loss

            return masked_lm_loss, att_output

        if not output_att:
            return prediction_scores
        return prediction_scores, att_output


class TinyBertForNextSentencePrediction(TinyBertPreTrainedModel):
    """
    BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertOnlyNSPHead(config)
        self.apply(self.init_model_weights)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        """construct."""
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False, output_att=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss

        return seq_relationship_score


class TinyBertForSentencePairClassification(TinyBertPreTrainedModel):
    """
    TinyBertForSentencePairClassification
    """

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = TinyBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size * 3, num_labels)
        self.apply(self.init_model_weights)

    def construct(self, a_input_ids, b_input_ids, a_token_type_ids=None, b_token_type_ids=None,
                  a_attention_mask=None, b_attention_mask=None, labels=None):
        """construct."""
        _, a_pooled_output = self.bert(
            a_input_ids, a_token_type_ids, a_attention_mask, output_all_encoded_layers=False, output_att=False)
        # a_pooled_output = self.dropout(a_pooled_output)

        _, b_pooled_output = self.bert(
            b_input_ids, b_token_type_ids, b_attention_mask, output_all_encoded_layers=False, output_att=False)
        # b_pooled_output = self.dropout(b_pooled_output)

        logits = self.classifier(ops.relu(ops.concat((a_pooled_output, b_pooled_output,
                                                      ops.abs(a_pooled_output - b_pooled_output)), -1)))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class TinyBertForSequenceClassification(TinyBertPreTrainedModel):
    """
    TinyBertForSequenceClassification
    """

    def __init__(self, config, num_labels, fit_size=768):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = TinyBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, num_labels)
        self.fit_dense = nn.Dense(config.hidden_size, fit_size)
        self.apply(self.init_model_weights)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None,
                  is_student=False):
        """construct"""
        sequence_output, att_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                               output_all_encoded_layers=True, output_att=True)

        logits = self.classifier(ops.relu(pooled_output))

        tmp = []
        if is_student:
            for _, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output

__all__ = [
    'TinyBertModel',
    'TinyBertForSequenceClassification',
    'TinyBertForMaskedLM',
    'TinyBertForNextSentencePrediction',
    'TinyBertForMaskedLM',
    'TinyBertForSentencePairClassification',
    'TinyBertForPreTraining',
    'TinyBertFitForPreTraining'
]
