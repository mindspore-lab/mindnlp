# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""MindNLP ALBERT model. """
import math
import warnings
import mindspore
import mindspore.numpy as mnp
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, TruncatedNormal, Normal
from mindnlp.abc import PreTrainedModel
from .albert_config import AlbertConfig

activation_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(False),
    'gelu_new': nn.GELU(),
    'swish': nn.SiLU()
}


class Matmul(nn.Cell):
    r"""
    Matmul Operation
    """
    def construct(self, a, b):
        return ops.matmul(a, b)


class AlbertEmbeddings(nn.Cell):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            embedding_table=TruncatedNormal(config.initializer_range))
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.embedding_size,
            embedding_table=TruncatedNormal(config.initializer_range))
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.embedding_size,
            embedding_table=TruncatedNormal(config.initializer_range),
            dtype=mindspore.int32)
        self.layer_norm = nn.LayerNorm(
            (config.embedding_size,),
            epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids=None, position_ids=None):
        seq_len = input_ids.shape[1]
        if position_ids is None:
            position_ids = mnp.arange(seq_len)
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


class AlbertAttention(nn.Cell):
    """
    Albert attention
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.pruned_heads = set()
        self.matmul = Matmul()

    def transpose_for_scores(self, input_x):
        """
        transpose
        """
        new_x_shape = input_x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        input_x = input_x.view(*new_x_shape)
        return input_x.permute(0, 2, 1, 3)

    def construct(self, input_ids, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = self.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)

        # Should find a better way to do this
        context_layer_w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        context_layer_b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = ops.einsum("bfnd,ndh->bfh", context_layer, context_layer_w) + context_layer_b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Cell):
    """
    Albert layer
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.full_layer_layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Dense(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Dense(config.intermediate_size, config.hidden_size)
        self.activation = activation_map[config.hidden_act]

    def construct(
            self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False
    ):
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class AlbertLayerGroup(nn.Cell):
    """
    Albert layer group
    """
    def __init__(self, config):
        super().__init__()
        self.albert_layers = nn.CellList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def construct(
            self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False,
            output_hidden_states=False
    ):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
            hidden_states = layer_output[0]
            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertTransformer(nn.Cell):
    """
    Albert transformer
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_hidden_mapping_in = nn.Dense(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.CellList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def construct(
            self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False,
            output_hidden_states=False
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()
        if output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]
            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class AlbertPretrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def _init_weights(self, cell):
        """ Initialize the weights."""
        if isinstance(cell, nn.Embedding):
            cell.embedding_table.set_data(initializer(Normal(self.config.initializer_range),
                                                      cell.embedding_table.shape,
                                                      cell.embedding_table.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                             cell.weight.shape,
                                             cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings


class AlbertModel(AlbertPretrainedModel):
    """
    Albert model
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Dense(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        """construct"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape)

        extended_attention_mask = attention_mask.expand_dims(1).expand_dims(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
                head_mask = mnp.broadcast_to(head_mask, (self.num_hidden_layers, -1, -1, -1, -1))
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs


class AlbertForPretraining(AlbertPretrainedModel):
    """
    Albert For Pretraining
    """
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            sentence_order_label=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs,
    ):

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, "
                "use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert not kwargs, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)
        outputs = (prediction_scores, sop_scores,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None and sentence_order_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_order_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1))
            total_loss = masked_lm_loss + sentence_order_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, sop_scores, (hidden_states), (attentions)


class AlbertMLMHead(nn.Cell):
    """
    Albert MLM head
    """
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm((config.embedding_size,))
        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))
        self.dense = nn.Dense(config.hidden_size, config.embedding_size)
        self.decoder = nn.Dense(config.embedding_size, config.vocab_size)
        self.activation = activation_map[config.hidden_act]
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        return prediction_scores


class AlbertSOPHead(nn.Cell):
    """
    Albert SOP head
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class AlbertForMaskedLM(AlbertPretrainedModel):
    """
    Albert for MaskedLM
    """
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, "
                "use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert not kwargs, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_outputs = outputs[0]
        prediction_scores = self.predictions(sequence_outputs)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs


class AlbertForSequenceClassification(AlbertPretrainedModel):
    """
    Albert for sequence classification
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, self.config.num_labels)

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertForTokenClassification(AlbertPretrainedModel):
    """
    Albert for token classification
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, self.config.num_labels)

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertForQuestionAnswering(AlbertPretrainedModel):
    """
    Albert for question answering
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class AlbertForMultipleChoice(AlbertPretrainedModel):
    """
    Albert for multiple choice
    """
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        resize the model position embeddings if necessary
        """
        self.embeddings.position_embeddings = new_num_position_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
