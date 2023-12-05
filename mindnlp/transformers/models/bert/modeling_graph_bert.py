# Copyright 2023 Huawei Technologies Co., Ltd
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
# pylint: disable=C0415
# pylint: disable=W0223
# pylint: disable=E0401
# pylint: disable=C0103

"""MindNLP bert model"""
import mindspore.common.dtype as mstype
from mindspore import nn, ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, Normal
from mindnlp.modules.functional import make_causal_mask, finfo
from .configuration_bert import BertConfig
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel


class MSBertEmbeddings(nn.Cell):
    """
    Embeddings for BERT, include word, position and token_type
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MSBertSelfAttention(nn.Cell):
    """
    Self attention layer for BERT.
    """

    def __init__(self, config, causal, init_cache=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )
        self.key = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )
        self.value = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(-1)

        self.causal = causal
        self.init_cache = init_cache

        self.causal_mask = make_causal_mask(
            ops.ones((1, config.max_position_embeddings), dtype=mstype.bool_),
            dtype=mstype.bool_,
        )

        if not init_cache:
            self.cache_key = None
            self.cache_value = None
            self.cache_index = None
        else:
            self.cache_key = Parameter(
                initializer(
                    "zeros",
                    (
                        config.max_length,
                        config.max_batch_size,
                        config.num_attention_heads,
                        config.attention_head_size,
                    ),
                )
            )
            self.cache_value = Parameter(
                initializer(
                    "zeros",
                    (
                        config.max_length,
                        config.max_batch_size,
                        config.num_attention_heads,
                        config.attention_head_size,
                    ),
                )
            )
            self.cache_index = Parameter(Tensor(0, mstype.int32))

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        if self.init_cache:
            batch_size = query.shape[0]
            num_updated_cache_vectors = query.shape[1]
            max_length = self.cache_key.shape[0]
            indices = ops.arange(
                self.cache_index, self.cache_index + num_updated_cache_vectors
            )
            key = ops.scatter_update(self.cache_key, indices, key.swapaxes(0, 1))
            value = ops.scatter_update(self.cache_value, indices, value.swapaxes(0, 1))

            self.cache_index += num_updated_cache_vectors

            pad_mask = ops.broadcast_to(
                ops.arange(max_length) < self.cache_index,
                (batch_size, 1, num_updated_cache_vectors, max_length),
            )
            attention_mask = ops.logical_and(attention_mask, pad_mask)

        return key, value, attention_mask

    def transpose_for_scores(self, input_x):
        r"""
        transpose for scores
        """
        new_x_shape = input_x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        input_x = input_x.view(*new_x_shape)
        return input_x.transpose(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        batch_size = hidden_states.shape[0]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_states = self.transpose_for_scores(mixed_query_layer)
        key_states = self.transpose_for_scores(mixed_key_layer)
        value_states = self.transpose_for_scores(mixed_value_layer)

        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = ops.slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = ops.broadcast_to(
                causal_mask, (batch_size,) + causal_mask.shape[1:]
            )
        else:
            causal_mask = None

        if attention_mask is not None and self.causal:
            attention_mask = ops.broadcast_to(
                attention_mask.expand_dims(-2).expand_dims(-3), causal_mask.shape
            )
            attention_mask = ops.logical_and(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = attention_mask.expand_dims(-2).expand_dims(-3)

        if self.causal and self.init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            # attention_bias = ops.select(
            #     attention_mask > 0,
            #     ops.full(attention_mask.shape, 0.0).astype(hidden_states.dtype),
            #     ops.full(attention_mask.shape, finfo(hidden_states.dtype, "min")).astype(
            #         hidden_states.dtype
            #     ),
            # )
            attention_bias = ops.select(
                attention_mask > 0,
                ops.zeros_like(attention_mask).astype(hidden_states.dtype),
                (ops.ones_like(attention_mask) * finfo(hidden_states.dtype, "min")).astype(
                    hidden_states.dtype
                ),
            )
        else:
            attention_bias = None

        # Take the dot product between "query" snd "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_states, key_states.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(
            Tensor(self.attention_head_size, mstype.float32)
        )
        # Apply the attention mask is (precommputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_bias

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class MSBertSelfOutput(nn.Cell):
    r"""
    Bert Self Output
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MSBertAttention(nn.Cell):
    r"""
    Bert Attention
    """

    def __init__(self, config, causal, init_cache=False):
        super().__init__()
        self.self = MSBertSelfAttention(config, causal, init_cache)
        self.output = MSBertSelfOutput(config)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MSBertIntermediate(nn.Cell):
    r"""
    Bert Intermediate
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.intermediate_size,
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MSBertOutput(nn.Cell):
    r"""
    Bert Output
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.intermediate_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MSBertLayer(nn.Cell):
    r"""
    Bert Layer
    """

    def __init__(self, config, init_cache=False):
        super().__init__()
        self.attention = MSBertAttention(config, causal=config.is_decoder, init_cache=init_cache)
        self.intermediate = MSBertIntermediate(config)
        self.output = MSBertOutput(config)
        if config.add_cross_attention:
            self.crossattention = MSBertAttention(config, causal=False, init_cache=init_cache)

    def construct(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states = None,
                encoder_attention_mask = None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
            )
            attention_output = cross_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class MSBertEncoder(nn.Cell):
    r"""
    Bert Encoder
    """

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList(
            [MSBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def _set_recompute(self):
        for layer in self.layer:
            layer.recompute()

    def construct(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states = None,
                encoder_attention_mask = None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask
                )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions += (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs += (all_hidden_states,)
        if self.output_attentions:
            outputs += (all_attentions,)
        return outputs


class MSBertPooler(nn.Cell):
    r"""
    Bert Pooler
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding.
        # to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MSBertPredictionHeadTransform(nn.Cell):
    r"""
    Bert Prediction Head Transform
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MSBertLMPredictionHead(nn.Cell):
    r"""
    Bert LM Prediction Head
    """

    def __init__(self, config):
        super().__init__()
        self.transform = MSBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(
            config.hidden_size,
            config.vocab_size,
            has_bias=False,
        )

        self.bias = Parameter(initializer("zeros", config.vocab_size), "bias")

    def construct(self, hidden_states, masked_lm_positions):
        batch_size, seq_len, hidden_size = hidden_states.shape
        if masked_lm_positions is not None:
            flat_offsets = ops.arange(batch_size) * seq_len
            flat_position = (masked_lm_positions + flat_offsets.reshape(-1, 1)).reshape(
                -1
            )
            flat_sequence_tensor = hidden_states.reshape(-1, hidden_size)
            hidden_states = ops.gather(flat_sequence_tensor, flat_position, 0)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class MSBertPreTrainingHeads(nn.Cell):
    r"""
    Bert PreTraining Heads
    """

    def __init__(self, config):
        super().__init__()
        self.predictions = MSBertLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output, masked_lm_positions):
        prediction_scores = self.predictions(sequence_output, masked_lm_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MSBertPreTrainedModel(PreTrainedModel):
    """BertPretrainedModel"""
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_recompute = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(
                initializer(
                    Normal(self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.has_bias:
                cell.bias.set_data(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            weight = initializer(
                Normal(self.config.initializer_range),
                cell.weight.shape,
                cell.weight.dtype,
            )
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer("ones", cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))


class MSBertModel(MSBertPreTrainedModel):
    r"""
    Bert Model
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.embeddings = MSBertEmbeddings(config)
        self.encoder = MSBertEncoder(config)
        self.pooler = MSBertPooler(config) if add_pooling_layer else None
        self.num_hidden_layers = config.num_hidden_layers

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states = None,
        encoder_attention_mask = None
    ):
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)
        if position_ids is None:
            position_ids = ops.broadcast_to(ops.arange(ops.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = (
                    head_mask.expand_dims(0)
                    .expand_dims(0)
                    .expand_dims(-1)
                    .expand_dims(-1)
                )
                head_mask = ops.broadcast_to(
                    head_mask, (self.num_hidden_layers, -1, -1, -1, -1)
                )
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class MSBertForPretraining(MSBertPreTrainedModel):
    r"""
    Bert For Pretraining
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = MSBertModel(config)
        self.cls = MSBertPreTrainingHeads(config)
        self.vocab_size = config.vocab_size

        self.cls.predictions.decoder.weight = (
            self.bert.embeddings.word_embeddings.weight
        )

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_lm_positions
        )

        outputs = (
            prediction_scores,
            seq_relationship_score,
        ) + outputs[2:]
        # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

        return outputs


class MSBertForSequenceClassification(MSBertPreTrainedModel):
    """Bert Model for classification tasks"""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = MSBertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.classifier = nn.Dense(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(p=classifier_dropout)

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]

        return output


__all__ = [
    "MSBertEmbeddings",
    "MSBertAttention",
    "MSBertEncoder",
    "MSBertIntermediate",
    "MSBertLayer",
    "MSBertModel",
    "MSBertForPretraining",
    "MSBertLMPredictionHead",
    "MSBertForSequenceClassification",
]
