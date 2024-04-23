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
""" ConvBERT model."""

from typing import Optional, Tuple, Union
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.nn import CrossEntropyLoss
from mindspore.common.initializer import initializer, Normal

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
)
from ...modeling_utils import PreTrainedModel
from .convbert_config import ConvBertConfig
from ...ms_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

MSCONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "conv-bert-base",
]


class MSConvBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )

        self.LayerNorm = nn.LayerNorm(
            config.embedding_size, epsilon=config.layer_norm_eps
        )
        self.dropout_p = config.hidden_dropout_prob
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to(
            (1, -1)
        )

        self.token_type_ids = ops.zeros(
            self.position_ids.shape, dtype=ms.int64)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:

        input_shape = input_ids.shape
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = ops.dropout(embeddings, p=self.dropout_p)
        return embeddings


class ConvBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvBertConfig
    base_model_prefix = "convbert"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight = ms.Parameter(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.bias is not None:
                cell.bias = ms.Parameter(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            cell.weight = ms.Parameter(
                initializer(
                    Normal(mean=0.0, sigma=self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            # if cell.padding_idx is not None:
            #     cell.weight.data[cell.padding_idx].set_data(
            #         initializer('zeros',
            #                     cell.weight.data[cell.padding_idx].shape,
            #                     cell.weight.data[cell.padding_idx].dtype)
            #     )
        elif isinstance(cell, nn.LayerNorm):
            cell.bias = ms.Parameter(
                initializer("zeros", cell.bias.shape, cell.bias.dtype)
            )
            cell.weight = ms.Parameter(
                initializer("ones", cell.weight.shape, cell.weight.dtype)
            )


class SeparableConv1D(nn.Cell):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, config, input_filters, output_filters, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            group=input_filters,
            pad_mode="pad",
            padding=kernel_size // 2,
            has_bias=False,
        )
        self.pointwise = nn.Conv1d(
            input_filters, output_filters, kernel_size=1, has_bias=False
        )
        self.bias = ms.Parameter(ops.zeros((output_filters, 1)))

        self.depthwise.weight = ms.Parameter(
            initializer(
                Normal(sigma=config.initializer_range, mean=0.0),
                self.depthwise.weight.shape,
                self.depthwise.weight.dtype,
            )
        )
        self.pointwise.weight = ms.Parameter(
            initializer(
                Normal(sigma=config.initializer_range, mean=0.0),
                self.pointwise.weight.shape,
                self.depthwise.weight.dtype,
            )
        )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias
        return x


class MSConvBertSelfAttention(nn.Cell):
    """
    MSConvBertSelfAttention
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        new_num_attention_heads = config.num_attention_heads // config.head_ratio
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio

        self.conv_kernel_size = config.conv_kernel_size
        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size should be divisible by num_attention_heads")

        self.attention_head_size = (
            config.hidden_size // self.num_attention_heads) // 2
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.key_conv_attn_layer = SeparableConv1D(
            config, config.hidden_size, self.all_head_size, self.conv_kernel_size
        )
        self.conv_kernel_layer = nn.Dense(
            self.all_head_size, self.num_attention_heads * self.conv_kernel_size
        )
        self.conv_out_layer = nn.Dense(config.hidden_size, self.all_head_size)
        self.unfold = nn.Unfold(
            ksizes=[1, self.conv_kernel_size, 1, 1],
            rates=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding="same",
        )
        self.dropout_p = config.attention_probs_dropout_prob

    def swapaxes_for_scores(self, x):
        """swapaxes for scores"""
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        batch_size = hidden_states.shape[0]
        # If this is instantiated as a cross-attention cell, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_key_conv_attn_layer = self.key_conv_attn_layer(
            hidden_states.swapaxes(1, 2)
        )
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.swapaxes(1, 2)

        query_layer = self.swapaxes_for_scores(mixed_query_layer)
        key_layer = self.swapaxes_for_scores(mixed_key_layer)
        value_layer = self.swapaxes_for_scores(mixed_value_layer)
        conv_attn_layer = ops.multiply(
            mixed_key_conv_attn_layer, mixed_query_layer)

        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = ops.reshape(
            conv_kernel_layer, [-1, self.conv_kernel_size, 1]
        )
        conv_kernel_layer = ops.softmax(conv_kernel_layer, axis=1)

        conv_out_layer = self.conv_out_layer(hidden_states)
        conv_out_layer = ops.reshape(
            conv_out_layer, [batch_size, -1, self.all_head_size]
        )
        conv_out_layer = conv_out_layer.swapaxes(1, 2).unsqueeze(-1)
        conv_out_layer = ops.unfold(
            conv_out_layer,
            kernel_size=[self.conv_kernel_size, 1],
            dilation=1,
            padding=[(self.conv_kernel_size - 1) // 2, 0],
            stride=1,
        )
        conv_out_layer = conv_out_layer.swapaxes(1, 2).reshape(
            batch_size, -1, self.all_head_size, self.conv_kernel_size
        )
        conv_out_layer = ops.reshape(
            conv_out_layer, [-1, self.attention_head_size,
                             self.conv_kernel_size]
        )
        conv_out_layer = ops.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = ops.reshape(conv_out_layer, [-1, self.all_head_size])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / \
            ops.sqrt(ms.Tensor(self.attention_head_size))

        # Apply the attention mask is (precomputed for all layers in ConvBertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = ops.dropout(attention_probs, p=self.dropout_p)

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)

        conv_out = ops.reshape(
            conv_out_layer,
            [batch_size, -1, self.num_attention_heads, self.attention_head_size],
        )
        context_layer = ops.cat([context_layer, conv_out], 2)

        # conv and context
        new_context_layer_shape = context_layer.shape[:-2] + (
            self.num_attention_heads * self.attention_head_size * 2,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        return outputs


class MSConvBertSelfOutput(nn.Cell):
    """
    MSConvBertSelfOutput
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout_p = config.hidden_dropout_prob

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout_p)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MSConvBertAttention(nn.Cell):
    """
    MSConvBertAttention
    """

    def __init__(self, config):
        super().__init__()
        self.self = MSConvBertSelfAttention(config)
        self.output = MSConvBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """prune heads"""
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
        self.output.dense = prune_linear_layer(
            self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - \
            len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MSConvBertIntermediate(nn.Cell):
    """
    MSConvBertIntermediate
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MSConvBertOutput(nn.Cell):
    """
    MSConvBertOutput
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout_p = config.hidden_dropout_prob

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout_p)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MSConvBertLayer(nn.Cell):
    """
    MSConvBertLayer
    """

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = MSConvBertAttention(config)
        self.intermediate = MSConvBertIntermediate(config)
        self.output = MSConvBertOutput(config)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class MSConvBertEncoder(nn.Cell):
    """
    MSConvBertEncoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList(
            [MSConvBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        for i, layer_cell in enumerate(self.layer):
            layer_outputs = layer_cell(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class MSConvBertModel(ConvBertPreTrainedModel):
    """
    MSConvBertModel
    """

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = MSConvBertEmbeddings(config)
        self.encoder = MSConvBertEncoder(config)
        self.config = config
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: ms.Tensor,
        attention_mask: ms.Tensor,
        token_type_ids: ms.Tensor,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.shape

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
        )

        return hidden_states


class MSConvBertForQuestionAnswering(ConvBertPreTrainedModel):
    """
    MSConvBertForQuestionAnswering
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convbert = MSConvBertModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:

        outputs = self.convbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.shape[1]
        start_positions = start_positions.clamp(0, ignored_index).to(ms.int32)
        end_positions = end_positions.clamp(0, ignored_index).to(ms.int32)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        return total_loss


__all__ = [
    "MSCONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MSConvBertForQuestionAnswering",
]
