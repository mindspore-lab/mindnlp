# coding=utf-8
# Copyright 2023 The HuggingFace Inc. & Google team. All rights reserved.
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
"""Pix2Struct modeling file"""

import math
from typing import Dict, List, Optional, Tuple, Union

import mindspore
from mindnlp.core import nn, ops

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS
from ....utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    logging,
)
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Pix2StructConfig"


# Adapted from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Pix2Struct
class Pix2StructLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = ops.mean(hidden_states.to(mindspore.float32).pow(2), -1, keepdim=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16, mindspore.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


ALL_LAYERNORM_LAYERS.append(Pix2StructLayerNorm)


class Pix2StructVisionEmbeddings(nn.Module):
    r"""
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, flattened_patches: mindspore.Tensor) -> mindspore.Tensor:
        # the row and column indices are stored in the first and second position of the flattened_patches
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()

        flattened_patches = flattened_patches[:, :, 2:]

        embeddings = self.patch_projection(flattened_patches)
        row_embeddings = self.row_embedder(row_indices)
        col_embeddings = self.column_embedder(col_indices)

        # sum all embeddings together
        embeddings = embeddings + row_embeddings + col_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class Pix2StructVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.query = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.key = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.value = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.output = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        Self-attention block
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        def to_projection_shape(states):
            """projection"""
            return ops.transpose(states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim), 1, 2)

        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = to_projection_shape(self.query(hidden_states))

        # get key/value states
        key_states = to_projection_shape(self.key(hidden_states))
        value_states = to_projection_shape(self.value(hidden_states))

        # compute scores
        # equivalent of ops.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = ops.matmul(query_states, ops.transpose(key_states, 3, 2))

        if position_bias is None:
            position_bias = ops.zeros(
                (1, self.n_heads, seq_length, seq_length), dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True

            if attention_mask is None:
                attention_mask = ops.ones((batch_size, seq_length), dtype=scores.dtype)

            if attention_mask.dim() == 2:
                position_bias = position_bias + attention_mask[:, None, None, :]
            else:
                # (batch_size, n_heads, seq_length, key_length)
                position_bias = position_bias + attention_mask
            position_bias = 1 - position_bias

        position_bias_masked = position_bias.masked_fill(position_bias == 1, float(ops.finfo(scores.dtype).min))
        scores += position_bias_masked
        scores = ops.maximum(scores, mindspore.tensor(ops.finfo(scores.dtype).min))

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=mindspore.float32).type_as(scores)

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = ops.matmul(attn_weights, value_states)

        # (batch_size, seq_length, dim)
        attn_output = ops.transpose(attn_output, 1, 2).view(batch_size, -1, self.inner_dim)

        attn_output = self.output(attn_output)

        outputs = (attn_output,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5DenseGatedActDense->Pix2StructVisionMlp,T5Config->Pix2StructVisionConfig,config.d_model->config.hidden_size,dropout_rate->dropout_rate
class Pix2StructVisionMlp(nn.Module):
    def __init__(self, config: Pix2StructVisionConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, mindspore.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != mindspore.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class Pix2StructVisionLayer(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Pix2StructVisionAttention(config)
        self.mlp = Pix2StructVisionMlp(config)
        self.pre_mlp_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_attention_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:
        residual = hidden_states

        # in Pix2StructVision, layernorm is applied before self-attention
        hidden_states = self.pre_attention_layer_norm(hidden_states)

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + residual

        # in Pix2StructVision, layernorm is also applied after self-attention
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        layer_output = self.mlp(layer_output) + hidden_states  # second residual connection

        outputs = (layer_output,) + outputs

        return outputs


class Pix2StructVisionEncoder(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Pix2StructVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

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
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Pix2StructPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Pix2StructConfig

    @property
    def dummy_inputs(self):
        input_ids = mindspore.tensor(DUMMY_INPUTS)
        input_mask = mindspore.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, Pix2StructLayerNorm):
            nn.init.constant_(module.weight, factor * 1.0)
        elif isinstance(module, Pix2StructTextDenseGatedActDense):
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, Pix2StructConfig)
                else self.config.hidden_size
            )
            d_ff = self.config.text_config.d_ff if isinstance(self.config, Pix2StructConfig) else self.config.d_ff

            nn.init.normal_(module.wi_0.weight, mean=0.0, std=factor * ((hidden_size) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                nn.init.zeros_(module.wi_0.bias)
            nn.init.normal_(module.wi_1.weight, mean=0.0, std=factor * ((hidden_size) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                nn.init.zeros_(module.wi_1.bias)
            nn.init.normal_(module.wo.weight, mean=0.0, std=factor * ((d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                nn.init.zeros_(module.wo.bias)
        elif isinstance(module, Pix2StructTextAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, Pix2StructConfig)
                else self.config.hidden_size
            )
            key_value_proj_dim = (
                self.config.text_config.d_kv if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            )
            n_heads = (
                self.config.text_config.num_heads
                if isinstance(self.config, Pix2StructConfig)
                else self.config.num_heads
            )

            nn.init.normal_(module.query.weight, mean=0.0, std=factor * ((hidden_size * key_value_proj_dim) ** -0.5))
            nn.init.normal_(module.key.weight, mean=0.0, std=factor * (hidden_size**-0.5))
            nn.init.normal_(module.value.weight, mean=0.0, std=factor * (hidden_size**-0.5))
            nn.init.normal_(module.output.weight, mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                nn.init.normal_(module.relative_attention_bias.weight, mean=0.0, std=factor * ((hidden_size) ** -0.5))
        elif isinstance(module, nn.Embedding):
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, Pix2StructConfig)
                else self.config.hidden_size
            )

            nn.init.normal_(module.weight, mean=0.0, std=factor * ((hidden_size) ** -0.5))
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0
        elif isinstance(module, Pix2StructTextModel):
            hidden_size = (
                self.config.text_config.hidden_size
                if isinstance(self.config, Pix2StructConfig)
                else self.config.hidden_size
            )

            nn.init.normal_(module.lm_head.weight, mean=0.0, std=factor * ((hidden_size) ** -0.5))
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight = nn.init.trunc_normal_(
                module.weight.to(mindspore.float32), mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Pix2StructLayerNorm):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0

    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right with T5->Pix2Struct
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pix2Struct it is usually set to the pad_token_id. "
                "See Pix2Struct docs for more information."
            )

        shifted_input_ids = ops.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        shifted_input_ids = ops.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class Pix2StructVisionModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructVisionConfig
    main_input_name = "flattened_patches"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Pix2StructVisionLayer"]

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = Pix2StructVisionEmbeddings(config)
        self.encoder = Pix2StructVisionEncoder(config)

        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        flattened_patches: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Example:

        ```python
        >>> import requests
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, Pix2StructVisionModel

        >>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with no_grad():
        ...     outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 2048, 768]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")

        if attention_mask is None:
            # check where `flattened_patches` is not 0
            attention_mask = (ops.sum(flattened_patches, dim=-1) != 0).float()

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(flattened_patches)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->Pix2StructText,d_model->hidden_size
class Pix2StructTextDenseGatedActDense(nn.Module):
    def __init__(self, config: Pix2StructTextConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, mindspore.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != mindspore.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class Pix2StructTextLayerFF(nn.Module):
    def __init__(self, config: Pix2StructTextConfig):
        super().__init__()
        self.DenseReluDense = Pix2StructTextDenseGatedActDense(config)

        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    # Copied from transformers.models.t5.modeling_t5.T5LayerFF.forward
    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class Pix2StructTextAttention(nn.Module):
    def __init__(self, config: Pix2StructTextConfig, has_relative_attention_bias=False):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    # Copied from transformers.models.t5.modeling_t5.T5Attention._relative_position_bucket
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.minimum(relative_position, ops.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += ops.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    # Adapted from transformers.models.t5.modeling_t5.T5Attention.compute_bias
    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = ops.arange(query_length, dtype=mindspore.int64)[:, None]
        memory_position = ops.arange(key_length, dtype=mindspore.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def to_projection_shape(states):
            """projection"""
            return ops.transpose(states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim), 1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = to_projection_shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = to_projection_shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = ops.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = to_projection_shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = to_projection_shape(self.query(hidden_states))

        # get key/value states
        key_states = project(
            hidden_states, self.key, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.value, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = ops.matmul(
            query_states, ops.transpose(key_states, 3, 2)
        )  # equivalent of ops.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.n_heads, real_seq_length, key_length), dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape[1] :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = ops.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = ops.matmul(attn_weights, value_states)
        # (batch_size, seq_length, dim)
        attn_output = ops.transpose(attn_output, 1, 2).view(batch_size, -1, self.inner_dim)

        attn_output = self.output(attn_output)

        present_key_value_state = (key_states, value_states) if use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5LayerNorm->Pix2StructLayerNorm,T5Attention->Pix2StructTextAttention,self.SelfAttention->self.attention,config.d_model->config.hidden_size
class Pix2StructTextLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5LayerNorm->Pix2StructLayerNorm,T5Attention->Pix2StructTextAttention,self.EncDecAttention->self.attention,config.d_model->config.hidden_size
class Pix2StructTextLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=False)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class Pix2StructTextBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()

        self.self_attention = Pix2StructTextLayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )

        self.encoder_decoder_attention = Pix2StructTextLayerCrossAttention(config)

        self.mlp = Pix2StructTextLayerFF(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = float(ops.finfo(hidden_states.dtype).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.encoder_decoder_attention(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
                clamp_value = float(ops.finfo(hidden_states.dtype).max) - 1000
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.mlp(hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = float(ops.finfo(hidden_states.dtype).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs


class Pix2StructTextModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructTextConfig
    _no_split_modules = ["Pix2StructTextBlock"]
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layer = nn.ModuleList(
            [Pix2StructTextBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.gradient_checkpointing = False

    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._reorder_cache
    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[mindspore.Tensor, ...], CausalLMOutputWithCrossAttentions]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, Pix2StructTextModel

        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructTextModel.from_pretrained("google/pix2struct-textcaps-base")

        >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        ```
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if attention_mask is None:
            attention_mask = ops.ones(batch_size, mask_seq_length)
        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = ops.ones(
                batch_size, encoder_seq_length, dtype=mindspore.int64
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.layer)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.layer, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class Pix2StructForConditionalGeneration(Pix2StructPreTrainedModel):
    config_class = Pix2StructConfig
    main_input_name = "flattened_patches"
    _tied_weights_keys = ["decoder.lm_head.weight"]

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)

        self.encoder = Pix2StructVisionModel(config.vision_config)
        self.decoder = Pix2StructTextModel(config.text_config)

        self.is_vqa = config.is_vqa

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)

        # update vocab size
        self.config.text_config.vocab_size = new_num_tokens

        return model_embeds

    def get_decoder(self):
        return self.decoder

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        flattened_patches: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        labels: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        Inference:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> # autoregressive generation
        >>> generated_ids = model.generate(**inputs, max_new_tokens=50)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> print(generated_text)
        A stop sign is on a street corner.

        >>> # conditional generation
        >>> text = "A picture of"
        >>> inputs = processor(text=text, images=image, return_tensors="pt", add_special_tokens=False)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=50)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> print(generated_text)
        A picture of a stop sign with a red stop sign
        ```

        Training:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-base")
        >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A stop sign is on the street corner."

        >>> inputs = processor(images=image, return_tensors="pt")
        >>> labels = processor(text=text, return_tensors="pt").input_ids

        >>> # forward pass
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> print(f"{loss.item():.5f}")
        5.94282
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = (
                decoder_attention_mask
                if decoder_attention_mask is not None
                else decoder_input_ids.ne(self.config.pad_token_id).float()
            )
            # Always attend to the first token
            decoder_attention_mask[:, 0] = 1

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if decoder_attention_mask is None:
            decoder_attention_mask = ops.ones_like(input_ids)

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

__all__ = [
        "Pix2StructPreTrainedModel",
        "Pix2StructForConditionalGeneration",
        "Pix2StructVisionModel",
        "Pix2StructTextModel",
]
