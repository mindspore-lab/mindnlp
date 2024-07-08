# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Data2VecAudio model."""
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import mindspore
from mindspore import nn ,ops
from mindspore.common.initializer import Uniform
from mindspore.common.initializer import HeNormal, initializer,Normal
from mindnlp.modules.functional import finfo
from mindnlp.utils import logging
from ...activations import ACT2FN
#from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_data2vec_audio import Data2VecAudioConfig


# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "Data2VecAudioConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[mindspore.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class Data2VecAudioConvLayer(nn.Cell):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            has_bias=config.conv_bias,
            pad_mode='valid',
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim)
        self.activation = ACT2FN[config.feat_extract_activation]

    def construct(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.swapaxes(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.swapaxes(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->Data2VecAudio
class Data2VecAudioPadLayer(nn.Cell):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def construct(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Data2VecAudioPositionalConvLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_pos_kernel_size,
            pad_mode="pad",
            padding=config.conv_pos_kernel_size // 2,
            group=config.num_conv_pos_embedding_groups,
        )

        self.padding = Data2VecAudioPadLayer(config.conv_pos_kernel_size)
        self.activation = ACT2FN[config.feat_extract_activation]
        # no learnable parameters
        self.layer_norm = nn.LayerNorm([config.hidden_size])

    def construct(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Data2VecAudioPositionalConvEmbedding(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.CellList(
            [Data2VecAudioPositionalConvLayer(config) for _ in range(config.num_conv_pos_embeddings)]
        )

    def construct(self, hidden_states):
        hidden_states = hidden_states.swapaxes(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states


class Data2VecAudioFeatureEncoder(nn.Cell):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.CellList(
            [Data2VecAudioConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        )
        self.gradient_checkpointing = False
        self._requires_grad = True

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder._freeze_parameters
    def _freeze_parameters(self):
        for _, param in self.parameters_and_names():
            param.requires_grad = False
        self._requires_grad = False

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder.forward
    def construct(self, input_values):
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->Data2VecAudio
class Data2VecAudioFeatureProjection(nn.Cell):
    def __init__(self, config:Data2VecAudioConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm([config.conv_dim[-1]], epsilon=config.layer_norm_eps)
        self.projection = nn.Dense(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(p= config.feat_proj_dropout)

    def construct(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Data2VecAudio
class Data2VecAudioAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Data2VecAudioConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        key_value_states: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.shape
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))
        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = ops.bmm(attn_probs, value_states)
        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value



DATA2VEC2AUDIO_ATTENTION_CLASSES = {
    "eager": Data2VecAudioAttention,
}


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->Data2VecAudio
class Data2VecAudioFeedForward(nn.Cell):
    def __init__(self, config:Data2VecAudioConfig):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(p = config.activation_dropout)

        self.intermediate_dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p= config.hidden_dropout)

    def construct(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer with Wav2Vec2->Data2VecAudio, WAV2VEC2->DATA2VEC2AUDIO
class Data2VecAudioEncoderLayer(nn.Cell):
    def __init__(self, config:Data2VecAudioConfig):
        super().__init__()
        #self.attention = DATA2VEC2AUDIO_ATTENTION_CLASSES[config._attn_implementation](
        self.attention = Data2VecAudioAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )

        self.dropout = nn.Dropout(p = config.hidden_dropout)
        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.feed_forward = Data2VecAudioFeedForward(config)
        #self.final_layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
    def construct(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder with Wav2Vec2->Data2VecAudio
class Data2VecAudioEncoder(nn.Cell):
    def __init__(self, config:Data2VecAudioConfig):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Data2VecAudioPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p = config.hidden_dropout)
        self.layers = nn.CellList([Data2VecAudioEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
                # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            #attention_mask = attention_mask * np.finfo(np.float32).min
            attention_mask = attention_mask * finfo(hidden_states.dtype, 'min')
            #attention_mask = attention_mask.broadcast_to((
            attention_mask = attention_mask.broadcast_to((
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            ))

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        #deepspeed_zero3_is_enabled = False#is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = ops.rand([])
            #skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            skip_the_layer = self.training and (dropout_probability < self.config.layerdrop)# else False
            if not skip_the_layer: #or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # if self.gradient_checkpointing and self.training:
                #     layer_outputs = self._gradient_checkpointing_func(
                #         layer.__call__,
                #         hidden_states,
                #         attention_mask,
                #         output_attentions,
                #     )
                # else:
                layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

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


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter with Wav2Vec2->Data2VecAudio
class Data2VecAudioAdapter(nn.Cell):
    def __init__(self, config:Data2VecAudioConfig):
        super().__init__()

        # feature dim might need to be down-projected
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Dense(config.hidden_size, config.output_hidden_size)
            #self.proj_layer_norm = nn.LayerNorm([config.output_hidden_size])
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        self.layers = nn.CellList([Data2VecAudioAdapterLayer(config) for _ in range(config.num_adapter_layers)])
        self.layerdrop = config.layerdrop

    def construct(self, hidden_states):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = hidden_states.swapaxes(1, 2)

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer with Wav2Vec2->Data2VecAudio
class Data2VecAudioAdapterLayer(nn.Cell):
    def __init__(self, config:Data2VecAudioConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
            pad_mode='pad',
            has_bias=True,
        )

    def construct(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = ops.glu(hidden_states, axis=1)

        return hidden_states


class Data2VecAudioPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Data2VecAudioConfig
    base_model_prefix = "data2vec_audio"
    main_input_name = "input_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, Data2VecAudioFeatureProjection):
            k = math.sqrt(1 / cell.projection.in_channels)
            cell.projection.weight.set_data(initializer(Uniform(scale=k), cell.projection.weight.shape, cell.projection.weight.dtype))
            cell.projection.bias.set_data(initializer(Uniform(scale=k), cell.projection.bias.shape, cell.projection.bias.dtype))
        elif isinstance(cell, Data2VecAudioPositionalConvLayer):
            cell.conv.bias.set_data(initializer(0, cell.conv.bias.shape, cell.conv.bias.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range), cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm)):

            if hasattr(cell, "bias") and cell.bias is not None:
                cell.bias.set_data(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
            if hasattr(cell, "weight") and cell.weight is not None:
                cell.weight.set_data(
                    initializer("ones", cell.weight.shape, cell.weight.dtype)
                )
        elif isinstance(cell, nn.Conv1d):
            cell.weight.set_data(initializer(HeNormal(), cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                k = math.sqrt(cell.group / (cell.in_channels * cell.kernel_size[0]))
                cell.bias.set_data(initializer(Uniform(scale=k), cell.bias.shape, cell.bias.dtype))

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PreTrainedModel._get_feat_extract_output_lengths with
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[mindspore.Tensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            #return ops.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PreTrainedModel._get_feature_vector_attention_mask
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: mindspore.Tensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(mindspore.int64)
        batch_size = attention_mask.shape[0]
        attention_mask = ops.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(ops.arange(attention_mask.shape[0]), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask



# DATA2VEC_AUDIO_INPUTS_DOCSTRING = r"""
#     Args:
#         input_values (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
#             Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
#             into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
#             soundfile*). To prepare the array into *input_values*, the [`AutoProcessor`] should be used for padding and
#             conversion into a tensor of type *mindspore.Tensor*. See [`Wav2Vec2Processor.__call__`] for details.
#         attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
#             1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#             [What are attention masks?](../glossary#attention-mask)

#             <Tip warning={true}>

#             `attention_mask` should be passed if the corresponding processor has `config.return_attention_mask ==
#             True`, which is the case for all pre-trained Data2Vec Audio models. Be aware that that even with
#             `attention_mask`, zero-padded inputs will have slightly different outputs compared to non-padded inputs
#             because there are more than one convolutional layer in the positional encodings. For a more detailed
#             explanation, see [here](https://github.com/huggingface/transformers/issues/25621#issuecomment-1713759349).

#             </Tip>

#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# """



class Data2VecAudioModel(Data2VecAudioPreTrainedModel):
    def __init__(self, config: Data2VecAudioConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Data2VecAudioFeatureEncoder(config)
        self.feature_projection = Data2VecAudioFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed= mindspore.Parameter(initializer(Uniform(), (config.hidden_size,), mindspore.float32),name ='masked_spec_embed')
        self.encoder = Data2VecAudioEncoder(config)
        self.adapter = Data2VecAudioAdapter(config) if config.add_adapter else None
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: mindspore.Tensor,
        mask_time_indices: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.shape

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = mindspore.Tensor(mask_time_indices, dtype=mindspore.bool_)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = mindspore.Tensor(mask_feature_indices, dtype=mindspore.bool_)
            mask_feature_indices = mask_feature_indices[:, None].broadcast_to((-1, sequence_length, -1))
            hidden_states[mask_feature_indices] = 0
        return hidden_states


    def construct(
        self,
        input_values: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        mask_time_indices: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.swapaxes(1, 2)
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel):
    def __init__(self, config:Data2VecAudioConfig):
        super().__init__(config)
        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(p = config.final_dropout)
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Dense(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()


    def construct(
        self,
        input_values: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            labels = labels.astype(mindspore.int32)
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
            attention_mask = (
                attention_mask if attention_mask is not None else ops.ones_like(input_values, dtype=mindspore.int64)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(mindspore.int64)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            log_probs = ops.log_softmax(logits, axis=-1).swapaxes(0, 1)
            loss,log_alpha = ops.ctc_loss(
                    log_probs,
                    labels,#flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )



class Data2VecAudioForSequenceClassification(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Data2VecAudio adapters (config.add_adapter=True)"
            )
        self.data2vec_audio = Data2VecAudioModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = mindspore.Parameter(ops.ones(num_layers) / num_layers)
            #self.layer_weights = mindspore.Parameter(ops.ones(num_layers) / num_layers,'layer_weights')
        self.projector = nn.Dense(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Dense(config.classifier_proj_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for _, param in self.data2vec_audio.parameters_and_names():
            param.requires_grad = False

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward with wav2vec2->data2vec_audio
    def construct(
        self,
        input_values: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = ops.stack(hidden_states, axis=1)
            norm_weights = ops.softmax(self.layer_weights, axis=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(axis=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(axis=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(axis=1) / padding_mask.sum(axis=1).view(-1, 1)

        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = ops.cross_entropy(input=logits.view(-1, self.config.num_labels), target=labels.view(-1))
            #loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class Data2VecAudioForAudioFrameClassification(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Data2VecAudio adapters"
                " (config.add_adapter=True)"
            )
        self.data2vec_audio = Data2VecAudioModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = mindspore.Parameter(ops.ones(num_layers) / num_layers,'layer_weights')
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.data2vec_audio.get_parameters():
            param.requires_grad = False
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.forward with wav2vec2->data2vec_audio
    def construct(
        self,
        input_values: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = ops.stack(hidden_states, axis=1)
            norm_weights = ops.softmax(self.layer_weights, axis=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(axis=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.astype(mindspore.int32)
            loss= ops.cross_entropy(input=logits.view(-1, self.num_labels), target=ops.argmax(labels.view(-1, self.num_labels),dim = 1))
            #loss = loss_fct(logits.view(-1, self.num_labels), ops.argmax(labels.view(-1, self.num_labels), axis=1))
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss
class AMSoftmaxLoss(nn.Cell):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = mindspore.Parameter(ops.randn(input_dim, num_labels), requires_grad=True,name ='weight')
        #self.loss = ops.cross_entropy()


    def construct(self, hidden_states, labels):
        labels = labels.flatten()
        #weight = ops.normalize(self.weight, axis=0)
        weight = self.weight / ops.norm(self.weight, dim=0, keepdim=True)
        #hidden_states = ops.normalize(hidden_states, axis=1)
        hidden_states = hidden_states / ops.norm(hidden_states, dim=1, keepdim=True)
        cos_theta = ops.mm(hidden_states, weight)
        psi = cos_theta - self.margin
        onehot = ops.one_hot(labels, self.num_labels)
        logits = self.scale * ops.where(onehot.bool(), psi, cos_theta)
        loss = ops.cross_entropy(logits, labels)

        return loss


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer
class TDNNLayer(nn.Cell):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]
        self.kernel = nn.Dense(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # for backward compatibility, we keep nn.Dense but call F.conv1d for speed up
        # hidden_states = hidden_states.swapaxes(1, 2)
        # weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).swapaxes(1, 2)
        # hidden_states = ops.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        # hidden_states = hidden_states.swapaxes(1, 2)

        # hidden_states = self.activation(hidden_states)
        # return hidden_states
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = ops.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Data2VecAudioForXVector(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.data2vec_audio = Data2VecAudioModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = mindspore.Parameter(ops.ones(num_layers) / num_layers,name = 'layer_weights')
        self.projector = nn.Dense(config.hidden_size, config.tdnn_dim[0])
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.CellList(tdnn_layers)
        self.feature_extractor = nn.Dense(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        self.classifier = nn.Dense(config.xvector_output_dim, config.xvector_output_dim)
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for _,param in self.data2vec_audio.parameters_and_names():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[mindspore.Tensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)
        return input_lengths

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector.forward with wav2vec2->data2vec_audio
    def construct(
        self,
        input_values: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = ops.stack(hidden_states, axis=1)
            norm_weights = ops.softmax(self.layer_weights, axis=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(axis=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(axis=1)
            #std_features = hidden_states.std(axis=1)
            std_features = ops.std(hidden_states, axis=1, keepdims=True).squeeze(1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(axis=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(axis=0))
                std_features.append(hidden_states[i, :length].std(axis=0))
            mean_features = ops.stack(mean_features)
            std_features = ops.stack(std_features)
        statistic_pooling = ops.cat([mean_features, std_features], axis=-1)

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
__all__ =[
        "Data2VecAudioForAudioFrameClassification",
        "Data2VecAudioForCTC",
        "Data2VecAudioForSequenceClassification",
        "Data2VecAudioForXVector",
        "Data2VecAudioModel",
        "Data2VecAudioPreTrainedModel",]
